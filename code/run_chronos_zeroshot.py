"""Chronos zero-shot probabilistic forecasting on JEPX.

Uses Amazon's Chronos-Bolt-small foundation model without any fine-tuning.
At each test-window issuance time t, we feed the past 336 half-hour prices
as the context and the model directly outputs nine quantile forecasts
{0.1, 0.2, ..., 0.9} of length 48. We interpolate/extrapolate to the
project's seven-quantile grid {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95};
the q=0.50 quantile is used as the point forecast.

The Bolt variant uses a direct quantile head (no autoregressive sampling)
and is roughly 100x faster than Chronos-T5 for the same output specification.

No rolling-window training (the model is frozen). We use the same test
window 2023-03-01 to 2023-12-30 and the same per-issuance evaluation as
the other forecasters. Inference is batched.

Saves results/chronos_<area>_<market>_quantiles.npz and chronos_metrics.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from data_loader import load_area
from feature_build import raw_lookback, build_targets, HORIZON, LOOKBACK
from metrics_prob import pinball_loss, crps_quantile, winkler_score, coverage, rmse, mae

OUT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a/results"
TEST_START = pd.Timestamp("2023-03-01 00:00:00")
TEST_END   = pd.Timestamp("2023-12-30 23:30:00")
QS_TARGET = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
BOLT_QS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
BATCH = 32
torch.manual_seed(0); np.random.seed(0)


def get_pipeline():
    from chronos import BaseChronosPipeline
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  loading chronos-bolt-small on {device}")
    return BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small", device_map=device, dtype=torch.float32,
    )


def remap_quantiles(bolt_q: np.ndarray) -> np.ndarray:
    """bolt_q: (B, H, 9) at BOLT_QS. Returns (B, H, 7) at QS_TARGET."""
    out = np.zeros((bolt_q.shape[0], bolt_q.shape[1], len(QS_TARGET)), dtype=bolt_q.dtype)
    out[..., 0] = bolt_q[..., 0] - 0.5 * (bolt_q[..., 1] - bolt_q[..., 0])  # 0.05 extrap
    out[..., 1] = bolt_q[..., 0]                                              # 0.10
    out[..., 2] = 0.5 * (bolt_q[..., 1] + bolt_q[..., 2])                     # 0.25
    out[..., 3] = bolt_q[..., 4]                                              # 0.50
    out[..., 4] = 0.5 * (bolt_q[..., 6] + bolt_q[..., 7])                     # 0.75
    out[..., 5] = bolt_q[..., 8]                                              # 0.90
    out[..., 6] = bolt_q[..., 8] + 0.5 * (bolt_q[..., 8] - bolt_q[..., 7])    # 0.95 extrap
    return np.sort(out, axis=-1)


def predict_block(pipe, ctx_batch: np.ndarray) -> np.ndarray:
    """ctx_batch: (B, L) chronological numpy; returns (B, H, 7) at QS_TARGET."""
    ctx = torch.tensor(ctx_batch.astype(np.float32))
    with torch.no_grad():
        q, mean = pipe.predict_quantiles(
            ctx, prediction_length=HORIZON,
            quantile_levels=BOLT_QS.tolist(),
        )
    # q: (B, H, 9) at BOLT_QS
    return remap_quantiles(q.cpu().numpy())


def run_one(area: str, market: str, pipe):
    print(f"\n=========== {area} {market} Chronos zero-shot ===========")
    df = load_area(area); y = df[market]
    X = raw_lookback(y, LOOKBACK)
    Y = build_targets(y, HORIZON)
    valid = X.dropna().index.intersection(Y.dropna().index)
    X = X.loc[valid]; Y = Y.loc[valid]
    # Test slice
    test_mask = (X.index >= TEST_START) & (X.index <= TEST_END)
    Xt = X.loc[test_mask]; Yt = Y.loc[test_mask]
    n_test = len(Xt)
    print(f"  n_test issuance: {n_test}")
    Q = np.zeros((n_test, HORIZON, len(QS_TARGET)), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n_test, BATCH):
        # raw_lookback returns columns lag0..lag335 with lag0=current; reverse to
        # chronological order (oldest -> newest) which is what the pretrained
        # foundation model expects as context.
        ctx = Xt.values[i:i+BATCH, ::-1].copy()
        Q[i:i+BATCH] = predict_block(pipe, ctx)  # (B, H, 7)
        if (i // BATCH) % 20 == 0:
            print(f"    block {i//BATCH:03d}  done={i+len(ctx)}/{n_test}  elapsed={time.time()-t0:.0f}s")
    print(f"  total inference: {time.time()-t0:.0f}s")

    # Save
    np.savez(os.path.join(OUT, f"chronos_{area}_{market}_quantiles.npz"),
             quantiles=Q, qs=QS_TARGET, index=np.array(Xt.index, dtype="datetime64[ns]"))
    # Also save point (median) as a CSV like the supervised models
    point_pred = Q[..., 3]  # q=0.50 index
    pd.DataFrame(point_pred, index=Xt.index,
                 columns=[f"h{h}" for h in range(1, HORIZON+1)]
                 ).to_csv(os.path.join(OUT, f"chronos_{area}_{market}_preds.csv"))

    # Metrics
    Y_arr = Yt.values
    valid_mask = ~np.isnan(Y_arr).any(axis=1)
    Y_v = Y_arr[valid_mask]; Q_v = Q[valid_mask]
    point_v = Q_v[..., 3]
    rmse_v = rmse(Y_v, point_v); mae_v = mae(Y_v, point_v)
    # WN baseline on same window
    P_wn = np.zeros((n_test, HORIZON))
    for h in range(1, HORIZON + 1):
        P_wn[:, h-1] = y.shift(336 - h).reindex(Xt.index).values
    P_wn_v = P_wn[valid_mask]
    rmse_wn = rmse(Y_v, P_wn_v); mae_wn = mae(Y_v, P_wn_v)
    # Probabilistic
    pin = pinball_loss(Y_v, Q_v, QS_TARGET)
    crps = crps_quantile(Y_v, Q_v, QS_TARGET)
    cov = {}; widths = {}; wsc = {}
    pi_map = {0.1: (0.05, 0.95), 0.2: (0.10, 0.90), 0.5: (0.25, 0.75)}
    for a, (lo_q, hi_q) in pi_map.items():
        li = int(np.argmin(np.abs(QS_TARGET - lo_q)))
        ui = int(np.argmin(np.abs(QS_TARGET - hi_q)))
        lo = Q_v[..., li]; hi = Q_v[..., ui]
        cov[f"alpha_{a:.2f}"] = coverage(Y_v, lo, hi)
        widths[f"alpha_{a:.2f}"] = float(np.mean(hi - lo))
        wsc[f"alpha_{a:.2f}"] = winkler_score(Y_v, lo, hi, alpha=a)

    return {"n_test": int(valid_mask.sum()),
            "RMSE": rmse_v, "MAE": mae_v,
            "WN_RMSE": rmse_wn, "WN_MAE": mae_wn,
            "rRMSE": rmse_v / rmse_wn, "rMAE": mae_v / mae_wn,
            "pinball": pin, "CRPS": crps,
            "coverage": cov, "mean_width": widths, "winkler": wsc}


def main():
    pipe = get_pipeline()
    out_path = os.path.join(OUT, "chronos_metrics.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            key = f"{area}_{market}"
            pred_file = os.path.join(OUT, f"chronos_{area}_{market}_preds.csv")
            if os.path.exists(pred_file) and key in out:
                print(f"  [skip {key}]")
                continue
            out[key] = run_one(area, market, pipe)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
    print("\n=== Chronos summary ===")
    for k, v in out.items():
        print(f"  {k}  RMSE={v['RMSE']:.3f}  rRMSE={v['rRMSE']:.3f}  CRPS={v['CRPS']:.3f}  cov90={v['coverage']['alpha_0.10']:.3f}")


if __name__ == "__main__":
    main()
