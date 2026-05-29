"""TimesFM zero-shot probabilistic forecasting on JEPX.

Uses Google's TimesFM-1.0-200M-pytorch foundation model (~200M params,
decoder-only Transformer) without any fine-tuning. At each test-window
issuance time t, we feed the past 336 half-hour prices as context and the
model produces a 48-step point forecast and 9 quantile forecasts
({0.1, 0.2, ..., 0.9}). We linearly extrapolate to {0.05, 0.95} and linearly
interpolate to {0.25, 0.75} to match the project's seven-quantile grid
{0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}.

No rolling-window training (the model is frozen). Inference is batched.
Saves results/timesfm_<area>_<market>_quantiles.npz and timesfm_metrics.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pandas as pd

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
TFM_QS    = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
BATCH = 16
np.random.seed(0)


def get_model():
    import timesfm, torch
    backend = "gpu" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
    print(f"  loading google/timesfm-1.0-200m-pytorch on {backend}")
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=backend, per_core_batch_size=BATCH, horizon_len=HORIZON,
            context_len=512, num_layers=20, use_positional_embedding=False,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )


def remap_quantiles(tfm_q: np.ndarray) -> np.ndarray:
    """tfm_q: (B, H, 9) at levels TFM_QS. Returns (B, H, 7) at levels QS_TARGET."""
    B, H, _ = tfm_q.shape
    out = np.zeros((B, H, len(QS_TARGET)), dtype=tfm_q.dtype)
    # 0.05: linear extrapolation from 0.1, 0.2
    out[..., 0] = tfm_q[..., 0] - 0.5 * (tfm_q[..., 1] - tfm_q[..., 0])
    out[..., 1] = tfm_q[..., 0]  # 0.10
    # 0.25 between 0.2 and 0.3
    out[..., 2] = 0.5 * (tfm_q[..., 1] + tfm_q[..., 2])
    out[..., 3] = tfm_q[..., 4]  # 0.50
    # 0.75 between 0.7 and 0.8
    out[..., 4] = 0.5 * (tfm_q[..., 6] + tfm_q[..., 7])
    out[..., 5] = tfm_q[..., 8]  # 0.90
    # 0.95: linear extrapolation from 0.8, 0.9
    out[..., 6] = tfm_q[..., 8] + 0.5 * (tfm_q[..., 8] - tfm_q[..., 7])
    # Enforce monotonicity
    out = np.sort(out, axis=-1)
    return out


def run_one(area: str, market: str, tfm):
    print(f"\n=========== {area} {market} TimesFM zero-shot ===========")
    df = load_area(area); y = df[market]
    X = raw_lookback(y, LOOKBACK)
    Y = build_targets(y, HORIZON)
    valid = X.dropna().index.intersection(Y.dropna().index)
    X = X.loc[valid]; Y = Y.loc[valid]
    test_mask = (X.index >= TEST_START) & (X.index <= TEST_END)
    Xt = X.loc[test_mask]; Yt = Y.loc[test_mask]
    n_test = len(Xt)
    print(f"  n_test issuance: {n_test}")
    Q = np.zeros((n_test, HORIZON, len(QS_TARGET)), dtype=np.float32)
    point_all = np.zeros((n_test, HORIZON), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n_test, BATCH):
        # raw_lookback returns columns lag0..lag335 with lag0=current; reverse to
        # chronological order (oldest -> newest) for the pretrained foundation model.
        ctx_batch = [Xt.values[j][::-1].astype(np.float32).copy() for j in range(i, min(i+BATCH, n_test))]
        freq = [0] * len(ctx_batch)
        point_fc, q_fc = tfm.forecast(inputs=ctx_batch, freq=freq)
        # point_fc: (B, H); q_fc: (B, H, 10) with col 0 = mean, cols 1-9 = q0.1..q0.9
        B = len(ctx_batch)
        point_all[i:i+B] = point_fc
        Q[i:i+B] = remap_quantiles(q_fc[..., 1:])  # drop col 0 (mean)
        if (i // BATCH) % 50 == 0:
            print(f"    block {i//BATCH:03d}  done={i+B}/{n_test}  elapsed={time.time()-t0:.0f}s")
    print(f"  total inference: {time.time()-t0:.0f}s")

    np.savez(os.path.join(OUT, f"timesfm_{area}_{market}_quantiles.npz"),
             quantiles=Q, qs=QS_TARGET, index=np.array(Xt.index, dtype="datetime64[ns]"))
    pd.DataFrame(point_all, index=Xt.index,
                 columns=[f"h{h}" for h in range(1, HORIZON+1)]
                 ).to_csv(os.path.join(OUT, f"timesfm_{area}_{market}_preds.csv"))

    Y_arr = Yt.values
    valid_mask = ~np.isnan(Y_arr).any(axis=1)
    Y_v = Y_arr[valid_mask]; Q_v = Q[valid_mask]; P_v = point_all[valid_mask]
    rmse_v = rmse(Y_v, P_v); mae_v = mae(Y_v, P_v)
    P_wn = np.zeros((n_test, HORIZON))
    for h in range(1, HORIZON + 1):
        P_wn[:, h-1] = y.shift(336 - h).reindex(Xt.index).values
    P_wn_v = P_wn[valid_mask]
    rmse_wn = rmse(Y_v, P_wn_v); mae_wn = mae(Y_v, P_wn_v)
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
    tfm = get_model()
    out_path = os.path.join(OUT, "timesfm_metrics.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            key = f"{area}_{market}"
            pred_file = os.path.join(OUT, f"timesfm_{area}_{market}_preds.csv")
            if os.path.exists(pred_file) and key in out:
                print(f"  [skip {key}]")
                continue
            out[key] = run_one(area, market, tfm)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
    print("\n=== TimesFM summary ===")
    for k, v in out.items():
        print(f"  {k}  RMSE={v['RMSE']:.3f}  rRMSE={v['rRMSE']:.3f}  CRPS={v['CRPS']:.3f}  cov90={v['coverage']['alpha_0.10']:.3f}")


if __name__ == "__main__":
    main()
