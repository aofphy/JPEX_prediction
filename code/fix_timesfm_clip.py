"""Apply post-hoc clipping to TimesFM zero-shot outputs (response to occasional
extreme tokenizer outliers in the foundation model).

Foundation-model decoding occasionally produces predictions far outside the
physically reasonable JEPX price range (0--250 JPY/kWh). The fraction of
out-of-range predictions is small (~0.3%) but those outliers dominate
mean-squared-error. We clip predictions to [0, 250] (the JEPX market cap),
which corresponds to what a practitioner would deploy in production.

Re-reads timesfm_<area>_<market>_preds.csv and timesfm_<area>_<market>_quantiles.npz,
clips, recomputes metrics, and overwrites the corresponding JSON entry.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_area
from feature_build import build_targets, HORIZON
from metrics_prob import pinball_loss, crps_quantile, winkler_score, coverage, rmse, mae

RES = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a/results"
CLIP_LO, CLIP_HI = 0.0, 250.0
QS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def run_one(area: str, market: str):
    print(f"\n=== {area} {market} TimesFM clip post-hoc ===")
    # Load raw outputs
    pred_csv = os.path.join(RES, f"timesfm_{area}_{market}_preds.csv")
    q_npz    = os.path.join(RES, f"timesfm_{area}_{market}_quantiles.npz")
    df_p = pd.read_csv(pred_csv, parse_dates=[0], index_col=0)
    arr  = np.load(q_npz, allow_pickle=True)
    Q = arr["quantiles"]; idx = pd.DatetimeIndex(arr["index"])
    n_extreme_p = int(((df_p.values < CLIP_LO) | (df_p.values > CLIP_HI)).sum())
    n_extreme_q = int(((Q < CLIP_LO) | (Q > CLIP_HI)).sum())
    print(f"  clipping point: {n_extreme_p} of {df_p.size} ({100*n_extreme_p/df_p.size:.3f}%)")
    print(f"  clipping quant: {n_extreme_q} of {Q.size} ({100*n_extreme_q/Q.size:.3f}%)")
    point_clipped = np.clip(df_p.values, CLIP_LO, CLIP_HI)
    Q_clipped = np.clip(Q, CLIP_LO, CLIP_HI)
    # Enforce monotonicity after clip
    Q_clipped = np.sort(Q_clipped, axis=-1)
    pd.DataFrame(point_clipped, index=df_p.index, columns=df_p.columns).to_csv(pred_csv)
    np.savez(q_npz, quantiles=Q_clipped, qs=QS, index=np.array(idx, dtype="datetime64[ns]"))

    # Recompute metrics
    df = load_area(area); y = df[market]
    Y_full = build_targets(y, HORIZON)
    common = df_p.index.intersection(Y_full.dropna().index)
    Y_arr = Y_full.loc[common].values
    P_v   = pd.DataFrame(point_clipped, index=df_p.index).loc[common].values
    # Align Q to common
    Q_df_idx = pd.DataFrame(np.arange(len(idx)), index=idx, columns=["pos"]).loc[common, "pos"].values
    Q_v = Q_clipped[Q_df_idx]
    valid = ~np.isnan(Y_arr).any(axis=1) & ~np.isnan(P_v).any(axis=1)
    Y_v = Y_arr[valid]; P_v = P_v[valid]; Q_v = Q_v[valid]
    rmse_v = rmse(Y_v, P_v); mae_v = mae(Y_v, P_v)
    P_wn = np.zeros((len(common), HORIZON))
    for h in range(1, HORIZON+1):
        P_wn[:, h-1] = y.shift(336 - h).reindex(common).values
    P_wn_v = P_wn[valid]
    rmse_wn = rmse(Y_v, P_wn_v); mae_wn = mae(Y_v, P_wn_v)
    pin = pinball_loss(Y_v, Q_v, QS)
    crps = crps_quantile(Y_v, Q_v, QS)
    cov = {}; widths = {}; wsc = {}
    pi_map = {0.1: (0.05, 0.95), 0.2: (0.10, 0.90), 0.5: (0.25, 0.75)}
    for a, (lo_q, hi_q) in pi_map.items():
        li = int(np.argmin(np.abs(QS - lo_q)))
        ui = int(np.argmin(np.abs(QS - hi_q)))
        lo = Q_v[..., li]; hi = Q_v[..., ui]
        cov[f"alpha_{a:.2f}"] = coverage(Y_v, lo, hi)
        widths[f"alpha_{a:.2f}"] = float(np.mean(hi - lo))
        wsc[f"alpha_{a:.2f}"] = winkler_score(Y_v, lo, hi, alpha=a)
    return {"n_test": int(valid.sum()),
            "RMSE": rmse_v, "MAE": mae_v,
            "WN_RMSE": rmse_wn, "WN_MAE": mae_wn,
            "rRMSE": rmse_v / rmse_wn, "rMAE": mae_v / mae_wn,
            "pinball": pin, "CRPS": crps,
            "coverage": cov, "mean_width": widths, "winkler": wsc,
            "clip_range": [CLIP_LO, CLIP_HI],
            "n_clipped_point": n_extreme_p,
            "n_clipped_quant": n_extreme_q}


def main():
    out = {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            out[f"{area}_{market}"] = run_one(area, market)
    with open(os.path.join(RES, "timesfm_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== TimesFM clipped summary ===")
    for k, v in out.items():
        print(f"  {k}  RMSE={v['RMSE']:.3f}  rRMSE={v['rRMSE']:.3f}  CRPS={v['CRPS']:.3f}  cov90={v['coverage']['alpha_0.10']:.3f}")


if __name__ == "__main__":
    main()
