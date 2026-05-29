"""
Rolling-window QRA combining LEAR + DDNN point forecasts.

Improves on the Month-2 single-30-day calibration by:
  - Sliding 90-day calibration window
  - Refitting the quantile regressions every 7 days
  - Evaluation begins after the first calibration window is filled
    (i.e., from 2023-06 onward)

This addresses the Month-2 under-coverage finding for QRA.

Saves:
  results/qra_rolling_<area>_<market>_quantiles.npz
  results/qra_rolling_metrics.json
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_area
from feature_build import build_targets, HORIZON
from models_qra import QRA
from metrics_prob import pinball_loss, crps_quantile, winkler_score, coverage

ROOT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a"
RES = os.path.join(ROOT, "results")
TEST_START = pd.Timestamp("2023-03-01 00:00:00")
TEST_END   = pd.Timestamp("2023-12-30 23:30:00")
CALIB_DAYS = 90
RECAL_DAYS = 7
QS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def load_paths(area: str, market: str):
    lear = pd.read_csv(os.path.join(RES, f"lear_{area}_{market}_preds.csv"),
                       parse_dates=[0], index_col=0)
    ddnn = pd.read_csv(os.path.join(RES, f"ddnn_{area}_{market}_mu.csv"),
                       parse_dates=[0], index_col=0)
    common = lear.index.intersection(ddnn.index)
    return lear.loc[common], ddnn.loc[common]


def run_one(area: str, market: str):
    print(f"\n=========== {area} {market} QRA-rolling ===========")
    lear_df, ddnn_df = load_paths(area, market)
    Y = build_targets(load_area(area)[market], HORIZON)
    common = lear_df.index.intersection(Y.dropna().index)
    lear_df = lear_df.loc[common]; ddnn_df = ddnn_df.loc[common]; Yc = Y.loc[common]
    P_lear = lear_df.values; P_ddnn = ddnn_df.values; Y_arr = Yc.values

    eval_start = TEST_START + pd.Timedelta(days=CALIB_DAYS)
    issuance = common[(common >= eval_start) & (common <= TEST_END)]
    n_eval = len(issuance)
    print(f"  Eval issuance n={n_eval} (rolling 90-day calibration + 7-day refit)")

    Q_pred = np.zeros((n_eval, HORIZON, len(QS)))
    Y_eval = np.zeros((n_eval, HORIZON))

    # Iterate refit blocks
    block_step = RECAL_DAYS * 48
    pos = 0; block_id = 0
    t0 = time.time()
    while pos < n_eval:
        block_idx = issuance[pos : pos + block_step]
        if len(block_idx) == 0: break
        block_start = block_idx[0]
        cal_start = block_start - pd.Timedelta(days=CALIB_DAYS)
        cal_mask = (common >= cal_start) & (common < block_start)
        cal_idx = common[cal_mask]

        # Fit one QRA per horizon
        for h in range(HORIZON):
            F_cal = np.column_stack([P_lear[cal_mask, h], P_ddnn[cal_mask, h]])
            y_cal = Y_arr[cal_mask, h]
            qra = QRA(qs=QS.tolist(), alpha=0.0).fit(F_cal, y_cal)
            # Predict on this block's slice for horizon h
            block_positions = np.where(np.isin(common, block_idx))[0]
            F_eval = np.column_stack([P_lear[block_positions, h], P_ddnn[block_positions, h]])
            block_eval_global_positions = np.arange(pos, pos + len(block_positions))
            Q_pred[block_eval_global_positions, h, :] = qra.predict(F_eval)
            Y_eval[block_eval_global_positions, h] = Y_arr[block_positions, h]
        if block_id % 5 == 0:
            elapsed = time.time() - t0
            print(f"    block {block_id:02d}  start={block_start}  cal_n={cal_mask.sum()}  elapsed={elapsed:.0f}s")

        pos += block_step; block_id += 1

    np.savez(os.path.join(RES, f"qra_rolling_{area}_{market}_quantiles.npz"),
             quantiles=Q_pred, y=Y_eval, qs=QS,
             index=np.array(issuance, dtype="datetime64[ns]"))

    # Metrics
    pin = pinball_loss(Y_eval, Q_pred, QS)
    crps = crps_quantile(Y_eval, Q_pred, QS)
    cov = {}; widths = {}; wsc = {}
    pi_map = {0.1: (0.05, 0.95), 0.2: (0.10, 0.90), 0.5: (0.25, 0.75)}
    for a, (lo_q, hi_q) in pi_map.items():
        li = int(np.argmin(np.abs(QS - lo_q)))
        ui = int(np.argmin(np.abs(QS - hi_q)))
        lo = Q_pred[..., li]; hi = Q_pred[..., ui]
        cov[f"alpha_{a:.2f}"] = coverage(Y_eval, lo, hi)
        widths[f"alpha_{a:.2f}"] = float(np.mean(hi - lo))
        wsc[f"alpha_{a:.2f}"] = winkler_score(Y_eval, lo, hi, alpha=a)
    rmse_med = float(np.sqrt(((Q_pred[..., 3] - Y_eval) ** 2).mean()))
    print(f"  pinball={pin:.3f}  CRPS={crps:.3f}  Q50 RMSE={rmse_med:.3f}")
    print(f"  cov90={cov['alpha_0.10']:.3f}  cov80={cov['alpha_0.20']:.3f}")
    return {"n_eval": int(n_eval),
            "pinball": pin, "CRPS": crps,
            "Q50_RMSE": rmse_med,
            "coverage": cov, "mean_width": widths, "winkler": wsc}


def main():
    out_path = os.path.join(RES, "qra_rolling_metrics.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            key = f"{area}_{market}"
            if key in out and os.path.exists(os.path.join(RES, f"qra_rolling_{area}_{market}_quantiles.npz")):
                print(f"  [skip {key}]")
                continue
            out[key] = run_one(area, market)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
    print("\n=== QRA-rolling summary ===")
    for k, v in out.items():
        print(f"  {k}  CRPS={v['CRPS']:.3f}  Q50_RMSE={v['Q50_RMSE']:.3f}  cov90={v['coverage']['alpha_0.10']:.3f}")


if __name__ == "__main__":
    main()
