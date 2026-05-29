"""
Quantile Regression Averaging combining LEAR + DDNN point forecasts.

QRA strategy (Uniejewski & Weron 2021):
  At each horizon h and quantile level tau, fit a quantile regression:
      Q_tau(y_h | f_LEAR_h, f_DDNN_h) = beta_0 + beta_1*LEAR + beta_2*DDNN

  - Training set:  first 30 days of test period (calibration slice)
  - Evaluation:    remaining ~270 days

Predicted quantiles are then read off the fitted regression at each
new (LEAR, DDNN) pair.

Outputs:
  results/qra_<area>_<market>_quantiles.npz  (n_eval, H, K) prediction matrix
  results/qra_metrics.json                    pooled + per-horizon CRPS, etc.
"""
from __future__ import annotations
import json, os, sys
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
CALIB_END  = pd.Timestamp("2023-03-30 23:30:00")
EVAL_START = pd.Timestamp("2023-03-31 00:00:00")
EVAL_END   = pd.Timestamp("2023-12-30 23:30:00")
QS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def load_preds(area: str, market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (LEAR_df, DDNN_mu_df) aligned on common index."""
    lear = pd.read_csv(os.path.join(RES, f"lear_{area}_{market}_preds.csv"),
                       parse_dates=[0], index_col=0)
    ddnn = pd.read_csv(os.path.join(RES, f"ddnn_{area}_{market}_mu.csv"),
                       parse_dates=[0], index_col=0)
    common = lear.index.intersection(ddnn.index)
    return lear.loc[common], ddnn.loc[common]


def run_one(area: str, market: str):
    print(f"\n=========== {area} {market} QRA ===========")
    lear_df, ddnn_df = load_preds(area, market)
    Y = build_targets(load_area(area)[market], HORIZON)
    common = lear_df.index.intersection(Y.dropna().index)
    lear_df = lear_df.loc[common]; ddnn_df = ddnn_df.loc[common]; Yc = Y.loc[common]

    P_lear = lear_df.values  # (n, H)
    P_ddnn = ddnn_df.values
    Y_arr = Yc.values

    # Splits
    idx = common
    mask_cal  = (idx >= TEST_START) & (idx <= CALIB_END)
    mask_eval = (idx >= EVAL_START) & (idx <= EVAL_END)

    # Fit one QRA per horizon
    n_eval = mask_eval.sum()
    Q_pred = np.zeros((n_eval, HORIZON, len(QS)))
    for h in range(HORIZON):
        F_cal = np.column_stack([P_lear[mask_cal, h], P_ddnn[mask_cal, h]])
        y_cal = Y_arr[mask_cal, h]
        qra = QRA(qs=QS.tolist(), alpha=0.0).fit(F_cal, y_cal)
        F_eval = np.column_stack([P_lear[mask_eval, h], P_ddnn[mask_eval, h]])
        Q_pred[:, h, :] = qra.predict(F_eval)

    Y_eval = Y_arr[mask_eval]
    np.savez(os.path.join(RES, f"qra_{area}_{market}_quantiles.npz"),
             quantiles=Q_pred, y=Y_eval, qs=QS)

    # Metrics
    pin = pinball_loss(Y_eval, Q_pred, QS)
    crps = crps_quantile(Y_eval, Q_pred, QS)
    # PI from QRA: nominal 0.9 -> use q05/q95, nominal 0.8 -> q10/q90, etc.
    pi_map = {0.1: (0.05, 0.95), 0.2: (0.10, 0.90), 0.5: (0.25, 0.75)}
    cov = {}; widths = {}; wsc = {}
    for a, (lo_q, hi_q) in pi_map.items():
        li = int(np.argmin(np.abs(QS - lo_q)))
        ui = int(np.argmin(np.abs(QS - hi_q)))
        lo = Q_pred[..., li]; hi = Q_pred[..., ui]
        cov[f"alpha_{a:.2f}"] = coverage(Y_eval, lo, hi)
        widths[f"alpha_{a:.2f}"] = float(np.mean(hi - lo))
        wsc[f"alpha_{a:.2f}"] = winkler_score(Y_eval, lo, hi, alpha=a)
    rmse_med = float(np.sqrt(((Q_pred[..., 3] - Y_eval) ** 2).mean()))  # q50 as point
    mae_med = float(np.abs(Q_pred[..., 3] - Y_eval).mean())

    print(f"  pinball={pin:.3f}  CRPS={crps:.3f}  Q50 RMSE={rmse_med:.3f}")
    print(f"  cov90={cov['alpha_0.10']:.3f}  cov80={cov['alpha_0.20']:.3f}")
    return {
        "n_eval": int(n_eval),
        "pinball": pin, "CRPS": crps,
        "Q50_RMSE": rmse_med, "Q50_MAE": mae_med,
        "coverage": cov, "mean_width": widths, "winkler": wsc,
    }


def main():
    out = {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            try:
                out[f"{area}_{market}"] = run_one(area, market)
            except FileNotFoundError as e:
                print(f"  [skip] {area} {market}: {e}")
    with open(os.path.join(RES, "qra_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== QRA summary ===")
    for k, v in out.items():
        print(f"  {k}  pinball={v['pinball']:.3f} CRPS={v['CRPS']:.3f} "
              f"cov80={v['coverage']['alpha_0.20']:.3f} cov90={v['coverage']['alpha_0.10']:.3f}")


if __name__ == "__main__":
    main()
