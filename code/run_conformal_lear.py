"""
Apply split + adaptive conformal prediction to LEAR rolling-window predictions.

Protocol:
  - Split test set into calibration (first 30 days, ~1440 issuance times)
    and evaluation (remainder, ~270 days, ~12,960 issuance times).
  - Split conformal: compute per-horizon quantile thresholds from
    calibration residuals, apply to evaluation.
  - Adaptive conformal: start from split-conformal thresholds, update
    q_h(t+1) = q_h(t) + gamma * (alpha - 1{outside interval}) per Gibbs
    & Candes (2021). Step gamma = 0.005.

Both produce prediction intervals at alphas in {0.1, 0.2, 0.5} (i.e.,
nominal coverages 90%, 80%, 50%).

Outputs:
  results/conformal_lear_<area>_<market>.json     -- coverage + width tables
  results/conformal_lear_<area>_<market>_intervals.npz  -- (lower, upper) per alpha
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_area, EXOG_COLS
from feature_build import build_targets, HORIZON
from models_conformal import calibrate, predict_intervals, adaptive_conformal_update
from metrics_prob import winkler_score, coverage

ROOT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a"
RES = os.path.join(ROOT, "results")
TEST_START = pd.Timestamp("2023-03-01 00:00:00")
CALIB_END  = pd.Timestamp("2023-03-30 23:30:00")  # 30 days = 1440 issuances
EVAL_START = pd.Timestamp("2023-03-31 00:00:00")
EVAL_END   = pd.Timestamp("2023-12-30 23:30:00")
ALPHAS = [0.5, 0.2, 0.1]   # nominal coverages 0.5, 0.8, 0.9
GAMMA  = 0.005


def run_one(area: str, market: str):
    print(f"\n=========== {area} {market} ===========")
    # Load LEAR predictions
    pred_path = os.path.join(RES, f"lear_{area}_{market}_preds.csv")
    df_pred = pd.read_csv(pred_path, parse_dates=[0], index_col=0)
    # Load actuals
    area_df = load_area(area)
    y = area_df[market]
    Y_full = build_targets(y, HORIZON)
    # Align
    common = df_pred.index.intersection(Y_full.dropna().index)
    df_pred = df_pred.loc[common]; Y = Y_full.loc[common]
    P = df_pred.values        # (n, 48)
    Y_arr = Y.values          # (n, 48)
    # Calibration vs evaluation split
    mask_cal  = (common >= TEST_START) & (common <= CALIB_END)
    mask_eval = (common >= EVAL_START) & (common <= EVAL_END)
    P_cal, P_eval = P[mask_cal], P[mask_eval]
    Y_cal, Y_eval = Y_arr[mask_cal], Y_arr[mask_eval]
    print(f"  Calibration n={len(P_cal)}  Evaluation n={len(P_eval)}")

    # Split conformal: calibrate thresholds, apply to evaluation
    q_dict_split = calibrate(Y_cal, P_cal, alphas=ALPHAS)
    split_results = {}
    for alpha, q_h in q_dict_split.items():
        lo, hi = predict_intervals(P_eval, q_h)
        cov = coverage(Y_eval, lo, hi)
        ws  = winkler_score(Y_eval, lo, hi, alpha=alpha)
        width = float(np.mean(hi - lo))
        split_results[f"alpha_{alpha:.2f}"] = {
            "nominal_coverage": 1 - alpha,
            "empirical_coverage": cov,
            "mean_width": width,
            "winkler_score": ws,
        }
        print(f"  Split conformal nominal={1-alpha:.2f}  empirical_cov={cov:.3f}  "
              f"width={width:.3f}  Winkler={ws:.3f}")

    # Adaptive conformal: update q_h after each issuance
    adaptive_results = {}
    for alpha in ALPHAS:
        q_h = q_dict_split[alpha].copy()
        lo_all = np.zeros_like(P_eval); hi_all = np.zeros_like(P_eval)
        for t in range(len(P_eval)):
            lo_t = P_eval[t] - q_h
            hi_t = P_eval[t] + q_h
            lo_all[t] = lo_t; hi_all[t] = hi_t
            in_interval = ((Y_eval[t] >= lo_t) & (Y_eval[t] <= hi_t)).astype(float)
            q_h = adaptive_conformal_update(q_h, in_interval, alpha=alpha, gamma=GAMMA)
            q_h = np.maximum(q_h, 0.0)
        cov = coverage(Y_eval, lo_all, hi_all)
        ws  = winkler_score(Y_eval, lo_all, hi_all, alpha=alpha)
        width = float(np.mean(hi_all - lo_all))
        adaptive_results[f"alpha_{alpha:.2f}"] = {
            "nominal_coverage": 1 - alpha,
            "empirical_coverage": cov,
            "mean_width": width,
            "winkler_score": ws,
        }
        print(f"  Adaptive conformal nominal={1-alpha:.2f}  empirical_cov={cov:.3f}  "
              f"width={width:.3f}  Winkler={ws:.3f}")

    return {"split": split_results, "adaptive": adaptive_results,
            "n_cal": int(len(P_cal)), "n_eval": int(len(P_eval))}


def main():
    out = {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            out[f"{area}_{market}"] = run_one(area, market)
    with open(os.path.join(RES, "conformal_lear_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    # Compact summary
    print("\n" + "=" * 80)
    print(f"{'area_market':<10} {'method':<10} {'nominal':>8} {'cov':>6} {'width':>8} {'Winkler':>10}")
    print("=" * 80)
    for key, data in out.items():
        for method in ["split", "adaptive"]:
            for ak, vals in data[method].items():
                print(f"{key:<10} {method:<10} {vals['nominal_coverage']:>8.2f} "
                      f"{vals['empirical_coverage']:>6.3f} "
                      f"{vals['mean_width']:>8.3f} {vals['winkler_score']:>10.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
