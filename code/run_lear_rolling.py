"""
Run LEAR with rolling-window recalibration on JEPX Tokyo & Kansai.

For each (area, market) ∈ {TK, KS} × {DA, IM}:
  - Sliding 365-day training window
  - Weekly recalibration (refit every 7 days)
  - 48-step direct multi-output forecast
  - Test window: 2023-03-01 to 2023-12-30
  - LEAR features: lagged y + lagged exogenous + slot/dow dummies (~75 features)

Produces:
  results/lear_<area>_<market>_preds.csv  -- predictions
  results/lear_metrics.json                -- overall + per-horizon metrics
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_area, EXOG_COLS
from feature_build import lear_features, build_targets, HORIZON
from models_lear import LEAR
from rolling_eval import rolling_predict, RollingConfig

OUT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a/results"
os.makedirs(OUT, exist_ok=True)

TEST_START = pd.Timestamp("2023-03-01 00:00:00")
TEST_END   = pd.Timestamp("2023-12-30 23:30:00")


def fit_predict_lear(X_tr: pd.DataFrame, Y_tr: pd.DataFrame, X_te: pd.DataFrame) -> np.ndarray:
    """Fit LEAR on training block; return (n_test, HORIZON) predictions."""
    lear = LEAR(horizon=HORIZON).fit(X_tr, Y_tr)
    return lear.predict(X_te)


def weekly_naive_pred(y: pd.Series, X_te_idx: pd.DatetimeIndex) -> np.ndarray:
    P = np.zeros((len(X_te_idx), HORIZON))
    for h in range(1, HORIZON + 1):
        P[:, h - 1] = y.shift(336 - h).reindex(X_te_idx).values
    return P


def run_one(area: str, market: str) -> dict:
    print(f"\n=========== {area} {market} ===========")
    df = load_area(area)
    y = df[market]
    exog = df[EXOG_COLS]
    X = lear_features(y, exog)
    Y = build_targets(y, HORIZON)

    # Run rolling LEAR
    print("[LEAR rolling]")
    test_idx, P_lear = rolling_predict(
        X, Y, TEST_START, TEST_END,
        fit_predict_lear,
        RollingConfig(train_window_days=365, recal_step_days=7, horizon=HORIZON),
        verbose=True,
    )
    # Save predictions
    df_pred = pd.DataFrame(P_lear, index=test_idx,
                           columns=[f"h{h}" for h in range(1, HORIZON + 1)])
    df_pred.to_csv(os.path.join(OUT, f"lear_{area}_{market}_preds.csv"))

    # Weekly-naive baseline
    P_wn = weekly_naive_pred(y, test_idx)

    # Realized
    Y_actual_df = Y.reindex(test_idx)
    Y_actual = Y_actual_df.values
    # Mask rows where any target is NaN
    valid = ~np.isnan(Y_actual).any(axis=1) & ~np.isnan(P_lear).any(axis=1) & ~np.isnan(P_wn).any(axis=1)
    Y_actual_v = Y_actual[valid]; P_lear_v = P_lear[valid]; P_wn_v = P_wn[valid]
    n_valid = int(valid.sum())

    # Metrics
    def metrics(P, name):
        rmse = float(np.sqrt(((P - Y_actual_v) ** 2).mean()))
        mae  = float(np.abs(P - Y_actual_v).mean())
        per_h_rmse = np.sqrt(((P - Y_actual_v) ** 2).mean(axis=0)).tolist()
        per_h_mae  = np.abs(P - Y_actual_v).mean(axis=0).tolist()
        return {"name": name, "RMSE": rmse, "MAE": mae,
                "RMSE_per_horizon": per_h_rmse, "MAE_per_horizon": per_h_mae}

    m_wn   = metrics(P_wn_v, "WeeklyNaive")
    m_lear = metrics(P_lear_v, "LEAR")
    m_lear["rRMSE"] = m_lear["RMSE"] / m_wn["RMSE"]
    m_lear["rMAE"]  = m_lear["MAE"]  / m_wn["MAE"]

    print(f"\n  WeeklyNaive  RMSE={m_wn['RMSE']:.3f}  MAE={m_wn['MAE']:.3f}")
    print(f"  LEAR         RMSE={m_lear['RMSE']:.3f}  MAE={m_lear['MAE']:.3f}  "
          f"rRMSE={m_lear['rRMSE']:.3f}  rMAE={m_lear['rMAE']:.3f}")

    return {
        "area": area, "market": market,
        "n_test_issuance": int(len(test_idx)),
        "n_valid": n_valid,
        "WeeklyNaive": m_wn,
        "LEAR": m_lear,
    }


def main():
    results = {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            key = f"{area}_{market}"
            results[key] = run_one(area, market)
    with open(os.path.join(OUT, "lear_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== Final Summary ===")
    print(f"{'area_market':<10} {'n_valid':>8} {'WN RMSE':>9} {'LEAR RMSE':>10} {'rRMSE':>7} {'rMAE':>7}")
    for k, v in results.items():
        print(f"{k:<10} {v['n_valid']:>8} {v['WeeklyNaive']['RMSE']:>9.3f} "
              f"{v['LEAR']['RMSE']:>10.3f} {v['LEAR']['rRMSE']:>7.3f} {v['LEAR']['rMAE']:>7.3f}")


if __name__ == "__main__":
    main()
