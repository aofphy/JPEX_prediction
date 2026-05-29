"""
Run N-HiTS under rolling-window protocol on 4 (area, market) pairs.

Uses raw 336-step lookback as the input (no engineered features). Same
weekly-recalibration / 365-day-window protocol as LEAR and DDNN runs.

Saves predictions to results/nhits_<area>_<market>_preds.csv and metrics
to results/nhits_metrics.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_area
from feature_build import raw_lookback, build_targets, HORIZON, LOOKBACK
from models_nhits import NHiTS, train_nhits
from rolling_eval import rolling_predict, RollingConfig
from metrics_prob import rmse, mae

OUT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a/results"
TEST_START = pd.Timestamp("2023-03-01 00:00:00")
TEST_END   = pd.Timestamp("2023-12-30 23:30:00")
torch.manual_seed(0); np.random.seed(0)


def make_fit_predict(hidden=128, dropout=0.1, lr=1e-3, wd=1e-5, epochs=12,
                      pool_sizes=(1, 4, 16)):
    def fit_predict(X_tr: pd.DataFrame, Y_tr: pd.DataFrame, X_te: pd.DataFrame) -> np.ndarray:
        Xtr_arr = X_tr.values.astype(np.float32)
        Xte_arr = X_te.values.astype(np.float32)
        Ytr_arr = Y_tr.values.astype(np.float32)
        sc = StandardScaler().fit(Xtr_arr)
        Xtr_s = sc.transform(Xtr_arr).astype(np.float32)
        Xte_s = sc.transform(Xte_arr).astype(np.float32)
        n_val = max(48, int(0.1 * len(Xtr_s)))
        model = NHiTS(lookback=LOOKBACK, horizon=HORIZON, pool_sizes=list(pool_sizes),
                       hidden=hidden, dropout=dropout)
        model, _ = train_nhits(model, Xtr_s[:-n_val], Ytr_arr[:-n_val],
                                 Xtr_s[-n_val:], Ytr_arr[-n_val:],
                                 epochs=epochs, batch=512, lr=lr, wd=wd)
        model.eval()
        with torch.no_grad():
            yhat = model(torch.tensor(Xte_s)).cpu().numpy()
        return yhat
    return fit_predict


def run_one(area: str, market: str):
    print(f"\n=========== {area} {market} N-HiTS ===========")
    df = load_area(area); y = df[market]
    X = raw_lookback(y, LOOKBACK)
    Y = build_targets(y, HORIZON)
    valid = X.dropna().index.intersection(Y.dropna().index)
    X = X.loc[valid]; Y = Y.loc[valid]

    test_idx, preds = rolling_predict(
        X, Y, TEST_START, TEST_END, make_fit_predict(),
        RollingConfig(train_window_days=365, recal_step_days=7, horizon=HORIZON),
        verbose=True,
    )
    # Save predictions
    pd.DataFrame(preds, index=test_idx,
                 columns=[f"h{h}" for h in range(1, HORIZON+1)]
                 ).to_csv(os.path.join(OUT, f"nhits_{area}_{market}_preds.csv"))

    # Metrics on common valid index
    Y_actual = Y.reindex(test_idx).values
    valid_mask = ~np.isnan(Y_actual).any(axis=1) & ~np.isnan(preds).any(axis=1)
    Y_v = Y_actual[valid_mask]; P_v = preds[valid_mask]
    rmse_v = rmse(Y_v, P_v); mae_v = mae(Y_v, P_v)
    # vs weekly-naive baseline
    P_wn = np.zeros((len(test_idx), HORIZON))
    for h in range(1, HORIZON + 1):
        P_wn[:, h-1] = y.shift(336 - h).reindex(test_idx).values
    P_wn_v = P_wn[valid_mask]
    rmse_wn = rmse(Y_v, P_wn_v); mae_wn = mae(Y_v, P_wn_v)
    per_h_rmse = np.sqrt(((P_v - Y_v) ** 2).mean(axis=0)).tolist()

    return {"n_test": int(valid_mask.sum()),
            "RMSE": rmse_v, "MAE": mae_v,
            "WN_RMSE": rmse_wn, "WN_MAE": mae_wn,
            "rRMSE": rmse_v / rmse_wn, "rMAE": mae_v / mae_wn,
            "RMSE_per_horizon": per_h_rmse}


def main():
    out_path = os.path.join(OUT, "nhits_metrics.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            key = f"{area}_{market}"
            pred_file = os.path.join(OUT, f"nhits_{area}_{market}_preds.csv")
            if os.path.exists(pred_file) and key in out:
                print(f"  [skip {key}: already done]")
                continue
            out[key] = run_one(area, market)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
    print("\n=== N-HiTS summary ===")
    for k, v in out.items():
        print(f"  {k}  RMSE={v['RMSE']:.3f}  rRMSE={v['rRMSE']:.3f}  rMAE={v['rMAE']:.3f}")


if __name__ == "__main__":
    main()
