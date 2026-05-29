"""
DDNN-Johnson-SU rolling-window benchmark on 4 (area, market) pairs.

Johnson SU is a flexible 4-parameter family allowing both heavier tails
and asymmetry — well-suited for electricity prices which often show
right-skewed spike tails.

Saves location (xi) + scale (lambda) + skew (gamma) + tail (delta) per
(issuance, horizon) plus the quantiles at standard levels.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_area, EXOG_COLS
from feature_build import lear_features, build_targets, HORIZON
from models_ddnn import DDNNJohnsonSU, train_ddnn
from rolling_eval import rolling_predict, RollingConfig
from metrics_prob import pinball_loss, crps_quantile, winkler_score, coverage

OUT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a/results"
TEST_START = pd.Timestamp("2023-03-01 00:00:00")
TEST_END   = pd.Timestamp("2023-12-30 23:30:00")
QS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
torch.manual_seed(0); np.random.seed(0)


def make_fit_predict():
    """Return fit_predict that emits (n, HORIZON * |QS|) flattened quantiles."""
    def fit_predict(X_tr: pd.DataFrame, Y_tr: pd.DataFrame, X_te: pd.DataFrame) -> np.ndarray:
        Xtr_arr = X_tr.values.astype(np.float32)
        Xte_arr = X_te.values.astype(np.float32)
        Ytr_arr = Y_tr.values.astype(np.float32)
        sc = StandardScaler().fit(Xtr_arr)
        Xtr_s = sc.transform(Xtr_arr).astype(np.float32)
        Xte_s = sc.transform(Xte_arr).astype(np.float32)
        n_val = max(48, int(0.1 * len(Xtr_s)))
        model = DDNNJohnsonSU(in_dim=Xtr_s.shape[1], hidden=256,
                                horizon=HORIZON, dropout=0.2)
        model, _ = train_ddnn(
            model, Xtr_s[:-n_val], Ytr_arr[:-n_val],
            Xtr_s[-n_val:], Ytr_arr[-n_val:],
            epochs=25, batch=512, lr=1e-3, wd=1e-5, distribution="jsu",
        )
        Q = model.predict_quantiles(torch.tensor(Xte_s), QS)
        # Flatten to (n, HORIZON * K)
        return Q.reshape(Q.shape[0], -1)
    return fit_predict


def run_one(area: str, market: str):
    print(f"\n=========== {area} {market} DDNN-JSU ===========")
    df = load_area(area); y = df[market]; exog = df[EXOG_COLS]
    X = lear_features(y, exog)
    Y = build_targets(y, HORIZON)
    valid = X.dropna().index.intersection(Y.dropna().index)
    X = X.loc[valid]; Y = Y.loc[valid]

    test_idx, preds_flat = rolling_predict(
        X, Y, TEST_START, TEST_END, make_fit_predict(),
        RollingConfig(train_window_days=365, recal_step_days=7, horizon=HORIZON),
        verbose=True,
    )
    K = len(QS)
    n = len(test_idx)
    Q = preds_flat.reshape(n, HORIZON, K)
    np.savez(os.path.join(OUT, f"ddnn_jsu_{area}_{market}_quantiles.npz"),
             quantiles=Q, qs=QS, index=np.array(test_idx, dtype="datetime64[ns]"))

    Y_actual = Y.reindex(test_idx).values
    valid_mask = ~np.isnan(Y_actual).any(axis=1) & ~np.isnan(Q).any(axis=(1, 2))
    Y_v = Y_actual[valid_mask]; Q_v = Q[valid_mask]

    # Q50 as point
    q50_idx = int(np.argmin(np.abs(QS - 0.5)))
    rmse_med = float(np.sqrt(((Q_v[..., q50_idx] - Y_v) ** 2).mean()))
    mae_med = float(np.abs(Q_v[..., q50_idx] - Y_v).mean())
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

    return {"n_test": int(valid_mask.sum()),
            "Q50_RMSE": rmse_med, "Q50_MAE": mae_med,
            "pinball": pin, "CRPS": crps,
            "coverage": cov, "mean_width": widths, "winkler": wsc}


def main():
    out_path = os.path.join(OUT, "ddnn_jsu_metrics.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            key = f"{area}_{market}"
            q_file = os.path.join(OUT, f"ddnn_jsu_{area}_{market}_quantiles.npz")
            if os.path.exists(q_file) and key in out:
                print(f"  [skip {key}: already done]")
                continue
            out[key] = run_one(area, market)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
    print("\n=== DDNN-JSU summary ===")
    for k, v in out.items():
        print(f"  {k}  Q50_RMSE={v['Q50_RMSE']:.3f}  CRPS={v['CRPS']:.3f}  cov80={v['coverage']['alpha_0.20']:.3f}  cov90={v['coverage']['alpha_0.10']:.3f}")


if __name__ == "__main__":
    main()
