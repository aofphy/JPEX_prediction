"""
Run DDNN-Normal under rolling-window protocol on 4 (area, market) pairs.

For each block (weekly refit on 365-day window):
  - Fit DDNN-Normal on the LEAR feature set (77 features) for ~25 epochs
    with patience-5 early stopping on the validation slice (last 10% of train).
  - Produce (mu, sigma) predictions for the next 7 days of test issuance.
  - Convert to quantiles at standard levels {0.1, 0.5, 0.9} for storage and
    metric computation.

Saves:
  results/ddnn_<area>_<market>_mu.csv     -- point predictions (n_test, 48)
  results/ddnn_<area>_<market>_sigma.csv  -- predicted std (n_test, 48)
  results/ddnn_metrics.json               -- pooled + per-horizon metrics
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_area, EXOG_COLS
from feature_build import lear_features, build_targets, HORIZON
from models_ddnn import DDNNNormal, train_ddnn
from rolling_eval import rolling_predict, RollingConfig
from metrics_prob import crps_normal, winkler_score, coverage

OUT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a/results"
os.makedirs(OUT, exist_ok=True)

TEST_START = pd.Timestamp("2023-03-01 00:00:00")
TEST_END   = pd.Timestamp("2023-12-30 23:30:00")
torch.manual_seed(0); np.random.seed(0)


def make_fit_predict(hidden=256, dropout=0.2, lr=1e-3, wd=1e-5, epochs=25):
    """Return a fit_predict callable that returns concatenated (mu, sigma).

    To match the rolling_predict interface (which expects a (n, HORIZON) array),
    we return (n, 2*HORIZON) where columns 0..47 are mu and 48..95 are sigma.
    Caller splits.
    """
    def fit_predict(X_tr: pd.DataFrame, Y_tr: pd.DataFrame, X_te: pd.DataFrame) -> np.ndarray:
        # standardize features
        Xtr_arr = X_tr.values.astype(np.float32)
        Xte_arr = X_te.values.astype(np.float32)
        Ytr_arr = Y_tr.values.astype(np.float32)
        sc = StandardScaler().fit(Xtr_arr)
        Xtr_s = sc.transform(Xtr_arr).astype(np.float32)
        Xte_s = sc.transform(Xte_arr).astype(np.float32)
        n_val = max(48, int(0.1 * len(Xtr_s)))
        model = DDNNNormal(in_dim=Xtr_s.shape[1], hidden=hidden,
                            horizon=HORIZON, dropout=dropout)
        model, _ = train_ddnn(
            model, Xtr_s[:-n_val], Ytr_arr[:-n_val],
            Xtr_s[-n_val:], Ytr_arr[-n_val:],
            epochs=epochs, batch=512, lr=lr, wd=wd, distribution="normal",
        )
        model.eval()
        with torch.no_grad():
            mu, sigma = model(torch.tensor(Xte_s))
        mu_np = mu.cpu().numpy(); sigma_np = sigma.cpu().numpy()
        return np.concatenate([mu_np, sigma_np], axis=1)
    return fit_predict


def run_one(area: str, market: str) -> dict:
    print(f"\n=========== {area} {market} ===========")
    df = load_area(area); y = df[market]; exog = df[EXOG_COLS]
    X = lear_features(y, exog)
    Y = build_targets(y, HORIZON)
    valid = X.dropna().index.intersection(Y.dropna().index)
    X = X.loc[valid]; Y = Y.loc[valid]

    fit_predict = make_fit_predict()
    test_idx, preds = rolling_predict(
        X, Y, TEST_START, TEST_END, fit_predict,
        RollingConfig(train_window_days=365, recal_step_days=7, horizon=HORIZON),
        verbose=True,
    )
    mu = preds[:, :HORIZON]; sigma = preds[:, HORIZON:]
    # Save
    pd.DataFrame(mu, index=test_idx,
                 columns=[f"h{h}" for h in range(1, HORIZON+1)]
                 ).to_csv(os.path.join(OUT, f"ddnn_{area}_{market}_mu.csv"))
    pd.DataFrame(sigma, index=test_idx,
                 columns=[f"h{h}" for h in range(1, HORIZON+1)]
                 ).to_csv(os.path.join(OUT, f"ddnn_{area}_{market}_sigma.csv"))

    # Metrics on common valid index
    Y_actual = Y.reindex(test_idx).values
    valid_mask = ~np.isnan(Y_actual).any(axis=1) & ~np.isnan(mu).any(axis=1)
    Y_v = Y_actual[valid_mask]; mu_v = mu[valid_mask]; sigma_v = sigma[valid_mask]

    rmse = float(np.sqrt(((mu_v - Y_v) ** 2).mean()))
    mae  = float(np.abs(mu_v - Y_v).mean())
    crps = crps_normal(Y_v, mu_v, sigma_v)
    from scipy.stats import norm
    z90 = norm.ppf(0.95); z80 = norm.ppf(0.9); z50 = norm.ppf(0.75)
    cov_pi = {}
    width_pi = {}
    ws_pi = {}
    for a, z in [(0.1, z90), (0.2, z80), (0.5, z50)]:
        lo = mu_v - z * sigma_v; hi = mu_v + z * sigma_v
        cov_pi[f"alpha_{a:.2f}"] = coverage(Y_v, lo, hi)
        width_pi[f"alpha_{a:.2f}"] = float(np.mean(hi - lo))
        ws_pi[f"alpha_{a:.2f}"] = winkler_score(Y_v, lo, hi, alpha=a)

    return {"n_test": int(valid_mask.sum()),
            "RMSE": rmse, "MAE": mae, "CRPS": crps,
            "coverage": cov_pi, "mean_width": width_pi, "winkler": ws_pi}


def main():
    out_path = os.path.join(OUT, "ddnn_metrics.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            key = f"{area}_{market}"
            mu_file = os.path.join(OUT, f"ddnn_{area}_{market}_mu.csv")
            if os.path.exists(mu_file) and key in out:
                print(f"  [skip {key}: already completed]")
                continue
            out[key] = run_one(area, market)
            # Save incrementally
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
    print("\n=== DDNN summary ===")
    for k, v in out.items():
        print(f"  {k}  RMSE={v['RMSE']:.3f}  CRPS={v['CRPS']:.3f}  cov80={v['coverage']['alpha_0.20']:.3f}  cov90={v['coverage']['alpha_0.10']:.3f}")


if __name__ == "__main__":
    main()
