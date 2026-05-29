"""
Apply Model Confidence Set (Hansen-Lunde-Nason 2011) across all Path A
point forecasters for each (area, market) pair.

Models included (loaded from saved prediction CSVs / NPZs):
  - WeeklyNaive baseline   (computed on the fly)
  - LEAR                    (lear_<area>_<market>_preds.csv)
  - DDNN-Normal (mu)        (ddnn_<area>_<market>_mu.csv)
  - DDNN-JSU (Q50)          (ddnn_jsu_<area>_<market>_quantiles.npz, optional)
  - N-HiTS                  (nhits_<area>_<market>_preds.csv, optional)
  - QRA (Q50)               (qra_<area>_<market>_quantiles.npz, optional)

For each model the pooled squared error and absolute error are computed on
the common test issuance times. MCS is run at alpha in {0.10, 0.25}.

Saves results/mcs_results.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_area
from feature_build import build_targets, HORIZON
from mcs import model_confidence_set

ROOT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a"
RES = os.path.join(ROOT, "results")


def collect_predictions(area: str, market: str) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, np.ndarray]]:
    """Return (common_index, Y_actual, dict[model_name -> P]) where P is (n, H)."""
    df = load_area(area); y = df[market]; Y_full = build_targets(y, HORIZON)
    preds: dict[str, np.ndarray | pd.DataFrame] = {}

    # LEAR
    p = os.path.join(RES, f"lear_{area}_{market}_preds.csv")
    if os.path.exists(p):
        preds["LEAR"] = pd.read_csv(p, parse_dates=[0], index_col=0)

    # DDNN-Normal mu
    p = os.path.join(RES, f"ddnn_{area}_{market}_mu.csv")
    if os.path.exists(p):
        preds["DDNN-Normal"] = pd.read_csv(p, parse_dates=[0], index_col=0)

    # N-HiTS
    p = os.path.join(RES, f"nhits_{area}_{market}_preds.csv")
    if os.path.exists(p):
        preds["N-HiTS"] = pd.read_csv(p, parse_dates=[0], index_col=0)

    # Build common index across all available DataFrames
    if not preds:
        raise RuntimeError(f"No prediction files found for {area} {market}")
    common = list(preds.values())[0].index
    for v in preds.values():
        common = common.intersection(v.index)
    common = common.intersection(Y_full.dropna().index)

    # DDNN-JSU Q50 (from NPZ); align to common index
    p = os.path.join(RES, f"ddnn_jsu_{area}_{market}_quantiles.npz")
    if os.path.exists(p):
        arr = np.load(p, allow_pickle=True)
        q_idx = pd.DatetimeIndex(arr["index"])
        qs = arr["qs"]; Q = arr["quantiles"]
        q50_i = int(np.argmin(np.abs(qs - 0.5)))
        q50_df = pd.DataFrame(Q[:, :, q50_i], index=q_idx,
                              columns=[f"h{h}" for h in range(1, HORIZON+1)])
        preds["DDNN-JSU"] = q50_df.loc[q50_df.index.intersection(common)]
        common = common.intersection(q50_df.index)

    # QRA Q50
    p = os.path.join(RES, f"qra_{area}_{market}_quantiles.npz")
    if os.path.exists(p):
        arr = np.load(p, allow_pickle=True)
        qs = arr["qs"]; Q = arr["quantiles"]
        # QRA was saved without index — it covers EVAL period (2023-03-31..)
        # Build an index matching common from the EVAL start
        eval_start = pd.Timestamp("2023-03-31 00:00:00")
        # Generate the right number of timestamps within common
        candidate_idx = pd.DatetimeIndex([t for t in common if t >= eval_start])
        if len(candidate_idx) == Q.shape[0]:
            q50_i = int(np.argmin(np.abs(qs - 0.5)))
            qra_df = pd.DataFrame(Q[:, :, q50_i], index=candidate_idx,
                                   columns=[f"h{h}" for h in range(1, HORIZON+1)])
            preds["QRA-Q50"] = qra_df
            # Restrict common to the QRA eval window for fair comparison
            common = candidate_idx
        else:
            print(f"  [warn QRA] mismatch n={Q.shape[0]} candidate={len(candidate_idx)}; skipping QRA")

    # Now slice everything to common
    out: dict[str, np.ndarray] = {}
    for name, p_df in preds.items():
        out[name] = p_df.loc[common].values

    # WeeklyNaive baseline (computed for the same common index)
    P_wn = np.zeros((len(common), HORIZON))
    for h in range(1, HORIZON + 1):
        P_wn[:, h - 1] = y.shift(336 - h).reindex(common).values
    out["WeeklyNaive"] = P_wn

    Y_actual = Y_full.loc[common].values
    valid = ~np.isnan(Y_actual).any(axis=1)
    for k in out:
        valid &= ~np.isnan(out[k]).any(axis=1)
    Y_actual = Y_actual[valid]
    out = {k: v[valid] for k, v in out.items()}
    common = common[valid]
    return common, Y_actual, out


def run_mcs_for_pair(area: str, market: str, n_boot: int = 1000):
    print(f"\n========== MCS for {area} {market} ==========")
    common, Y_actual, preds = collect_predictions(area, market)
    print(f"  models: {list(preds.keys())}")
    print(f"  common n_test: {len(common)}")
    sq_losses = {k: ((v - Y_actual) ** 2).mean(axis=1) for k, v in preds.items()}
    # Pool across (n_test, H) by averaging per row → per-day squared loss
    # MCS uses one loss per period
    res = model_confidence_set(sq_losses, alphas=[0.10, 0.25], n_boot=n_boot)
    # Compute per-model mean loss for ranking
    ranking = sorted(((k, float(v.mean())) for k, v in sq_losses.items()), key=lambda x: x[1])
    res["mean_loss_ranking"] = ranking
    print("  Ranking (best → worst):")
    for name, ml in ranking:
        print(f"    {name:<14}  mean_sq_loss={ml:.3f}")
    print(f"  MCS @ alpha=0.10: {res['mcs_10']}")
    print(f"  MCS @ alpha=0.25: {res['mcs_25']}")
    return res


def main():
    out = {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            out[f"{area}_{market}"] = run_mcs_for_pair(area, market)
    with open(os.path.join(RES, "mcs_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n=== Final MCS summary ===")
    for k, v in out.items():
        print(f"{k:<8} MCS_10={v['mcs_10']}  MCS_25={v['mcs_25']}")


if __name__ == "__main__":
    main()
