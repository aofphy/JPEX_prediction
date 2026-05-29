"""Extended MCS with all single-model forecasters including PatchTST,
iTransformer, Chronos zero-shot, and TimesFM zero-shot (response to R3 X1, X2).

Models included (when their prediction files exist on disk):
  WeeklyNaive, LEAR, DDNN-Normal, DDNN-JSU, N-HiTS,
  PatchTST, iTransformer, Chronos (median), TimesFM (point), QRA-Q50.

Same Hansen-Lunde-Nason MCS procedure as run_mcs.py. Runs at three block
lengths for robustness reporting.
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
BLOCK_LENGTHS = [None, 336.0, 672.0]
BLOCK_LABELS  = ["default_T^(1/3)", "weekly_336", "biweekly_672"]


def collect_predictions(area: str, market: str) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, np.ndarray]]:
    df = load_area(area); y = df[market]; Y_full = build_targets(y, HORIZON)
    preds: dict[str, pd.DataFrame] = {}

    sources = {
        "LEAR":       f"lear_{area}_{market}_preds.csv",
        "DDNN-Normal":f"ddnn_{area}_{market}_mu.csv",
        "N-HiTS":     f"nhits_{area}_{market}_preds.csv",
        "PatchTST":   f"patchtst_{area}_{market}_preds.csv",
        "iTransformer": f"itransformer_{area}_{market}_preds.csv",
        "Chronos":    f"chronos_{area}_{market}_preds.csv",
        "TimesFM":    f"timesfm_{area}_{market}_preds.csv",
    }
    for name, fname in sources.items():
        p = os.path.join(RES, fname)
        if os.path.exists(p):
            preds[name] = pd.read_csv(p, parse_dates=[0], index_col=0)

    common = None
    for df_ in preds.values():
        common = df_.index if common is None else common.intersection(df_.index)
    common = common.intersection(Y_full.dropna().index)

    # DDNN-JSU Q50
    p = os.path.join(RES, f"ddnn_jsu_{area}_{market}_quantiles.npz")
    if os.path.exists(p):
        arr = np.load(p, allow_pickle=True)
        q_idx = pd.DatetimeIndex(arr["index"]); qs = arr["qs"]; Q = arr["quantiles"]
        q50_i = int(np.argmin(np.abs(qs - 0.5)))
        q50_df = pd.DataFrame(Q[:, :, q50_i], index=q_idx,
                              columns=[f"h{h}" for h in range(1, HORIZON+1)])
        preds["DDNN-JSU"] = q50_df
        common = common.intersection(q50_df.index)

    # QRA Q50 (rolling)
    p = os.path.join(RES, f"qra_rolling_{area}_{market}_quantiles.npz")
    if os.path.exists(p):
        arr = np.load(p, allow_pickle=True)
        qs = arr["qs"]; Q = arr["quantiles"]
        if "index" in arr.files:
            q_idx = pd.DatetimeIndex(arr["index"])
            q50_i = int(np.argmin(np.abs(qs - 0.5)))
            qra_df = pd.DataFrame(Q[:, :, q50_i], index=q_idx,
                                   columns=[f"h{h}" for h in range(1, HORIZON+1)])
            preds["QRA-Q50"] = qra_df
            common = common.intersection(qra_df.index)

    out = {name: df_.loc[common].values for name, df_ in preds.items()}

    # WeeklyNaive
    P_wn = np.zeros((len(common), HORIZON))
    for h in range(1, HORIZON + 1):
        P_wn[:, h-1] = y.shift(336 - h).reindex(common).values
    out["WeeklyNaive"] = P_wn

    Y_actual = Y_full.loc[common].values
    valid = ~np.isnan(Y_actual).any(axis=1)
    for k in out:
        valid &= ~np.isnan(out[k]).any(axis=1)
    Y_actual = Y_actual[valid]
    out = {k: v[valid] for k, v in out.items()}
    common = common[valid]
    return common, Y_actual, out


def main():
    summary: dict = {}
    detail: dict = {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            key = f"{area}_{market}"
            print(f"\n========== {key} ==========")
            common, Y_actual, preds = collect_predictions(area, market)
            print(f"  models: {list(preds.keys())}; n_common={len(common)}")
            sq_losses = {k: ((v - Y_actual) ** 2).mean(axis=1) for k, v in preds.items()}
            ranking = sorted(((k, float(v.mean())) for k, v in sq_losses.items()), key=lambda x: x[1])
            detail[key] = {"ranking": ranking, "n_common": int(len(common)), "by_block": {}}
            for mb, label in zip(BLOCK_LENGTHS, BLOCK_LABELS):
                res = model_confidence_set(sq_losses, alphas=[0.10, 0.25], n_boot=1000, mean_block=mb)
                detail[key]["by_block"][label] = {
                    "mcs_10": res["mcs_10"], "mcs_25": res["mcs_25"],
                    "eliminations": res["eliminations"],
                }
                print(f"  block={label:<20}  MCS_10={res['mcs_10']}  MCS_25={res['mcs_25']}")
    with open(os.path.join(RES, "mcs_extended_results.json"), "w") as f:
        json.dump(detail, f, indent=2, default=str)
    print("\nSaved -> mcs_extended_results.json")


if __name__ == "__main__":
    main()
