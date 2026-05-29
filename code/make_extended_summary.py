"""Aggregate metrics across all 11 forecasters into a single summary table
for the extended-benchmark revision (response to R-M6 R3 X1/X2).

Reads existing per-model JSON metric files plus the four new model outputs:
  - patchtst_metrics.json
  - itransformer_metrics.json
  - chronos_metrics.json
  - timesfm_metrics.json

Writes:
  - results/extended_summary.csv    (long-format flat table)
  - results/extended_summary.json   (nested per-cell)
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd

RES = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a/results"

MODELS_POINT = [
    ("WeeklyNaive", None),                       # computed inline (rRMSE := 1.0)
    ("LEAR",       "lear_metrics.json"),
    ("DDNN-Normal","ddnn_metrics.json"),
    ("DDNN-JSU",   "ddnn_jsu_metrics.json"),
    ("N-HiTS",     "nhits_metrics.json"),
    ("PatchTST",   "patchtst_metrics.json"),
    ("iTransformer","itransformer_metrics.json"),
    ("Chronos",    "chronos_metrics.json"),
    ("TimesFM",    "timesfm_metrics.json"),
    ("QRA-rolling","qra_rolling_metrics.json"),
]

MODELS_PROB = [
    ("DDNN-Normal","ddnn_metrics.json"),
    ("DDNN-JSU",   "ddnn_jsu_metrics.json"),
    ("QRA-rolling","qra_rolling_metrics.json"),
    ("LEAR+cf",    "conformal_lear_metrics.json"),
    ("Chronos",    "chronos_metrics.json"),
    ("TimesFM",    "timesfm_metrics.json"),
]


def load_json(name: str | None) -> dict | None:
    if name is None: return None
    path = os.path.join(RES, name)
    if not os.path.exists(path): return None
    return json.load(open(path))


def fmt(x): return f"{x:.3f}" if isinstance(x, (int, float)) and not np.isnan(x) else "--"


def main():
    cells = ["TK_DA", "TK_IM", "KS_DA", "KS_IM"]
    rows: list[dict] = []
    nested: dict = {c: {} for c in cells}

    # Point table
    print("\n=== POINT (rRMSE) ===")
    print(f"{'Model':<15}" + "".join(f"{c:>10}" for c in cells))
    for model_name, src in MODELS_POINT:
        data = load_json(src)
        rrmse_row = {}
        for c in cells:
            if model_name == "WeeklyNaive":
                rrmse = 1.0
            elif data and c in data and "rRMSE" in data[c]:
                rrmse = data[c]["rRMSE"]
            else:
                rrmse = float("nan")
            rrmse_row[c] = rrmse
            nested[c].setdefault(model_name, {})["rRMSE"] = rrmse
        rows.append({"model": model_name, "metric": "rRMSE", **{c: rrmse_row[c] for c in cells}})
        print(f"{model_name:<15}" + "".join(f"{fmt(rrmse_row[c]):>10}" for c in cells))

    # Probabilistic table
    print("\n=== PROBABILISTIC (CRPS) ===")
    print(f"{'Model':<15}" + "".join(f"{c:>10}" for c in cells))
    for model_name, src in MODELS_PROB:
        data = load_json(src)
        crps_row = {}
        for c in cells:
            if data and c in data and "CRPS" in data[c]:
                crps = data[c]["CRPS"]
            else:
                crps = float("nan")
            crps_row[c] = crps
            nested[c].setdefault(model_name, {})["CRPS"] = crps
        rows.append({"model": model_name, "metric": "CRPS", **{c: crps_row[c] for c in cells}})
        print(f"{model_name:<15}" + "".join(f"{fmt(crps_row[c]):>10}" for c in cells))

    # Coverage @ 90%
    print("\n=== COVERAGE @ 90% (target 0.900) ===")
    print(f"{'Model':<15}" + "".join(f"{c:>10}" for c in cells))
    for model_name, src in MODELS_PROB:
        data = load_json(src)
        cov_row = {}
        for c in cells:
            if data and c in data and "coverage" in data[c]:
                cov = data[c]["coverage"].get("alpha_0.10", float("nan"))
            else:
                cov = float("nan")
            cov_row[c] = cov
            nested[c].setdefault(model_name, {})["coverage_90"] = cov
        rows.append({"model": model_name, "metric": "coverage_90", **{c: cov_row[c] for c in cells}})
        print(f"{model_name:<15}" + "".join(f"{fmt(cov_row[c]):>10}" for c in cells))

    # Winkler @ 90%
    print("\n=== WINKLER @ 90% (lower is better) ===")
    print(f"{'Model':<15}" + "".join(f"{c:>10}" for c in cells))
    for model_name, src in MODELS_PROB:
        data = load_json(src)
        w_row = {}
        for c in cells:
            if data and c in data and "winkler" in data[c]:
                w = data[c]["winkler"].get("alpha_0.10", float("nan"))
            else:
                w = float("nan")
            w_row[c] = w
            nested[c].setdefault(model_name, {})["winkler_90"] = w
        rows.append({"model": model_name, "metric": "winkler_90", **{c: w_row[c] for c in cells}})
        print(f"{model_name:<15}" + "".join(f"{fmt(w_row[c]):>10}" for c in cells))

    pd.DataFrame(rows).to_csv(os.path.join(RES, "extended_summary.csv"), index=False)
    with open(os.path.join(RES, "extended_summary.json"), "w") as f:
        json.dump(nested, f, indent=2)
    print("\nSaved -> extended_summary.csv, extended_summary.json")


if __name__ == "__main__":
    main()
