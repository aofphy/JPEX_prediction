"""Block-length sensitivity for MCS (response to R1 M1, DA C2).

Re-runs the Hansen-Lunde-Nason MCS with three stationary-bootstrap mean block
lengths: the default T^(1/3) ~= 24 (12 h), the dominant weekly cycle 336
(7 d), and twice that 672 (14 d). The expectation under the reviewer's
concern is that a sub-seasonal block length under-states serial dependence
and may inflate the apparent singleton structure of the MCS; we check by
sweeping the block length and reporting MCS composition at both alpha=0.10
and alpha=0.25 for each.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_mcs import collect_predictions
from mcs import model_confidence_set

ROOT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a"
RES = os.path.join(ROOT, "results")

BLOCK_LENGTHS = [None, 336.0, 672.0]   # None = default T^(1/3)
BLOCK_LABELS  = ["default_T^(1/3)", "weekly_336", "biweekly_672"]


def main():
    out: dict = {}
    for area in ["TK", "KS"]:
        for market in ["DA", "IM"]:
            key = f"{area}_{market}"
            print(f"\n========== {key} ==========")
            _, Y_actual, preds = collect_predictions(area, market)
            sq_losses = {k: ((v - Y_actual) ** 2).mean(axis=1) for k, v in preds.items()}
            per_block: dict = {}
            for mb, label in zip(BLOCK_LENGTHS, BLOCK_LABELS):
                res = model_confidence_set(sq_losses, alphas=[0.10, 0.25],
                                            n_boot=1000, mean_block=mb)
                per_block[label] = {
                    "mcs_10": res["mcs_10"],
                    "mcs_25": res["mcs_25"],
                    "eliminations": res["eliminations"],
                }
                print(f"  block={label:<20} MCS_10={res['mcs_10']}  MCS_25={res['mcs_25']}")
            out[key] = per_block
    with open(os.path.join(RES, "mcs_block_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSaved -> mcs_block_sensitivity.json")


if __name__ == "__main__":
    main()
