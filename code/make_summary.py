"""
Path A Month-1 summary: tables + per-horizon RMSE figure across 4 (area, market) pairs.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})
C = {"blue": "#0072B2", "orange": "#E69F00", "vermillion": "#D55E00",
     "green": "#009E73", "gray": "#888"}

ROOT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a"
RES = json.load(open(os.path.join(ROOT, "results", "lear_metrics.json")))
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

# ----- Table: pooled metrics -----
rows = []
for key in ["TK_DA", "TK_IM", "KS_DA", "KS_IM"]:
    r = RES.get(key, {})
    if not r: continue
    rows.append({
        "area_market": key,
        "n_test": r["n_valid"],
        "WN_RMSE": r["WeeklyNaive"]["RMSE"],
        "WN_MAE":  r["WeeklyNaive"]["MAE"],
        "LEAR_RMSE": r["LEAR"]["RMSE"],
        "LEAR_MAE":  r["LEAR"]["MAE"],
        "rRMSE": r["LEAR"]["rRMSE"],
        "rMAE":  r["LEAR"]["rMAE"],
    })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "lear_summary.csv"), index=False)
print("\n" + "=" * 84)
print(df.to_string(index=False))
print("=" * 84)

# ----- Figure: per-horizon RMSE for all 4 (area, market) pairs -----
fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True, sharey=False)
horizons = np.arange(1, 49)

for ax, key in zip(axes.flat, ["TK_DA", "TK_IM", "KS_DA", "KS_IM"]):
    r = RES.get(key, {})
    if not r:
        ax.set_title(f"{key} (no data)")
        continue
    wn = np.array(r["WeeklyNaive"]["RMSE_per_horizon"])
    le = np.array(r["LEAR"]["RMSE_per_horizon"])
    ax.plot(horizons, wn, color=C["gray"], lw=1.2, label="WeeklyNaive")
    ax.plot(horizons, le, color=C["blue"], lw=1.2, label="LEAR")
    ax.set_xticks([1, 12, 24, 36, 48])
    ax.set_xticklabels(["+0.5h", "+6h", "+12h", "+18h", "+24h"])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("RMSE (JPY/kWh)")
    ax.set_title(f"{key.replace('_',' ')}  (rRMSE={r['LEAR']['rRMSE']:.3f})")
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.legend(loc="upper left", fontsize=7)

plt.savefig(os.path.join(FIG, "fig_pathA_lear_per_horizon.pdf"))
plt.savefig(os.path.join(FIG, "fig_pathA_lear_per_horizon.png"))
plt.close()
print(f"\nSaved figure to {FIG}/fig_pathA_lear_per_horizon.pdf")
