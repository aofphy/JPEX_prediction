"""
Consolidate Month-3 results into a comprehensive table + figure.

Models compared (all rolling-window):
  - WeeklyNaive (computed on the fly)
  - LEAR
  - DDNN-Normal
  - DDNN-JSU
  - N-HiTS
  - QRA-rolling (LEAR + DDNN; 90-day rolling calibration)

Outputs:
  results/month3_summary.csv   -- master comparison table
  figures/fig_month3_main.pdf  -- 2x2 panel: per-horizon RMSE for all 4 pairs
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family":"serif","font.size":9,"axes.labelsize":9,"axes.titlesize":10,
    "legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":120,"savefig.dpi":300,"savefig.bbox":"tight",
})
C = {"black":"#000000","gray":"#888","blue":"#0072B2","skyblue":"#56B4E9",
     "green":"#009E73","vermillion":"#D55E00","orange":"#E69F00","purple":"#CC79A7"}

ROOT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a"
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# Load all metrics JSONs
lear  = json.load(open(os.path.join(RES, "lear_metrics.json")))
ddnn  = json.load(open(os.path.join(RES, "ddnn_metrics.json")))
jsu   = json.load(open(os.path.join(RES, "ddnn_jsu_metrics.json")))
nhits = json.load(open(os.path.join(RES, "nhits_metrics.json")))
qra_r = json.load(open(os.path.join(RES, "qra_rolling_metrics.json")))
conf  = json.load(open(os.path.join(RES, "conformal_lear_metrics.json")))
mcs   = json.load(open(os.path.join(RES, "mcs_results.json")))

pairs = ["TK_DA", "TK_IM", "KS_DA", "KS_IM"]

# --- Comprehensive comparison table ---
rows = []
for p in pairs:
    wn_rmse = lear[p]["WeeklyNaive"]["RMSE"]
    rows.append({
        "pair": p,
        "WN_RMSE": wn_rmse,
        "LEAR_rRMSE":     lear[p]["LEAR"]["rRMSE"],
        "DDNN-N_rRMSE":   ddnn[p]["RMSE"] / wn_rmse,
        "DDNN-JSU_rRMSE": jsu[p]["Q50_RMSE"] / wn_rmse,
        "N-HiTS_rRMSE":   nhits[p]["rRMSE"],
        "QRAroll_rRMSE":  qra_r[p]["Q50_RMSE"] / wn_rmse,
        "DDNN-N_CRPS":    ddnn[p]["CRPS"],
        "DDNN-JSU_CRPS":  jsu[p]["CRPS"],
        "QRAroll_CRPS":   qra_r[p]["CRPS"],
        "DDNN-N_cov90":   ddnn[p]["coverage"]["alpha_0.10"],
        "DDNN-JSU_cov90": jsu[p]["coverage"]["alpha_0.10"],
        "QRAroll_cov90":  qra_r[p]["coverage"]["alpha_0.10"],
        "LEARconf_cov90": conf[p]["adaptive"]["alpha_0.10"]["empirical_coverage"],
        "DDNN-JSU_Wnk90": jsu[p]["winkler"]["alpha_0.10"],
        "QRAroll_Wnk90":  qra_r[p]["winkler"]["alpha_0.10"],
        "LEARconf_Wnk90": conf[p]["adaptive"]["alpha_0.10"]["winkler_score"],
        "MCS_10":         ", ".join(mcs[p]["mcs_10"]),
    })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "month3_summary.csv"), index=False)

print("\n=" * 30)
print("MONTH-3 COMPREHENSIVE COMPARISON")
print("=" * 30)
print("\nPoint accuracy (rRMSE, lower = better; 1.0 = weekly-naive baseline):")
print(f"{'pair':<8} {'LEAR':>7} {'DDNN-N':>7} {'JSU':>7} {'N-HiTS':>7} {'QRAroll':>8}")
for r in rows:
    print(f"{r['pair']:<8} {r['LEAR_rRMSE']:>7.3f} {r['DDNN-N_rRMSE']:>7.3f} "
          f"{r['DDNN-JSU_rRMSE']:>7.3f} {r['N-HiTS_rRMSE']:>7.3f} {r['QRAroll_rRMSE']:>8.3f}")

print("\nProbabilistic accuracy (CRPS, lower = better):")
print(f"{'pair':<8} {'DDNN-N':>7} {'DDNN-JSU':>9} {'QRAroll':>8}")
for r in rows:
    print(f"{r['pair']:<8} {r['DDNN-N_CRPS']:>7.3f} {r['DDNN-JSU_CRPS']:>9.3f} {r['QRAroll_CRPS']:>8.3f}")

print("\n90% empirical coverage (closer to 0.90 = better):")
print(f"{'pair':<8} {'DDNN-N':>7} {'DDNN-JSU':>9} {'QRAroll':>8} {'LEARconf':>9}")
for r in rows:
    print(f"{r['pair']:<8} {r['DDNN-N_cov90']:>7.3f} {r['DDNN-JSU_cov90']:>9.3f} "
          f"{r['QRAroll_cov90']:>8.3f} {r['LEARconf_cov90']:>9.3f}")

print("\nWinkler @ 90% (lower = better):")
print(f"{'pair':<8} {'DDNN-JSU':>9} {'QRAroll':>8} {'LEARconf':>9}")
for r in rows:
    print(f"{r['pair']:<8} {r['DDNN-JSU_Wnk90']:>9.2f} {r['QRAroll_Wnk90']:>8.2f} {r['LEARconf_Wnk90']:>9.2f}")

print("\nModel Confidence Set @ alpha=0.10:")
for r in rows:
    print(f"  {r['pair']:<8}  MCS = {{{r['MCS_10']}}}")

# --- Figure: per-horizon RMSE for all 4 pairs ---
fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), constrained_layout=True)
horizons = np.arange(1, 49)
xticks = [1, 12, 24, 36, 48]
xlabels = ["+0.5h", "+6h", "+12h", "+18h", "+24h"]

for ax, p in zip(axes.flat, pairs):
    wn = np.array(lear[p]["WeeklyNaive"]["RMSE_per_horizon"])
    le = np.array(lear[p]["LEAR"]["RMSE_per_horizon"])
    nh = np.array(nhits[p]["RMSE_per_horizon"]) if p in nhits else None
    ax.plot(horizons, wn, color=C["gray"], lw=1.3, label="WeeklyNaive")
    ax.plot(horizons, le, color=C["blue"], lw=1.3, label="LEAR")
    if nh is not None:
        ax.plot(horizons, nh, color=C["green"], lw=1.3, label="N-HiTS")
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("RMSE (JPY/kWh)")
    mcs_set = mcs[p]["mcs_10"]
    ax.set_title(f"{p.replace('_', ' ')}  —  MCS: {{{', '.join(mcs_set)}}}")
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.legend(loc="upper left", fontsize=7)

plt.savefig(os.path.join(FIG, "fig_month3_main.pdf"))
plt.savefig(os.path.join(FIG, "fig_month3_main.png"))
plt.close()
print(f"\nSaved fig_month3_main to {FIG}")
