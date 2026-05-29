"""Figure: split vs adaptive conformal calibration across 4 (area, market) pairs."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family":"serif","font.size":9,"axes.labelsize":9,
    "axes.titlesize":10,"legend.fontsize":8,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":120,"savefig.dpi":300,"savefig.bbox":"tight",
})

ROOT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a"
RES = json.load(open(os.path.join(ROOT, "results", "conformal_lear_metrics.json")))

pairs = ["TK_DA", "TK_IM", "KS_DA", "KS_IM"]
alphas = [0.5, 0.2, 0.1]
nominal = [1 - a for a in alphas]

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), constrained_layout=True)

# (a) Empirical vs nominal coverage
ax = axes[0]
ax.plot([0, 1], [0, 1], color="black", lw=0.7, ls="--", label="ideal calibration")
for pair in pairs:
    cov_split = [RES[pair]["split"][f"alpha_{a:.2f}"]["empirical_coverage"] for a in alphas]
    cov_adapt = [RES[pair]["adaptive"][f"alpha_{a:.2f}"]["empirical_coverage"] for a in alphas]
    ax.plot(nominal, cov_split, "o-", lw=1.0, alpha=0.5, label=f"{pair} split")
    ax.plot(nominal, cov_adapt, "s-", lw=1.2, label=f"{pair} adaptive")
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Empirical coverage")
ax.set_title("(a) Conformal calibration")
ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.0)
ax.legend(loc="lower right", fontsize=6, ncol=2)
ax.grid(True, alpha=0.3, lw=0.4)

# (b) Winkler scores at alpha=0.1 (90% intervals)
ax = axes[1]
x = np.arange(len(pairs))
width = 0.35
ws_split = [RES[p]["split"]["alpha_0.10"]["winkler_score"] for p in pairs]
ws_adapt = [RES[p]["adaptive"]["alpha_0.10"]["winkler_score"] for p in pairs]
ax.bar(x - width/2, ws_split, width, label="Split", color="#56B4E9")
ax.bar(x + width/2, ws_adapt, width, label="Adaptive", color="#0072B2")
ax.set_xticks(x)
ax.set_xticklabels([p.replace("_", " ") for p in pairs])
ax.set_ylabel("Winkler score (lower = better, 90% PI)")
ax.set_title("(b) Interval sharpness at 90%")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3, axis="y", lw=0.4)

plt.savefig(os.path.join(ROOT, "figures", "fig_conformal.pdf"))
plt.savefig(os.path.join(ROOT, "figures", "fig_conformal.png"))
plt.close()
print("Saved fig_conformal.{pdf,png}")
