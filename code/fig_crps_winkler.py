"""Final figures: CRPS bar chart + Winkler comparison + per-horizon CRPS."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ROOT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a"
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

ddnn = json.load(open(os.path.join(RES, "ddnn_metrics.json")))
jsu  = json.load(open(os.path.join(RES, "ddnn_jsu_metrics.json")))
qra  = json.load(open(os.path.join(RES, "qra_rolling_metrics.json")))
conf = json.load(open(os.path.join(RES, "conformal_lear_metrics.json")))
mcs  = json.load(open(os.path.join(RES, "mcs_results.json")))

pairs = ["TK_DA", "TK_IM", "KS_DA", "KS_IM"]
labels = ["Tokyo DA", "Tokyo IM", "Kansai DA", "Kansai IM"]

# Color palette
C = {"ddnn_n": "#56B4E9", "ddnn_jsu": "#0072B2", "qra": "#D55E00",
     "lear_conf": "#009E73"}

# ============================================================
# Figure: CRPS bar chart
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0), constrained_layout=True)
x = np.arange(len(pairs))
width = 0.27

crps_ddnn   = [ddnn[p]["CRPS"] for p in pairs]
crps_jsu    = [jsu[p]["CRPS"]  for p in pairs]
crps_qra    = [qra[p]["CRPS"]  for p in pairs]

bars1 = ax.bar(x - width, crps_ddnn, width, label="DDNN-Normal", color=C["ddnn_n"])
bars2 = ax.bar(x,         crps_jsu,  width, label="DDNN-Johnson SU", color=C["ddnn_jsu"])
bars3 = ax.bar(x + width, crps_qra,  width, label="QRA-rolling (LEAR+DDNN)", color=C["qra"])

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("CRPS (JPY/kWh, lower = better)")
ax.set_title("Probabilistic accuracy across (area, market) pairs")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3, axis="y", lw=0.4)

# Annotate winners
for i, p in enumerate(pairs):
    vals = [crps_ddnn[i], crps_jsu[i], crps_qra[i]]
    win_idx = int(np.argmin(vals))
    win_x = x[i] + [-width, 0, width][win_idx]
    win_v = vals[win_idx]
    ax.annotate(f"{win_v:.2f}", xy=(win_x, win_v), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=7, fontweight="bold")

plt.savefig(os.path.join(FIG, "fig_crps.pdf"))
plt.savefig(os.path.join(FIG, "fig_crps.png"))
plt.close()
print("Saved fig_crps.pdf")

# ============================================================
# Figure: Winkler @ 90% comparison
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0), constrained_layout=True)

wnk_jsu  = [jsu[p]["winkler"]["alpha_0.10"]                        for p in pairs]
wnk_qra  = [qra[p]["winkler"]["alpha_0.10"]                        for p in pairs]
wnk_conf = [conf[p]["adaptive"]["alpha_0.10"]["winkler_score"]     for p in pairs]

bars1 = ax.bar(x - width, wnk_jsu,  width, label="DDNN-Johnson SU", color=C["ddnn_jsu"])
bars2 = ax.bar(x,         wnk_qra,  width, label="QRA-rolling",     color=C["qra"])
bars3 = ax.bar(x + width, wnk_conf, width, label="LEAR + adaptive conformal", color=C["lear_conf"])

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Winkler score @ 90% (lower = better)")
ax.set_title("Probabilistic interval quality (sharpness penalised by miscoverage)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3, axis="y", lw=0.4)

for i in range(len(pairs)):
    vals = [wnk_jsu[i], wnk_qra[i], wnk_conf[i]]
    win_idx = int(np.argmin(vals))
    win_x = x[i] + [-width, 0, width][win_idx]
    win_v = vals[win_idx]
    ax.annotate(f"{win_v:.1f}", xy=(win_x, win_v), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=7, fontweight="bold")

plt.savefig(os.path.join(FIG, "fig_winkler.pdf"))
plt.savefig(os.path.join(FIG, "fig_winkler.png"))
plt.close()
print("Saved fig_winkler.pdf")

# ============================================================
# Figure: MCS visualization — best model per (area, market)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5), constrained_layout=True)
ax.axis("off")
ax.set_title("Model Confidence Set (Hansen-Lunde-Nason 2011) at $\\alpha = 0.10$",
             fontsize=11)

# Build 2x2 grid of cells
cell_w = 0.4
cell_h = 0.35
x_pos = [0.1, 0.5]
y_pos = [0.55, 0.1]
mcs_label = {"N-HiTS": "#009E73", "LEAR": "#0072B2"}

ax.text(0.3, 0.95, "DA market", ha="center", fontsize=10, fontweight="bold")
ax.text(0.7, 0.95, "IM market", ha="center", fontsize=10, fontweight="bold")
ax.text(0.02, 0.725, "Tokyo", ha="center", va="center", rotation=90, fontsize=10, fontweight="bold")
ax.text(0.02, 0.275, "Kansai", ha="center", va="center", rotation=90, fontsize=10, fontweight="bold")

cells = [
    ("TK_DA", 0, 0), ("TK_IM", 1, 0),
    ("KS_DA", 0, 1), ("KS_IM", 1, 1),
]
for key, col, row in cells:
    model = mcs[key]["mcs_10"][0]
    rect = plt.Rectangle((x_pos[col], y_pos[row]), cell_w, cell_h,
                          facecolor=mcs_label.get(model, "#888"), alpha=0.7,
                          edgecolor="black", lw=1)
    ax.add_patch(rect)
    ax.text(x_pos[col] + cell_w/2, y_pos[row] + cell_h/2,
            f"MCS:\n{{{model}}}", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.savefig(os.path.join(FIG, "fig_mcs.pdf"))
plt.savefig(os.path.join(FIG, "fig_mcs.png"))
plt.close()
print("Saved fig_mcs.pdf")

print("\nAll Month-4 figures generated.")
