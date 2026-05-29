"""
Consolidate Month-2 probabilistic results across LEAR+Conformal, DDNN-Normal,
and QRA(LEAR,DDNN). Produces a comparison table and a multi-panel figure.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family":"serif","font.size":9,"axes.labelsize":9,
    "axes.titlesize":10,"legend.fontsize":7,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":120,"savefig.dpi":300,"savefig.bbox":"tight",
})
C = {"blue":"#0072B2","green":"#009E73","orange":"#E69F00","vermillion":"#D55E00","gray":"#888"}

ROOT = "/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised/path_a"
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# Load all metrics
conf = json.load(open(os.path.join(RES, "conformal_lear_metrics.json")))

ddnn_path = os.path.join(RES, "ddnn_metrics.json")
ddnn = json.load(open(ddnn_path)) if os.path.exists(ddnn_path) else {}

qra_path = os.path.join(RES, "qra_metrics.json")
qra = json.load(open(qra_path)) if os.path.exists(qra_path) else {}

pairs = ["TK_DA", "TK_IM", "KS_DA", "KS_IM"]

# --- Table: coverage, width, Winkler at 90% across methods ---
rows = []
for p in pairs:
    row = {"pair": p}
    # LEAR+Adaptive-Conformal
    if p in conf:
        a = conf[p]["adaptive"].get("alpha_0.10", {})
        row["LEAR_conf_cov90"] = a.get("empirical_coverage")
        row["LEAR_conf_width90"] = a.get("mean_width")
        row["LEAR_conf_Winkler90"] = a.get("winkler_score")
    # DDNN-Normal
    if p in ddnn:
        d = ddnn[p]
        row["DDNN_RMSE"] = d.get("RMSE")
        row["DDNN_CRPS"] = d.get("CRPS")
        row["DDNN_cov90"] = d["coverage"]["alpha_0.10"]
        row["DDNN_width90"] = d["mean_width"]["alpha_0.10"]
        row["DDNN_Winkler90"] = d["winkler"]["alpha_0.10"]
    # QRA
    if p in qra:
        q = qra[p]
        row["QRA_pinball"] = q.get("pinball")
        row["QRA_CRPS"] = q.get("CRPS")
        row["QRA_cov90"] = q["coverage"]["alpha_0.10"]
        row["QRA_width90"] = q["mean_width"]["alpha_0.10"]
        row["QRA_Winkler90"] = q["winkler"]["alpha_0.10"]
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "month2_summary.csv"), index=False)
print("\n========== Month-2 Probabilistic Summary ==========")
print(df.to_string(index=False))
print("===================================================")

# --- Figure: side-by-side coverage and Winkler at 90% ---
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
x = np.arange(len(pairs))
width = 0.27

# (a) Empirical coverage at 90% nominal
ax = axes[0]
cov_lear  = [conf[p]["adaptive"]["alpha_0.10"]["empirical_coverage"] if p in conf else np.nan for p in pairs]
cov_ddnn  = [ddnn[p]["coverage"]["alpha_0.10"] if p in ddnn else np.nan for p in pairs]
cov_qra   = [qra[p]["coverage"]["alpha_0.10"] if p in qra else np.nan for p in pairs]
ax.bar(x - width, cov_lear, width, label="LEAR + adaptive conformal", color=C["blue"])
ax.bar(x,         cov_ddnn, width, label="DDNN-Normal", color=C["green"])
ax.bar(x + width, cov_qra,  width, label="QRA(LEAR,DDNN)", color=C["orange"])
ax.axhline(0.9, color="black", lw=0.7, ls="--", label="nominal 90%")
ax.set_ylabel("Empirical coverage")
ax.set_title("(a) 90% prediction interval coverage")
ax.set_xticks(x); ax.set_xticklabels([p.replace("_"," ") for p in pairs])
ax.set_ylim(0.7, 1.0)
ax.legend(loc="lower right", fontsize=7)
ax.grid(True, alpha=0.3, axis="y", lw=0.4)

# (b) Winkler at 90%
ax = axes[1]
ws_lear = [conf[p]["adaptive"]["alpha_0.10"]["winkler_score"] if p in conf else np.nan for p in pairs]
ws_ddnn = [ddnn[p]["winkler"]["alpha_0.10"] if p in ddnn else np.nan for p in pairs]
ws_qra  = [qra[p]["winkler"]["alpha_0.10"] if p in qra else np.nan for p in pairs]
ax.bar(x - width, ws_lear, width, label="LEAR + adaptive conformal", color=C["blue"])
ax.bar(x,         ws_ddnn, width, label="DDNN-Normal", color=C["green"])
ax.bar(x + width, ws_qra,  width, label="QRA(LEAR,DDNN)", color=C["orange"])
ax.set_ylabel("Winkler score (lower = better)")
ax.set_title("(b) 90% PI sharpness penalised by miscoverage")
ax.set_xticks(x); ax.set_xticklabels([p.replace("_"," ") for p in pairs])
ax.legend(loc="upper right", fontsize=7)
ax.grid(True, alpha=0.3, axis="y", lw=0.4)

plt.savefig(os.path.join(FIG, "fig_month2_probabilistic.pdf"))
plt.savefig(os.path.join(FIG, "fig_month2_probabilistic.png"))
plt.close()
print(f"\nSaved fig_month2_probabilistic.{{pdf,png}}")
