"""
Combine the v4 FLAML-tuned results (single 80/20 split) with the Path A
LEAR rolling-window results into a unified comparison table.

This shows reviewers the value-add of moving from "single-split + FLAML-tuned
complex models" (v4) to "rolling-window LEAR" (Path A).
"""
import json
from pathlib import Path

ROOT = Path("/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/revised")
V4_RES = ROOT / "results" / "results_flaml.json"
PA_RES = ROOT / "path_a" / "results" / "lear_metrics.json"

v4 = json.load(V4_RES.open())
pa = json.load(PA_RES.open())

print("\n" + "=" * 90)
print(f"{'Setup':<40} {'TK DA':>12} {'TK IM':>12} {'KS DA':>12} {'KS IM':>12}")
print("=" * 90)
# v4 single-split metrics (Tokyo only)
def fmt(x): return f"{x:.3f}" if x else "  --  "
v4_tk_da = v4.get("DA", {}).get("metrics", {})
v4_tk_im = v4.get("IM", {}).get("metrics", {})

models_v4 = ["WeeklyNaive", "RollingMean", "LightGBM-FE", "LightGBM-Raw", "XGBoost-FE", "MLP", "LSTM"]
print("\nv4 (single 80/20 split, FLAML-tuned, Tokyo only) — rRMSE:")
for m in models_v4:
    da_v = v4_tk_da.get(m, {}).get("rRMSE")
    im_v = v4_tk_im.get(m, {}).get("rRMSE")
    print(f"  {m:<38} {fmt(da_v):>12} {fmt(im_v):>12}   --             --")

print("\nPath A (rolling-window, weekly recal, 4 area-market pairs) — rRMSE:")
# Aggregate Path A
tkda = pa.get("TK_DA", {})
tkim = pa.get("TK_IM", {})
ksda = pa.get("KS_DA", {})
ksim = pa.get("KS_IM", {})
print(f"  {'WeeklyNaive (baseline)':<38} {'1.000':>12} {'1.000':>12} {'1.000':>12} {'1.000':>12}")
def get_lear(d):
    return d.get("LEAR", {}).get("rRMSE")
print(f"  {'LEAR':<38} {fmt(get_lear(tkda)):>12} {fmt(get_lear(tkim)):>12} {fmt(get_lear(ksda)):>12} {fmt(get_lear(ksim)):>12}")

# Also show test sizes
print("\nTest set sizes (n issuance × 48 horizons):")
print(f"  {'v4 (single split)':<38} {'14640':>12} {'14640':>12}    --             --")
for area_market in ["TK_DA", "TK_IM", "KS_DA", "KS_IM"]:
    if area_market in pa:
        n_v = pa[area_market]['n_valid']
        print(f"  Path A {area_market:<32} {n_v:>12}")

print("=" * 90)
