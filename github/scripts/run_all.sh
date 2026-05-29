#!/usr/bin/env bash
# Reproduce the full JEPX-EPF-Benchmark in one go.
# Estimated runtime on Apple M2, 16 GB RAM: ~80 minutes.
set -euo pipefail

PY="${PY:-python}"
cd "$(dirname "$0")/.."

echo "=== Stage 1/9: LEAR rolling (~30 min) ==="
$PY code/run_lear_rolling.py

echo "=== Stage 2/9: DDNN-Normal rolling (~5 min) ==="
$PY code/run_ddnn_rolling.py

echo "=== Stage 3/9: DDNN-Johnson-SU rolling (~15 min) ==="
$PY code/run_ddnn_jsu_rolling.py

echo "=== Stage 4/9: N-HiTS rolling (~25 min) ==="
$PY code/run_nhits_rolling.py

echo "=== Stage 5/9: QRA-rolling (~10 min) ==="
$PY code/run_qra_rolling.py

echo "=== Stage 6/9: Adaptive conformal on LEAR (<1 min) ==="
$PY code/run_conformal_lear.py

echo "=== Stage 7/9: Model Confidence Set (<1 min) ==="
$PY code/run_mcs.py

echo "=== Stage 8/9: Summary tables + figure (Month-3 main) (<1 min) ==="
$PY code/make_month3_summary.py

echo "=== Stage 9/9: CRPS / Winkler / MCS figures (<1 min) ==="
$PY code/fig_crps_winkler.py

echo
echo "=== Done. Results in results/, figures in figures/. ==="
