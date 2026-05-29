# jepx-epf-benchmark

> Reproducible probabilistic 48-step electricity-price-forecasting benchmark for the **Japan Electric Power Exchange (JEPX)**, Tokyo and Kansai areas, day-ahead and imbalance markets.

[![License: MIT (code)](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![License: CC-BY-4.0 (data)](https://img.shields.io/badge/data-CC--BY--4.0-blue.svg)](LICENSE-DATA)
[![DOI](https://img.shields.io/badge/Zenodo-DOI%20pending-orange.svg)](#citation)
[![Paper](https://img.shields.io/badge/Paper-Applied%20Energy%20(submitted)-green.svg)](#citation)

Companion repository for the manuscript *A Probabilistic 48-Step Forecasting Benchmark for the Japan Electric Power Exchange: Conformal, Distributional, and Quantile-Averaging Methods Across Day-Ahead and Imbalance Markets* by **Sriboriboon & Fongkaew** (Suranaree University of Technology), submitted to **Applied Energy**.

---

## TL;DR

| Pair | Best single model (MCS @ α=0.10) | Best probabilistic forecaster | rRMSE | CRPS | Coverage @ 90% |
|------|----------------------------------|------------------------------|-------|------|----------------|
| Tokyo DA   | **N-HiTS** | QRA-rolling | 0.756 | 1.229 | 0.897 |
| Tokyo IM   | **LEAR**   | QRA-rolling | 0.751 | 1.947 | 0.918 |
| Kansai DA  | **N-HiTS** | QRA-rolling | 0.839 | 1.592 | 0.894 |
| Kansai IM  | **LEAR**   | QRA-rolling | 0.789 | 1.908 | 0.893 |

- rRMSE relative to the weekly-naive seasonal baseline; lower = better.
- CRPS in JPY/kWh; lower = better.
- Empirical coverage of the 90% prediction interval; closer to 0.900 = better.

---

## Quick start

### Prerequisites

- Python 3.8+ (tested on 3.8.20 via miniforge/conda)
- ~20 GB free disk (for prediction matrices)
- ~80 minutes of CPU time on Apple M2 16 GB (no GPU required)

### Install

```bash
git clone https://github.com/<your-org>/jepx-epf-benchmark.git
cd jepx-epf-benchmark
pip install -r requirements.txt
```

`requirements.txt`:
```
lightgbm>=3.3
xgboost>=1.7
torch>=2.0
scikit-learn>=1.0
statsmodels>=0.14
pandas>=1.5
numpy>=1.21
matplotlib>=3.5
scipy>=1.10
```

### Reproduce all results

```bash
# Run the full pipeline (sequential, ~80 min):
bash scripts/run_all.sh

# OR step-by-step:
python code/run_lear_rolling.py
python code/run_ddnn_rolling.py
python code/run_ddnn_jsu_rolling.py
python code/run_nhits_rolling.py
python code/run_qra_rolling.py
python code/run_conformal_lear.py
python code/run_mcs.py
python code/make_month3_summary.py
python code/fig_crps_winkler.py
```

Results land in `results/` (JSON metrics + CSV/NPZ prediction matrices) and figures in `figures/`.

### Compile the manuscript

```bash
cd tex/
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex cover_letter
```

---

## What's in this repo

```
jepx-epf-benchmark/
├── README.md                  ← this file
├── LICENSE                    ← MIT for code
├── LICENSE-DATA               ← CC-BY-4.0 for data and figures
├── CITATION.cff               ← citation metadata
├── requirements.txt
├── scripts/
│   └── run_all.sh             ← single-command reproduction
├── dataset/
│   ├── IM_DA_TK_ALL.csv       ← Tokyo half-hourly DA + IM + exog, 2022-03 to 2023-12
│   └── IM_DA_KS_ALL.csv       ← Kansai, same
├── code/
│   ├── data_loader.py
│   ├── feature_build.py
│   ├── models_{lear,ddnn,nhits,qra,conformal}.py
│   ├── rolling_eval.py
│   ├── mcs.py
│   ├── metrics_prob.py
│   ├── run_*_rolling.py        ← 6 experiment drivers
│   ├── run_conformal_lear.py
│   ├── run_mcs.py
│   ├── make_*_summary.py
│   └── fig_*.py                ← figure generators
├── results/                   ← saved metrics, predictions (ignored by .gitignore in clean clone; populated by running the pipeline)
├── figures/                   ← saved PDF + PNG figures
├── tex/                       ← manuscript and cover letter LaTeX source
│   ├── main.tex
│   ├── main.pdf
│   ├── cover_letter.tex
│   ├── cover_letter.pdf
│   └── references.bib
└── docs/
    └── PROTOCOL.md            ← detailed methodology notes
```

---

## Methodology snapshot

- **Target.** At every 30-min issuance time *t*, predict the next-24h 48-vector \(\hat y_{t+1}, \ldots, \hat y_{t+48}\).
- **Protocol.** Lago et al.\ (2021) rolling-window with a 365-day sliding training window and weekly recalibration (refit every 7 days).
- **Test window.** 2023-03-01 → 2023-12-30 (305 days × 48 issuance/day = 14,640 issuance times per area-market pair; each issuance produces a 48-vector → 702,720 predictions per pair per model).
- **Markets.** Day-ahead (DA) and imbalance (IM) for Tokyo (TK) and Kansai (KS) → 4 (area, market) pairs.

### Models compared

| Model | Type | Reference |
|-------|------|-----------|
| **WeeklyNaive** | seasonal baseline | --- |
| **Rolling-mean** | smoothed baseline | --- |
| **LEAR** | LASSO-LARS-IC regression | [Lago et al. 2021](https://doi.org/10.1016/j.apenergy.2021.116983) |
| **DDNN-Normal** | feed-forward NN with Gaussian predictive head | [Marcjasz et al. 2023](https://doi.org/10.1016/j.eneco.2023.106843) |
| **DDNN-Johnson-SU** | same backbone with heavy-tailed predictive head | [Marcjasz et al. 2023](https://doi.org/10.1016/j.eneco.2023.106843) |
| **N-HiTS** | hierarchical multi-rate deep forecaster | [Challu et al. 2023](https://doi.org/10.1609/aaai.v37i6.25854) |
| **QRA-rolling** | per-horizon quantile-regression ensemble (LEAR + DDNN-N) with 90-day rolling calibration | [Uniejewski & Weron 2021](https://doi.org/10.1016/j.eneco.2021.105121) |
| **LEAR + adaptive conformal** | post-hoc conformal wrapper on LEAR | [Gibbs & Candès 2021](https://arxiv.org/abs/2106.00170) |

### Evaluation

- **Point**: RMSE, MAE, rRMSE, rMAE (relative to WeeklyNaive)
- **Probabilistic**: CRPS, pinball loss, Winkler score, empirical coverage at α ∈ {0.5, 0.2, 0.1}
- **Significance**: Diebold–Mariano with Newey–West HAC variance and HLN correction; Model Confidence Set (Hansen–Lunde–Nason 2011) with 1000-replication stationary-bootstrap critical values.

---

## Citation

If you use this code or dataset, please cite both:

```bibtex
@article{jepx_epf_benchmark_paper,
  author  = {Sriboriboon, Panithan and Fongkaew, Ittipon},
  title   = {A Probabilistic 48-Step Forecasting Benchmark for the Japan
             Electric Power Exchange},
  journal = {Applied Energy},
  year    = {2026},
  doi     = {<to-be-supplied-on-acceptance>}
}

@dataset{jepx_epf_benchmark_zenodo,
  author    = {Sriboriboon, Panithan and Fongkaew, Ittipon},
  title     = {{JEPX-EPF-Benchmark}: Probabilistic 48-Step Electricity Price
               Forecasting Dataset and Code for the Japan Electric Power Exchange},
  year      = 2026,
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {<to-be-supplied-on-deposit>}
}
```

A machine-readable `CITATION.cff` is included in the repository root.

---

## Contributing

This repository is the frozen reproducibility package for the Applied Energy submission, **not** a long-term maintained library. Pull requests are welcome for:

- Bug reports / numerical-precision issues
- Extensions to additional JEPX areas (Hokkaido, Tohoku, Chubu, etc.)
- Additional probabilistic models with rolling protocol
- Speed-up of the experiment runners

Please open an issue before submitting a substantial PR so we can discuss the scope.

---

## Licence

- **Code** (`code/`, `scripts/`): MIT License --- see [LICENSE](LICENSE)
- **Processed dataset** (`dataset/`), **prediction matrices** (`results/`), and **figures** (`figures/`): Creative Commons Attribution 4.0 International --- see [LICENSE-DATA](LICENSE-DATA)
- **Raw JEPX prices** are publicly available from the JEPX information disclosure portal under the JEPX disclosure terms.

---

## Contact

**Ittipon Fongkaew** (corresponding author)
School of Physics, Institute of Science
Suranaree University of Technology, Thailand
ittipon@sut.ac.th

---

## Acknowledgments

We thank the JEPX information disclosure team and the area TSOs for the public data feed. We also thank the anonymous reviewers of the v1e preprint for substantive corrections that materially shaped this final version. This work was supported by the School of Physics, Institute of Science, Suranaree University of Technology.
