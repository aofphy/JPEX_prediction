# JEPX EPF Benchmark — Reproducibility Package

This repository accompanies the manuscript

> **A Probabilistic 48-Step Forecasting Benchmark for the Japan Electric Power Exchange: Conformal, Distributional, and Quantile-Averaging Methods Across Day-Ahead and Imbalance Markets**
> Panithan Sriboriboon, Ittipon Fongkaew (corresponding author).
> Submitted to *Applied Energy*.

It contains the full Python pipeline, processed dataset references, per-issuance prediction metrics, and the LaTeX source of the manuscript. The large per-issuance prediction matrices (`.npz`, ~600 MB total) are deposited on Zenodo per the manuscript's Data Availability statement; the JSON metric summaries and predicted medians are included in this repo for full result verification.

## What's in the benchmark

A Lago-protocol-compliant rolling-window evaluation of ten 48-step forecasters across four (area, market) pairs of JEPX (Tokyo + Kansai x Day-Ahead + Imbalance):

| Model | Family | Source |
|---|---|---|
| WeeklyNaive | seasonal baseline | -- |
| LEAR | LASSO-LARS regression | Lago et al. 2021 |
| DDNN-Normal | distributional NN, Gaussian head | Marcjasz et al. 2023 |
| DDNN-JSU | distributional NN, Johnson-SU head | Marcjasz et al. 2023 |
| N-HiTS | hierarchical multi-rate decomposition | Olivares et al. 2023 |
| **PatchTST** | patching Transformer | Nie et al. 2023 |
| **iTransformer** | variate-inverted Transformer | Liu et al. 2024 |
| **Chronos-Bolt** | pretrained foundation model (zero-shot) | Ansari et al. 2024 |
| **TimesFM** | pretrained foundation model (zero-shot) | Das et al. 2024 |
| QRA-rolling | ensemble of LEAR + DDNN-N (Q regression) | Uniejewski & Weron 2021 |

Models in **bold** are new in this version of the benchmark. The headline result is that the pretrained zero-shot foundation model Chronos-Bolt is in the Hansen-Lunde-Nason Model Confidence Set (alpha in {0.10, 0.25}) on every (area, market) pair and is the sole MCS singleton on Tokyo IM, statistically tying with the strongest supervised method (iTransformer).

## Reproducing the benchmark

### Prerequisites

- Python 3.10
- ~16 GB RAM
- Apple Silicon (MPS) or NVIDIA GPU recommended; CPU-only works but is slower
- ~5 GB free disk (model weights + intermediate predictions)

### Install

```bash
git clone https://github.com/aofphy/JPEX_prediction.git
cd JPEX_prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `chronos-forecasting` and `timesfm` will download ~1 GB of pretrained weights on first invocation
- On macOS with conflicting TF/numpy installs, set `USE_TF=0 TRANSFORMERS_NO_TF=1` before invoking the foundation-model runners (the zero-shot runners do this automatically via `os.environ.setdefault`)

### Data

The processed half-hourly JEPX dataset (Tokyo + Kansai DA + IM, 2022-03 to 2023-12) lives in `results/` as `IM_DA_TK_ALL.csv` and `IM_DA_KS_ALL.csv` once the Zenodo deposit is downloaded. The raw JEPX prices are publicly available at `jepx.jp`.

### Run

```bash
cd code
python run_lear_rolling.py              # LEAR baseline             ~10 min
python run_ddnn_rolling.py              # DDNN-Normal               ~15 min on MPS
python run_ddnn_jsu_rolling.py          # DDNN-JSU                  ~15 min on MPS
python run_nhits_rolling.py             # N-HiTS                    ~20 min on MPS
python run_patchtst_rolling.py          # PatchTST                  ~25 min on MPS
python run_itransformer_rolling.py      # iTransformer              ~10 min on MPS
python run_chronos_zeroshot.py          # Chronos-Bolt (zero-shot)  ~5 min on MPS
python run_timesfm_zeroshot.py          # TimesFM (zero-shot)       ~7 min on CPU
python fix_timesfm_clip.py              # physical-range clipping for TimesFM
python run_qra_rolling.py               # QRA-rolling ensemble      variable
python run_conformal_lear.py            # LEAR + adaptive conformal
python run_mcs_extended.py              # Hansen-Lunde-Nason MCS    ~5 min
python make_extended_summary.py         # Aggregate Tables 2-5      < 1 min
```

Total wall-clock on commodity Apple M2 hardware: ~2 hours.

## Repository layout

```
.
|-- code/                       # All Python pipeline (22 files)
|   |-- data_loader.py          # JEPX CSV loader
|   |-- feature_build.py        # Lookback + target construction
|   |-- rolling_eval.py         # Lago rolling-window evaluator
|   |-- mcs.py                  # Hansen-Lunde-Nason Model Confidence Set
|   |-- metrics_prob.py         # RMSE, MAE, CRPS, pinball, Winkler, coverage
|   |-- models_*.py             # Per-model implementations
|   |-- run_*_rolling.py        # Rolling-window runners (supervised)
|   |-- run_*_zeroshot.py       # Zero-shot runners (foundation models)
|   `-- make_extended_summary.py
|-- tex/                        # LaTeX manuscript
|   |-- main.tex                # Manuscript
|   |-- main.pdf                # Built PDF
|   |-- references.bib          # Bibliography
|   |-- cover_letter.tex        # Cover letter
|   `-- RESPONSE_TO_REVIEWERS.md
|-- results/                    # Metrics JSONs + summary CSVs
|-- figures/                    # PDF/PNG figures used in the manuscript
|-- github/                     # GitHub Actions workflow + community files
|-- zenodo/                     # Zenodo deposit metadata
|-- CITATION.cff                # GitHub Citation File Format
|-- LICENSE                     # MIT (code)
|-- LICENSE-DATA                # CC-BY-4.0 (processed dataset)
`-- requirements.txt
```

## Citing

If you use this benchmark, please cite both the manuscript (see `CITATION.cff`) and the Zenodo deposit (DOI to be supplied on acceptance).

## License

- **Code**: MIT (see `LICENSE`)
- **Processed dataset**: CC-BY-4.0 (see `LICENSE-DATA`); the underlying raw JEPX prices are public-domain market data published by the Japan Electric Power Exchange

## Contact

Ittipon Fongkaew - `ittipon@sut.ac.th` - School of Physics, Institute of Science, Suranaree University of Technology.
