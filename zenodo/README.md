# JEPX-EPF-Benchmark — Zenodo Archive

**Version 1.0.0** | **License: CC-BY-4.0** | **Language: English**

Reproducibility package for the manuscript:

> Sriboriboon, P. & Fongkaew, I. (2026). *A Probabilistic 48-Step Forecasting Benchmark for the Japan Electric Power Exchange: Conformal, Distributional, and Quantile-Averaging Methods Across Day-Ahead and Imbalance Markets.* Submitted to *Applied Energy*.

## What this archive contains

This Zenodo deposit is the canonical reproducibility package for the JEPX-EPF benchmark. It bundles:

1. **Processed dataset** (`dataset/`):
   - `IM_DA_TK_ALL.csv` (3.0 MB) — Tokyo area, 30-min, 2022-03-01 → 2023-12-31, 32,208 rows × 24 columns
   - `IM_DA_KS_ALL.csv` (3.0 MB) — Kansai area, same span, same schema
   - Each CSV contains: `DATETIME`, `DA_<area>`, `IM_<area>`, `IMV_<area>` (imbalance volume), `DA_SYSTEM_PRICE`, weather (`TEMPERATURE`, `RADIATION`, `WIND_SPEED`, `PRECIPITATION`, `WIND_DIRECTION`), fuel/macro (`GAS_USD`, `USD_JPY`, `GAS_JPY`), and calendar columns (`DAY_OF_WEEK`, `DAY_TYPE`, `PEAK_OFFPEAK`).

2. **Python benchmarking pipeline** (`code/`):
   - `data_loader.py` — unified loader for both areas
   - `feature_build.py` — LEAR / engineered / raw lookback features
   - `models_lear.py` — LASSO-LARS-IC (Lago 2021)
   - `models_ddnn.py` — DDNN-Normal + DDNN-Johnson-SU (Marcjasz 2023)
   - `models_nhits.py` — Self-contained N-HiTS (Olivares 2023)
   - `models_qra.py` — Quantile Regression Averaging (Uniejewski-Weron 2021)
   - `models_conformal.py` — Split + adaptive conformal prediction (Gibbs-Candès 2021)
   - `rolling_eval.py` — generic rolling-window evaluator (365-day window, weekly refit)
   - `mcs.py` — Hansen-Lunde-Nason Model Confidence Set with stationary bootstrap
   - `metrics_prob.py` — CRPS, pinball, Winkler, coverage
   - `run_lear_rolling.py`, `run_ddnn_rolling.py`, `run_ddnn_jsu_rolling.py`,
     `run_nhits_rolling.py`, `run_qra_rolling.py`, `run_conformal_lear.py`,
     `run_mcs.py` — end-to-end experiment drivers
   - `make_*_summary.py`, `fig_*.py` — table and figure generation

3. **Per-issuance prediction matrices** (`results/`):
   - For each of 6 rolling models × 4 (area, market) pairs:
     - Mean / median predictions: `<model>_<area>_<market>_preds.csv` or `_mu.csv`
     - Standard deviation (where applicable): `_sigma.csv`
     - Full quantile cubes (DDNN-JSU, QRA): `.npz` arrays
   - Aggregated metrics JSON: `lear_metrics.json`, `ddnn_metrics.json`, `ddnn_jsu_metrics.json`, `nhits_metrics.json`, `qra_rolling_metrics.json`, `conformal_lear_metrics.json`, `mcs_results.json`
   - Master summary CSV: `month3_summary.csv`

4. **Figures** (`figures/`):
   - 7 PDF + PNG figures used in the manuscript (price dynamics, model scatter, per-horizon RMSE, conformal calibration, CRPS comparison, Winkler comparison, MCS visualization).

5. **Manuscript source** (`tex/`):
   - `main.tex`, `references.bib`, vendored Elsevier `elsarticle.cls` + bibstyle
   - Cover letter source
   - `main.pdf` (17 pages) + `cover_letter.pdf` (2 pages)

## Reproduce in 80 minutes

```bash
# Minimum Python environment (3.8+):
pip install lightgbm xgboost torch scikit-learn statsmodels pandas numpy matplotlib

# Run the full pipeline (sequential, ~80 minutes on Apple M2):
python code/run_lear_rolling.py            # ~30 min  (LEAR rolling on 4 pairs)
python code/run_ddnn_rolling.py            #  ~5 min  (DDNN-Normal rolling)
python code/run_ddnn_jsu_rolling.py        # ~15 min  (DDNN-Johnson-SU rolling)
python code/run_nhits_rolling.py           # ~25 min  (N-HiTS rolling, hidden=128)
python code/run_qra_rolling.py             # ~10 min  (Rolling QRA, 90-day calibration)
python code/run_conformal_lear.py          #   <1 min (Split + adaptive conformal)
python code/run_mcs.py                     #   <1 min (Model Confidence Set)
python code/make_month3_summary.py         #   <1 min (Summary tables + figures)
python code/fig_crps_winkler.py            #   <1 min (Final figures)
```

## Key findings reproducible from this archive

| Metric | Tokyo DA | Tokyo IM | Kansai DA | Kansai IM |
|--------|----------|----------|-----------|-----------|
| QRA-rolling rRMSE | **0.756** | **0.751** | **0.839** | **0.789** |
| QRA-rolling CRPS  | **1.229** | **1.947** | **1.592** | **1.908** |
| MCS @ α=0.10      | {N-HiTS}  | {LEAR}   | {N-HiTS}  | {LEAR}   |

## Citation

If you use this archive, please cite **both** the Zenodo record and the Applied Energy paper:

```bibtex
@dataset{jepx_epf_benchmark_zenodo,
  author       = {Sriboriboon, Panithan and Fongkaew, Ittipon},
  title        = {{JEPX-EPF-Benchmark}: Probabilistic 48-Step Electricity Price
                  Forecasting Dataset and Code for the Japan Electric Power Exchange},
  month        = {<month of acceptance>},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {<to-be-supplied-on-deposit>},
  url          = {<to-be-supplied-on-deposit>}
}

@article{jepx_epf_benchmark_paper,
  author  = {Sriboriboon, Panithan and Fongkaew, Ittipon},
  title   = {A Probabilistic 48-Step Forecasting Benchmark for the Japan
             Electric Power Exchange},
  journal = {Applied Energy},
  year    = {2026},
  doi     = {<to-be-supplied-on-acceptance>}
}
```

## License

- **Code**: MIT License (see `LICENSE` in the GitHub repository)
- **Processed data + prediction matrices + figures**: Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Raw JEPX prices**: public domain via JEPX information disclosure portal; redistribution permitted under the JEPX disclosure terms (https://www.jepx.jp/en/)

## Data sources

| Source | Variables | URL |
|--------|-----------|-----|
| JEPX | DA, IM, IMV, DA_SYSTEM_PRICE, DA_SELL_V, DA_BUY_V, DA_TOTAL_V | https://www.jepx.jp/en/ |
| Japan TSOs (via JEPX aggregation) | IM_TK, IM_KS | publicly aggregated by area TSOs |
| TOCOM / public LNG | GAS_USD, GAS_JPY | via Bank of Japan / TOCOM |
| Bank of Japan | USD_JPY | https://www.boj.or.jp/en/ |
| JMA AMeDAS | weather | https://www.data.jma.go.jp/ |

## Contact

For questions about the dataset, methodology, or code, please contact:

**Ittipon Fongkaew** (corresponding author)
School of Physics, Institute of Science
Suranaree University of Technology
111 University Avenue, Nakhon Ratchasima 30000, Thailand
Email: ittipon@sut.ac.th

## Acknowledgments

We thank the JEPX information disclosure team and the area transmission system operators for the public data feed. This work was supported by the School of Physics, Institute of Science, Suranaree University of Technology.

## Changelog

- **1.0.0** (2026): Initial deposit corresponding to the Applied Energy submission.
