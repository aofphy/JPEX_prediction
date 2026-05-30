# JEPX Electricity-Price Forecasting Benchmark

A reproducible probabilistic forecasting benchmark for the **Japan Electric Power Exchange (JEPX)**, covering Tokyo and Kansai areas across day-ahead and imbalance markets at 30-minute resolution.

The benchmark follows the rolling-window protocol of Lago et al. (2021) and compares ten 48-step forecasters, including classical regression, distributional neural networks, modern Transformer architectures, and pretrained zero-shot foundation models.

---

## Headline result

A pretrained zero-shot foundation model (Chronos-Bolt) is statistically tied with the strongest supervised method (iTransformer) across every (area, market) pair, and is the sole singleton of the Hansen-Lunde-Nason Model Confidence Set on Tokyo imbalance, at zero training cost.

## Models

| Model | Family |
|---|---|
| WeeklyNaive | seasonal baseline |
| LEAR | LASSO-LARS regression |
| DDNN-Normal / DDNN-JSU | distributional neural networks |
| N-HiTS | hierarchical multi-rate decomposition |
| PatchTST | patching Transformer |
| iTransformer | variate-inverted Transformer |
| Chronos-Bolt | pretrained foundation model (zero-shot) |
| TimesFM | pretrained foundation model (zero-shot) |
| QRA-rolling | quantile regression averaging ensemble |

## Install

```bash
git clone https://github.com/aofphy/JPEX_prediction.git
cd JPEX_prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. Apple Silicon (MPS) or NVIDIA GPU recommended; CPU also works. The two foundation models will download approximately 1 GB of pretrained weights on first invocation.

## Run

The processed half-hourly JEPX dataset (Tokyo + Kansai, day-ahead + imbalance, 2022-03 to 2023-12) is hosted on Zenodo. The raw prices are publicly available at [jepx.jp](https://www.jepx.jp).

```bash
cd code
python run_lear_rolling.py
python run_ddnn_rolling.py
python run_ddnn_jsu_rolling.py
python run_nhits_rolling.py
python run_patchtst_rolling.py
python run_itransformer_rolling.py
python run_chronos_zeroshot.py
python run_timesfm_zeroshot.py
python fix_timesfm_clip.py
python run_qra_rolling.py
python run_conformal_lear.py
python run_mcs_extended.py
python make_extended_summary.py
```

Total wall-clock on commodity Apple M2 hardware is approximately two hours.

## Repository layout

```
code/        Python pipeline (rolling-window evaluator, model implementations, MCS, metrics)
results/    Per-model JSON metric files and aggregated summary tables
figures/    Publication figures
tex/        LaTeX source of the accompanying manuscript
```

## Citation

If you use this benchmark or the released prediction matrices in your work, please cite the manuscript and the Zenodo deposit. See [`CITATION.cff`](CITATION.cff) for the GitHub Citation File Format entry.

## License

- **Code** — [MIT](LICENSE)
- **Processed dataset** — [CC-BY-4.0](LICENSE-DATA). The underlying raw JEPX prices are public-domain market data published by the Japan Electric Power Exchange.

## Contact

[Ittipon Fongkaew](mailto:ittipon@sut.ac.th) — School of Physics, Institute of Science, Suranaree University of Technology.
