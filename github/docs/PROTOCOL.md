# Benchmark protocol (detailed)

Detailed methodology notes for `jepx-epf-benchmark`. The Applied Energy manuscript is the primary reference for everything below; this document fills in implementation-level detail that a reader may want when reproducing the benchmark or extending it to a new model class.

---

## 1. Data

### 1.1 Sources
The processed CSVs in `dataset/` are derived from:

- **JEPX information disclosure portal** for DA spot price (`DA_TK`, `DA_KS`), DA volumes (`DA_SELL_V`, `DA_BUY_V`, `DA_TOTAL_V`), and the area-system reference price (`DA_SYSTEM_PRICE`).
- **Area Transmission System Operators (TSOs)** for imbalance prices (`IM_TK`, `IM_KS`) and imbalance volumes (`IMV_TK`, `IMV_KS`), as aggregated by JEPX.
- **Bank of Japan / TOCOM** for `USD_JPY`, `GAS_USD`, `GAS_JPY`.
- **Japan Meteorological Agency (JMA) AMeDAS** for `TEMPERATURE`, `RADIATION`, `WIND_SPEED`, `WIND_DIRECTION`, `PRECIPITATION` (station-aggregated to the area level).

### 1.2 Schema
Both `IM_DA_TK_ALL.csv` and `IM_DA_KS_ALL.csv` share the same 24-column schema (column 8 differs in the price label):

```
DATETIME, DATE, TIME_CODE, DA_SELL_V, DA_BUY_V, DA_TOTAL_V,
DA_SYSTEM_PRICE, DA_<area>, TIME, IM_<area>, IMV_<area>,
PRECIPITATION, RADIATION, TEMPERATURE, WIND_DIRECTION, WIND_SPEED,
GAS_USD, USD_JPY, GAS_JPY, DAY_OF_WEEK, DAY_TYPE, PEAK_OFFPEAK,
DAY_TYPE_CODE, PEAK_OFFPEAK_CODE
```

The Tokyo CSV stores `DATETIME` in ISO format (`2022-03-01 00:00:00`); the Kansai CSV uses US-locale (`3/1/2022 0:00`). `data_loader.load_area(area)` normalises both to a 30-min DatetimeIndex.

### 1.3 Time span
2022-03-01 00:00 → 2023-12-31 23:30, half-hourly. 32,208 rows per CSV.

---

## 2. Forecasting target and splits

- **Target**: at every issuance time *t*, the 48-vector \(y_{t+1}, y_{t+2}, \ldots, y_{t+48}\) (the next 24 hours from *t*).
- **Train**: a 365-day sliding window ending strictly before each test block (17,520 issuance times).
- **Validation**: last 10% of the current training window (for early stopping in DDNN, N-HiTS).
- **Test**: 2023-03-01 → 2023-12-30, recalibrated every 7 days (44 refit blocks × 336 issuance/block = 14,640 issuance times per (area, market) pair).
- **Look-ahead avoidance**: each test block's predictions are produced by a model fit on data strictly before the block start.

---

## 3. Model details

### 3.1 LEAR (Lago 2021)
- 77 features per issuance: lagged y at {0,1,2,47,48,49,95,96,97,144,191,192,193,335,336,337}; 7 exogenous regressors at *t*; 47 slot-of-day dummies; 6 day-of-week dummies; weekend indicator.
- One `LassoLarsIC` per horizon (BIC-tuned), trained on standardised features.

### 3.2 DDNN-Normal / DDNN-Johnson-SU (Marcjasz 2023)
- Backbone: 77 → 256 → 128 (ReLU + dropout 0.2).
- DDNN-Normal head: two 48-dim outputs (μ, log σ → softplus).
- DDNN-JSU head: four 48-dim outputs (ξ, λ, γ, δ; positivity via softplus where needed).
- Trained by NLL, Adam(lr=1e-3, weight_decay=1e-5), batch 512, up to 25 epochs with patience-5 early stopping on validation NLL.

### 3.3 N-HiTS (Olivares 2023, simplified)
- Backbone: 3 stacks at pool sizes {1, 4, 16}; each stack has a 2-layer MLP (hidden 128) with backcast and forecast heads.
- Trained by MSE with Adam(lr=1e-3, wd=1e-5), batch 512, up to 12 epochs with patience-5 early stopping.
- Input: 336-step raw (standardised) lookback.

### 3.4 QRA-rolling (Uniejewski-Weron 2021)
- Base learners: LEAR + DDNN-Normal point forecasts.
- Per-horizon `sklearn.linear_model.QuantileRegressor` at quantile levels {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}.
- Calibration window: 90 days (4,320 issuance times) sliding strictly before the eval window.
- Refit cadence: every 7 days.
- Quantile post-processing: per-row sort to enforce monotonicity.

### 3.5 LEAR + adaptive conformal (Gibbs-Candès 2021)
- Split conformal calibration on the first 30 days of the test period.
- Adaptive update: \(q_{h, t+1} = q_{h, t} + \gamma (\alpha - \mathbf 1\{y \notin \text{interval}\})\), \(\gamma = 0.005\).
- Lower bound on \(q_{h, t}\): 0.

---

## 4. Statistical testing

### 4.1 Diebold-Mariano (per-horizon)
- Loss differential per (issuance, horizon).
- HAC variance via Newey-West with lag \(\lfloor T^{1/3} \rfloor\).
- Harvey-Leybourne-Newbold small-sample correction: \(k_{\text{HLN}} = \sqrt{(T + 1 - 2h + h(h-1)/T)/T}\).
- Two-sided p-value against Student-t with \(T-1\) df.

### 4.2 Model Confidence Set (Hansen-Lunde-Nason 2011)
- Loss per model per issuance: mean across horizons of squared error.
- T_R range statistic over all pairwise standardised loss differentials.
- Stationary-bootstrap critical values (1000 replications, mean block length \(T^{1/3}\)).
- Iterative elimination of worst model until null of equal predictive ability fails to be rejected.

---

## 5. Metrics

### 5.1 Point
- RMSE, MAE on the pooled (issuance × horizon) loss matrix.
- rRMSE = RMSE_model / RMSE_weekly_naive (similarly rMAE).

### 5.2 Probabilistic
- CRPS:
  - DDNN-Normal: closed form Φ-based expression.
  - DDNN-JSU, QRA, conformal: \(\mathrm{CRPS} \approx 2 \cdot \mathrm{mean\,pinball}\) over the K reported quantile levels (Gneiting-Ranjan approximation).
- Pinball: \(\rho_\tau(u) = u(\tau - \mathbf 1\{u < 0\})\), averaged across (issuance, horizon, quantile).
- Winkler@(1-α): \((U - L) + \frac{2}{\alpha}[(L - y)\mathbf 1\{y<L\} + (y - U)\mathbf 1\{y>U\}]\).
- Coverage: empirical proportion of \(y \in [L, U]\).

---

## 6. Reproducibility hygiene

- Deterministic seeds set globally: `numpy.random.seed(0)`, `torch.manual_seed(0)`.
- Single-thread CPU determinism is **not** guaranteed (`torch` on CPU still has minor non-determinism in some kernels); we observed up to ±1% drift in DDNN final test RMSE across reruns on Apple M2.
- All experiment runners use a "skip-if-already-done" guard, so re-running the pipeline after a partial completion will not redo finished work.

---

## 7. Known limitations

See Section 5 (Discussion) of the paper for the four named limitations: (1) 305-day test window may not include a major spike event, (2) weekly recalibration cadence (vs Lago's daily), (3) QRA-rolling base learners limited to LEAR + DDNN-Normal, (4) intraday and 1-hour-ahead JEPX products not covered.

---

For questions about the implementation that aren't covered in this document, please open an issue on the GitHub repository.
