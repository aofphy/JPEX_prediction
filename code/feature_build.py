"""
Feature construction for JEPX 48-step EPF.

Two feature sets:
  - lear_features(s, exog): LEAR-style (Lago 2021 adapted to 30-min, 48 horizons).
    Per-horizon features include lagged y, lagged exog, and calendar dummies.
    Linear models (LEAR / LASSO) trained on these.
  - rich_features(s, exog): full engineered feature set used by GBDT/DL models
    (lags, rolling stats, EWMA, calendar, Fourier, exogenous).

Targets:
  - build_targets(s, horizon=48): (T, 48) matrix where row t holds y_{t+1..t+48}.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

LOOKBACK = 336  # 7 days * 48 slots
HORIZON = 48    # default forecast horizon (48 half-hour steps = 24 hours)


def build_targets(y: pd.Series, horizon: int = 48) -> pd.DataFrame:
    return pd.DataFrame({f"y_h{h}": y.shift(-h) for h in range(1, horizon + 1)}, index=y.index)


def lear_features(y: pd.Series, exog: pd.DataFrame) -> pd.DataFrame:
    """LEAR-style feature set: lagged y, lagged exog, calendar dummies.

    Following Lago et al. (2021) but adapted to 30-min granularity and a
    48-step forecast horizon. Features are computed at issuance time t and
    used to predict y_{t+h} for all h.
    """
    out = pd.DataFrame(index=y.index)
    # Lagged prices: current + key seasonal lags
    out["y_lag0"] = y.shift(0)
    for k in [1, 2, 47, 48, 49, 95, 96, 97, 144, 191, 192, 193, 335, 336, 337]:
        out[f"y_lag{k}"] = y.shift(k)
    # Lagged exogenous (use lag 0 = available at issuance)
    for c in exog.columns:
        out[f"x_{c}"] = exog[c].values
    # Hour-of-day dummies (47 dummies; one slot dropped to avoid collinearity)
    slot = y.index.hour * 2 + (y.index.minute // 30)
    for s in range(1, 48):
        out[f"slot_{s}"] = (slot == s).astype(float)
    # Day-of-week dummies (6, dropping Monday)
    dow = y.index.dayofweek
    for d in range(1, 7):
        out[f"dow_{d}"] = (dow == d).astype(float)
    # Weekend
    out["is_weekend"] = (dow >= 5).astype(float)
    return out


def rich_features(y: pd.Series, exog: pd.DataFrame) -> pd.DataFrame:
    """Engineered feature set used by GBDT/DL models (24 features)."""
    out = pd.DataFrame(index=y.index)
    out["y_lag0"] = y.shift(0)
    for lag in [1, 2, 47, 48, 49, 95, 96, 97, 335, 336, 337]:
        out[f"lag_{lag}"] = y.shift(lag)
    for win in [48, 336]:
        out[f"rmean_{win}"] = y.rolling(win).mean()
        out[f"rstd_{win}"] = y.rolling(win).std()
    for span in [48, 336]:
        out[f"ewma_{span}"] = y.ewm(span=span, adjust=False).mean()
    idx = y.index
    out["hour"] = idx.hour
    out["minute"] = idx.minute
    out["dow"] = idx.dayofweek
    out["doy"] = idx.dayofyear
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["sin_h"] = np.sin(2 * np.pi * idx.hour / 24)
    out["cos_h"] = np.cos(2 * np.pi * idx.hour / 24)
    out["sin_dow"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["cos_dow"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    for c in exog.columns:
        out[c] = exog[c].values
    return out


def raw_lookback(y: pd.Series, lookback: int = LOOKBACK) -> pd.DataFrame:
    return pd.DataFrame(
        {f"lag{k}": y.shift(k) for k in range(0, lookback)}, index=y.index
    )
