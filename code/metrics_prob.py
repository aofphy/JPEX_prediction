"""
Probabilistic forecasting metrics for EPF.

Used downstream of distributional / quantile / ensemble models that emit
either:
  - a set of K quantile predictions per (issuance, horizon), or
  - a parametric distribution (location, scale) per (issuance, horizon).

All metrics return one scalar per pooled set or a per-horizon array.

References:
  - Gneiting & Raftery (2007), JASA - Strictly proper scoring rules.
  - Hong et al. (2016), IJF - GEFCom2014 probabilistic forecast benchmarks.
  - Marcjasz et al. (2023), Energy Economics - DDNN evaluation conventions.
"""
from __future__ import annotations
import numpy as np


def pinball_loss(y_true: np.ndarray, quantiles: np.ndarray, qs: np.ndarray) -> float:
    """Pinball (quantile) loss averaged across (n, h, q).

    y_true:    (n, h) realised values.
    quantiles: (n, h, q) predicted quantiles for each of K quantile levels.
    qs:        (q,) quantile levels in (0, 1).
    """
    y = y_true[..., None]
    diff = y - quantiles
    pos = qs * np.where(diff > 0, diff, 0.0)
    neg = (qs - 1) * np.where(diff <= 0, diff, 0.0)
    losses = pos + neg
    return float(losses.mean())


def crps_quantile(y_true: np.ndarray, quantiles: np.ndarray, qs: np.ndarray) -> float:
    """Approximate CRPS via quantile-based trapezoidal integration.

    For a vector of quantile levels qs and the corresponding quantile values
    in `quantiles`, CRPS ≈ 2 * mean(pinball_loss across quantile levels).
    """
    # Pinball-based CRPS approximation (Gneiting & Ranjan 2011, eq. 17)
    return 2.0 * pinball_loss(y_true, quantiles, qs)


def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.1) -> float:
    """Winkler interval score for a (1-alpha) prediction interval.

    Reward = (upper - lower) plus a penalty if y_true falls outside.
    Average over all (n, h) entries.
    """
    width = upper - lower
    below = (y_true < lower).astype(float)
    above = (y_true > upper).astype(float)
    penalty = (2 / alpha) * (lower - y_true) * below + (2 / alpha) * (y_true - upper) * above
    return float((width + penalty).mean())


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage of a prediction interval."""
    return float(((y_true >= lower) & (y_true <= upper)).mean())


def crps_normal(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form CRPS for Normal predictive distribution.

    CRPS(N(mu, sigma), y) = sigma * (z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi))
    where z = (y - mu) / sigma.
    """
    from scipy.stats import norm
    sigma = np.clip(sigma, 1e-6, None)
    z = (y_true - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(crps.mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    n, h, K = 100, 48, 9
    qs = np.linspace(0.1, 0.9, K)
    mu = np.random.randn(n, h)
    sigma = np.ones((n, h)) * 0.5
    from scipy.stats import norm
    quants = mu[..., None] + sigma[..., None] * norm.ppf(qs)
    y = mu + np.random.randn(n, h) * 0.5
    print("pinball:", pinball_loss(y, quants, qs))
    print("CRPS (quantile):", crps_quantile(y, quants, qs))
    print("CRPS (normal):", crps_normal(y, mu, sigma))
    print("Winkler@90:", winkler_score(y, quants[..., 0], quants[..., -1], alpha=0.2))
    print("coverage@90:", coverage(y, quants[..., 0], quants[..., -1]))
