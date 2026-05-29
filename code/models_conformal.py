"""
Split conformal prediction wrapper.

Wraps any point forecaster f(X) -> Y_hat to produce calibrated prediction
intervals with marginal coverage guarantee 1 - alpha.

Protocol (per Lei, G'Sell, Rinaldo, Tibshirani & Wasserman 2018):
  - Split training data into proper training set + calibration set.
  - Fit f on proper training.
  - Compute absolute residuals on calibration set: r_i = |y_i - f(x_i)|.
  - For test input x, the (1 - alpha) prediction interval is
        [f(x) - q, f(x) + q]
    where q = (1 - alpha) empirical quantile of {r_i}, with a +1
    finite-sample correction.

For multi-output (48-step) forecasting, we compute one quantile per horizon
to allow horizon-specific interval widths. This is the standard practice
in EPF conformal applications (Kath & Ziel 2021, EJOR).

This module provides a thin, framework-agnostic wrapper that takes
calibration-set y / y_hat arrays and returns the quantile thresholds.
"""
from __future__ import annotations
import numpy as np


def calibrate(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    alphas: list[float] = [0.1, 0.2, 0.5],
) -> dict[float, np.ndarray]:
    """Compute conformal quantile thresholds per horizon.

    Args:
        y_cal: (n_cal, H) realized targets on the calibration set.
        yhat_cal: (n_cal, H) point predictions on the calibration set.
        alphas: list of miscoverage levels (e.g., [0.1] for 90% PI).

    Returns:
        Dict mapping alpha -> (H,) array of quantile thresholds q_h.
        The (1-alpha) interval for horizon h+1 is [f(x)_h - q_h, f(x)_h + q_h].
    """
    residuals = np.abs(y_cal - yhat_cal)  # (n_cal, H)
    n_cal = residuals.shape[0]
    out: dict[float, np.ndarray] = {}
    for alpha in alphas:
        # Finite-sample correction: ceil((n_cal + 1) * (1 - alpha)) / n_cal
        q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
        q_h = np.quantile(residuals, q_level, axis=0)
        out[alpha] = q_h
    return out


def predict_intervals(
    yhat_test: np.ndarray,
    q_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build conformal prediction intervals.

    Args:
        yhat_test: (n_test, H) test-set point predictions.
        q_h: (H,) conformal quantile thresholds.

    Returns:
        (lower, upper) each (n_test, H).
    """
    lower = yhat_test - q_h[None, :]
    upper = yhat_test + q_h[None, :]
    return lower, upper


def adaptive_conformal_update(q_t: np.ndarray, in_interval_t: np.ndarray,
                                alpha: float = 0.1, gamma: float = 0.005) -> np.ndarray:
    """One-step adaptive conformal update (Gibbs & Candes 2021).

    For each horizon h, update q_h based on whether the realisation was
    inside the previous interval:
        q_{t+1} = q_t + gamma * (alpha - 1{y_t outside interval})

    Args:
        q_t: (H,) current quantile thresholds.
        in_interval_t: (H,) 0/1 indicators of whether y_t was inside the
                       previous interval at each horizon.
        alpha: target miscoverage.
        gamma: learning rate.

    Returns:
        Updated q_{t+1}.
    """
    err_t = (1.0 - in_interval_t).astype(float)  # 1 if outside, 0 if inside
    return q_t + gamma * (err_t - alpha)


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    n_cal, n_te, H = 1000, 300, 48
    y_cal = np.random.randn(n_cal, H)
    yhat_cal = y_cal + np.random.randn(n_cal, H) * 0.5
    q90 = calibrate(y_cal, yhat_cal, [0.1])[0.1]
    print(f"Calibration q90 per horizon (first 5): {q90[:5]}")
    y_te = np.random.randn(n_te, H)
    yhat_te = y_te + np.random.randn(n_te, H) * 0.5
    lo, hi = predict_intervals(yhat_te, q90)
    cov = ((y_te >= lo) & (y_te <= hi)).mean()
    print(f"Empirical 90% coverage: {cov:.3f}  (target 0.9)")
