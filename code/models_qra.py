"""
Quantile Regression Averaging (QRA) for probabilistic EPF.

Per Nowotarski & Weron (2015) and Uniejewski & Weron (2021):

  Given K point forecasts {f1, ..., fK} of y_t from K models, fit a quantile
  regression at each quantile level tau:

      Q_tau(y | f1, ..., fK) = beta_0(tau) + sum_k beta_k(tau) * f_k

  Predicted quantiles for new test inputs are then beta(tau)^T [1, f1, ..., fK].

This module implements the standard QRA. The smoothed-QRA variant
(Uniejewski & Weron 2021) replaces the LP-based quantile regression with an
L1-regularised smoothed loss; we use sklearn's QuantileRegressor with l1 penalty
as a reasonable proxy.

Inputs:
  F_train: (n_train, K) stacked point forecasts on a training set (e.g.,
           leave-one-out forecasts or rolling out-of-sample predictions).
  y_train: (n_train,) realised targets.
  F_test:  (n_test, K) point forecasts on the test set.
  qs:      list of quantile levels in (0, 1).

Output: (n_test, K_q) quantile predictions.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import QuantileRegressor


class QRA:
    """Quantile Regression Averaging — single-horizon version."""

    def __init__(self, qs: list[float] | np.ndarray | None = None, alpha: float = 0.0):
        if qs is None:
            qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        self.qs = np.asarray(qs)
        self.alpha = alpha  # L1 regularisation strength (0 = standard QRA)
        self.models: list[QuantileRegressor] = []

    def fit(self, F_train: np.ndarray, y_train: np.ndarray) -> "QRA":
        self.models = []
        for q in self.qs:
            m = QuantileRegressor(quantile=float(q), alpha=self.alpha, solver="highs")
            m.fit(F_train, y_train)
            self.models.append(m)
        return self

    def predict(self, F_test: np.ndarray) -> np.ndarray:
        """Return (n_test, K_q) quantile predictions."""
        Q = np.column_stack([m.predict(F_test) for m in self.models])
        # Enforce monotonicity (sort quantiles per row)
        Q = np.sort(Q, axis=1)
        return Q


class QRAMultiHorizon:
    """Apply QRA independently per horizon."""

    def __init__(self, horizon: int = 48,
                 qs: list[float] | np.ndarray | None = None, alpha: float = 0.0):
        self.horizon = horizon
        self.qs = qs
        self.alpha = alpha
        self.qras: list[QRA] = []

    def fit(self, F_train: np.ndarray, Y_train: np.ndarray) -> "QRAMultiHorizon":
        """F_train: (n, H, K)  Y_train: (n, H)."""
        assert F_train.shape[0] == Y_train.shape[0]
        assert F_train.shape[1] == self.horizon
        self.qras = []
        for h in range(self.horizon):
            qra = QRA(qs=self.qs, alpha=self.alpha)
            qra.fit(F_train[:, h, :], Y_train[:, h])
            self.qras.append(qra)
        return self

    def predict(self, F_test: np.ndarray) -> np.ndarray:
        """F_test: (n_test, H, K).  Returns (n_test, H, K_q)."""
        out = np.stack([self.qras[h].predict(F_test[:, h, :])
                         for h in range(self.horizon)], axis=1)
        return out


if __name__ == "__main__":
    # Smoke test: 2 base forecasts that bracket the true value
    np.random.seed(0)
    n_tr, n_te = 500, 200
    y_tr = np.random.randn(n_tr)
    y_te = np.random.randn(n_te)
    # f1 = y - 0.5 + noise; f2 = y + 0.3 + noise
    f1_tr = y_tr - 0.5 + 0.2 * np.random.randn(n_tr)
    f2_tr = y_tr + 0.3 + 0.2 * np.random.randn(n_tr)
    f1_te = y_te - 0.5 + 0.2 * np.random.randn(n_te)
    f2_te = y_te + 0.3 + 0.2 * np.random.randn(n_te)
    F_tr = np.column_stack([f1_tr, f2_tr])
    F_te = np.column_stack([f1_te, f2_te])

    qra = QRA(qs=[0.1, 0.5, 0.9]).fit(F_tr, y_tr)
    Q = qra.predict(F_te)
    print(f"QRA q10/q50/q90 (first 3 test points):")
    for i in range(3):
        print(f"  y={y_te[i]:+.3f}  Q10={Q[i,0]:+.3f}  Q50={Q[i,1]:+.3f}  Q90={Q[i,2]:+.3f}")
    cov80 = ((y_te >= Q[:, 0]) & (y_te <= Q[:, 2])).mean()
    print(f"Empirical 80% PI coverage: {cov80:.3f}  (target 0.8)")
