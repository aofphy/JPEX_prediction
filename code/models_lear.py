"""
LEAR — LASSO-Estimated AutoRegressive model for JEPX EPF.

Follows the Lago et al. (2021) "lear" specification, adapted to 30-min /
48-step forecasting:

  - One LassoLars model per forecast horizon h = 1..48.
  - LASSO regularization parameter selected by BIC (LassoLarsIC), avoiding
    expensive cross-validation that breaks time-series ordering.
  - Features (constructed by feature_build.lear_features):
      * Lagged y at {0, 1, 2, 47, 48, 49, 95, 96, 97, 144, 191, 192, 193,
        335, 336, 337}
      * Lagged exogenous regressors at lag 0
      * 47 slot-of-day dummies
      * 6 day-of-week dummies
      * Weekend dummy
  - Inputs are standardised per fold (LassoLars expects centred features).
  - The model returns point predictions; a residual-bootstrap wrapper builds
    quantile bands for probabilistic evaluation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import StandardScaler


class LEAR:
    """LEAR with one LassoLars-IC per horizon and BIC-based regularisation."""

    def __init__(self, horizon: int = 48, criterion: str = "bic", max_iter: int = 1000):
        self.horizon = horizon
        self.criterion = criterion
        self.max_iter = max_iter
        self.models: list[LassoLarsIC] = []
        self.scalers: list[StandardScaler] = []
        self.feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "LEAR":
        """X: (n, p) features at issuance time t; Y: (n, H) targets y_{t+h}."""
        assert Y.shape[1] == self.horizon, f"Y must have {self.horizon} cols"
        self.models = []
        self.scalers = []
        self.feature_names = list(X.columns)
        Xv = X.values
        for h in range(1, self.horizon + 1):
            sc = StandardScaler()
            Xs = sc.fit_transform(Xv)
            yh = Y[f"y_h{h}"].values
            model = LassoLarsIC(criterion=self.criterion, max_iter=self.max_iter,
                                 fit_intercept=True)
            model.fit(Xs, yh)
            self.models.append(model)
            self.scalers.append(sc)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return (n, H) prediction matrix."""
        n = len(X)
        P = np.zeros((n, self.horizon), dtype=float)
        Xv = X.values
        for i, (model, sc) in enumerate(zip(self.models, self.scalers)):
            Xs = sc.transform(Xv)
            P[:, i] = model.predict(Xs)
        return P

    def n_nonzero(self) -> np.ndarray:
        """Sparsity per horizon — number of nonzero coefficients each h."""
        return np.array([np.count_nonzero(m.coef_) for m in self.models])

    def selected_features(self, h: int) -> list[str]:
        """Names of features with nonzero coefficient at horizon h (1-indexed)."""
        m = self.models[h - 1]
        return [self.feature_names[i] for i, c in enumerate(m.coef_) if abs(c) > 1e-12]


if __name__ == "__main__":
    # Smoke test on synthetic data
    np.random.seed(0)
    n = 2000
    p = 30
    H = 48
    X = pd.DataFrame(np.random.randn(n, p), columns=[f"x{i}" for i in range(p)])
    # Y_h = x0 + 0.5*x1 + 0.2*h + noise
    Y = pd.DataFrame(
        {f"y_h{h}": X["x0"].values + 0.5 * X["x1"].values + 0.2 * h + np.random.randn(n) * 0.3
         for h in range(1, H + 1)},
        index=X.index,
    )
    lear = LEAR(horizon=H).fit(X.iloc[:1500], Y.iloc[:1500])
    pred = lear.predict(X.iloc[1500:])
    actual = Y.iloc[1500:].values
    rmse = float(np.sqrt(((pred - actual) ** 2).mean()))
    print(f"LEAR smoke-test RMSE = {rmse:.3f} (expect ~ 0.30)")
    print(f"nonzero coefs per horizon: min={lear.n_nonzero().min()}, max={lear.n_nonzero().max()}, mean={lear.n_nonzero().mean():.1f}")
