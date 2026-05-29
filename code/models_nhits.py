"""
N-HiTS — Simplified Neural Hierarchical Interpolation for Time Series.

Reference: Challu, Olivares, Oreshkin, Garza, Mergenthaler-Canseco, Dubrawski
(2023) "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting",
AAAI 2023. Implementation here is a self-contained PyTorch version, simplified
from the official `neuralforecast` library (which had Numba compatibility
issues in the deployment environment).

Architecture (3 stacks, K = 3):
  - Each stack k consumes the 336-step raw lookback x[-L:].
  - Stack-k MaxPool the lookback with kernel = pool_size[k] -> downsampled
    representation (capturing different frequency bands).
  - 2-layer MLP maps downsampled lookback -> (backcast, forecast) outputs.
  - Backcast is linearly upsampled back to L steps and subtracted from x.
  - The next stack receives the residual.
  - Final 48-step forecast = sum of per-stack forecast outputs.

This implementation produces a deterministic 48-vector per (issuance) row.
For probabilistic output we use the post-hoc adaptive conformal wrapper.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NHiTSBlock(nn.Module):
    def __init__(self, lookback: int, horizon: int, pool_size: int,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.pool_size = pool_size
        pooled_len = math.ceil(lookback / pool_size)
        # Downsampling via max-pool happens in forward; MLP operates on pooled
        # 1D feature vector.
        self.mlp = nn.Sequential(
            nn.Linear(pooled_len, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        # Two heads: backcast (lookback-length) and forecast (horizon-length)
        # via small intermediate (coefficient) representation.
        self.backcast_head = nn.Linear(hidden, lookback)
        self.forecast_head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (n, lookback)
        # Max-pool by treating x as (n, 1, lookback)
        x_pooled = F.max_pool1d(
            x.unsqueeze(1), kernel_size=self.pool_size,
            stride=self.pool_size, ceil_mode=True,
        ).squeeze(1)
        h = self.mlp(x_pooled)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        return backcast, forecast


class NHiTS(nn.Module):
    def __init__(self, lookback: int = 336, horizon: int = 48,
                 pool_sizes: list[int] = (1, 4, 16),
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.blocks = nn.ModuleList([
            NHiTSBlock(lookback, horizon, ps, hidden=hidden, dropout=dropout)
            for ps in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, lookback)
        residual = x
        forecast_sum = torch.zeros(x.size(0), self.horizon, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast
        return forecast_sum


def train_nhits(model, Xtr, Ytr, Xv, Yv, epochs=30, batch=512, lr=1e-3, wd=1e-5):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(Xtr); Yt = torch.tensor(Ytr)
    Xv_ = torch.tensor(Xv); Yv_ = torch.tensor(Yv)
    n = len(Xt); best = float("inf"); state = None; bad = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            l = loss_fn(model(Xt[idx]), Yt[idx])
            l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(Xv_), Yv_).item())
        if v < best - 1e-5:
            best = v
            state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 5: break
    if state is not None: model.load_state_dict(state)
    return model, best


if __name__ == "__main__":
    np.random.seed(0); torch.manual_seed(0)
    n, L, H = 800, 336, 48
    x = np.random.randn(n, L).astype(np.float32)
    # Synthetic target: lag-48 + small noise
    y = np.stack([x[:, L - 48 + h] + 0.2 * np.random.randn(n) for h in range(H)], axis=1).astype(np.float32)
    m = NHiTS(lookback=L, horizon=H, pool_sizes=[1, 4, 16], hidden=128)
    m, vloss = train_nhits(m, x[:600], y[:600], x[600:], y[600:], epochs=20)
    print(f"N-HiTS val MSE: {vloss:.4f}")
    test_pred = m(torch.tensor(x[600:])).detach().numpy()
    rmse = float(np.sqrt(((test_pred - y[600:]) ** 2).mean()))
    print(f"N-HiTS smoke RMSE: {rmse:.3f}  (expect ~ 0.2)")
