"""iTransformer — A simplified univariate PyTorch implementation.

Reference: Liu, Hu, Wang, Long, Wen (2024), "iTransformer: Inverted
Transformers Are Effective for Time Series Forecasting", ICLR 2024.

iTransformer inverts the standard Transformer-for-time-series formulation:
each variate's full L-length series is embedded as a single token, then
self-attention is performed across variates. For univariate forecasting we
synthesise a small variate dimension by adding the lookback-mean, the
lookback-std, and per-quartile slot indicators as auxiliary "variates", to
exercise the inverted-attention mechanism. The result is a strong univariate
Transformer baseline.

Architecture (V variates, each embedded to d_model):
  1. Build V variate channels from the univariate lookback:
     [raw, mean-removed, last-day, last-week]
  2. Embed each L-length variate via linear(L -> d_model);
  3. 2-layer Transformer encoder across V tokens (self-attention over variates);
  4. Projection: token vector for variate 0 (the price residual) -> H-step forecast.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


class ITransformer(nn.Module):
    def __init__(self, lookback: int = 336, horizon: int = 48,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_variates = 4
        self.embed = nn.Linear(lookback, d_model)
        self.var_pos = nn.Parameter(torch.randn(1, self.n_variates, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, horizon)
        self.norm = nn.LayerNorm(d_model)

    def build_variates(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        L = x.size(1)
        # Variate 1: raw lookback
        v1 = x
        # Variate 2: mean-removed (centred)
        v2 = x - x.mean(dim=1, keepdim=True)
        # Variate 3: roll back one day (48 steps); pad with previous values
        v3 = torch.roll(x, shifts=48, dims=1)
        v3[:, :48] = x[:, :48]
        # Variate 4: roll back one week (336 steps); since L == 336 this is the
        # full-series shifted by L which is invalid; instead build week-difference
        v4 = x - v3
        variates = torch.stack([v1, v2, v3, v4], dim=1)  # (B, V, L)
        return variates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # Per-instance reversible instance normalisation
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-5)
        x_n = (x - mean) / std
        variates = self.build_variates(x_n)  # (B, V, L)
        tokens = self.embed(variates) + self.var_pos  # (B, V, d_model)
        tokens = self.norm(self.encoder(tokens))
        # Project token 0 (raw lookback) to horizon
        out = self.head(tokens[:, 0, :])
        return out * std + mean


def train_itransformer(model, Xtr, Ytr, Xv, Yv, epochs=30, batch=256, lr=5e-4, wd=1e-5,
                        device=None):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(Xtr, device=device); Yt = torch.tensor(Ytr, device=device)
    Xv_ = torch.tensor(Xv, device=device); Yv_ = torch.tensor(Yv, device=device)
    n = len(Xt); best = float("inf"); state = None; bad = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), Yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(Xv_), Yv_).item())
        if v < best - 1e-5:
            best = v
            state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 4: break
    if state is not None: model.load_state_dict(state)
    model.to("cpu")
    return model, best


if __name__ == "__main__":
    np.random.seed(0); torch.manual_seed(0)
    n, L, H = 800, 336, 48
    x = np.random.randn(n, L).astype(np.float32)
    y = np.stack([x[:, L - 48 + h] + 0.2 * np.random.randn(n) for h in range(H)], axis=1).astype(np.float32)
    m = ITransformer(lookback=L, horizon=H)
    m, vloss = train_itransformer(m, x[:600], y[:600], x[600:], y[600:], epochs=10)
    print(f"iTransformer val MSE: {vloss:.4f}")
