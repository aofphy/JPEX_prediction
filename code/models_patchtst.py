"""PatchTST — A simplified univariate PyTorch implementation.

Reference: Nie, Nguyen, Sinthong, Kalagnanam (2023), "A Time Series is Worth
64 Words: Long-term Forecasting with Transformers", ICLR 2023.

This is a minimal channel-independent univariate variant suitable for the
JEPX EPF benchmark. The model:
  - Takes a univariate lookback of length L = 336;
  - Splits it into overlapping patches of length P = 16 with stride S = 8;
  - Linearly embeds each patch into d_model = 64;
  - Adds learned position encodings across patches;
  - Applies a 2-layer Transformer encoder (4 heads, d_ff=128);
  - Flattens the patch tokens and projects to a 48-step forecast.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(self, lookback: int = 336, horizon: int = 48,
                 patch_len: int = 16, stride: int = 8,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        # Number of patches with padding
        self.n_patches = (lookback - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patches * d_model, horizon)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        B = x.size(0)
        # Per-instance reversible instance normalisation (RevIN-lite)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-5)
        x_n = (x - mean) / std
        # Patch
        patches = x_n.unfold(1, self.patch_len, self.stride)  # (B, n_patches, patch_len)
        tokens = self.embed(patches) + self.pos  # (B, n_patches, d_model)
        tokens = self.norm(self.encoder(tokens))
        flat = tokens.reshape(B, -1)
        out = self.head(flat)
        # De-normalise to original price scale
        return out * std + mean


def train_patchtst(model, Xtr, Ytr, Xv, Yv, epochs=30, batch=256, lr=5e-4, wd=1e-5,
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
    m = PatchTST(lookback=L, horizon=H)
    m, vloss = train_patchtst(m, x[:600], y[:600], x[600:], y[600:], epochs=10)
    print(f"PatchTST val MSE: {vloss:.4f}")
