"""
DDNN — Distributional Deep Neural Network for EPF.

Follows Marcjasz, Uniejewski & Weron (2023) "Distributional neural networks
for electricity price forecasting", Energy Economics 125, 106843, adapted
to JEPX 30-min half-hourly data with 48-step multi-output forecasting.

Two flavours:

  - DDNN-N  -- Normal predictive distribution; output head produces
              (mu_h, log_sigma_h) for each horizon h = 1..H.
  - DDNN-JSU -- Johnson SU predictive distribution; output head produces
              (mu_h, log_sigma_h, gamma_h, delta_h) for heavy-tail asymmetry.

Training: negative log-likelihood (NLL) loss, Adam, early stopping on the
NLL of the validation slice. Standard input feature: 336-step raw lookback
+ standardised exogenous regressors at issuance time.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class DDNNNormal(nn.Module):
    """DDNN with Normal predictive distribution per horizon."""

    def __init__(self, in_dim: int, hidden: int = 512, horizon: int = 48,
                 dropout: float = 0.2):
        super().__init__()
        self.horizon = horizon
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        # Two heads: mu and log_sigma
        self.mu_head = nn.Linear(hidden // 2, horizon)
        self.log_sigma_head = nn.Linear(hidden // 2, horizon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.mu_head(h)
        # softplus(log_sigma) > 0 with floor for stability
        log_sigma = self.log_sigma_head(h)
        sigma = F.softplus(log_sigma) + 1e-3
        return mu, sigma

    def predict_quantiles(self, x: torch.Tensor, qs: np.ndarray) -> np.ndarray:
        """Return (n, H, K) quantile predictions for quantile levels qs."""
        self.eval()
        with torch.no_grad():
            mu, sigma = self.forward(x)
        from scipy.stats import norm
        mu_np = mu.cpu().numpy()[..., None]
        sigma_np = sigma.cpu().numpy()[..., None]
        z = norm.ppf(qs)[None, None, :]
        return mu_np + sigma_np * z


def nll_normal(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood for Normal predictive distribution."""
    dist = Normal(mu, sigma)
    return -dist.log_prob(y).mean()


class DDNNJohnsonSU(nn.Module):
    """DDNN with Johnson's SU predictive distribution per horizon.

    Johnson SU parameters: location ξ (xi), scale λ (lambda), shape γ
    (gamma), shape δ (delta). We parametrise as
        ξ = mu, λ = softplus(s) + eps, δ = softplus(d) + eps, γ = g.
    """

    def __init__(self, in_dim: int, hidden: int = 512, horizon: int = 48,
                 dropout: float = 0.2):
        super().__init__()
        self.horizon = horizon
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.xi_head = nn.Linear(hidden // 2, horizon)
        self.s_head = nn.Linear(hidden // 2, horizon)
        self.g_head = nn.Linear(hidden // 2, horizon)
        self.d_head = nn.Linear(hidden // 2, horizon)

    def forward(self, x):
        h = self.backbone(x)
        xi = self.xi_head(h)
        lam = F.softplus(self.s_head(h)) + 1e-3
        gam = self.g_head(h)
        delt = F.softplus(self.d_head(h)) + 1e-1   # delta > 0; floor for stability
        return xi, lam, gam, delt

    def predict_quantiles(self, x, qs):
        self.eval()
        with torch.no_grad():
            xi, lam, gam, delt = self.forward(x)
        xi = xi.cpu().numpy(); lam = lam.cpu().numpy()
        gam = gam.cpu().numpy(); delt = delt.cpu().numpy()
        # Johnson SU quantile: x = xi + lam * sinh((Phi^{-1}(q) - gamma) / delta)
        from scipy.stats import norm
        zq = norm.ppf(qs)
        n, H = xi.shape; K = len(qs)
        out = np.empty((n, H, K))
        for k, z in enumerate(zq):
            out[..., k] = xi + lam * np.sinh((z - gam) / delt)
        return out


def nll_jsu(y, xi, lam, gam, delt) -> torch.Tensor:
    """Negative log-likelihood for Johnson SU.

    pdf(x) = (delta / (lam * sqrt(2*pi*(1 + ((x-xi)/lam)^2))))
              * exp(- 0.5 * (gamma + delta * asinh((x-xi)/lam))^2)
    """
    z = (y - xi) / lam
    asinh_z = torch.asinh(z)
    log_pdf = (torch.log(delt) - torch.log(lam)
               - 0.5 * math.log(2 * math.pi)
               - 0.5 * torch.log1p(z * z)
               - 0.5 * (gam + delt * asinh_z) ** 2)
    return -log_pdf.mean()


def train_ddnn(model, Xtr, Ytr, Xv, Yv, epochs=40, batch=512, lr=1e-3, wd=1e-5,
               distribution: str = "normal", device="cpu", verbose=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(Xtr, device=device); Yt = torch.tensor(Ytr, device=device)
    Xv_t = torch.tensor(Xv, device=device); Yv_t = torch.tensor(Yv, device=device)
    n = len(Xt); best = float("inf"); state = None; bad = 0; patience = 5
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            if distribution == "normal":
                mu, sigma = model(Xt[idx])
                loss = nll_normal(Yt[idx], mu, sigma)
            else:  # jsu
                xi, lam, gam, delt = model(Xt[idx])
                loss = nll_jsu(Yt[idx], xi, lam, gam, delt)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            if distribution == "normal":
                mu, sigma = model(Xv_t)
                v = float(nll_normal(Yv_t, mu, sigma).item())
            else:
                xi, lam, gam, delt = model(Xv_t)
                v = float(nll_jsu(Yv_t, xi, lam, gam, delt).item())
        if verbose:
            print(f"  ep {ep:3d}  val_nll {v:.4f}")
        if v < best - 1e-5:
            best = v
            state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: model.load_state_dict(state)
    return model, best


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0); torch.manual_seed(0)
    n, p, H = 1000, 30, 48
    X = np.random.randn(n, p).astype(np.float32)
    Y = (X[:, 0:1] + 0.5 * X[:, 1:2] + np.random.randn(n, H).astype(np.float32) * 0.3)
    m = DDNNNormal(in_dim=p, hidden=128, horizon=H)
    m, best = train_ddnn(m, X[:700], Y[:700], X[700:], Y[700:], epochs=30,
                          distribution="normal", verbose=False)
    print(f"DDNN-Normal best val NLL: {best:.4f}")

    m2 = DDNNJohnsonSU(in_dim=p, hidden=128, horizon=H)
    m2, best2 = train_ddnn(m2, X[:700], Y[:700], X[700:], Y[700:], epochs=30,
                             distribution="jsu", verbose=False)
    print(f"DDNN-JSU best val NLL: {best2:.4f}")
