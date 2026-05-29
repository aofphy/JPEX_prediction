"""
Model Confidence Set (MCS) — Hansen, Lunde & Nason (2011) Econometrica.

Iteratively eliminates worst-performing models until the null hypothesis of
equal predictive ability (EPA) is no longer rejected. The remaining set is
the Model Confidence Set at significance level alpha, i.e., the subset
of models statistically indistinguishable from the best with probability
at least 1 - alpha.

We implement the T_R (range) variant with stationary-bootstrap critical
values (Politis & Romano 1994). Bootstrap block length parameter is
selected per recommendation of Patton, Politis & White (2009): mean block
length ≈ T^(1/3).

Inputs:
  losses: dict[str, np.ndarray]  -- per-model 1-D loss arrays of equal length
                                    (e.g., flattened squared errors)

Outputs:
  dict with keys:
    "mcs_10": list of model names in MCS at alpha=0.10
    "mcs_25": list of model names in MCS at alpha=0.25
    "eliminations": ordered list of (eliminated model name, p-value)
"""
from __future__ import annotations
import numpy as np


def stationary_bootstrap(T: int, n_boot: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Generate (n_boot, T) bootstrap index matrices via Politis-Romano stationary bootstrap."""
    p = 1.0 / mean_block
    indices = np.empty((n_boot, T), dtype=np.int64)
    for b in range(n_boot):
        idx = np.empty(T, dtype=np.int64)
        idx[0] = rng.integers(0, T)
        for t in range(1, T):
            if rng.random() < p:
                idx[t] = rng.integers(0, T)
            else:
                idx[t] = (idx[t - 1] + 1) % T
        indices[b] = idx
    return indices


def model_confidence_set(losses: dict[str, np.ndarray],
                          alphas: list[float] = (0.10, 0.25),
                          n_boot: int = 1000,
                          mean_block: float | None = None,
                          seed: int = 0) -> dict:
    """Compute MCS via the T_R range statistic with stationary bootstrap.

    Args:
        losses: dict mapping model name -> (T,) loss array (e.g., squared errors).
                All arrays must have the same length.
        alphas: significance levels for MCS construction.
        n_boot: bootstrap replications.
        mean_block: mean block length for stationary bootstrap. If None,
                    defaults to T^(1/3).

    Returns:
        dict with keys "mcs_<int(100*alpha)>" -> list of model names, plus
        "eliminations" -> list of (model_name, p_value) tuples.
    """
    rng = np.random.default_rng(seed)
    names = list(losses.keys())
    L_mat = np.stack([losses[k] for k in names], axis=0)  # (M, T)
    M, T = L_mat.shape
    if mean_block is None:
        mean_block = max(2.0, T ** (1.0 / 3.0))

    # Pre-compute bootstrap indices once and re-use across iterations
    boot_idx = stationary_bootstrap(T, n_boot, mean_block, rng)

    # Iterative elimination
    active = list(range(M))
    eliminations: list[tuple[str, float]] = []
    mcs_sets: dict[float, list[str]] = {}
    sorted_alphas = sorted(alphas)
    pvals_sequence: list[float] = []

    while len(active) > 1:
        L_sub = L_mat[active]                       # (m, T)
        m = L_sub.shape[0]
        loss_means = L_sub.mean(axis=1)             # (m,)
        # Pairwise mean loss differences d_{ij} = mean(L_i - L_j)
        D = loss_means[:, None] - loss_means[None, :]  # (m, m)
        # Stationary-bootstrap variance of d_{ij}
        # For each bootstrap rep b, compute mean(L_sub[:, idx_b]) per model
        # then differences. Variance is across bootstrap reps.
        boot_means = np.stack(
            [L_sub[:, boot_idx[b]].mean(axis=1) for b in range(n_boot)], axis=0
        )  # (n_boot, m)
        # Centred bootstrap means
        boot_means_c = boot_means - loss_means[None, :]
        # Bootstrap variance of (mean_i - mean_j) under stationary bootstrap:
        # var = mean over b of (d_ij_boot - 0)^2  where d_ij_boot = (boot_mean_i - mean_i) - (boot_mean_j - mean_j)
        # Compute as (m, m) variance matrix
        var_D = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i == j: continue
                diffs = boot_means_c[:, i] - boot_means_c[:, j]
                var_D[i, j] = (diffs ** 2).mean()
        # Standardised t-statistics
        t_stat = np.divide(D, np.sqrt(var_D + 1e-12), where=(var_D > 0))
        # Range statistic T_R = max_{i, j in M} |t_ij|
        T_R = float(np.max(np.abs(t_stat)))
        # Bootstrap distribution of T_R under H0 (equal predictive ability)
        # Compute the same statistic on bootstrap-centred loss differences
        T_R_boot = np.empty(n_boot)
        for b in range(n_boot):
            mb = boot_means_c[b]
            Db = mb[:, None] - mb[None, :]
            tb = np.divide(Db, np.sqrt(var_D + 1e-12), where=(var_D > 0))
            T_R_boot[b] = float(np.max(np.abs(tb)))
        # p-value = P(T_R_boot >= T_R)
        p_val = float((T_R_boot >= T_R).mean())
        pvals_sequence.append(p_val)

        # If H0 not rejected at the most stringent alpha, stop
        if p_val > min(sorted_alphas):
            break

        # Otherwise eliminate the worst model (highest mean loss in active set)
        worst_local_idx = int(np.argmax(loss_means))
        worst_global_idx = active[worst_local_idx]
        eliminations.append((names[worst_global_idx], p_val))
        del active[worst_local_idx]

    # Build MCS sets: at each alpha, MCS contains all models active before
    # the first elimination with p_val <= alpha.
    final_active = [names[i] for i in active]
    out = {"eliminations": eliminations}
    for alpha in alphas:
        # MCS_alpha = models that survived after all eliminations with p<=alpha
        mcs = list(final_active)
        # Add back any models eliminated when p_val > alpha
        for k, (model_name, p_val) in enumerate(reversed(eliminations)):
            if p_val > alpha:
                mcs.append(model_name)
            else:
                break
        out[f"mcs_{int(100 * alpha):02d}"] = sorted(mcs)
    return out


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(0)
    T = 500
    losses = {
        "best":   rng.exponential(1.0, T),
        "ok_a":   rng.exponential(1.1, T),
        "ok_b":   rng.exponential(1.15, T),
        "bad":    rng.exponential(2.0, T),
        "awful":  rng.exponential(5.0, T),
    }
    res = model_confidence_set(losses, alphas=[0.10, 0.25], n_boot=500)
    print("Eliminations:", res["eliminations"])
    print("MCS @ alpha=0.10:", res["mcs_10"])
    print("MCS @ alpha=0.25:", res["mcs_25"])
