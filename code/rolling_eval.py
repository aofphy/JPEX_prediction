"""
Rolling-window recalibration framework for JEPX 48-step EPF benchmark.

Lago et al. (2021) prescribe daily recalibration on a sliding training
window. For 30-min granularity and seven models on two areas, daily refit
is computationally heavy. We use a practical compromise:

    - Sliding training window of 365 days (1 year)
    - Models refit weekly (every 7 days = 336 issuance times) within the test
      period; predictions for those 7 days use the model fit at the start
      of that block.
    - Evaluation aggregates predictions across all test issuance times.

This is the protocol recommended by Marcjasz et al. (2023) for distributional
EPF models, balancing fidelity to Lago 2021 against computational tractability.

The pipeline operates on a single (area, market) pair. Outputs are
(test_idx_array, horizon_array) prediction matrices, one per model.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Callable

WEEK_STEPS = 336      # 7 days * 48 slots
YEAR_STEPS = 365 * 48 # one-year sliding window
HORIZON = 48


@dataclass
class RollingConfig:
    train_window_days: int = 365      # length of sliding training window
    recal_step_days: int = 7          # refit every N days
    horizon: int = 48                 # forecast horizon (30-min steps)
    val_frac: float = 0.10            # tail of training window held out for ES/HPO


def rolling_predict(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], np.ndarray],
    config: RollingConfig | None = None,
    verbose: bool = True,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Generic rolling-window evaluator.

    Args:
        X: full feature matrix indexed by issuance time t. NaNs are kept;
           caller is responsible for dropping rows where features/targets
           are unobserved.
        Y: full target matrix (n, HORIZON), aligned to X by index.
        test_start, test_end: inclusive bounds for evaluating predictions.
        fit_predict_fn: callable taking (X_tr, Y_tr, X_te) and returning
                        a (len(X_te), HORIZON) prediction array. Internally
                        the caller can do any training/HPO it likes, but
                        sees ONLY data with index <= last issuance time
                        of X_tr.
        config: rolling-window configuration.

    Returns:
        test_idx, predictions
            test_idx: pd.DatetimeIndex of issuance times where predictions
                      were produced.
            predictions: ndarray (n_test, HORIZON).
    """
    cfg = config or RollingConfig()
    test_idx_all = X.loc[test_start:test_end].index
    n_test = len(test_idx_all)
    preds = None  # will be allocated on first fit_predict call once output width is known

    block_size = cfg.recal_step_days * 48
    train_window = cfg.train_window_days * 48

    # Iterate over test blocks
    pos = 0
    block_id = 0
    t0 = time.time()
    while pos < n_test:
        test_block_idx = test_idx_all[pos : pos + block_size]
        if len(test_block_idx) == 0:
            break
        block_start = test_block_idx[0]

        # Training data: [block_start - train_window, block_start) i.e. strictly before
        train_end_t = block_start - pd.Timedelta(minutes=30)  # last train obs
        train_start_t = block_start - pd.Timedelta(minutes=30 * train_window)
        X_tr = X.loc[train_start_t:train_end_t].dropna()
        # Align Y to X_tr index — drop rows where Y has NaN (last HORIZON rows)
        Y_tr = Y.loc[X_tr.index].dropna()
        X_tr = X_tr.loc[Y_tr.index]
        X_te = X.loc[test_block_idx].dropna()
        test_block_actual = X_te.index

        if len(X_tr) < 100 or len(X_te) == 0:
            pos += block_size; block_id += 1
            continue

        yhat = fit_predict_fn(X_tr, Y_tr, X_te)
        # Allocate preds on first block using observed output width
        if preds is None:
            out_width = yhat.shape[1] if yhat.ndim == 2 else cfg.horizon
            preds = np.zeros((n_test, out_width), dtype=float)
        # Place into preds at positions corresponding to test_block_actual
        positions = test_idx_all.get_indexer(test_block_actual)
        valid = positions >= 0
        preds[positions[valid]] = yhat[valid]

        if verbose:
            elapsed = time.time() - t0
            print(f"  block {block_id:02d}  {block_start} -> {test_block_idx[-1]}  "
                  f"train n={len(X_tr)} test n={len(X_te)}  elapsed={elapsed:.0f}s")

        pos += block_size
        block_id += 1

    if preds is None:
        # No successful blocks; return empty predictions
        preds = np.zeros((n_test, cfg.horizon), dtype=float)
    return test_idx_all, preds
