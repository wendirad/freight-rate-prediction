"""Temporal cross-validation fold assignment.

Expanding-window scheme mirroring sklearn's TimeSeriesSplit, exposed as a
small dedicated function so fold row-index assignments can be computed once,
saved, and reused identically across every experiment run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_expanding_folds(
    df: pd.DataFrame, n_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Splits a date-sorted dataframe into n_folds expanding-window folds.

    df must already be sorted by date (as DataLoader.load produces). Row n
    is split into n_folds + 1 contiguous chunks; fold i trains on chunks
    0..i and validates on chunk i+1, so training data always precedes
    validation data in time and each successive fold sees strictly more
    training data.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")

    n_rows = len(df)
    if n_rows < n_folds + 1:
        raise ValueError("not enough rows to build the requested number of folds")

    indices = np.arange(n_rows)
    chunk_bounds = np.linspace(0, n_rows, n_folds + 2, dtype=int)

    folds = []
    for i in range(n_folds):
        train_idx = indices[: chunk_bounds[i + 1]]
        val_idx = indices[chunk_bounds[i + 1] : chunk_bounds[i + 2]]
        folds.append((train_idx, val_idx))
    return folds
