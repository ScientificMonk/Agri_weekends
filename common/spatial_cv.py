"""Spatial cross-validation helpers.

Neighbouring fields share soil, weather and farming practice, so a random
train/test split leaks information and inflates scores. Grouping samples into
spatial blocks and holding out whole blocks gives an honest estimate of how a
model does on a new area.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold


def block_ids(x_m: np.ndarray, y_m: np.ndarray, block_size_m: float) -> np.ndarray:
    """Assign each sample (projected coordinates, metres) to a square block."""
    bx = np.floor(x_m / block_size_m).astype(np.int64)
    by = np.floor(y_m / block_size_m).astype(np.int64)
    _, ids = np.unique(np.stack([bx, by], axis=1), axis=0, return_inverse=True)
    return ids.ravel()


def make_folds(y, groups=None, n_splits: int = 5, seed: int = 42):
    """List of (train_idx, test_idx). Spatial folds if `groups` is given, else random stratified."""
    idx = np.arange(len(y))
    if groups is None:
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(idx, y))
    # Shuffle block order so folds are not spatial stripes
    rng = np.random.default_rng(seed)
    perm = rng.permutation(groups.max() + 1)
    return list(GroupKFold(n_splits).split(idx, y, perm[groups]))
