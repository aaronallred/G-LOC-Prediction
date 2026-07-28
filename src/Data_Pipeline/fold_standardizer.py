"""Fold-aware standardizers for the traditional data pipeline.

Replaces the leaky intra-trial (s1) and inter-trial (s2) z-score computations
with statistics fit on training rows only and applied to test rows. Both
classes follow a fit/transform API and operate on a row-major matrix
``X: (N, F)`` plus a parallel per-row trial-id array.

- ``TrialAwareStandardizer`` replaces the per-trial s1 z-score blocks: each
  trial's own training-window μ/σ is used when the trial contributed training
  rows; otherwise (trial fully in the test fold) the pooled training μ/σ is
  used.

- ``GlobalStandardizer`` replaces ``_inter_trial_standardization``: a single
  per-column μ/σ fit on all training rows, applied to every row.
"""

from typing import Any, Optional

import numpy as np


def _guard_divide_by_zero(
    data: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Z-score with the same σ==0 guard the legacy code uses.

    For columns where std is zero, return zeros instead of producing NaNs.
    Inputs are assumed 1-D (per-column) statistics broadcast against rows.
    """
    data = np.asarray(data, dtype=np.float64)
    Z = np.zeros_like(data, dtype=np.float64)
    non_zero = std != 0
    if np.any(non_zero):
        Z[:, non_zero] = (data[:, non_zero] - mean[non_zero]) / std[non_zero]
    return Z


class GlobalStandardizer:
    """Single per-column μ/σ fit on training rows, applied to all rows."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray) -> "GlobalStandardizer":
        X_train = np.asarray(X_train, dtype=np.float64)
        self.mean_ = np.nanmean(X_train, axis=0, keepdims=False)
        self.std_ = np.nanstd(X_train, axis=0, keepdims=False)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("GlobalStandardizer.transform called before fit")
        return _guard_divide_by_zero(X, self.mean_, self.std_)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)


class TrialAwareStandardizer:
    """Per-trial s1 z-score with fold-aware fallback to pooled training stats.

    Fit expects the full stacked feature matrix ``X``, a parallel
    ``trial_id_per_row`` array, and a boolean mask of training rows.
    Statistics are computed per-trial over that trial's training rows only,
    with a pooled (all-training-rows) μ/σ held in reserve for trials not
    present in the training mask.
    """

    def __init__(self) -> None:
        self._per_trial_mean: dict[Any, np.ndarray] = {}
        self._per_trial_std: dict[Any, np.ndarray] = {}
        self._pooled_mean: Optional[np.ndarray] = None
        self._pooled_std: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        trial_id_per_row: np.ndarray,
        train_mask: np.ndarray,
    ) -> "TrialAwareStandardizer":
        X = np.asarray(X, dtype=np.float64)
        train_ids = np.unique(trial_id_per_row[train_mask])
        pooled_rows = []
        for tid in train_ids:
            row_mask = (trial_id_per_row == tid) & train_mask
            rows = X[row_mask]
            self._per_trial_mean[tid] = np.nanmean(rows, axis=0, keepdims=False)
            self._per_trial_std[tid] = np.nanstd(rows, axis=0, keepdims=False)
            pooled_rows.append(rows)

        if pooled_rows:
            all_train = np.vstack(pooled_rows)
            self._pooled_mean = np.nanmean(all_train, axis=0, keepdims=False)
            self._pooled_std = np.nanstd(all_train, axis=0, keepdims=False)
        else:
            # Defensive fallback (no training rows in any trial).
            n_cols = X.shape[1]
            self._pooled_mean = np.zeros(n_cols, dtype=np.float64)
            self._pooled_std = np.zeros(n_cols, dtype=np.float64)
        return self

    def transform(self, X: np.ndarray, trial_id_per_row: np.ndarray) -> np.ndarray:
        if self._pooled_mean is None or self._pooled_std is None:
            raise RuntimeError("TrialAwareStandardizer.transform called before fit")
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros_like(X, dtype=np.float64)
        # Vectorize: assign each row a mean/std from per-trial if known, else pooled.
        per_row_mean = self._pooled_mean.copy()
        per_row_std = self._pooled_std.copy()
        # Build (N, F) per-row stats by selecting per-trial or pooled per row.
        mean_matrix = np.broadcast_to(self._pooled_mean, X.shape).copy()
        std_matrix = np.broadcast_to(self._pooled_std, X.shape).copy()
        for tid, mean in self._per_trial_mean.items():
            sel = trial_id_per_row == tid
            mean_matrix[sel] = mean
            std_matrix[sel] = self._per_trial_std[tid]
        non_zero = std_matrix != 0
        out[non_zero] = (X[non_zero] - mean_matrix[non_zero]) / std_matrix[non_zero]
        return out
