from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

import numpy as np


def _is_trial_id_column(candidate: np.ndarray) -> bool:
    """Heuristic for detecting legacy trial-id column in the final feature position."""
    if candidate.ndim != 1 or candidate.size < 4:
        return False
    if not np.all(np.isfinite(candidate)):
        return False
    if not np.allclose(candidate, np.round(candidate), atol=1e-6):
        return False

    # Trial ids are expected to repeat in contiguous segments.
    rounded = np.round(candidate).astype(np.int64)
    if np.unique(rounded).size < 2:
        return False

    transitions = np.where(np.diff(rounded) != 0)[0] + 1
    segment_values = np.concatenate(([rounded[0]], rounded[transitions]))
    appears_once_per_segment = np.unique(segment_values).size == segment_values.size
    if not appears_once_per_segment:
        return False

    _, counts = np.unique(rounded, return_counts=True)
    return int(np.min(counts)) >= 2


def split_features_and_trials(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split matrix into features and trial ids; fallback to one synthetic trial."""
    if X.ndim != 2:
        raise ValueError("split_features_and_trials expects a 2D array.")
    if X.shape[1] < 2:
        return X.astype(np.float32), np.zeros(X.shape[0], dtype=np.int64)

    last_col = X[:, -1]
    if _is_trial_id_column(last_col):
        return X[:, :-1].astype(np.float32), np.round(last_col).astype(np.int64)

    return X.astype(np.float32), np.zeros(X.shape[0], dtype=np.int64)


def max_trial_sequence_length(X: np.ndarray) -> int:
    """Return max contiguous trial length for dynamic sequence search bounds."""
    if X.ndim == 3:
        return int(X.shape[1])
    if X.ndim != 2 or X.shape[0] == 0:
        return 1
    _, trial_ids = split_features_and_trials(X)
    _, counts = np.unique(trial_ids, return_counts=True)
    if counts.size == 0:
        return 1
    return int(np.max(counts))


def _create_windows(
    sequence: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step_size: int,
    end_label: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    windows: list[np.ndarray] = []
    window_labels: list[np.ndarray] = []

    if sequence.shape[0] == 0:
        return windows, window_labels

    usable_window = max(1, min(window_size, sequence.shape[0]))
    usable_step = max(1, step_size)
    for start in range(0, sequence.shape[0] - usable_window + 1, usable_step):
        end = start + usable_window
        windows.append(sequence[start:end])
        if end_label:
            window_labels.append(np.asarray(labels[end - 1], dtype=np.float32))
        else:
            window_labels.append(labels[start:end].astype(np.float32))
    return windows, window_labels


def create_trial_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    step_size: int,
    end_label: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Create legacy-compatible trial windows from either 2D+trial or pre-windowed 3D arrays."""
    if X.ndim == 3:
        X_windows = X.astype(np.float32)
        if end_label:
            if y.ndim == 2:
                y_windows = y[:, -1].astype(np.float32)
            else:
                y_windows = y.astype(np.float32)
        else:
            if y.ndim == 1:
                y_windows = np.repeat(y.reshape(-1, 1), X_windows.shape[1], axis=1).astype(np.float32)
            else:
                y_windows = y.astype(np.float32)
        return X_windows, y_windows

    if X.ndim != 2:
        raise ValueError("create_trial_windows expects a 2D or 3D feature array.")

    features, trial_ids = split_features_and_trials(X)
    if y.ndim != 1:
        y = y.reshape(-1)

    order_preserving_trials = list(OrderedDict.fromkeys(trial_ids.tolist()))
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for trial_id in order_preserving_trials:
        trial_mask = trial_ids == trial_id
        trial_sequence = features[trial_mask]
        trial_labels = y[trial_mask]
        trial_windows, trial_window_labels = _create_windows(
            trial_sequence,
            trial_labels,
            window_size=window_size,
            step_size=step_size,
            end_label=end_label,
        )
        windows.extend(trial_windows)
        labels.extend(trial_window_labels)

    if len(windows) == 0:
        fallback_X = features.reshape(features.shape[0], 1, features.shape[1]).astype(np.float32)
        if end_label:
            fallback_y = y.astype(np.float32)
        else:
            fallback_y = y.reshape(-1, 1).astype(np.float32)
        return fallback_X, fallback_y

    X_windows = np.asarray(windows, dtype=np.float32)
    y_windows = np.asarray(labels, dtype=np.float32)
    return X_windows, y_windows


def summarize_windows_mean(X_windows: np.ndarray) -> np.ndarray:
    """Legacy GAM feature summary: mean over sequence length per feature."""
    if X_windows.ndim != 3:
        raise ValueError("summarize_windows_mean expects a 3D array.")
    return X_windows.mean(axis=1).astype(np.float32)
