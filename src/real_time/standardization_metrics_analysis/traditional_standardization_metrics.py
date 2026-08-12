"""Compare train-only vs all-rows (legacy leaky) standardization metrics.

For each traditional model in ``configs/real_time_sensor_ablation.yaml``, this
script extracts the raw feature matrix that ``_standardize_raw`` consumes, re-apply both
standardization styles, and saves per-feature delta statistics for test rows.

For the test rows with the largest per-feature standardization-metric deltas, the
script also records three per-row context flags — ``is_beginning_of_trial``,
``positive_gloc`` and ``had_imputed_input`` — and emits an overlay plot
(train-only vs all-rows standardized value across test rows) for the top-N
features per fold by ``max_abs_delta``.

Usage::

    python -m src.real_time.traditional_standardization_metrics
    python -m src.real_time.traditional_standardization_metrics --config configs/real_time_sensor_ablation.yaml
    python -m src.real_time.traditional_standardization_metrics --no-plots    # skip plot generation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Data_Pipeline.data_pipeline import DataPipeline, TraditionalDataPipeline
from src.Data_Pipeline.fold_standardizer import GlobalStandardizer, TrialAwareStandardizer
from src.models.model_factory import ModelFactory
from src.model_type import ModelType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = "configs/real_time_sensor_ablation.yaml"
OUTPUT_DIR_NAME = "Results/Traditional_Standardization_Metrics"
# Auto-appended columns that are stream-independent and should be excluded from
# the per-feature largest-delta analysis regardless of which streams are
# requested. The pipeline's _apply_substring_filter already drops AFE for
# stream-filter requests; the name here becomes relevant only when the
# AFE-indicator column survives (e.g., when feature_streams is None and the
# filter falls back to "include all").
EXCLUDE_FEATURE_NAMES = {"AFE_indicator_windowed"}
TOP_N_FEATURES = 10
# Plot palette: blue = current fold-aware standardization, orange = leaky all-rows.
_PLOT_PALETTE = {"train_only": "#1f77b4", "all_rows": "#ff7f0e"}
_PLOT_DPI = 300

# ---------------------------------------------------------------------------
# monkey-patch machinery
# ---------------------------------------------------------------------------
# Per-fold captures.
_CapturedStandardization: dict[int, dict[str, np.ndarray]] = {}
_CapturedFeatures: dict[int, dict[str, Any]] = {}
# Per-sample pre-feature KNN imputation mask (shape (n_samples, n_raw_features)),
# captured at the FIRST _faster_knn_impute call within a fold. Subsequent calls
# (post-feature KNN) are ignored — pre-feature is the relevant phase.
_CapturedImputeMask: dict[int, np.ndarray] = {}
# Per-sample trial-id and time columns + the raw feature names list, captured
# from _reduce_memory (which runs AFTER pre-feature imputation but BEFORE
# _feature_generation). Used to reduce the per-sample impute mask per window.
_CapturedExperimentMetadata: dict[int, dict[str, Any]] = {}
# Indices of post-_process_NaN_temporal row removals, used to remap the
# pre-removal per-row context flags onto the surviving rows.
_CapturedRemovedRows: dict[int, np.ndarray] = {}
# Per-fold hyperparameters required to mirror _sliding_window_mean_calc windowing.
_CapturedWindowingParams: dict[int, dict[str, float]] = {}
_CURRENT_FOLD: int = -1

_OriginalStandardizeRaw = TraditionalDataPipeline._standardize_raw
_OriginalFeatureGeneration = TraditionalDataPipeline._feature_generation
_OriginalFasterKnnImpute = TraditionalDataPipeline._faster_knn_impute
_OriginalReduceMemory = TraditionalDataPipeline._reduce_memory
_OriginalProcessNaN = TraditionalDataPipeline._process_NaN_temporal


def _capturing_standardize_raw(self, x_raw, trial_id_per_row, train_mask):
    global _CURRENT_FOLD, _CapturedStandardization
    _CapturedStandardization[_CURRENT_FOLD] = {
        "X_raw": np.asarray(x_raw, dtype=np.float64).copy(),
        "trial_id_per_row": np.array(trial_id_per_row, copy=True),
        "train_mask": np.array(train_mask, dtype=bool, copy=True),
    }
    return _OriginalStandardizeRaw(self, x_raw, trial_id_per_row, train_mask)


def _capturing_feature_generation(self, *args, **kwargs):
    global _CURRENT_FOLD, _CapturedFeatures, _CapturedWindowingParams
    result = _OriginalFeatureGeneration(self, *args, **kwargs)
    _CapturedFeatures[_CURRENT_FOLD] = {
        "y_gloc_labels": np.array(result[0], copy=True),
        "x_feature_matrix": np.array(result[1], copy=True),
        "all_features": list(result[2]),
        "trial_id_per_row": np.array(result[3], copy=True),
    }
    # _feature_generation's positional args unpack to the hyperparameters that
    # drove windowing: (time_start, offset, stride, window_size, ...). Stash
    # these so the per-window impute reduction mirrors _sliding_window_mean_calc
    # exactly.
    if args:
        _CapturedWindowingParams[_CURRENT_FOLD] = {
            "time_start": float(args[0]),
            "offset": float(args[1]),
            "stride": float(args[2]),
            "window_size": float(args[3]),
        }
    return result


def _capturing_faster_knn_impute(self, X, k=5, M=32, efSearch=64):
    """Snapshot the pre-fill NaN mask of the FIRST KNN impute call per fold.

    The first ``_faster_knn_impute`` call inside ``TraditionalDataPipeline.get_data``
    is the pre-feature imputation (operates on the per-sample raw column matrix).
    Subsequent calls operate on the per-window feature matrix; we ignore those
    because per-window vs per-sample mask propagation differs.
    """
    global _CURRENT_FOLD, _CapturedImputeMask
    if _CURRENT_FOLD not in _CapturedImputeMask:
        _CapturedImputeMask[_CURRENT_FOLD] = np.isnan(np.asarray(X, dtype=np.float64)).copy()
    return _OriginalFasterKnnImpute(self, X, k=k, M=M, efSearch=efSearch)


def _capturing_reduce_memory(self, gloc_data, gloc_labels, features, output_feature_dtype=np.dtype(np.float32)):
    """Snapshot experiment_metadata (trial_id, Time) and the raw feature names.

    These define the per-sample row order that the captured pre-feature KNN
    impute mask aligns with, enabling a per-window reduction independent of
    the production pipeline.
    """
    global _CURRENT_FOLD, _CapturedExperimentMetadata
    result = _OriginalReduceMemory(self, gloc_data, gloc_labels, features, output_feature_dtype)
    _, _, experiment_metadata = result
    _CapturedExperimentMetadata[_CURRENT_FOLD] = {
        "trial_id": np.array(experiment_metadata["trial_id"], copy=True),
        "Time (s)": np.array(experiment_metadata["Time (s)"], copy=True),
        "feature_names": list(features["All"]),
    }
    return result


def _capturing_process_nan(self, y_gloc_labels, x_feature_matrix, all_features):
    """Capture the indices of rows removed by _process_NaN_temporal.

    These post-_feature_generation row removals must be mirrored to every
    per-row context flag (pos_gloc, beginning_of_trial, impute) so they align
    with the test-row deltas that _analyse_fold computes.
    """
    global _CURRENT_FOLD, _CapturedRemovedRows
    y_noNaN, x_noNaN, all_features_out, removed = _OriginalProcessNaN(self, y_gloc_labels, x_feature_matrix, all_features)
    _CapturedRemovedRows[_CURRENT_FOLD] = np.array(removed, copy=True)
    return y_noNaN, x_noNaN, all_features_out, removed


# apply the monkey-patches (after both capturing functions are defined)
TraditionalDataPipeline._standardize_raw = _capturing_standardize_raw
TraditionalDataPipeline._feature_generation = _capturing_feature_generation
TraditionalDataPipeline._faster_knn_impute = _capturing_faster_knn_impute
TraditionalDataPipeline._reduce_memory = _capturing_reduce_memory
TraditionalDataPipeline._process_NaN_temporal = _capturing_process_nan

# ---------------------------------------------------------------------------
# feature-name helpers
# ---------------------------------------------------------------------------
def _extract_baseline_method(feature_name: str) -> str | None:
    for bm in ("v0", "v1", "v2", "v5", "v6"):
        if f"_{bm}" in feature_name or feature_name.endswith(f"_{bm}"):
            return bm
    return None


def _extract_feature_type(feature_name: str) -> str | None:
    for ft in ("mean", "stddev", "max", "range", "additional"):
        if f"_{ft}_" in feature_name or feature_name.endswith(f"_{ft}"):
            return ft
    return None


def _extract_standardization(feature_name: str) -> str | None:
    for s in ("s1", "s2"):
        if feature_name.endswith(f"_{s}"):
            return s
    return None


def _is_target_feature(name: str, filter_substrings: Optional[list[str]] = None) -> bool:
    """Return True if a feature column is in-scope for the top-N largest-delta analysis.

    When ``filter_substrings`` is None or empty, every non-excluded column is
    in-scope (back-compat for runs that don't request any streams). Otherwise,
    a column is in-scope if its lowercased name contains any of the requested
    stream keywords — the same union-substring matcher the production pipeline
    uses in ``_apply_substring_filter``. This auto-covers derived columns that
    lack the device suffix (e.g., ``HRV (SDNN)`` is caught by the ``"hr"``
    substring) and drops AFE-indicator / unrelated-stream columns.
    """
    if name in EXCLUDE_FEATURE_NAMES:
        return False
    if not filter_substrings:
        return True
    lower = name.lower()
    return any(s in lower for s in filter_substrings)


# ---------------------------------------------------------------------------
# per-window impute-mask reduction + context flags
# ---------------------------------------------------------------------------
def _reduce_impute_mask_per_window(
    impute_mask_per_sample: np.ndarray,
    trial_id_per_sample: np.ndarray,
    time_per_sample: np.ndarray,
    time_start: float,
    offset: float,
    stride: float,
    window_size: float,
    feature_names: list[str],
    trial_id_per_row: np.ndarray,
) -> np.ndarray:
    """Reduce the per-sample KNN-imputation NaN mask to a per-window flag.

    Mirrors ``_sliding_window_mean_calc`` exactly: trials iterated in
    ``pd.unique`` order; per trial, windows advance by ``stride`` starting at
    ``time_start``; rows concatenated in the same order as
    ``trial_id_per_row``. For each window, the per-row boolean flag is ``True``
    if ANY sample in the feature window had an imputed value across the
    raw feature columns spanned by ``feature_names``.

    Returns a 1-D boolean array of length ``len(trial_id_per_row)`` aligned 1:1
    with the rows produced by ``_feature_generation`` (pre-``_process_NaN_temporal``).
    """
    if impute_mask_per_sample is None or impute_mask_per_sample.size == 0 or len(trial_id_per_row) == 0:
        return np.zeros(len(trial_id_per_row), dtype=bool)

    impute_mask_per_sample = np.asarray(impute_mask_per_sample, dtype=bool)
    trial_id_per_sample = np.asarray(trial_id_per_sample)
    time_per_sample = np.asarray(time_per_sample, dtype=np.float64)

    n_samples, n_raw_features = impute_mask_per_sample.shape
    # Resolved columns of the per-sample matrix must map onto the raw feature
    # names produced by _reduce_memory. _sliding_window_* operate on the
    # combined_baseline matrix (same column count as the raw all-features
    # matrix); the per-window reduction here uses the union of any column.
    out = np.zeros(len(trial_id_per_row), dtype=bool)
    pos = 0
    for trial in pd.unique(trial_id_per_sample):
        # Same time-trim per trial as the sliding-window source.
        trial_sel = trial_id_per_sample == trial
        time_trimmed = time_per_sample[trial_sel]
        if time_trimmed.size == 0:
            continue
        time_end = float(np.max(time_trimmed))
        n_windows = int(((time_end - offset) // stride) - (window_size // stride - 1))
        if n_windows <= 0:
            continue
        if pos + n_windows > len(out):
            break
        time_iteration = time_start
        for j in range(n_windows):
            time_period = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))
            # Combined-baseline windowing reductions operate on impute_mask_per_sample
            # rows of this trial spanned by time_period — the same sample span
            # used by _sliding_window_mean_calc's nanmean; use np.any across
            # both rows and columns to yield a single per-window "any input
            # imputed" indicator.
            imputed_window = impute_mask_per_sample[trial_sel][time_period]
            out[pos + j] = bool(np.any(imputed_window)) if imputed_window.size else False
            time_iteration += stride
        pos += n_windows
    return out[:pos] if pos < len(out) else out


def _beginning_of_trial_mask(trial_id_per_row: np.ndarray) -> np.ndarray:
    """Boolean mask True at the first row of each contiguous trial block."""
    if len(trial_id_per_row) == 0:
        return np.zeros(0, dtype=bool)
    a = np.asarray(trial_id_per_row)
    flag = np.ones(len(a), dtype=bool)
    flag[1:] = a[1:] != a[:-1]
    return flag


def _remap_through_survivor(arr: np.ndarray, removed_indices: np.ndarray, n_pre_rows: int) -> np.ndarray:
    """Apply the same survivor remap as train_mask_pre[survivor_mask_pre]."""
    removed_mask = np.zeros(n_pre_rows, dtype=bool)
    removed_mask[np.asarray(removed_indices, dtype=int)] = True
    survivor = ~removed_mask
    return arr[survivor]


# ---------------------------------------------------------------------------
# re-standardization
# ---------------------------------------------------------------------------
def _standardize_train_only(
    X_raw: np.ndarray, trial_id: np.ndarray, train_mask: np.ndarray
) -> np.ndarray:
    s1 = TrialAwareStandardizer().fit(X_raw, trial_id, train_mask).transform(X_raw, trial_id)
    s2 = GlobalStandardizer().fit(X_raw[train_mask]).transform(X_raw)
    return np.hstack([s1, s2])


def _standardize_all_rows(X_raw: np.ndarray, trial_id: np.ndarray) -> np.ndarray:
    all_true = np.ones(X_raw.shape[0], dtype=bool)
    s1 = TrialAwareStandardizer().fit(X_raw, trial_id, all_true).transform(X_raw, trial_id)
    s2 = GlobalStandardizer().fit(X_raw).transform(X_raw)
    return np.hstack([s1, s2])


# ---------------------------------------------------------------------------
# per-feature deltas
# ---------------------------------------------------------------------------
def _per_feature_statistics(
    train_only_test: np.ndarray,
    all_rows_test: np.ndarray,
    feature_names: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, dict[str, float]]]]]:
    deltas = train_only_test - all_rows_test
    per_feature: dict[str, dict[str, float]] = {}
    stratified_totals: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = {}

    for j, fname in enumerate(feature_names):
        col_deltas = deltas[:, j]
        stats: dict[str, float] = {
            "mean_abs_delta": float(np.mean(np.abs(col_deltas))),
            "max_abs_delta": float(np.max(np.abs(col_deltas))),
            "mean_delta": float(np.mean(col_deltas)),
            "std_delta": float(np.std(col_deltas)),
            "s1_or_s2": _extract_standardization(fname) or "unknown",
            "feature_type": _extract_feature_type(fname) or "unknown",
            "baseline_method": _extract_baseline_method(fname) or "none",
        }
        per_feature[fname] = stats

        strat_s1_s2 = stats["s1_or_s2"]
        strat_ft = stats["feature_type"]
        strat_bm = stats["baseline_method"]
        stratified_totals.setdefault(strat_s1_s2, {}).setdefault(strat_ft, {}).setdefault(strat_bm, []).append(stats)

    # aggregate stratified totals into mean across features in each bucket
    stratified_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for s1s2, ft_dict in stratified_totals.items():
        for ft, bm_dict in ft_dict.items():
            for bm, stat_list in bm_dict.items():
                keys = ["mean_abs_delta", "max_abs_delta", "mean_delta", "std_delta"]
                aggregated = {
                    k: float(np.mean([s[k] for s in stat_list]))
                    for k in keys
                }
                aggregated["n_features"] = len(stat_list)
                stratified_summary.setdefault(s1s2, {}).setdefault(ft, {})[bm] = aggregated

    return per_feature, stratified_summary


# ---------------------------------------------------------------------------
# per-fold analysis
# ---------------------------------------------------------------------------
def _analyse_fold(
    X_raw: np.ndarray,
    trial_id: np.ndarray,
    train_mask: np.ndarray,
    all_features: list[str],
    y_gloc_labels: np.ndarray,
    impute_mask_per_window: Optional[np.ndarray],
    filter_substrings: Optional[list[str]] = None,
    fold_dir: Optional[Path] = None,
    model_name: str = "",
    fold_id: int = -1,
    generate_plots: bool = True,
) -> dict[str, Any]:
    std_train_only = _standardize_train_only(X_raw, trial_id, train_mask)
    std_all_rows = _standardize_all_rows(X_raw, trial_id)
    test_mask = ~train_mask
    n_test_rows = int(test_mask.sum())

    target_idx = [i for i, n in enumerate(all_features) if _is_target_feature(n, filter_substrings)]
    target_features = [all_features[i] for i in target_idx]
    n_target = len(target_features)

    per_feature, stratified = _per_feature_statistics(
        std_train_only[test_mask][:, target_idx],
        std_all_rows[test_mask][:, target_idx],
        target_features,
    )

    abs_deltas = [
        abs(std_train_only[test_mask, i] - std_all_rows[test_mask, i])
        for i in target_idx
    ]
    all_flat = np.concatenate(abs_deltas) if abs_deltas else np.zeros(0)
    summary = {
        "mean_abs_delta": float(np.mean(all_flat)) if len(all_flat) else 0.0,
        "median_abs_delta": float(np.median(all_flat)) if len(all_flat) else 0.0,
        "max_abs_delta": float(np.max(all_flat)) if len(all_flat) else 0.0,
        "std_abs_delta": float(np.std(all_flat)) if len(all_flat) else 0.0,
    }

    # ---- per-row context flags for the worst rows -------------------------
    # Per-row delta matrix restricted to test rows × target features
    # (aligned with the captured `y_gloc_labels` and the per-row impute mask;
    # `_feature_generation` returns BEFORE `_process_NaN_temporal`, and
    # `_CapturedRemovedRows` records the rows consequently dropped, so
    # `X_raw`/`trial_id`/`y_gloc_labels` here all index the PRE-removal
    # ordering).
    delta_matrix = np.abs(
        std_train_only[test_mask][:, target_idx] - std_all_rows[test_mask][:, target_idx]
    ) if n_target else np.zeros((n_test_rows, 0))

    positive_gloc_test = (
        np.asarray(y_gloc_labels).ravel().astype(bool)[test_mask]
        if len(y_gloc_labels) > 0 else np.zeros(n_test_rows, dtype=bool)
    )
    beginning_test = (
        _beginning_of_trial_mask(trial_id)[test_mask]
        if len(trial_id) > 0 else np.zeros(n_test_rows, dtype=bool)
    )
    imputed_test = (
        impute_mask_per_window[test_mask]
        if impute_mask_per_window is not None else np.zeros(n_test_rows, dtype=bool)
    )

    # Attach the worst row's three-flag context to each per-feature entry.
    rank_features: list[tuple[float, int, str]] = []
    for j, fname in enumerate(target_features):
        if n_test_rows == 0:
            continue
        worst_row_local = int(np.argmax(delta_matrix[:, j]))
        per_feature[fname]["worst_row_context"] = {
            "row_local_idx": worst_row_local,
            "is_beginning_of_trial": bool(beginning_test[worst_row_local]),
            "positive_gloc": bool(positive_gloc_test[worst_row_local]),
            "had_imputed_input": bool(imputed_test[worst_row_local]),
            "abs_delta_value": float(delta_matrix[worst_row_local, j]),
            "std_train_only_value": float(std_train_only[test_mask][worst_row_local, target_idx[j]]),
            "std_all_rows_value": float(std_all_rows[test_mask][worst_row_local, target_idx[j]]),
        }
        rank_features.append((float(delta_matrix[worst_row_local, j]), j, fname))

    # Top-N features by max abs delta (default 10, see TOP_N_FEATURES).
    rank_features.sort(key=lambda t: t[0], reverse=True)
    top_n = rank_features[:TOP_N_FEATURES]
    top_summary: list[dict[str, Any]] = []
    for rank, (_, j, fname) in enumerate(top_n, start=1):
        entry = {
            "rank": rank,
            "feature": fname,
            "max_abs_delta": per_feature[fname]["max_abs_delta"],
            "mean_abs_delta": per_feature[fname]["mean_abs_delta"],
            "worst_row_context": per_feature[fname]["worst_row_context"],
        }
        top_summary.append(entry)

    # Plot top-N per-fold overlay onto ``fold_dir/plots/``.
    if generate_plots and fold_dir is not None and top_n:
        plots_dir = fold_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for rank, (_, j, fname) in enumerate(top_n, start=1):
            plot_path = plots_dir / f"{rank:02d}_{_sanitize_filename(fname)}.png"
            _plot_top_feature_overlay(
                feature_name=fname,
                train_only_values=std_train_only[test_mask][:, target_idx[j]],
                all_rows_values=std_all_rows[test_mask][:, target_idx[j]],
                is_beginning=beginning_test,
                positive_gloc=positive_gloc_test,
                had_imputed=imputed_test,
                out_path=plot_path,
                title=f"{model_name} | fold {fold_id} | {fname}",
            )
            top_summary[rank - 1]["plot_path"] = str(plot_path.relative_to(fold_dir.parent.parent.parent))
    summary["top_features_count"] = len(top_summary)
    # Aggregate flag counts over the top-N worst rows.
    top_worst_flag_counts = {
        "beginning_of_trial": sum(1 for e in top_summary if e["worst_row_context"]["is_beginning_of_trial"]),
        "positive_gloc": sum(1 for e in top_summary if e["worst_row_context"]["positive_gloc"]),
        "had_imputed_input": sum(1 for e in top_summary if e["worst_row_context"]["had_imputed_input"]),
    }

    return {
        "n_train_rows": int(train_mask.sum()),
        "n_test_rows": n_test_rows,
        "n_total_features_cols": int(std_train_only.shape[1]),
        "n_target_features": n_target,
        "per_feature": per_feature,
        "stratified_summary": stratified,
        "summary": summary,
        "top_features": top_summary,
        "top_features_ranked_by": "max_abs_delta",
        "top_worst_row_flag_counts": top_worst_flag_counts,
    }


def _sanitize_filename(name: str) -> str:
    """Replace characters unsafe for filenames so features can become plot names."""
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", ".", " ", "(", ")"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip().replace(" ", "_")


# ---------------------------------------------------------------------------
# model-level aggregation
# ---------------------------------------------------------------------------
def _aggregate_model_summary(folds: list[dict[str, Any]]) -> dict[str, float]:
    mean_abs = [f["summary"]["mean_abs_delta"] for f in folds]
    median_abs = [f["summary"]["median_abs_delta"] for f in folds]
    max_abs = [f["summary"]["max_abs_delta"] for f in folds]
    return {
        "mean_abs_delta_across_folds": float(np.mean(mean_abs)),
        "median_abs_delta_across_folds": float(np.median(median_abs)),
        "max_abs_delta_across_folds": float(np.max(max_abs)),
    }


# ---------------------------------------------------------------------------
# text report writers
# ---------------------------------------------------------------------------
def _save_text_per_fold(fold_report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("FOLD REPORT")
    lines.append("=" * 80)
    lines.append(f"  n_train_rows: {fold_report['n_train_rows']}")
    lines.append(f"  n_test_rows:  {fold_report['n_test_rows']}")
    lines.append(f"  n_total_features_cols: {fold_report['n_total_features_cols']}")
    lines.append(f"  n_target_features:     {fold_report['n_target_features']}")
    lines.append("")
    s = fold_report["summary"]
    lines.append(f"  mean_abs_delta:   {s['mean_abs_delta']:.6f}")
    lines.append(f"  median_abs_delta: {s['median_abs_delta']:.6f}")
    lines.append(f"  max_abs_delta:    {s['max_abs_delta']:.6f}")
    lines.append(f"  std_abs_delta:    {s['std_abs_delta']:.6f}")
    lines.append("")

    lines.append("-" * 40)
    lines.append("STRATIFIED SUMMARY (s1/s2 x feature_type x baseline_method)")
    lines.append("-" * 40)
    for s1s2 in ("s1", "s2"):
        ft_dict = fold_report.get("stratified_summary", {}).get(s1s2, {})
        if not ft_dict:
            continue
        for ft in ("mean", "stddev", "max", "range", "additional"):
            bm_dict = ft_dict.get(ft, {})
            if not bm_dict:
                continue
            for bm in ("v0", "v1", "v2", "v5", "v6", "none"):
                stats = bm_dict.get(bm)
                if stats is None:
                    continue
                lines.append(f"  {s1s2:>3s} | {ft:>10s} | {bm:>4s} -> "
                             f"mean_abs_delta={stats['mean_abs_delta']:.6f}  "
                             f"max_abs_delta={stats['max_abs_delta']:.6f}  "
                             f"n_features={stats['n_features']}")
    lines.append("")

    lines.append("-" * 40)
    lines.append("PER-FEATURE DELTA STATISTICS (test rows, train_only - all_rows)")
    lines.append("-" * 40)
    for fname, stats_outer in fold_report.get("per_feature", {}).items():
        lines.append(f"  {fname}")
        lines.append(f"    mean_abs_delta={stats_outer['mean_abs_delta']:.6f}  "
                     f"max_abs_delta={stats_outer['max_abs_delta']:.6f}  "
                     f"mean_delta={stats_outer['mean_delta']:.6f}  "
                     f"std_delta={stats_outer['std_delta']:.6f}  "
                     f"s1s2={stats_outer.get('s1_or_s2','?')}  "
                     f"ft={stats_outer.get('feature_type','?')}  "
                     f"bm={stats_outer.get('baseline_method','?')}")

    # Top-N features ranked by max_abs_delta + worst-row three-flag context
    top_features = fold_report.get("top_features", [])
    if top_features:
        lines.append("")
        lines.append("-" * 40)
        lines.append(f"TOP-{len(top_features)} FEATURES BY MAX_ABS_DELTA (worst-row context)")
        lines.append("-" * 40)
        for entry in top_features:
            wrc = entry["worst_row_context"]
            plot_rel = entry.get("plot_path", "")
            lines.append(f"  {entry['rank']:2d}. {entry['feature']}")
            lines.append(f"        max_abs_delta={entry['max_abs_delta']:.6f}  "
                         f"mean_abs_delta={entry['mean_abs_delta']:.6f}")
            lines.append(f"        worst row (local idx={wrc['row_local_idx']}): "
                         f"beginning_of_trial={wrc['is_beginning_of_trial']}  "
                         f"positive_gloc={wrc['positive_gloc']}  "
                         f"had_imputed_input={wrc['had_imputed_input']}")
            lines.append(f"        std_train_only={wrc['std_train_only_value']:.6f}  "
                         f"std_all_rows={wrc['std_all_rows_value']:.6f}  "
                         f"abs_delta={wrc['abs_delta_value']:.6f}")
            if plot_rel:
                lines.append(f"        plot: {plot_rel}")
        counts = fold_report.get("top_worst_row_flag_counts", {})
        if counts:
            n_t = len(top_features)
            lines.append("")
            lines.append(f"  Top-{n_t} worst-row flag breakdown:")
            lines.append(f"    beginning_of_trial:  {counts.get('beginning_of_trial',0)}/{n_t}")
            lines.append(f"    positive_gloc:        {counts.get('positive_gloc',0)}/{n_t}")
            lines.append(f"    had_imputed_input:    {counts.get('had_imputed_input',0)}/{n_t}")
    lines.append("=" * 80)
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_model_text_summary(
    model_name: str,
    summary: dict[str, float],
    model_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"MODEL SUMMARY: {model_name}")
    lines.append("=" * 80)
    lines.append(f"  mean_abs_delta_across_folds:  {summary['mean_abs_delta_across_folds']:.6f}")
    lines.append(f"  median_abs_delta_across_folds: {summary['median_abs_delta_across_folds']:.6f}")
    lines.append(f"  max_abs_delta_across_folds:    {summary['max_abs_delta_across_folds']:.6f}")
    lines.append("=" * 80)
    (model_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _save_text_overall(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("TRADITIONAL STANDARDIZATION METRICS COMPARISON")
    lines.append("Train-Only (current) vs All-Rows (legacy leaky) Standardization")
    lines.append("=" * 80)
    lines.append("")

    cfg = report["config"]
    lines.append("-" * 40)
    lines.append("CONFIG")
    lines.append("-" * 40)
    for k, v in cfg.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("-" * 40)
    lines.append("PER-MODEL SUMMARY")
    lines.append("-" * 40)
    header = f"{'Model':>6s} | {'mean_abs_delta':>14s} | {'median_abs_delta':>16s} | {'max_abs_delta':>13s}"
    lines.append(header)
    lines.append("-" * len(header))
    for mdata in report["models"]:
        name = mdata["name"]
        s = mdata["summary"]
        lines.append(f"{name:>6s} | {s['mean_abs_delta_across_folds']:14.6f} | "
                     f"{s['median_abs_delta_across_folds']:16.6f} | "
                     f"{s['max_abs_delta_across_folds']:13.6f}")
    lines.append("")

    lines.append("-" * 40)
    lines.append("PER-FOLD TABLE (one row per model×fold)")
    lines.append("-" * 40)
    header2 = f"{'Model':>6s} | {'Fold':>4s} | {'#train':>6s} | {'#test':>5s} | {'n_target':>8s} | {'mean_abs':>10s} | {'max_abs':>10s}"
    lines.append(header2)
    lines.append("-" * len(header2))
    for mdata in report["models"]:
        name = mdata["name"]
        for f in mdata["folds"]:
            s = f["summary"]
            lines.append(f"{name:>6s} | {f['fold_id']:4d} | {f['n_train_rows']:6d} | "
                         f"{f['n_test_rows']:5d} | {f['n_target_features']:8d} | "
                         f"{s['mean_abs_delta']:10.6f} | {s['max_abs_delta']:10.6f}")
    lines.append("")

    # Top-N worst features aggregated across all (model × fold) with their
    # worst-row three-flag context, plus a global histogram of those flags.
    lines.append("-" * 40)
    lines.append("TOP-N WORST FEATURES ACROSS ALL (model×fold) — worst-row context")
    lines.append("-" * 40)
    header3 = f"{'Model':>6s} | {'Fold':>4s} | {'Rk':>2s} | {'feature':>40s} | {'max_abs':>10s} | {'B':>1s} | {'G':>1s} | {'I':>1s}"
    lines.append(header3)
    lines.append("-" * len(header3))
    totals = {"beginning_of_trial": 0, "positive_gloc": 0, "had_imputed_input": 0}
    total_top = 0
    for mdata in report["models"]:
        name = mdata["name"]
        for f in mdata["folds"]:
            top = f.get("top_features", [])
            count = f.get("top_worst_row_flag_counts", {})
            totals["beginning_of_trial"] += count.get("beginning_of_trial", 0)
            totals["positive_gloc"] += count.get("positive_gloc", 0)
            totals["had_imputed_input"] += count.get("had_imputed_input", 0)
            total_top += len(top)
            for entry in top:
                wrc = entry["worst_row_context"]
                lines.append(
                    f"{name:>6s} | {f['fold_id']:4d} | {entry['rank']:2d} | "
                    f"{entry['feature'][:40]:40s} | {entry['max_abs_delta']:10.6f} | "
                    f"{'Y' if wrc['is_beginning_of_trial'] else 'N':>1s} | "
                    f"{'Y' if wrc['positive_gloc'] else 'N':>1s} | "
                    f"{'Y' if wrc['had_imputed_input'] else 'N':>1s}"
                )
    if total_top:
        lines.append("")
        lines.append(f"  Top-N worst-row flag totals across {total_top} entries:")
        lines.append(f"    beginning_of_trial:  {totals['beginning_of_trial']}/{total_top}")
        lines.append(f"    positive_gloc:        {totals['positive_gloc']}/{total_top}")
        lines.append(f"    had_imputed_input:   {totals['had_imputed_input']}/{total_top}")
    lines.append("")

    lines.append("=" * 80)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved JSON to %s", path)


# ---------------------------------------------------------------------------
# plotting (headless)
# ---------------------------------------------------------------------------
def _plot_top_feature_overlay(
    feature_name: str,
    train_only_values: np.ndarray,
    all_rows_values: np.ndarray,
    is_beginning: np.ndarray,
    positive_gloc: np.ndarray,
    had_imputed: np.ndarray,
    out_path: Path,
    title: str = "",
) -> None:
    """Plot a single feature's standardized value across test rows.

    Two overlaid lines (train-only vs leaky all-rows); markers highlight the
    three per-row context flags: beginning-of-trial rows (dashed vertical
    lines), positive-GLOC rows (red squares on the train-only curve), and
    had-imputed-input rows (shaded band behind the curves).
    """
    train_only_values = np.asarray(train_only_values, dtype=np.float64)
    all_rows_values = np.asarray(all_rows_values, dtype=np.float64)
    is_beginning = np.asarray(is_beginning, dtype=bool)
    positive_gloc = np.asarray(positive_gloc, dtype=bool)
    had_imputed = np.asarray(had_imputed, dtype=bool)
    n_rows = train_only_values.shape[0]
    x = np.arange(n_rows)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    if np.any(had_imputed) and n_rows > 0:
        for idx in np.flatnonzero(had_imputed):
            ax.axvspan(idx - 0.5, idx + 0.5, color="lightgray", alpha=0.35, lw=0)
    ax.plot(x, train_only_values, color=_PLOT_PALETTE["train_only"], lw=1.2,
            label="train_only (fold-aware)", zorder=3)
    ax.plot(x, all_rows_values, color=_PLOT_PALETTE["all_rows"], lw=1.2,
            label="all_rows (legacy leaky)", zorder=3)
    for idx in np.flatnonzero(is_beginning):
        ax.axvline(idx, color="green", ls="--", lw=0.8, alpha=0.6, zorder=2)
    if np.any(positive_gloc):
        ax.scatter(np.flatnonzero(positive_gloc), train_only_values[positive_gloc],
                   facecolors="none", edgecolors="red", s=40, lw=1.0,
                   label="positive_gloc", zorder=4)
    ax.set_xlabel("test-row index (post row-removal)")
    ax.set_ylabel("standardized feature value")
    ax.set_title(title or feature_name)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_PLOT_DPI)
    plt.close(fig)
    logger.info("Saved plot to %s", out_path)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


def run_standardization_comparison(
    config_path: Path | None = None,
    project_root: Path | None = None,
    output_dir: Path | None = None,
    data_path: Path | None = None,
    generate_plots: bool = True,
) -> dict[str, Any]:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    if config_path is None:
        config_path = project_root / DEFAULT_CONFIG
    if output_dir is None:
        output_dir = project_root / OUTPUT_DIR_NAME

    from src.config_loader import load_experiment_config

    config = load_experiment_config(config_path)
    if data_path is not None:
        config["data_path"] = str(data_path)
    training_cfg = config["sensor_ablation"]["training"]
    num_splits: int = training_cfg["num_splits"]
    random_seed: int = training_cfg["random_seed"]
    model_names: list[str] = training_cfg["models"]
    model_type: ModelType = training_cfg["model_type"]
    feature_streams: list[str] = training_cfg["streams"][0]

    pipeline = DataPipeline(config=config)
    pipeline.set_model_type(model_type)
    pipeline.set_random_seed(random_seed)
    factory = ModelFactory()

    # Compute the stream-aware target filter substrings by reusing the
    # pipeline's canonical `_resolve_feature_groups_for_streams` helper. The
    # facade's `DataPipeline` doesn't expose this method directly (it lives on
    # the BaseGLOCDataPipeline backends), so instantiate a throwaway
    # TraditionalDataPipeline bound to the same config and query the resolver.
    # This produces the same lowercased stream-keyword list the production
    # pipeline uses in `_apply_substring_filter`, so the top-N largest-delta
    # analysis targets exactly the columns the YAML requests — covering
    # derived columns like HRV (SDNN)/HRV (RMSSD) that lack the device
    # suffix and automatically dropping non-requested sensor-group columns.
    resolver = TraditionalDataPipeline(
        data_path=config.get("data_path", ""), random_seed=random_seed, config=config,
    )
    default_feature_groups = resolver.FEATURE_GROUPS_BY_MODEL_TYPE[model_type]
    _, _, filter_substrings = resolver._resolve_feature_groups_for_streams(
        feature_streams, default_feature_groups
    )
    logger.info(
        "Stream-aware target filter substrings (from feature_streams=%s): %s",
        feature_streams, filter_substrings,
    )

    out_root = output_dir / model_type.get_folder_name()
    out_root.mkdir(parents=True, exist_ok=True)

    overall_report: dict[str, Any] = {
        "config": {
            "num_splits": num_splits,
            "random_seed": random_seed,
            "model_type_string": str(model_type),
            "feature_streams": feature_streams,
            "target_substrings": list(filter_substrings or []),
            "top_n_features": TOP_N_FEATURES,
            "generate_plots": generate_plots,
        },
        "models": [],
    }

    global _CURRENT_FOLD

    try:
        for model_name in model_names:
            model = factory.create_model(model_name)
            model_out = out_root / model.name
            model_out.mkdir(parents=True, exist_ok=True)

            fold_reports: list[dict[str, Any]] = []

            for fold_id in range(num_splits):
                fold_json_path = model_out / f"fold_{fold_id}" / "fold_result.json"
                if fold_json_path.exists():
                    logger.info(
                        "Skipping fold %d for model %s (already exists at %s)",
                        fold_id, model.name, fold_json_path,
                    )
                    try:
                        existing_report = json.loads(fold_json_path.read_text())
                        fold_reports.append(existing_report)
                        fold_txt_path = fold_json_path.with_name("fold_result.txt")
                        if not fold_txt_path.exists():
                            _save_text_per_fold(existing_report, fold_txt_path)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Could not load existing fold_result.json for fold %d; will recompute",
                            fold_id,
                        )
                    continue
                _CURRENT_FOLD = fold_id

                try:
                    _ = pipeline.get_data(
                        model=model,
                        kfold_id=fold_id,
                        num_splits=num_splits,
                        feature_streams=feature_streams,
                        traditional_feature_selection="raw",
                        return_feature_names=True,
                    )
                except Exception:
                    logger.error(
                        "get_data failed for model=%s fold=%d",
                        model_name,
                        fold_id,
                        exc_info=True,
                    )
                    # Drop any partial captures from this fold so they don't
                    # leak into subsequent folds (e.g., a pre-fill KNN impute
                    # snapshot but no matching _standardize_raw capture).
                    for store in (
                        _CapturedStandardization, _CapturedFeatures,
                        _CapturedImputeMask, _CapturedExperimentMetadata,
                        _CapturedRemovedRows, _CapturedWindowingParams,
                    ):
                        store.pop(fold_id, None)
                    continue

                cap_std = _CapturedStandardization.pop(fold_id, None)
                cap_feat = _CapturedFeatures.pop(fold_id, None)
                if cap_std is None or cap_feat is None:
                    logger.error(
                        "Missing captured data for model=%s fold=%d. "
                        "std=%s feat=%s",
                        model_name,
                        fold_id,
                        cap_std is not None,
                        cap_feat is not None,
                    )
                    continue

                # Build the per-window impute mask from the pre-feature KNN
                # snapshot + per-sample trial/time + the same windowing params
                # _feature_generation used this fold.
                impute_mask_per_sample = _CapturedImputeMask.pop(fold_id, None)
                cap_meta = _CapturedExperimentMetadata.pop(fold_id, None)
                cap_win = _CapturedWindowingParams.pop(fold_id, None)
                impute_mask_per_window: Optional[np.ndarray] = None
                if (
                    impute_mask_per_sample is not None
                    and cap_meta is not None
                    and cap_win is not None
                ):
                    try:
                        impute_mask_per_window = _reduce_impute_mask_per_window(
                            impute_mask_per_sample=impute_mask_per_sample,
                            trial_id_per_sample=cap_meta["trial_id"],
                            time_per_sample=cap_meta["Time (s)"],
                            time_start=cap_win["time_start"],
                            offset=cap_win["offset"],
                            stride=cap_win["stride"],
                            window_size=cap_win["window_size"],
                            feature_names=cap_meta["feature_names"],
                            trial_id_per_row=cap_feat["trial_id_per_row"],
                        )
                    except Exception:
                        logger.error(
                            "Per-window impute mask reduction failed for model=%s fold=%d",
                            model_name, fold_id, exc_info=True,
                        )

                # Align the captured PRE-_process_NaN arrays with the
                # surviving rows. _analyse_fold consumes test_mask applied to
                # X_raw/trial_id; both arrays below need to keep only surviving
                # rows so the three per-row flags are 1:1 with test-row deltas.
                removed_rows = _CapturedRemovedRows.pop(fold_id, None)
                n_pre = cap_feat["y_gloc_labels"].shape[0]
                if removed_rows is not None and removed_rows.size:
                    y_gloc_prep = cap_feat["y_gloc_labels"].ravel()
                    trial_prep = cap_feat["trial_id_per_row"]
                    if y_gloc_prep.shape[0] != n_pre:
                        y_gloc_prep = y_gloc_prep.reshape(-1)[:n_pre]
                    if trial_prep.shape[0] != n_pre:
                        trial_prep = trial_prep[:n_pre]
                    y_gloc_surv = _remap_through_survivor(y_gloc_prep, removed_rows, n_pre)
                    trial_surv = _remap_through_survivor(trial_prep, removed_rows, n_pre)
                    impute_surv: Optional[np.ndarray] = (
                        _remap_through_survivor(impute_mask_per_window, removed_rows, n_pre)
                        if impute_mask_per_window is not None else None
                    )
                    # train_mask was captured inside _standardize_raw (operating
                    # on the pre-removal matrix); mirror the survivor remap so
                    # test_mask inside _analyse_fold selects the right rows.
                    train_mask_surv = _remap_through_survivor(
                        cap_std["train_mask"], removed_rows, n_pre
                    )
                    X_raw_surv = _remap_through_survivor(
                        cap_std["X_raw"], removed_rows, n_pre
                    )
                else:
                    y_gloc_surv = cap_feat["y_gloc_labels"].ravel()
                    trial_surv = cap_feat["trial_id_per_row"]
                    impute_surv = impute_mask_per_window
                    train_mask_surv = cap_std["train_mask"]
                    X_raw_surv = cap_std["X_raw"]

                fold_dir = model_out / f"fold_{fold_id}"
                fold_dir.mkdir(parents=True, exist_ok=True)

                fold_report = _analyse_fold(
                    X_raw_surv,
                    trial_surv,
                    train_mask_surv,
                    cap_feat["all_features"],
                    y_gloc_surv,
                    impute_surv,
                    filter_substrings=filter_substrings,
                    fold_dir=fold_dir,
                    model_name=model.name,
                    fold_id=fold_id,
                    generate_plots=generate_plots,
                )
                fold_report["fold_id"] = fold_id

                _save_json(fold_report, fold_dir / "fold_result.json")
                _save_text_per_fold(fold_report, fold_dir / "fold_result.txt")
                if fold_report.get("top_features"):
                    _save_json(
                        {"fold_id": fold_id, "top_features": fold_report["top_features"]},
                        fold_dir / "top_features.json",
                    )

                fold_reports.append(fold_report)

            model_summary = _aggregate_model_summary(fold_reports)
            model_entry: dict[str, Any] = {
                "name": model.name,
                "baseline_window": model.data_pipeline_hyperparameters.get("baseline_window"),
                "window_size": model.data_pipeline_hyperparameters.get("window_size"),
                "stride": model.data_pipeline_hyperparameters.get("stride"),
                "summary": model_summary,
                "folds": fold_reports,
            }
            overall_report["models"].append(model_entry)

            _save_json({"summary": model_summary}, model_out / "summary.json")
            _save_model_text_summary(model.name, model_summary, model_out)

    finally:
        TraditionalDataPipeline._standardize_raw = _OriginalStandardizeRaw
        TraditionalDataPipeline._feature_generation = _OriginalFeatureGeneration
        TraditionalDataPipeline._faster_knn_impute = _OriginalFasterKnnImpute
        TraditionalDataPipeline._reduce_memory = _OriginalReduceMemory
        TraditionalDataPipeline._process_NaN_temporal = _OriginalProcessNaN

    _save_json(overall_report, out_root.parent / "summary.json")
    _save_text_overall(overall_report, out_root.parent / "summary.txt")

    return overall_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compare train-only vs all-rows standardization metrics for traditional models."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output reports",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override the data_path from the YAML config (e.g., 'data_reduced' to save memory)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating per-feature overlay plots for the top-N features.",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    data_path = Path(args.data_path) if args.data_path else None

    report = run_standardization_comparison(
        config_path=config_path,
        output_dir=output_dir,
        data_path=data_path,
        generate_plots=not args.no_plots,
    )
    print("Standardization metrics comparison complete.")
    print(f"Results saved to {OUTPUT_DIR_NAME}/")