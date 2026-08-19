"""Save per-fold train-only vs all-rows (legacy leaky) standardization data.

For each traditional model in ``configs/real_time_sensor_ablation.yaml``, this
script re-runs ``TraditionalDataPipeline.get_data`` TWICE per fold under five
monkey-patches that capture the raw per-window feature matrix, trial ids, train
mask, GLOC labels, per-window imputation flag, windowing parameters, and the
per-sample raw sensor DataFrame:

- run 1 (``train_only``): the normal fold-aware pipeline — s1/s2 statistics are
  fit on training rows only.
- run 2 (``all_rows``): the legacy leaky pipeline — the fold train mask is
  overridden to all-True so s1/s2 statistics are fit on every row.

Each run's standardized feature matrix (the full ``[s1 | s2]`` output of
``_feature_generation``) is captured. All per-window arrays are survivor-remapped
through ``_process_NaN_temporal`` so every file is row-aligned 1:1.

No plots or summary reports are produced — only the raw data is saved; a
separate script reads it later for interactive visualization.

Per fold::

    Results/Traditional_Standardization_Metrics/<ModelType>/<model>/fold_N/
        standardized_data.npz       train_only, all_rows   (n_rows, n_cols_doubled)
        delta_data.npz              delta_s1_mean, delta_s1_std   (n_rows, n_raw_cols)
                                    delta_s2_mean, delta_s2_std   (n_raw_cols,)
        trial_data.npz              trial_id, subject, trial      (n_rows,)
        time_data.npz               time (window start)           (n_rows,)
        label_data.npz              y_gloc, train_mask, imputed   (n_rows,)
        raw_per_sample_data.npz     per-sample raw sensor rows for the configured
                                    streams: time, trial_id, subject, trial
                                    (n_samples,) + sensor (n_samples, n_stream_cols)
        fold_metadata.json

``standardized_data.npz`` holds the final standardized feature matrices exactly
as the pipeline emits them (columns are ``[s1 | s2]``, names in
``fold_metadata.json`` under ``feature_names``). The delta arrays are the
all-rows-fit minus train-only-fit μ/σ, computed over ALL surviving rows
(train + test) for every feature: ``delta_s1_*`` vary per row (s1 is per-trial),
``delta_s2_*`` are per-column (s2 is a single global fit) and are identical for
every row.

The unstandardized per-window matrix is NOT saved; it is used internally only to
compute the μ/σ fits. File 6's ``sensor`` matrix holds the raw per-sample values
(captured at ``_reduce_memory`` time, i.e. after pre-feature KNN imputation,
before any feature generation or standardization), restricted to the sensor
streams requested in the YAML ``streams`` section, plus enough ids to subset a
single subject + trial for plotting.

Usage::

    python -m src.real_time.standardization_metrics_analysis.traditional_standardization_metrics
    python -m src.real_time.standardization_metrics_analysis.traditional_standardization_metrics \\
        --config configs/real_time_sensor_ablation.yaml
    python -m src.real_time.standardization_metrics_analysis.traditional_standardization_metrics \\
        --data-path data_reduced
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

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

# Standardization modes the orchestrator drives per pipeline run.
_MODE_TRAIN_ONLY = "train_only"
_MODE_ALL_ROWS = "all_rows"

# ---------------------------------------------------------------------------
# monkey-patch machinery
# ---------------------------------------------------------------------------
# Which standardization fit the current pipeline run should produce. The
# orchestrator sets this to _MODE_TRAIN_ONLY / _MODE_ALL_ROWS before each
# get_data call.
_CURRENT_STANDARDIZATION_MODE: str = _MODE_TRAIN_ONLY

# Per-fold captures.
_CapturedStandardization: dict[int, dict[str, np.ndarray]] = {}
# Standardized pipeline output matrix per mode, keyed by fold id.
_CapturedPipelineOutputs: dict[int, dict[str, np.ndarray]] = {}
_CapturedFeatures: dict[int, dict[str, Any]] = {}
# Per-sample pre-feature KNN imputation mask (shape (n_samples, n_raw_features)),
# captured at the FIRST _faster_knn_impute call within a fold. Subsequent calls
# (post-feature KNN) are ignored — pre-feature is the relevant phase.
_CapturedImputeMask: dict[int, np.ndarray] = {}
# Per-sample trial-id and time columns + the raw feature names list, captured
# from _reduce_memory (which runs AFTER pre-feature imputation but BEFORE
# _feature_generation). Used to reduce the per-sample impute mask per window.
_CapturedExperimentMetadata: dict[int, dict[str, Any]] = {}
# Per-sample raw sensor DataFrame (as passed to _reduce_memory), used to build
# the raw_per_sample_data.npz file.
_CapturedRawGlocData: dict[int, pd.DataFrame] = {}
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
    """Capture the raw matrix / trial ids / fold mask, and on the all-rows run
    override the mask so the standardizers fit on every row."""
    global _CURRENT_FOLD, _CapturedStandardization, _CURRENT_STANDARDIZATION_MODE
    if _CURRENT_STANDARDIZATION_MODE == _MODE_ALL_ROWS:
        train_mask = np.ones_like(train_mask, dtype=bool)
    else:
        _CapturedStandardization[_CURRENT_FOLD] = {
            "X_raw": np.asarray(x_raw, dtype=np.float64).copy(),
            "trial_id_per_row": np.array(trial_id_per_row, copy=True),
            "train_mask": np.array(train_mask, dtype=bool, copy=True),
        }
    return _OriginalStandardizeRaw(self, x_raw, trial_id_per_row, train_mask)


def _capturing_feature_generation(self, *args, **kwargs):
    global _CURRENT_FOLD, _CapturedFeatures, _CapturedWindowingParams
    global _CapturedPipelineOutputs, _CURRENT_STANDARDIZATION_MODE
    result = _OriginalFeatureGeneration(self, *args, **kwargs)
    # Standardized pipeline output matrix (post-standardization, float32).
    _CapturedPipelineOutputs.setdefault(_CURRENT_FOLD, {})[_CURRENT_STANDARDIZATION_MODE] = np.array(
        result[1], copy=True
    )
    if _CURRENT_STANDARDIZATION_MODE == _MODE_TRAIN_ONLY:
        _CapturedFeatures[_CURRENT_FOLD] = {
            "y_gloc_labels": np.array(result[0], copy=True),
            "x_feature_matrix": np.array(result[1], copy=True),
            "all_features": list(result[2]),
            "trial_id_per_row": np.array(result[3], copy=True),
        }
        # _feature_generation's positional args unpack to the hyperparameters
        # that drove windowing: (time_start, offset, stride, window_size, ...).
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
    """Snapshot experiment_metadata (trial_id, Time), the raw feature names,
    and the raw per-sample DataFrame.

    These define the per-sample row order that the captured pre-feature KNN
    impute mask aligns with, and provide the raw sensor values for the
    per-sample output file.
    """
    global _CURRENT_FOLD, _CapturedExperimentMetadata, _CapturedRawGlocData
    global _CURRENT_STANDARDIZATION_MODE
    result = _OriginalReduceMemory(self, gloc_data, gloc_labels, features, output_feature_dtype)
    if _CURRENT_STANDARDIZATION_MODE == _MODE_TRAIN_ONLY:
        _, _, experiment_metadata = result
        _CapturedExperimentMetadata[_CURRENT_FOLD] = {
            "trial_id": np.array(experiment_metadata["trial_id"], copy=True),
            "Time (s)": np.array(experiment_metadata["Time (s)"], copy=True),
            "feature_names": list(features["All"]),
        }
        _CapturedRawGlocData[_CURRENT_FOLD] = gloc_data
    return result


def _capturing_process_nan(self, y_gloc_labels, x_feature_matrix, all_features):
    """Capture the indices of rows removed by _process_NaN_temporal.

    These post-_feature_generation row removals must be mirrored to every
    per-row array so they align with the surviving rows.
    """
    global _CURRENT_FOLD, _CapturedRemovedRows, _CURRENT_STANDARDIZATION_MODE
    y_noNaN, x_noNaN, all_features_out, removed = _OriginalProcessNaN(self, y_gloc_labels, x_feature_matrix, all_features)
    if _CURRENT_STANDARDIZATION_MODE == _MODE_TRAIN_ONLY:
        _CapturedRemovedRows[_CURRENT_FOLD] = np.array(removed, copy=True)
    return y_noNaN, x_noNaN, all_features_out, removed


# apply the monkey-patches (after both capturing functions are defined)
TraditionalDataPipeline._standardize_raw = _capturing_standardize_raw
TraditionalDataPipeline._feature_generation = _capturing_feature_generation
TraditionalDataPipeline._faster_knn_impute = _capturing_faster_knn_impute
TraditionalDataPipeline._reduce_memory = _capturing_reduce_memory
TraditionalDataPipeline._process_NaN_temporal = _capturing_process_nan

# ---------------------------------------------------------------------------
# per-window impute-mask reduction + row remapping
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


def _remap_through_survivor(arr: np.ndarray, removed_indices: np.ndarray, n_pre_rows: int) -> np.ndarray:
    """Apply the same survivor remap as train_mask_pre[survivor_mask_pre]."""
    removed_mask = np.zeros(n_pre_rows, dtype=bool)
    removed_mask[np.asarray(removed_indices, dtype=int)] = True
    survivor = ~removed_mask
    return arr[survivor]


# ---------------------------------------------------------------------------
# standardization-metric fits (μ/σ only, no z-score)
# ---------------------------------------------------------------------------
def _per_row_s1_stats(X: np.ndarray, trial_id: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row s1 μ/σ matrices from a TrialAwareStandardizer fit.

    Mirrors ``TrialAwareStandardizer.transform``'s per-row mean/std construction
    (pooled broadcast overridden per-trial) without computing z-scores.
    """
    fitter = TrialAwareStandardizer().fit(X, trial_id, fit_mask)
    mean = np.broadcast_to(fitter._pooled_mean, X.shape).copy()
    std = np.broadcast_to(fitter._pooled_std, X.shape).copy()
    for tid, m in fitter._per_trial_mean.items():
        sel = trial_id == tid
        mean[sel] = m
        std[sel] = fitter._per_trial_std[tid]
    return mean, std


def _per_col_s2_stats(X: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column s2 μ/σ vectors from a GlobalStandardizer fit."""
    fitter = GlobalStandardizer().fit(X[fit_mask])
    return fitter.mean_, fitter.std_


def _raw_feature_names(all_features: list[str]) -> list[str]:
    """Dedupe the ``[s1 | s2]`` name list to raw feature names (strip _s1/_s2)."""
    names = []
    for name in all_features:
        base = name[:-3] if name.endswith("_s1") or name.endswith("_s2") else name
        if base not in names:
            names.append(base)
    return names


def _window_start_times(trial_id_per_row: np.ndarray, time_start: float, stride: float) -> np.ndarray:
    """Window start times: ``time_start + j*stride`` within each trial block.

    Rows are contiguous per-trial blocks, so each block gets an arithmetic
    progression starting at ``time_start``.
    """
    out = np.zeros(len(trial_id_per_row), dtype=np.float64)
    for tid in pd.unique(trial_id_per_row):
        sel = trial_id_per_row == tid
        n = int(sel.sum())
        out[sel] = time_start + stride * np.arange(n)
    return out


# ---------------------------------------------------------------------------
# per-fold dataset assembly + save
# ---------------------------------------------------------------------------
def _build_fold_dataset(
    X_raw: np.ndarray,
    train_only: np.ndarray,
    all_rows: np.ndarray,
    trial_id: np.ndarray,
    train_mask: np.ndarray,
    y_gloc: np.ndarray,
    imputed: np.ndarray,
    all_features: list[str],
    time_start: float,
    stride: float,
) -> dict[str, Any]:
    """Assemble the row-aligned per-fold dataset from captured arrays.

    All inputs must be survivor-remapped (post ``_process_NaN_temporal``) so
    the returned arrays are aligned 1:1. ``train_only`` / ``all_rows`` are the
    standardized pipeline outputs; the μ/σ delta arrays are the all-rows fit
    minus train-only fit, computed over ALL rows (train + test).
    """
    n_rows, n_raw_cols = X_raw.shape
    all_true = np.ones(n_rows, dtype=bool)

    s1_mean_train, s1_std_train = _per_row_s1_stats(X_raw, trial_id, train_mask)
    s1_mean_all, s1_std_all = _per_row_s1_stats(X_raw, trial_id, all_true)
    s2_mean_train, s2_std_train = _per_col_s2_stats(X_raw, train_mask)
    s2_mean_all, s2_std_all = _per_col_s2_stats(X_raw, all_true)

    parts = np.array([str(t).split("-") for t in trial_id], dtype=object)
    subject = np.array([p[0] for p in parts], dtype=str)
    trial = np.array([p[1] if len(p) > 1 else "" for p in parts], dtype=str)
    trial_id = np.asarray(trial_id).astype(str)

    return {
        "train_only": train_only,
        "all_rows": all_rows,
        "delta_s1_mean": s1_mean_all - s1_mean_train,
        "delta_s1_std": s1_std_all - s1_std_train,
        "delta_s2_mean": s2_mean_all - s2_mean_train,
        "delta_s2_std": s2_std_all - s2_std_train,
        "trial_id": trial_id,
        "subject": subject,
        "trial": trial,
        "time": _window_start_times(trial_id, time_start, stride),
        "y_gloc": y_gloc,
        "train_mask": train_mask,
        "imputed": imputed,
        "feature_names": list(all_features),
        "raw_feature_names": _raw_feature_names(all_features),
    }


def _build_raw_sample_dataset(
    gloc_data: pd.DataFrame,
    filter_substrings: list[str],
) -> dict[str, Any]:
    """Per-sample raw sensor rows restricted to the configured streams.

    ``gloc_data`` is captured at ``_reduce_memory`` time (after pre-feature KNN
    imputation, before feature generation / standardization). Columns whose
    lowercased name contains any stream keyword are kept, plus the ids needed
    to subset a single subject + trial for plotting.
    """
    keep = [c for c in gloc_data.columns if any(s in c.lower() for s in filter_substrings)]
    # Raw sensor columns can hold 'NO VALUE' string placeholders for missing
    # samples; coerce them to NaN so the matrix stays numeric.
    sensor = gloc_data[keep].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return {
        "time": gloc_data["Time (s)"].to_numpy(dtype=np.float64),
        "trial_id": gloc_data["trial_id"].to_numpy().astype(str),
        "subject": gloc_data["subject"].to_numpy().astype(str),
        "trial": gloc_data["trial"].to_numpy().astype(str),
        "sensor": sensor,
        "raw_column_names": keep,
    }


def _save_fold_files(
    dataset: dict[str, Any],
    raw_sample_dataset: dict[str, Any],
    fold_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Write the six per-fold .npz files + fold_metadata.json."""
    fold_dir.mkdir(parents=True, exist_ok=True)

    file_arrays = {
        "standardized_data.npz": ("train_only", "all_rows"),
        "delta_data.npz": ("delta_s1_mean", "delta_s1_std", "delta_s2_mean", "delta_s2_std"),
        "trial_data.npz": ("trial_id", "subject", "trial"),
        "time_data.npz": ("time",),
        "label_data.npz": ("y_gloc", "train_mask", "imputed"),
    }
    for filename, keys in file_arrays.items():
        arrays = {k: np.asarray(dataset[k]) for k in keys}
        np.savez_compressed(fold_dir / filename, **arrays)

    raw_arrays = {
        k: np.asarray(v) for k, v in raw_sample_dataset.items() if k != "raw_column_names"
    }
    np.savez_compressed(fold_dir / "raw_per_sample_data.npz", **raw_arrays)

    with open(fold_dir / "fold_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved fold files to %s", fold_dir)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def run_standardization_comparison(
    config_path: Path | None = None,
    project_root: Path | None = None,
    output_dir: Path | None = None,
    data_path: Path | None = None,
) -> dict[str, Any]:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]
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

    # Resolve the stream-aware sensor-column substrings for the per-sample raw
    # output. The facade doesn't expose the resolver, so use a throwaway backend
    # bound to the same config (produces the same lowercased keyword list the
    # production pipeline applies in _apply_substring_filter).
    resolver = TraditionalDataPipeline(
        data_path=config.get("data_path", ""), random_seed=random_seed, config=config,
    )
    default_feature_groups = resolver.FEATURE_GROUPS_BY_MODEL_TYPE[model_type]
    _, _, filter_substrings = resolver._resolve_feature_groups_for_streams(
        feature_streams, default_feature_groups
    )
    logger.info(
        "Stream-aware sensor-column filter substrings (from feature_streams=%s): %s",
        feature_streams, filter_substrings,
    )

    out_root = output_dir / model_type.get_folder_name()
    out_root.mkdir(parents=True, exist_ok=True)

    saved: dict[str, dict[int, str]] = {}
    global _CURRENT_FOLD, _CURRENT_STANDARDIZATION_MODE

    try:
        for model_name in model_names:
            model = factory.create_model(model_name)
            model_out = out_root / model.name
            model_out.mkdir(parents=True, exist_ok=True)
            saved[model.name] = {}

            for fold_id in range(num_splits):
                fold_dir = model_out / f"fold_{fold_id}"
                resume_path = fold_dir / "standardized_data.npz"
                if resume_path.exists():
                    logger.info(
                        "Skipping fold %d for model %s (already saved at %s)",
                        fold_id, model.name, resume_path,
                    )
                    saved[model.name][fold_id] = str(resume_path)
                    continue

                _CURRENT_FOLD = fold_id
                try:
                    for mode in (_MODE_TRAIN_ONLY, _MODE_ALL_ROWS):
                        _CURRENT_STANDARDIZATION_MODE = mode
                        pipeline.get_data(
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
                        model_name, fold_id, exc_info=True,
                    )
                    # Drop any partial captures from this fold so they don't
                    # leak into subsequent folds.
                    for store in (
                        _CapturedStandardization, _CapturedPipelineOutputs,
                        _CapturedFeatures, _CapturedImputeMask,
                        _CapturedExperimentMetadata, _CapturedRawGlocData,
                        _CapturedRemovedRows, _CapturedWindowingParams,
                    ):
                        store.pop(fold_id, None)
                    continue

                cap_std = _CapturedStandardization.pop(fold_id, None)
                cap_feat = _CapturedFeatures.pop(fold_id, None)
                pipe_outputs = _CapturedPipelineOutputs.pop(fold_id, None)
                if (
                    cap_std is None or cap_feat is None or pipe_outputs is None
                    or _MODE_TRAIN_ONLY not in pipe_outputs or _MODE_ALL_ROWS not in pipe_outputs
                ):
                    logger.error(
                        "Missing captured data for model=%s fold=%d",
                        model_name, fold_id,
                    )
                    continue

                removed_rows = _CapturedRemovedRows.pop(fold_id, None)
                n_pre = cap_std["X_raw"].shape[0]

                def _remap_if_needed(arr: np.ndarray) -> np.ndarray:
                    if removed_rows is not None and removed_rows.size:
                        if arr.shape[0] != n_pre:
                            arr = arr.ravel()[:n_pre] if arr.ndim == 1 else arr[:n_pre]
                        return _remap_through_survivor(arr, removed_rows, n_pre)
                    return arr

                X_raw_surv = _remap_if_needed(cap_std["X_raw"])
                trial_surv = _remap_if_needed(cap_std["trial_id_per_row"])
                train_mask_surv = _remap_if_needed(cap_std["train_mask"])
                y_gloc_surv = _remap_if_needed(cap_feat["y_gloc_labels"].ravel())
                train_only_surv = _remap_if_needed(pipe_outputs[_MODE_TRAIN_ONLY])
                all_rows_surv = _remap_if_needed(pipe_outputs[_MODE_ALL_ROWS])

                # Build the per-window impute flag from the pre-feature KNN
                # snapshot + per-sample trial/time + this fold's windowing params.
                imputed = np.zeros(len(trial_surv), dtype=bool)
                impute_mask_per_sample = _CapturedImputeMask.pop(fold_id, None)
                cap_meta = _CapturedExperimentMetadata.pop(fold_id, None)
                cap_win = _CapturedWindowingParams.pop(fold_id, None)
                if (
                    impute_mask_per_sample is not None
                    and cap_meta is not None
                    and cap_win is not None
                ):
                    try:
                        imputed_pre = _reduce_impute_mask_per_window(
                            impute_mask_per_sample=impute_mask_per_sample,
                            trial_id_per_sample=cap_meta["trial_id"],
                            time_per_sample=cap_meta["Time (s)"],
                            time_start=cap_win["time_start"],
                            offset=cap_win["offset"],
                            stride=cap_win["stride"],
                            window_size=cap_win["window_size"],
                            feature_names=cap_meta["feature_names"],
                            trial_id_per_row=cap_std["trial_id_per_row"],
                        )
                        imputed = _remap_if_needed(imputed_pre)
                    except Exception:
                        logger.error(
                            "Per-window impute mask reduction failed for model=%s fold=%d",
                            model_name, fold_id, exc_info=True,
                        )

                time_start = cap_win["time_start"] if cap_win else 0.0
                stride = cap_win["stride"] if cap_win else 1.0

                dataset = _build_fold_dataset(
                    X_raw_surv, train_only_surv, all_rows_surv, trial_surv,
                    train_mask_surv, y_gloc_surv, imputed,
                    cap_feat["all_features"], time_start, stride,
                )

                gloc_data = _CapturedRawGlocData.pop(fold_id, None)
                if gloc_data is None:
                    logger.error(
                        "Missing raw gloc_data for model=%s fold=%d",
                        model_name, fold_id,
                    )
                    continue
                raw_sample_dataset = _build_raw_sample_dataset(gloc_data, filter_substrings)

                metadata = {
                    "model_name": model.name,
                    "fold_id": fold_id,
                    "feature_names": dataset["feature_names"],
                    "raw_feature_names": dataset["raw_feature_names"],
                    "raw_column_names": raw_sample_dataset["raw_column_names"],
                    "shapes": {
                        "standardized_data.npz": {
                            k: list(dataset[k].shape) for k in ("train_only", "all_rows")
                        },
                        "delta_data.npz": {
                            k: list(dataset[k].shape)
                            for k in ("delta_s1_mean", "delta_s1_std", "delta_s2_mean", "delta_s2_std")
                        },
                        "trial_data.npz": {
                            k: list(dataset[k].shape) for k in ("trial_id", "subject", "trial")
                        },
                        "time_data.npz": {"time": list(dataset["time"].shape)},
                        "label_data.npz": {
                            k: list(dataset[k].shape) for k in ("y_gloc", "train_mask", "imputed")
                        },
                        "raw_per_sample_data.npz": {
                            k: list(v.shape)
                            for k, v in raw_sample_dataset.items() if k != "raw_column_names"
                        },
                    },
                    "num_splits": num_splits,
                    "random_seed": random_seed,
                    "model_type_string": str(model_type),
                    "feature_streams": feature_streams,
                }
                _save_fold_files(dataset, raw_sample_dataset, fold_dir, metadata)
                saved[model.name][fold_id] = str(resume_path)

    finally:
        TraditionalDataPipeline._standardize_raw = _OriginalStandardizeRaw
        TraditionalDataPipeline._feature_generation = _OriginalFeatureGeneration
        TraditionalDataPipeline._faster_knn_impute = _OriginalFasterKnnImpute
        TraditionalDataPipeline._reduce_memory = _OriginalReduceMemory
        TraditionalDataPipeline._process_NaN_temporal = _OriginalProcessNaN

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Save per-fold train-only vs all-rows standardization data for traditional models."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output data",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override the data_path from the YAML config (e.g., 'data_reduced' to save memory)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    data_path = Path(args.data_path) if args.data_path else None

    saved = run_standardization_comparison(
        config_path=config_path,
        output_dir=output_dir,
        data_path=data_path,
    )
    print("Standardization data saved.")
    for model_name, folds in saved.items():
        print(f"  {model_name}: {len(folds)} fold(s)")
