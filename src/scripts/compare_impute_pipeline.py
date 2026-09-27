import os
import pickle
import sys

# Lets this file run directly (e.g. PyCharm's play button) as well as via
#   python -m src.scripts.compare_impute_pipeline
# The pipeline modules use relative imports, so the script needs to know it lives in the src.scripts package.
if not __package__:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    __package__ = "src.scripts"

# Same modules as the GLOC main script
from .GLOC_data_processing_traditional import *
from .imputation_traditional import *
from .baseline_methods_traditional import *
from .features_traditional import *
from .feature_selection import *
from .GLOC_classifier_traditional import *
from .GLOC_visualization_traditional import *
from .imbalance_techniques_traditional import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
Compares x_train / x_test with imputation (impute_type = 1) against without (impute_type = 0), for one shared
train/test split, using the GLOC main-script functions 

Why a shared split needs help: the main script runs data load -> impute -> baseline -> features -> process_NaN ->
split, all inside one pass. process_NaN drops whichever rows still have NaN after that run's imputation, so the
no-impute and impute runs keep different sets of rows -- calling pre_classification_training_test_split separately
in each run, as main script does, gives two different splits over two different row sets, and x_train from one run
is then not comparable cell-by-cell against x_train from the other.

So this script instead:
  1. Runs the pipeline through remove_constant_columns for both impute types (same as the main script up to that
     point), BEFORE process_NaN and BEFORE the split.
  2. Calls pre_classification_training_test_split ONCE, on that common pre-process_NaN row range, so both runs
     share the exact same train/test assignment.
  3. Works out which of those rows process_NaN would keep in EACH run separately (its drop rule, replicated
     here only to track row identity).
  4. Compares x_train / x_test on the rows process_NaN keeps in BOTH runs, and separately reports how many rows
     process_NaN would keep in only one run (i.e. rescued, or lost, by imputation).

This relies on impute_type = 0 keeping the raw NaNs (instead of the main script's own impute_type 0, which
deletes raw NaN rows via process_NaN_raw -- that leaves features_phys/ecg/eeg at the old row count and opens time
gaps that leave sliding windows with zero rows, crashing feature_generation). Keeping the NaNs means both runs
process the same trial/time grid, so build_full_matrix(0) and build_full_matrix(1) return the same row count in
the same order, which is what step 2 above depends on.
"""

# Pipeline settings, copied from the GLOC main script (must stay identical between the two runs)
DATAFOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
RANDOM_STATE = 42
REMOVE_NAN_TRIALS = True
KNN_K = 3  # matches faster_knn_impute(features, k=3) in the main script

MODEL_TYPE = ['complete', 'explicit']
if 'noAFE' in MODEL_TYPE and 'explicit' in MODEL_TYPE:
    FEATURE_GROUPS = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                      'rawEEG', 'processedEEG', 'strain', 'demographics']
if 'noAFE' in MODEL_TYPE and 'implicit' in MODEL_TYPE:
    FEATURE_GROUPS = ['ECG', 'BR', 'temp', 'eyetracking', 'rawEEG', 'processedEEG']

# FROM DL PIPELINE
## Model Parameters
if 'noAFE' in MODEL_TYPE and 'explicit' in MODEL_TYPE:
    FEATURE_GROUPS = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                 'rawEEG', 'processedEEG', 'demographics', 'strain']

if 'noAFE' in MODEL_TYPE and 'implicit' in MODEL_TYPE:
    FEATURE_GROUPS = ['ECG', 'BR', 'temp', 'eyetracking', 'rawEEG']

if 'complete' in MODEL_TYPE and 'implicit' in MODEL_TYPE:
    FEATURE_GROUPS = ['ECG', 'BR', 'temp', 'eyetracking', 'rawEEG', 'AFE']  # AFE is removed downstream

if 'complete' in MODEL_TYPE and 'explicit' in MODEL_TYPE:
    FEATURE_GROUPS = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                 'rawEEG', 'processedEEG', 'demographics', 'strain']

# NOTE:
# AFE indicator is required for EEG imputation in complete models,
# but is only included as a predictive feature for explicit models.

# Set baseline characteristics. Depends on model type
if 'noAFE' in MODEL_TYPE:
    BASELINE_METHODS = ['v0', 'v1', 'v2', 'v5', 'v6', 'v7', 'v8']
else:
    BASELINE_METHODS = ['v0', 'v1', 'v2', 'v5', 'v6']


ANALYSIS_TYPE = 2
BASELINE_WINDOW = 10  # seconds
WINDOW_SIZE = 10      # seconds
STRIDE = 1            # seconds
OFFSET = 0            # seconds
TIME_START = 0        # seconds
TRAINING_RATIO = 0.8
SUBJECT_TO_ANALYZE = '01'
TRIAL_TO_ANALYZE = '02'

# Comparison settings
DEFAULT_CHUNK_ROWS = 100_000    # Rows processed at a time. Lower this if memory is tight
DEFAULT_ATOL = 1e-9              # Fixed floor tolerance, applied at every rtol level below (handles near-zero cells)
RTOL_LEVELS = [0.0, 1e-9, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 0.75, 1]  # Relative tolerances to sweep
REF_RTOL = 1e-6                  # Which of the above levels gets a per-feature breakdown in the saved CSV

# Absolute tolerances to sweep (units of the feature itself, i.e. no scaling by the value's own magnitude).
# NOTE: baseline_methods_to_use mixes subtract-based methods (v2, v6, v8 -- roughly z-score-like units, centered
# near 0) with divide-based methods (v1, v5, v7 -- ratio units, centered near 1). A single absolute threshold
# means something different for the two groups, so treat cross-feature comparisons at a given atol with that
# in mind; it's still the right tool for "how many z-score-like units did this shift", which %-of-value can't
# answer for values near zero.
ATOL_LEVELS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0, 2.0]

# Which of the ATOL_LEVELS get a per-modality (eeg/ecg/hr/etc.) breakdown -- both must already be in ATOL_LEVELS
MODALITY_ATOL_LEVELS = [0.2, 1.0]


############################################### BUILD FEATURE MATRIX ###############################################
def build_full_matrix(impute_type):
    """
    Runs the GLOC main script's pipeline for the given impute_type (0 = no imputation, raw NaNs kept;
    1 = KNN impute raw data), through remove_constant_columns. Stops BEFORE process_NaN and the split -- see
    module docstring for why. Otherwise identical to the main script's import_feature_matrix == 0 branch.
    """
    (filename, baseline_data_filename, demographic_data_filename,
     list_of_eeg_data_files, list_of_baseline_eeg_processed_files) = data_locations(DATAFOLDER)

    (gloc_data_reduced, features, features_phys, features_ecg, features_eeg, all_features, all_features_phys,
     all_features_ecg, all_features_eeg) = (
        analysis_driven_csv_processing(ANALYSIS_TYPE, filename, FEATURE_GROUPS, demographic_data_filename,
                                       MODEL_TYPE, list_of_eeg_data_files, TRIAL_TO_ANALYZE, SUBJECT_TO_ANALYZE))

    gloc = label_gloc_events(gloc_data_reduced)

    # 'complete' keeps both AFE and non-AFE trials (no afe_subset), but AFE equipment records different EEG
    # channels than non-AFE, so eeg_condition_impute mean-fills whichever channels the other condition's rows
    # are missing. The AFE/non-AFE 'condition' indicator itself is set aside here (removed from `features`) and
    # stashed on gloc_data_reduced as 'AFE_indicator' so it survives remove_all_nan_trials/KNN impute below and
    # can be windowed and re-attached to x_feature_matrix after feature_generation, as its own feature.
    if 'complete' in MODEL_TYPE and 'explicit' in MODEL_TYPE:
        condition_idx = all_features.index('condition')
        afe_indicator_column = features[:, condition_idx]

        gloc_data_reduced, features, features_phys, features_eeg = (
            eeg_condition_impute(gloc_data_reduced, all_features, all_features_phys, all_features_eeg,
                                 afe_indicator_column))

        features = np.delete(features, condition_idx, axis=1)
        all_features = [stream for stream in all_features if stream != 'condition']

        gloc_data_reduced["AFE_indicator"] = afe_indicator_column
    if 'noAFE' in MODEL_TYPE:
        gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc = (
            afe_subset(MODEL_TYPE, gloc_data_reduced, all_features,
                       features, features_phys, features_ecg, features_eeg, gloc))

    if REMOVE_NAN_TRIALS:
        gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc, nan_proportion_df = (
            remove_all_nan_trials(gloc_data_reduced, all_features,
                                  features, features_phys, features_ecg, features_eeg, gloc))

    if impute_type == 1:
        features = faster_knn_impute(features, k=KNN_K)

    trial_column = gloc_data_reduced['trial_id']
    time_column = gloc_data_reduced['Time (s)']
    event_validated_column = gloc_data_reduced['event_validated']
    subject_column = gloc_data_reduced['subject']

    # Grab the AFE indicator back out before gloc_data_reduced is freed (it was stashed on it above so it would
    # survive remove_all_nan_trials, which can drop rows/trials)
    if 'complete' in MODEL_TYPE and 'explicit' in MODEL_TYPE:
        afe_indicator_column = gloc_data_reduced["AFE_indicator"].to_numpy(dtype=np.float32).reshape(-1, 1)

    del gloc_data_reduced

    combined_baseline, combined_baseline_names, baseline_v0, baseline_names_v0 = (
        baseline_data(BASELINE_METHODS, trial_column, time_column, event_validated_column, subject_column, features,
                      all_features, gloc, BASELINE_WINDOW, features_phys, all_features_phys, features_ecg,
                      all_features_ecg, features_eeg, all_features_eeg, baseline_data_filename,
                      list_of_baseline_eeg_processed_files, MODEL_TYPE))

    y_gloc_labels, x_feature_matrix, all_features = (
        feature_generation(TIME_START, OFFSET, STRIDE, WINDOW_SIZE, combined_baseline, gloc, trial_column,
                           time_column, combined_baseline_names, baseline_names_v0, baseline_v0, FEATURE_GROUPS))

    # Re-attach the AFE indicator as its own windowed feature (max over the window -- it's constant 0/1 within
    # a trial anyway, so this just carries it through the same windowing as everything else)
    if 'complete' in MODEL_TYPE and 'explicit' in MODEL_TYPE:
        afe_indicator_column_windowed, gloc_compare, _ = sliding_window_max(
            afe_indicator_column, trial_column, time_column, gloc, OFFSET, STRIDE, WINDOW_SIZE, TIME_START)
        x_feature_matrix = np.hstack([x_feature_matrix, afe_indicator_column_windowed])
        all_features.append('AFE_indicator_windowed')

    x_feature_matrix, all_features = remove_constant_columns(x_feature_matrix, all_features)

    return y_gloc_labels, x_feature_matrix, list(all_features)


def build_and_cache_full(impute_type, root):
    """
    Builds the pre-process_NaN feature matrix for one impute type and writes it to disk, then frees it.
    If the cache exists it is reused, so DELETE the impute_typeN folder after changing any pipeline setting.
    """
    d = os.path.join(root, f"impute_type{impute_type}")
    names = ["y_full.npy", "x_full.npy", "features.pkl"]

    if all(os.path.exists(os.path.join(d, n)) for n in names):
        print(f"Using cached data for impute_type={impute_type}: {d}")
        return d

    os.makedirs(d, exist_ok=True)
    print(f"Building data for impute_type={impute_type} ...")
    y_full, x_full, feats = build_full_matrix(impute_type)

    np.save(os.path.join(d, "y_full.npy"), np.asarray(y_full))
    np.save(os.path.join(d, "x_full.npy"), np.asarray(x_full))
    with open(os.path.join(d, "features.pkl"), "wb") as f:
        pickle.dump(feats, f)
    return d


def load_cached_full(d):
    """Memory-mapped x (read from disk as needed), small y fully loaded."""
    y_full = np.load(os.path.join(d, "y_full.npy"))
    x_full = np.load(os.path.join(d, "x_full.npy"), mmap_mode="r")
    with open(os.path.join(d, "features.pkl"), "rb") as f:
        feats = pickle.load(f)
    return y_full, x_full, feats


############################################# PROCESS_NAN BOOKKEEPING #############################################
def process_nan_keep_mask(x_full, chunk_rows):
    """
    Replicates process_NaN's own row/column selection (GLOC_data_processing_traditional.py: drop all-NaN
    columns, then drop any row still NaN in the surviving columns) chunk by chunk, purely to learn which
    ORIGINAL row positions it would keep -- process_NaN itself doesn't return that. The values this lets us
    read out of x_full for the kept rows/columns are identical to what process_NaN's own output would contain,
    since it only selects rows/columns and never transforms values.

    Returns (col_all_nan, keep_mask), both booleans over the full column / row axis.
    """
    n_rows, n_feat = x_full.shape

    # Matches process_NaN's index_column_all_NaN: is every value in this column NaN?
    any_valid = np.zeros(n_feat, dtype=bool)
    for s in range(0, n_rows, chunk_rows):
        c = np.asarray(x_full[s:s + chunk_rows], dtype=np.float64)
        any_valid |= (~np.isnan(c)).any(axis=0)
    col_all_nan = ~any_valid

    # Matches process_NaN's row filter, restricted to the surviving columns
    surv_cols = ~col_all_nan
    keep_mask = np.zeros(n_rows, dtype=bool)
    for s in range(0, n_rows, chunk_rows):
        c = np.asarray(x_full[s:s + chunk_rows][:, surv_cols], dtype=np.float64)
        keep_mask[s:s + chunk_rows] = ~np.isnan(c).any(axis=1)

    return col_all_nan, keep_mask


################################################### COMPARISON ###################################################
def feature_stats(x, col_idx, names, chunk_rows, row_idx):
    """
    Per-feature mean and std over the given (arbitrary, original-position) rows and columns, computed chunk by
    chunk with a numerically stable running merge (Chan et al.). These rows are already known NaN-free in these
    columns (guaranteed by process_nan_keep_mask), so nan_count is included only as a sanity check.
    """
    n_rows = len(row_idx)
    n_feat = len(col_idx)
    n = np.zeros(n_feat)
    mean = np.zeros(n_feat)
    m2 = np.zeros(n_feat)

    for s in range(0, n_rows, chunk_rows):
        c = np.asarray(x[row_idx[s:s + chunk_rows]][:, col_idx], dtype=np.float64)
        n_b = (~np.isnan(c)).sum(axis=0).astype(np.float64)

        mean_b = np.divide(np.nansum(c, axis=0), n_b, out=np.zeros(n_feat), where=n_b > 0)
        m2_b = np.nansum((c - mean_b) ** 2, axis=0)

        tot = n + n_b
        delta = mean_b - mean
        mean = mean + np.divide(delta * n_b, tot, out=np.zeros(n_feat), where=tot > 0)
        m2 = m2 + m2_b + delta ** 2 * np.divide(n * n_b, tot, out=np.zeros(n_feat), where=tot > 0)
        n = tot

    has_data = n > 0
    return pd.DataFrame({
        "nan_count": (n_rows - n).astype(np.int64),
        "mean": np.where(has_data, mean, np.nan),
        "std": np.where(has_data, np.sqrt(np.divide(m2, n, out=np.zeros(n_feat), where=has_data)), np.nan),
    }, index=names)


def build_sweep_levels(rtol_levels, atol_levels, fixed_atol):
    """
    Builds one combined list of thresholds to sweep in a single pass: relative levels (diff > fixed_atol +
    rtol * |impute value|, i.e. numpy's isclose rule -- "differs by more than X% of the value") and absolute
    levels (diff > atol, rtol = 0 -- "differs by more than X units, regardless of the value's own size").
    Each entry is a dict: kind ("rtol"/"atol"), value (the swept number), and the actual (rtol, atol) to test.
    """
    levels = [{"kind": "rtol", "value": r, "rtol": r, "atol": fixed_atol} for r in rtol_levels]
    levels += [{"kind": "atol", "value": a, "rtol": 0.0, "atol": a} for a in atol_levels]
    return levels


def classify_baseline_group(feature_name, methods=BASELINE_METHODS):
    """
    Maps a feature name back to the baseline method that produced it. combine_all_baseline names columns
    "<base>_<method>[_derivative|_second_derivative]", and feature_generation then appends a window-stat suffix
    ("_mean_s1", "_stddev_s2", ...), so the method tag always shows up as "_v0_", "_v1_", etc. somewhere in the
    middle of the name (or, rarely, right at the end with no stat suffix). Falls back to "untagged" for anything
    that doesn't carry one of BASELINE_METHODS (there shouldn't be much, if any).
    """
    for m in methods:
        if f"_{m}_" in feature_name or feature_name.endswith(f"_{m}"):
            return m
    return "untagged"


# Sensor/modality keywords to look for in a feature name, checked in this order (first match wins), based on
# the raw column names analysis_driven_csv_processing pulls per FEATURE_GROUPS (GLOC_data_processing_traditional.py):
# ECG group  -> 'HR (bpm) - Equivital', 'ECG Lead 1/2 - Equivital', 'HR_instant/average/w_average - Equivital'
# EEG groups -> '..._delta/theta/alpha/beta - EEG', rawEEG / processedEEG columns
# BR         -> 'BR (rpm) - Equivital'
# temp       -> 'Skin Temperature - IR Thermometer (C) - Equivital'
# eyetracking-> 'Pupil position/diameter ... - Tobii'
# 'hr' is checked separately from 'ecg' (not folded together) since that's what was asked for -- it also
# catches the HRV features (hrv_sdnn, hrv_rmssd), which are heart-rate-derived, not raw ECG waveform.
MODALITY_KEYWORDS = [
    ("ECG", ["ecg"]),
    ("HR", ["hr"]),
    ("EEG", ["eeg"]),
    ("BR", ["br "]),
    ("temp", ["temp"]),
    ("eyetracking", ["pupil", "tobii"]),
    ("strain", ["strain"]),
    ("AFE", ["afe"]),
    ("demographics", ["age", "weight", "height", "sex", "demographic"]),
]


def classify_modality(feature_name, keywords=MODALITY_KEYWORDS):
    """
    Maps a feature name to a sensor modality by case-insensitive keyword search (see MODALITY_KEYWORDS for the
    exact keywords and where they come from). Falls back to "other" for anything unmatched -- mostly G-force
    and any demographic/derived columns the keyword list above doesn't happen to catch.
    """
    lower = feature_name.lower()
    for label, kws in keywords:
        if any(kw in lower for kw in kws):
            return label
    return "other"


def cell_diff_sweep(x0, idx0_cols, x1, idx1_cols, row_idx, chunk_rows, levels, ref_rtol, ref_atol):
    """
    Cell-by-cell comparison at the SAME original row positions (row_idx) in both runs, on matched columns
    (idx0_cols / idx1_cols can differ, since process_NaN can drop different all-NaN columns per run).

    Computes |no-impute - impute| once per chunk, then checks it against every threshold in levels (each its
    own (rtol, atol) pair -- see build_sweep_levels) in the same pass, so sweeping several tolerances costs one
    extra boolean comparison per level rather than another full read of the data. Per-feature counts are kept
    at EVERY level (cheap: n_levels x n_features ints), not just the reference one, so callers can break the
    sweep down by feature group (e.g. by baseline method) at any tolerance without re-scanning the data.

    Returns:
      overall_df       -- one row per level: cells_changed and frac_changed, summed over ALL shared cells
      feat_changed      -- per-feature changed-cell count, at the level matching (ref_rtol, ref_atol) only
                           (for the per-feature CSV)
      feat_max_abs       -- per-feature largest absolute change (tolerance-independent)
      level_feat_changed -- (n_levels, n_features) changed-cell counts, every level (for group breakdowns)
    """
    n_feat = len(idx0_cols)
    n_rows = len(row_idx)
    total_cells = n_rows * n_feat

    level_feat_changed = np.zeros((len(levels), n_feat), dtype=np.int64)
    ref_pos = next(i for i, lv in enumerate(levels) if lv["rtol"] == ref_rtol and lv["atol"] == ref_atol)
    feat_max_abs = np.zeros(n_feat)

    for s in range(0, n_rows, chunk_rows):
        rows = row_idx[s:s + chunk_rows]
        ca = np.asarray(x0[rows][:, idx0_cols], dtype=np.float64)
        cb = np.asarray(x1[rows][:, idx1_cols], dtype=np.float64)
        diff = np.abs(ca - cb)
        abs_cb = np.abs(cb)
        feat_max_abs = np.maximum(feat_max_abs, diff.max(axis=0))

        for li, lv in enumerate(levels):
            mask = diff > (lv["atol"] + lv["rtol"] * abs_cb)
            level_feat_changed[li] += mask.sum(axis=0)

    overall_changed = level_feat_changed.sum(axis=1)
    overall_df = pd.DataFrame(levels)
    overall_df["cells_changed"] = overall_changed
    overall_df["total_cells"] = total_cells
    overall_df["frac_changed"] = overall_changed / total_cells if total_cells else np.full(len(levels), np.nan)
    return overall_df, level_feat_changed[ref_pos], feat_max_abs, level_feat_changed


def compare_split_shared(name, split_idx, x0, keep0, x1, keep1, idx0_cols, idx1_cols, shared_features,
                         chunk_rows, levels, ref_rtol, ref_atol, modality_atol_levels=MODALITY_ATOL_LEVELS):
    """
    Compares one shared split (original row positions assigned to train or test by the ONE split done on the
    common pre-process_NaN row range). Rows process_NaN keeps in BOTH runs are compared cell-by-cell, at every
    threshold in levels (see build_sweep_levels); rows it keeps in only one run are counted separately, to show
    how many windows imputation rescues (or costs).
    """
    split_idx = np.asarray(split_idx)
    k0 = keep0[split_idx]
    k1 = keep1[split_idx]

    shared_idx = np.sort(split_idx[k0 & k1])
    only_imp_idx = split_idx[k1 & ~k0]
    only_noimp_idx = split_idx[k0 & ~k1]
    neither_idx = split_idx[~k0 & ~k1]

    print(f"\n===== {name} =====")
    print(f"Rows assigned to this split: {len(split_idx)}")
    print(f"  Kept by process_NaN in both runs (compared below): {len(shared_idx)}")
    print(f"  Rescued by imputation only (process_NaN drops without it): {len(only_imp_idx)}")
    print(f"  Kept without imputation but dropped by process_NaN with it: {len(only_noimp_idx)}")
    print(f"  Dropped by process_NaN in both runs: {len(neither_idx)}")

    out = pd.concat([
        feature_stats(x0, idx0_cols, shared_features, chunk_rows, shared_idx).add_suffix("_noimp"),
        feature_stats(x1, idx1_cols, shared_features, chunk_rows, shared_idx).add_suffix("_imp"),
    ], axis=1)
    out["mean_diff"] = out["mean_imp"] - out["mean_noimp"]
    out["std_diff"] = out["std_imp"] - out["std_noimp"]

    group_sweep_df = pd.DataFrame(columns=["split", "kind", "value", "rtol", "atol", "baseline_group",
                                          "n_features", "cells_changed", "total_cells", "frac_changed"])

    if len(shared_idx) > 0:
        overall_df, feat_changed, max_abs, level_feat_changed = cell_diff_sweep(
            x0, idx0_cols, x1, idx1_cols, shared_idx, chunk_rows, levels, ref_rtol, ref_atol)
        out[f"cells_changed_rtol_{ref_rtol}"] = feat_changed
        out["max_abs_change"] = max_abs
        overall_df.insert(0, "split", name)

        total_shared_cells = len(shared_idx) * len(shared_features)
        pretty = overall_df.assign(frac_changed=lambda d: d["frac_changed"].map(lambda v: f"{v:.4%}"))
        print(f"\nFraction of {total_shared_cells} shared cells that differ, by RELATIVE tolerance "
              f"(> X% of the imputed value):")
        print(pretty.loc[pretty["kind"] == "rtol", ["value", "cells_changed", "frac_changed"]]
              .rename(columns={"value": "rtol"}).to_string(index=False))
        print(f"\nFraction that differ, by ABSOLUTE tolerance (> X units, regardless of the value's size):")
        print(pretty.loc[pretty["kind"] == "atol", ["value", "cells_changed", "frac_changed"]]
              .rename(columns={"value": "atol"}).to_string(index=False))

        # Break the changed-cell counts down by baseline method (v0, v1, v2, ...), at every level already
        # computed above -- no extra pass over the data needed, this just re-sums the per-feature counts.
        groups = np.array([classify_baseline_group(f) for f in shared_features])
        n_shared_rows = len(shared_idx)
        rows = []
        for li, lv in enumerate(levels):
            for g in np.unique(groups):
                gm = groups == g
                n_cells_g = int(gm.sum()) * n_shared_rows
                changed_g = int(level_feat_changed[li, gm].sum())
                rows.append({**lv, "baseline_group": g, "n_features": int(gm.sum()), "cells_changed": changed_g,
                            "total_cells": n_cells_g, "frac_changed": changed_g / n_cells_g if n_cells_g else np.nan})
        group_sweep_df = pd.DataFrame(rows)
        group_sweep_df.insert(0, "split", name)

        focused = (group_sweep_df[(group_sweep_df["rtol"] == ref_rtol) & (group_sweep_df["atol"] == ref_atol)]
                  .sort_values("frac_changed", ascending=False))
        print(f"\nWhere the differences live, by baseline method (rtol={ref_rtol}, atol={ref_atol}):")
        print(focused.assign(frac_changed=lambda d: d["frac_changed"].map(lambda v: f"{v:.4%}"))
              [["baseline_group", "n_features", "cells_changed", "total_cells", "frac_changed"]]
              .to_string(index=False))

        # Same idea, but by sensor modality (eeg/ecg/hr/etc. -- see MODALITY_KEYWORDS), at the specific
        # absolute-tolerance levels in modality_atol_levels only (not the whole sweep).
        atol_pos = {lv["value"]: i for i, lv in enumerate(levels) if lv["kind"] == "atol"}
        missing = [a for a in modality_atol_levels if a not in atol_pos]
        if missing:
            print(f"\nWARNING: atol level(s) {missing} not in ATOL_LEVELS, so no modality breakdown at "
                  f"{missing} -- add them to ATOL_LEVELS to include.")

        modalities = np.array([classify_modality(f) for f in shared_features])
        mod_rows = []
        for a in modality_atol_levels:
            if a not in atol_pos:
                continue
            li = atol_pos[a]
            for mgroup in np.unique(modalities):
                mm = modalities == mgroup
                n_cells_m = int(mm.sum()) * n_shared_rows
                changed_m = int(level_feat_changed[li, mm].sum())
                mod_rows.append({"atol": a, "modality": mgroup, "n_features": int(mm.sum()),
                                 "cells_changed": changed_m, "total_cells": n_cells_m,
                                 "frac_changed": changed_m / n_cells_m if n_cells_m else np.nan})
        modality_df = pd.DataFrame(mod_rows)
        modality_df.insert(0, "split", name)

        if len(modality_df) > 0:
            pivot_mod = modality_df.pivot(index="modality", columns="atol", values="frac_changed") * 100
            print(f"\n% of shared cells that differ, by modality (rows = modality, columns = atol):")
            print(pivot_mod.round(3).to_string())
    else:
        print("No rows survived process_NaN in both runs for this split; nothing to compare cell-by-cell.")
        overall_df = pd.DataFrame(columns=["split", "kind", "value", "rtol", "atol", "cells_changed",
                                          "total_cells", "frac_changed"])
        modality_df = pd.DataFrame(columns=["split", "atol", "modality", "n_features", "cells_changed",
                                           "total_cells", "frac_changed"])

    summary = {
        "split": name,
        "rows_assigned": len(split_idx),
        "kept_both": len(shared_idx),
        "rescued_by_impute": len(only_imp_idx),
        "kept_noimp_only": len(only_noimp_idx),
        "dropped_both": len(neither_idx),
    }
    return out, summary, overall_df, group_sweep_df, modality_df


def save_fig(fig, save_dir, stem):
    """Saves a raster PNG (for quick viewing) plus vector SVG and PDF (for editing / print) of the same figure."""
    fig.savefig(os.path.join(save_dir, f"{stem}.png"), dpi=150)
    fig.savefig(os.path.join(save_dir, f"{stem}.svg"))
    fig.savefig(os.path.join(save_dir, f"{stem}.pdf"))


def plot_comparison(name, comp, save_dir=None):
    """Bar charts of mean/std shift and count of changed cells (at ref_rtol), per feature."""
    features = list(comp.index)
    changed_col = next((c for c in comp.columns if c.startswith("cells_changed_rtol_")), None)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    x = np.arange(len(features))

    axes[0].bar(x, comp["mean_diff"])
    axes[0].set_ylabel("Mean (impute - no impute)")

    axes[1].bar(x, comp["std_diff"])
    axes[1].set_ylabel("Std (impute - no impute)")

    if changed_col:
        axes[2].bar(x, comp[changed_col])
        axes[2].set_ylabel(changed_col)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(features, rotation=90, fontsize=5)

    fig.suptitle(f"{name}: impute vs no impute")
    fig.tight_layout()

    if save_dir:
        save_fig(fig, save_dir, f"compare_{name}")
    return fig


def plot_tolerance_sweep(sweep_df, save_dir=None):
    """
    Two line plots of fraction of shared cells changed: one vs. relative tolerance (rtol, % of the value),
    one vs. absolute tolerance (atol, units of the value). One line per split in each.
    """
    fig, (ax_rel, ax_abs) = plt.subplots(1, 2, figsize=(13, 5))

    rel = sweep_df[sweep_df["kind"] == "rtol"]
    for split_name, g in rel.groupby("split"):
        g = g.sort_values("value")
        # value=0 doesn't plot on a log x-axis; nudge it to a small positive value just for plotting
        pos_vals = g.loc[g["value"] > 0, "value"]
        x_vals = g["value"].replace(0.0, pos_vals.min() / 10 if len(pos_vals) else 1e-12)
        ax_rel.plot(x_vals, g["frac_changed"] * 100, marker="o", label=split_name)
    ax_rel.set_xscale("log")
    ax_rel.set_xlabel("Relative error tolerance")
    ax_rel.set_ylabel("% of items that differ")
    ax_rel.set_title("By relative tolerance")
    ax_rel.legend()
    ax_rel.grid(True, which="both", alpha=0.3)

    ab = sweep_df[sweep_df["kind"] == "atol"]
    for split_name, g in ab.groupby("split"):
        g = g.sort_values("value")
        ax_abs.plot(g["value"], g["frac_changed"] * 100, marker="o", label=split_name)
    ax_abs.set_xscale("log")
    ax_abs.set_xlabel("Absolute error tolerance")
    ax_abs.set_ylabel("% of items that differ")
    ax_abs.set_title("By absolute tolerance")
    ax_abs.legend()
    ax_abs.grid(True, which="both", alpha=0.3)

    fig.suptitle("Impute vs no-impute: cell difference rate by tolerance")
    fig.tight_layout()

    if save_dir:
        save_fig(fig, save_dir, "tolerance_sweep")
    return fig


def compare_impute(root="./impute_compare", chunk_rows=DEFAULT_CHUNK_ROWS, rtol_levels=RTOL_LEVELS,
                   atol_levels=ATOL_LEVELS, ref_rtol=REF_RTOL, ref_atol=DEFAULT_ATOL, save_results=True):
    """
    Builds the feature matrix with impute_type 0 and 1 (both stopping right before process_NaN and the
    train/test split), defines ONE shared split with the main script's own pre_classification_training_test_split
    over that common row range, applies process_NaN separately per run on top of it, and compares x_train / x_test
    on the rows process_NaN kept in both -- at every relative tolerance in rtol_levels ("differs by more than X%
    of the value") and every absolute tolerance in atol_levels ("differs by more than X units"), so you can see
    how the % of cells that "differ" shrinks as either tolerance loosens. Returns dict with the per-feature
    comparison tables, a row-survival summary, and the tolerance sweep table.
    """
    os.makedirs(root, exist_ok=True)

    dir0 = build_and_cache_full(0, root)
    dir1 = build_and_cache_full(1, root)

    y0, x0, feat0 = load_cached_full(dir0)
    y1, x1, feat1 = load_cached_full(dir1)

    if x0.shape[0] != x1.shape[0]:
        raise ValueError(
            f"impute_type=0 produced {x0.shape[0]} windows but impute_type=1 produced {x1.shape[0]}. A shared "
            "split needs both runs to generate the same windows in the same order, which only holds because "
            "impute_type=0 here keeps raw NaNs instead of dropping raw rows (see build_full_matrix's docstring). "
            "If this fires, something upstream changed the row count independent of imputation."
        )
    n = x0.shape[0]

    labels_match = np.array_equal(y0, y1)
    print(f"Labels identical before process_NaN: {labels_match} ({n} rows)")
    if not labels_match:
        print("WARNING: y differs between runs at the same row position, even though imputation should not "
              "change GLOC labels. The shared split below still uses impute_type=0's labels for stratification.")

    # One shared split, using the main script's own split function, over the common pre-process_NaN row range.
    # Passing an index array in place of x_feature_matrix makes it split (and return) row positions instead.
    train_idx, test_idx, _, _ = pre_classification_training_test_split(y0, np.arange(n), TRAINING_RATIO,
                                                                       RANDOM_STATE)

    print("Finding which rows process_NaN would keep in each run ...")
    col_all_nan0, keep0 = process_nan_keep_mask(x0, chunk_rows)
    col_all_nan1, keep1 = process_nan_keep_mask(x1, chunk_rows)
    print(f"process_NaN keeps {int(keep0.sum())}/{n} rows without imputation, {int(keep1.sum())}/{n} with it")

    pos0 = {f: i for i, f in enumerate(feat0)}
    pos1 = {f: i for i, f in enumerate(feat1)}
    survivors0 = {feat0[i] for i in range(len(feat0)) if not col_all_nan0[i]}
    survivors1 = {feat1[i] for i in range(len(feat1)) if not col_all_nan1[i]}
    shared_features = [f for f in feat0 if f in survivors0 and f in survivors1]
    idx0_cols = [pos0[f] for f in shared_features]
    idx1_cols = [pos1[f] for f in shared_features]
    print(f"Feature columns: {len(feat0)} total, {len(survivors0)} survive process_NaN without imputation, "
          f"{len(survivors1)} survive with it, {len(shared_features)} shared and usable for comparison")

    levels = build_sweep_levels(rtol_levels, atol_levels, DEFAULT_ATOL)

    comp = {}
    summary_rows = []
    sweep_dfs = []
    group_sweep_dfs = []
    modality_dfs = []
    for split_name, split_idx in [("x_train", train_idx), ("x_test", test_idx)]:
        out, summary, sweep_df, group_sweep_df, modality_df = compare_split_shared(
            split_name, split_idx, x0, keep0, x1, keep1, idx0_cols, idx1_cols, shared_features, chunk_rows,
            levels, ref_rtol, ref_atol)
        plot_comparison(split_name, out, root)
        if save_results:
            out.to_csv(os.path.join(root, f"compare_{split_name}.csv"))
        comp[split_name] = out
        summary_rows.append(summary)
        sweep_dfs.append(sweep_df)
        group_sweep_dfs.append(group_sweep_df)
        modality_dfs.append(modality_df)

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))
    if save_results:
        summary_df.to_csv(os.path.join(root, "row_survival_summary.csv"), index=False)

    tolerance_sweep_df = pd.concat(sweep_dfs, ignore_index=True)
    plot_tolerance_sweep(tolerance_sweep_df, root)
    if save_results:
        tolerance_sweep_df.to_csv(os.path.join(root, "tolerance_sweep.csv"), index=False)

    baseline_group_sweep_df = pd.concat(group_sweep_dfs, ignore_index=True)
    if save_results:
        baseline_group_sweep_df.to_csv(os.path.join(root, "baseline_group_sweep.csv"), index=False)

    modality_breakdown_df = pd.concat(modality_dfs, ignore_index=True)
    if save_results:
        modality_breakdown_df.to_csv(os.path.join(root, "modality_breakdown.csv"), index=False)

    return {"comp_train": comp["x_train"], "comp_test": comp["x_test"], "row_survival_summary": summary_df,
           "tolerance_sweep": tolerance_sweep_df, "baseline_group_sweep": baseline_group_sweep_df,
           "modality_breakdown": modality_breakdown_df}


if __name__ == "__main__":

    results = compare_impute(root="./impute_compare_complete2")

    plt.show()
