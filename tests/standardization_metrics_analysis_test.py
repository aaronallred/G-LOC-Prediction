"""Tests for the standardization-metrics data saver.

Validates the per-window imputation reduction, the survivor-mask remap, the
μ/σ fit helpers (train-only vs all-rows), the per-fold dataset assembly, the
per-sample raw sensor extraction, and the six-file .npz + metadata save on
synthetic data without exercising the pipeline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.real_time.standardization_metrics_analysis.traditional_standardization_metrics import (
    _build_fold_dataset,
    _build_raw_sample_dataset,
    _per_col_s2_stats,
    _per_row_s1_stats,
    _raw_feature_names,
    _reduce_impute_mask_per_window,
    _remap_through_survivor,
    _save_fold_files,
    _window_start_times,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _synthetic_setup(n_trials=2, n_windows_per_trial=10, n_raw_features=5, seed=7):
    rng = np.random.default_rng(seed)
    # Sliding-window math: with (stride=1, window_size=1, offset=0) a trial
    # spanning time t=0..T yields T windows. To get exactly
    # ``n_windows_per_trial`` windows per trial, use (n_windows_per_trial + 1)
    # samples per trial so max(time) = n_windows_per_trial.
    n_samples_per_trial = n_windows_per_trial + 1
    n_rows = n_trials * n_windows_per_trial

    raw_feature_names = [f"f{i}_mean_s1" for i in range(n_raw_features)]
    # Match the column count of _feature_generation's output (doubled).
    all_features = raw_feature_names + [n.replace("_s1", "_s2") for n in raw_feature_names]

    # Per-sample arrays used to drive the impute reduction.
    trial_id_per_sample = np.array(
        [f"t{i}" for i in range(n_trials) for _ in range(n_samples_per_trial)]
    )
    time_per_sample = np.concatenate(
        [np.arange(0.0, n_samples_per_trial, dtype=np.float64) for _ in range(n_trials)]
    )
    pre_impute_mask = np.zeros((len(time_per_sample), n_raw_features), dtype=bool)
    # Mark sample index 3 of trial 0 imputed (falls into window j=3 with
    # stride=1, window_size=1).
    pre_impute_mask[3, 0] = True

    # Hyperparameters matching _sliding_window_mean_calc.
    time_start, offset, stride, window_size = 0.0, 0.0, 1.0, 1.0
    trial_id_per_row = np.array(
        [f"t{i}" for i in range(n_trials) for _ in range(n_windows_per_trial)]
    )

    # Raw feature matrix: Gaussian noise per row.
    X_raw = rng.standard_normal((n_rows, n_raw_features))
    # Standardized pipeline outputs (full [s1 | s2] width).
    train_only = rng.standard_normal((n_rows, 2 * n_raw_features))
    all_rows = rng.standard_normal((n_rows, 2 * n_raw_features))

    train_mask = np.ones(n_rows, dtype=bool)
    # Mark the last row of each trial block as a test row.
    for t in range(n_trials):
        train_mask[(t + 1) * n_windows_per_trial - 1] = False

    y_gloc_labels = np.zeros(n_rows, dtype=np.float32)
    if n_windows_per_trial > 3:
        y_gloc_labels[3] = 1.0  # GLOC-positive row inside trial 0

    return dict(
        n_rows=n_rows,
        n_raw_features=n_raw_features,
        all_features=all_features,
        trial_id_per_sample=trial_id_per_sample,
        time_per_sample=time_per_sample,
        pre_impute_mask=pre_impute_mask,
        time_start=time_start, offset=offset, stride=stride, window_size=window_size,
        trial_id_per_row=trial_id_per_row,
        X_raw=X_raw,
        train_only=train_only,
        all_rows=all_rows,
        train_mask=train_mask,
        y_gloc_labels=y_gloc_labels,
    )


def _impute_reduced(s):
    return _reduce_impute_mask_per_window(
        impute_mask_per_sample=s["pre_impute_mask"],
        trial_id_per_sample=s["trial_id_per_sample"],
        time_per_sample=s["time_per_sample"],
        time_start=s["time_start"], offset=s["offset"], stride=s["stride"],
        window_size=s["window_size"],
        feature_names=[f"raw_{i}" for i in range(s["n_raw_features"])],
        trial_id_per_row=s["trial_id_per_row"],
    )


def _build_dataset(s, trial_id=None, imputed=None, removed=None):
    n_pre = s["n_rows"]
    if trial_id is None:
        trial_id = np.array(
            [f"S1-T{i}" for i in range(2) for _ in range(s["n_rows"] // 2)]
        )
    if imputed is None:
        imputed = _impute_reduced(s)
    if removed is not None and removed.size:
        remap = lambda arr: _remap_through_survivor(arr, removed, n_pre)
        X_raw, trial_id, train_mask, y_gloc, imputed = (
            remap(s["X_raw"]), remap(trial_id), remap(s["train_mask"]),
            remap(s["y_gloc_labels"]), remap(imputed),
        )
        train_only, all_rows = remap(s["train_only"]), remap(s["all_rows"])
    else:
        X_raw, train_mask, y_gloc = s["X_raw"], s["train_mask"], s["y_gloc_labels"]
        train_only, all_rows = s["train_only"], s["all_rows"]
    return _build_fold_dataset(
        X_raw=X_raw, train_only=train_only, all_rows=all_rows,
        trial_id=trial_id, train_mask=train_mask, y_gloc=y_gloc, imputed=imputed,
        all_features=s["all_features"], time_start=0.0, stride=1.0,
    )


# ---------------------------------------------------------------------------
# Pure-function tests (no pipeline)
# ---------------------------------------------------------------------------
class TestPerWindowImputeReduction:
    def test_mirrors_sliding_window_order_and_window_count(self):
        s = _synthetic_setup(n_trials=2, n_windows_per_trial=10, n_raw_features=5)
        out = _impute_reduced(s)
        assert out.shape == (20,)
        assert out.dtype == bool
        # Trial 0 window 3 should be flagged imputed (sample index 3 in trial
        # 0 corresponds to time t=3, which falls into window j=3 since
        # stride=1, window_size=1).
        assert bool(out[3]) is True
        # All other windows have no imputed samples.
        assert int(out.sum()) == 1

    def test_returns_empty_with_no_rows(self):
        out = _reduce_impute_mask_per_window(
            impute_mask_per_sample=np.zeros((0, 5)),
            trial_id_per_sample=np.array([]),
            time_per_sample=np.array([]),
            time_start=0.0, offset=0.0, stride=1.0, window_size=1.0,
            feature_names=[f"raw_{i}" for i in range(5)],
            trial_id_per_row=np.array([]),
        )
        assert out.shape == (0,)


class TestSurvivorRemap:
    def test_drops_removed_rows(self):
        arr = np.arange(10)
        removed = np.array([2, 5, 8])
        out = _remap_through_survivor(arr, removed, n_pre_rows=10)
        assert out.tolist() == [0, 1, 3, 4, 6, 7, 9]

    def test_passthrough_when_no_removals(self):
        arr = np.arange(5)
        out = _remap_through_survivor(arr, np.array([], dtype=int), n_pre_rows=5)
        assert out.tolist() == [0, 1, 2, 3, 4]


class TestWindowStartTimes:
    def test_arithmetic_progression_within_each_block(self):
        trial_id = np.array(["t0", "t0", "t0", "t1", "t1"])
        out = _window_start_times(trial_id, time_start=10.0, stride=2.0)
        assert out.tolist() == [10.0, 12.0, 14.0, 10.0, 12.0]

    def test_empty_input(self):
        out = _window_start_times(np.array([]), time_start=0.0, stride=1.0)
        assert out.shape == (0,)
        assert out.dtype == np.float64


class TestRawFeatureNames:
    def test_strips_s1_s2_suffixes_and_dedupes(self):
        all_features = ["a_mean_s1", "b_mean_s1", "a_mean_s2", "b_mean_s2", "no_suffix"]
        out = _raw_feature_names(all_features)
        assert out == ["a_mean", "b_mean", "no_suffix"]


class TestPerRowS1Stats:
    def _two_trial_matrix(self):
        # Rows 0-1 = trial t0 (train), rows 2-3 = trial t1 (test-only).
        X = np.array(
            [[10.0, 5.0],
             [12.0, 5.0],
             [100.0, 5.0],
             [110.0, 5.0]],
        )
        trial_id = np.array(["t0", "t0", "t1", "t1"])
        train_mask = np.array([True, True, False, False])
        return X, trial_id, train_mask

    def test_train_only_uses_pooled_stats_for_test_only_trials(self):
        X, trial_id, train_mask = self._two_trial_matrix()
        mean, std = _per_row_s1_stats(X, trial_id, train_mask)
        # t0 rows get trial-t0 stats (mean 11, std 1); t1 rows fall back to the
        # pooled training stats (also mean 11, std 1); the constant column has
        # std 0 everywhere.
        np.testing.assert_allclose(mean, np.tile([11.0, 5.0], (4, 1)))
        np.testing.assert_allclose(std, np.tile([1.0, 0.0], (4, 1)))

    def test_all_rows_uses_per_trial_stats_for_every_trial(self):
        X, trial_id, _ = self._two_trial_matrix()
        all_true = np.ones(len(X), dtype=bool)
        mean, std = _per_row_s1_stats(X, trial_id, all_true)
        # t0 rows use t0 stats; t1 rows use t1 stats (mean 105, std 5).
        np.testing.assert_allclose(mean, [[11.0, 5.0]] * 2 + [[105.0, 5.0]] * 2)
        np.testing.assert_allclose(std, [[1.0, 0.0]] * 2 + [[5.0, 0.0]] * 2)


class TestPerColS2Stats:
    def _matrix(self):
        X = np.array(
            [[10.0, 5.0],
             [12.0, 5.0],
             [100.0, 5.0],
             [110.0, 5.0]],
        )
        return X

    def test_train_only_fits_on_training_rows(self):
        X = self._matrix()
        train_mask = np.array([True, True, False, False])
        mean, std = _per_col_s2_stats(X, train_mask)
        np.testing.assert_allclose(mean, [11.0, 5.0])
        np.testing.assert_allclose(std, [1.0, 0.0])

    def test_all_rows_fits_on_all_rows(self):
        X = self._matrix()
        all_true = np.ones(len(X), dtype=bool)
        mean, std = _per_col_s2_stats(X, all_true)
        np.testing.assert_allclose(mean, [58.0, 5.0])
        # Population std of [10, 12, 100, 110].
        np.testing.assert_allclose(std[0], np.std([10.0, 12.0, 100.0, 110.0]))
        np.testing.assert_allclose(std[1], 0.0)


class TestBuildFoldDataset:
    def test_shapes_keys_and_row_alignment(self):
        s = _synthetic_setup(n_trials=2, n_windows_per_trial=4, n_raw_features=3)
        trial_id = np.array([f"S1-T{i}" for i in range(2) for _ in range(4)])
        dataset = _build_dataset(s, trial_id=trial_id)
        n_rows = s["n_rows"]
        n_cols = s["n_raw_features"]
        # Standardized pipeline outputs keep the full doubled width.
        for k in ("train_only", "all_rows"):
            assert dataset[k].shape == (n_rows, 2 * n_cols)
        # s1 deltas are per-row.
        for k in ("delta_s1_mean", "delta_s1_std"):
            assert dataset[k].shape == (n_rows, n_cols)
        # s2 deltas are per-column.
        for k in ("delta_s2_mean", "delta_s2_std"):
            assert dataset[k].shape == (n_cols,)
        for k in ("time", "trial_id", "subject", "trial", "y_gloc", "train_mask", "imputed"):
            assert len(dataset[k]) == n_rows
        # All per-row arrays aligned with the raw rows.
        assert np.array_equal(dataset["train_mask"], s["train_mask"])
        assert np.array_equal(dataset["trial_id"], trial_id)
        # subject/trial split from "S1-T0"/"S1-T1" ids.
        assert dataset["subject"].tolist() == ["S1"] * 8
        assert dataset["trial"].tolist() == ["T0"] * 4 + ["T1"] * 4
        # time = window start within each trial block.
        assert dataset["time"].tolist() == [0.0, 1.0, 2.0, 3.0] * 2
        # feature_names maps 1:1 to standardized columns; raw_feature_names is
        # the deduped counterpart for the delta arrays.
        assert dataset["feature_names"] == s["all_features"]
        assert dataset["raw_feature_names"] == _raw_feature_names(s["all_features"])

    def test_s2_deltas_equal_columnwise_fit_differences(self):
        s = _synthetic_setup(n_trials=2, n_windows_per_trial=4, n_raw_features=3)
        dataset = _build_dataset(s)
        s2_mean_train, s2_std_train = _per_col_s2_stats(s["X_raw"], s["train_mask"])
        s2_mean_all, s2_std_all = _per_col_s2_stats(
            s["X_raw"], np.ones(s["n_rows"], dtype=bool)
        )
        np.testing.assert_allclose(dataset["delta_s2_mean"], s2_mean_all - s2_mean_train)
        np.testing.assert_allclose(dataset["delta_s2_std"], s2_std_all - s2_std_train)

    def test_survivor_remap_before_build(self):
        s = _synthetic_setup(n_trials=2, n_windows_per_trial=5, n_raw_features=3)
        removed = np.array([1, 7])
        trial_pre = np.array([f"S1-T{i}" for i in range(2) for _ in range(5)])
        trial_surv = _remap_through_survivor(trial_pre, removed, s["n_rows"])
        mask_surv = _remap_through_survivor(s["train_mask"], removed, s["n_rows"])
        y_surv = _remap_through_survivor(s["y_gloc_labels"], removed, s["n_rows"])
        imputed_surv = _remap_through_survivor(_impute_reduced(s), removed, s["n_rows"])

        dataset = _build_dataset(s, trial_id=trial_pre, removed=removed)
        n_pre = s["n_rows"]
        assert dataset["train_only"].shape[0] == n_pre - len(removed)
        assert dataset["all_rows"].shape[0] == n_pre - len(removed)
        assert dataset["delta_s1_mean"].shape[0] == n_pre - len(removed)
        # Every per-row array agrees with the survivor-remapped source.
        assert np.array_equal(dataset["trial_id"], trial_surv)
        assert np.array_equal(dataset["train_mask"], mask_surv)
        assert np.array_equal(dataset["imputed"], imputed_surv)
        assert np.array_equal(dataset["y_gloc"], y_surv)


class TestBuildRawSampleDataset:
    def test_filters_to_requested_streams_only(self):
        gloc_data = pd.DataFrame(
            {
                "Time (s)": [0.0, 1.0, 2.0],
                "trial_id": ["01-01", "01-01", "01-01"],
                "subject": ["01", "01", "01"],
                "trial": ["01", "01", "01"],
                "ECG Lead 1 - Equivital_v0": [1.0, 2.0, 3.0],
                "HR (bpm) - Equivital_v0": [70.0, 71.0, 72.0],
                "HRV (SDNN)_v0": [10.0, 11.0, 12.0],
                "Pupil diameter left [mm] - Tobii_v0": [3.0, 3.1, 3.2],
                "AFE_indicator": [0, 1, 0],
            }
        )
        out = _build_raw_sample_dataset(gloc_data, ["ecg", "hr"])
        # Only ECG + HR(+HRV) stream columns survive.
        assert out["raw_column_names"] == [
            "ECG Lead 1 - Equivital_v0",
            "HR (bpm) - Equivital_v0",
            "HRV (SDNN)_v0",
        ]
        assert out["sensor"].shape == (3, 3)
        np.testing.assert_allclose(out["sensor"][:, 1], [70.0, 71.0, 72.0])
        assert out["trial_id"].tolist() == ["01-01"] * 3
        assert out["subject"].tolist() == ["01"] * 3
        assert out["trial"].tolist() == ["01"] * 3
        assert out["time"].tolist() == [0.0, 1.0, 2.0]

    def test_empty_match_yields_zero_sensor_columns(self):
        gloc_data = pd.DataFrame(
            {
                "Time (s)": [0.0],
                "trial_id": ["01-01"],
                "subject": ["01"],
                "trial": ["01"],
                "EEG Fz_v0": [1.0],
            }
        )
        out = _build_raw_sample_dataset(gloc_data, ["ecg", "hr"])
        assert out["raw_column_names"] == []
        assert out["sensor"].shape == (1, 0)


class TestSaveFoldFiles:
    def _dataset_and_raw(self):
        s = _synthetic_setup(n_trials=1, n_windows_per_trial=3, n_raw_features=2)
        dataset = _build_dataset(s, trial_id=np.array(["S1-T1"] * 3))
        raw_sample = _build_raw_sample_dataset(
            pd.DataFrame(
                {
                    "Time (s)": [0.0, 1.0, 2.0],
                    "trial_id": ["01-01"] * 3,
                    "subject": ["01"] * 3,
                    "trial": ["01"] * 3,
                    "ECG Lead 1 - Equivital_v0": [1.0, 2.0, 3.0],
                }
            ),
            ["ecg"],
        )
        return dataset, raw_sample

    def test_writes_six_npz_files_and_metadata(self, tmp_path):
        dataset, raw_sample = self._dataset_and_raw()
        fold_dir = tmp_path / "fold_0"
        metadata = {
            "model_name": "TEST",
            "fold_id": 0,
            "feature_names": dataset["feature_names"],
            "raw_feature_names": dataset["raw_feature_names"],
            "raw_column_names": raw_sample["raw_column_names"],
            "num_splits": 3,
            "random_seed": 7,
            "model_type_string": "Complete_Explicit",
            "feature_streams": ["ECG", "HR"],
        }
        _save_fold_files(dataset, raw_sample, fold_dir, metadata)

        expected_files = {
            "standardized_data.npz": ("train_only", "all_rows"),
            "delta_data.npz": ("delta_s1_mean", "delta_s1_std", "delta_s2_mean", "delta_s2_std"),
            "trial_data.npz": ("trial_id", "subject", "trial"),
            "time_data.npz": ("time",),
            "label_data.npz": ("y_gloc", "train_mask", "imputed"),
            "raw_per_sample_data.npz": ("time", "trial_id", "subject", "trial", "sensor"),
        }
        for fname, keys in expected_files.items():
            path = fold_dir / fname
            assert path.exists(), f"missing {fname}"
            loaded = np.load(path)
            assert set(loaded.files) == set(keys)

        loaded = np.load(fold_dir / "standardized_data.npz")
        np.testing.assert_allclose(loaded["train_only"], dataset["train_only"])
        np.testing.assert_allclose(loaded["all_rows"], dataset["all_rows"])
        loaded_d = np.load(fold_dir / "delta_data.npz")
        np.testing.assert_allclose(loaded_d["delta_s1_mean"], dataset["delta_s1_mean"])
        loaded_t = np.load(fold_dir / "trial_data.npz")
        np.testing.assert_array_equal(loaded_t["subject"], dataset["subject"])
        loaded_r = np.load(fold_dir / "raw_per_sample_data.npz")
        np.testing.assert_allclose(loaded_r["sensor"], raw_sample["sensor"])

        saved_meta = json.loads((fold_dir / "fold_metadata.json").read_text())
        assert saved_meta["model_name"] == "TEST"
        assert saved_meta["fold_id"] == 0
        assert saved_meta["feature_names"] == dataset["feature_names"]
        assert saved_meta["raw_feature_names"] == dataset["raw_feature_names"]
        assert saved_meta["raw_column_names"] == raw_sample["raw_column_names"]
