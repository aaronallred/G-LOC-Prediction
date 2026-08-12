"""Smoke tests for the standardization-metrics analysis producer.

Validates the three per-row context flags (beginning-of-trial, positive-GLOC,
imputed input), the per-row delta attribution, and the survivor-mask remap
on synthetic data without exercising the data pipeline.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from src.real_time.standardization_metrics_analysis.traditional_standardization_metrics import (
    TOP_N_FEATURES,
    _analyse_fold,
    _beginning_of_trial_mask,
    _is_target_feature,
    _per_feature_statistics,
    _plot_top_feature_overlay,
    _reduce_impute_mask_per_window,
    _remap_through_survivor,
)

# The synthesized "Equivital/Centrifuge" feature names below contain "ecg",
# "hr", and "centrifuge" substrings (the Equivital device's HR/BR/Temp/ECG
# columns carry the device name in the column suffix, which the union-substring
# matcher exploits via the per-stream keywords matching on "hr", "ecg", etc.).
# These lowercased keywords reproduce what the orchestrator would obtain from
# ``pipeline._resolve_feature_groups_for_streams([ECG, HR, BR, Temperature,
# Centrifuge], ...)`` so the test exercises the same code path as production.
TEST_FILTER_SUBSTRINGS = ["ecg", "hr", "br", "temp", "centrifuge"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _synthetic_setup(n_trials=2, n_windows_per_trial=10, n_raw_features=5, seed=7,
                     rows_to_remove=(2, 12)):
    rng = np.random.default_rng(seed)
    # Sliding-window math: with (stride=1, window_size=1, offset=0) a trial
    # spanning time t=0..T yields T windows. To get exactly
    # ``n_windows_per_trial`` windows per trial, use (n_windows_per_trial + 1)
    # samples per trial so max(time) = n_windows_per_trial.
    n_samples_per_trial = n_windows_per_trial + 1
    n_rows = n_trials * n_windows_per_trial

    # Five raw feature columns of interest, derived from the G-LOC pipeline's
    # actual FEATURE_REGISTRY naming conventions:
    #   - HR (bpm) - Equivital_*      : HR sub-stream (matches "hr")
    #   - magnitude - Centrifuge_*    : Centrifuge stream (matches "centrifuge")
    #   - BR (rpm) - Equivital_*      : BR stream (matches "br")
    #   - Skin Temperature - IR Thermometer (°C) - Equivital_*  : Temperature (matches "temp")
    #   - HRV (SDNN)_*                : ECG-derived HRV column NOT carrying the
    #                                    "- Equivital" device suffix; relies on
    #                                    the union-substring "hr" matcher to be
    #                                    retained by the new filter (regression
    #                                    for the previously-dropped HRV columns).
    raw_feature_names = [
        "HR (bpm) - Equivital_v0_mean_s1",
        "magnitude - Centrifuge_v0_mean_s1",
        "BR (rpm) - Equivital_v0_mean_s1",
        "Skin Temperature - IR Thermometer (°C) - Equivital_v0_mean_s1",
        "HRV (SDNN)_v0_mean_s1",
    ]
    # If a caller asks for more raw features than this canonical set, append
    # plain filler columns (which the stream-aware filter drops).
    if n_raw_features > len(raw_feature_names):
        raw_feature_names += [
            f"filler_{i}_mean_s1" for i in range(n_raw_features - len(raw_feature_names))
        ]
    else:
        raw_feature_names = raw_feature_names[:n_raw_features]
    # Match the column count of _standardize_* output (doubled).
    all_features = raw_feature_names + [n.replace("_s1", "_s2") for n in raw_feature_names]

    # Per-sample arrays used to drive the impute reduction.
    trial_id_per_sample = np.array(
        [f"t{i}" for i in range(n_trials) for _ in range(n_samples_per_trial)]
    )
    time_per_sample = np.concatenate(
        [np.arange(0.0, n_samples_per_trial, dtype=np.float64) for _ in range(n_trials)]
    )
    n_raw_cols = len(raw_feature_names)
    pre_impute_mask = np.zeros((len(time_per_sample), n_raw_cols), dtype=bool)
    # Mark some samples imputed in trial 0, window 3 of feature 0.
    # Sample index 3 in trial 0 corresponds to time t=3, which falls into
    # window j=3 (since stride=1, window_size=1: window j spans [j, j+1)).
    pre_impute_mask[3, 0] = True
    # Impute flag expected to land in trial 0's third window.

    # Hyperparameters matching _sliding_window_mean_calc.
    time_start, offset, stride, window_size = 0.0, 0.0, 1.0, 1.0
    trial_id_per_row = np.array(
        [f"t{i}" for i in range(n_trials) for _ in range(n_windows_per_trial)]
    )

    # Standardized test matrix placeholders (linear in time)
    std_train_only = rng.standard_normal((n_rows, len(all_features)))
    std_all_rows = rng.standard_normal((n_rows, len(all_features)))
    # Force large deltas on rows 1 and 11 at target column 0.
    std_all_rows[1, 0] = std_train_only[1, 0] + 5.0
    std_all_rows[11, 0] = std_train_only[11, 0] + 4.5

    train_mask = np.ones(n_rows, dtype=bool)
    train_mask[np.array(rows_to_remove, dtype=int)] = False

    y_gloc_labels = np.zeros(n_rows, dtype=np.float32)
    y_gloc_labels[3] = 1.0
    y_gloc_labels[13] = 1.0

    return dict(
        n_rows=n_rows,
        all_features=all_features,
        trial_id_per_sample=trial_id_per_sample,
        time_per_sample=time_per_sample,
        pre_impute_mask=pre_impute_mask,
        time_start=time_start, offset=offset, stride=stride, window_size=window_size,
        trial_id_per_row=trial_id_per_row,
        std_train_only=std_train_only,
        std_all_rows=std_all_rows,
        train_mask=train_mask,
        y_gloc_labels=y_gloc_labels,
    )


# ---------------------------------------------------------------------------
# Pure-function tests (no pipeline)
# ---------------------------------------------------------------------------
class TestPerWindowImputeReduction:
    def test_mirrors_sliding_window_order_and_window_count(self):
        s = _synthetic_setup(n_trials=2, n_windows_per_trial=10, n_raw_features=5)
        out = _reduce_impute_mask_per_window(
            impute_mask_per_sample=s["pre_impute_mask"],
            trial_id_per_sample=s["trial_id_per_sample"],
            time_per_sample=s["time_per_sample"],
            time_start=s["time_start"], offset=s["offset"], stride=s["stride"],
            window_size=s["window_size"],
            feature_names=[f"raw_{i}" for i in range(5)],
            trial_id_per_row=s["trial_id_per_row"],
        )
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


class TestBeginningOfTrialMask:
    def test_flags_first_row_of_each_block(self):
        trial_id = np.array(["a", "a", "a", "b", "b", "c"])
        flag = _beginning_of_trial_mask(trial_id)
        assert flag.tolist() == [True, False, False, True, False, True]

    def test_empty_input(self):
        assert _beginning_of_trial_mask(np.array([])).tolist() == []


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


class TestPerFeatureStatistics:
    def test_returns_correct_shape_and_keys(self):
        s = _synthetic_setup(n_raw_features=5)
        # Indices 0, 1 of the all_features list (= HR-Centrifuge s1 columns).
        target_idx = [0, 1]
        target_features = [s["all_features"][i] for i in target_idx]
        per_feat, stratified = _per_feature_statistics(
            s["std_train_only"][:5, :][..., target_idx],
            s["std_all_rows"][:5, :][..., target_idx],
            target_features,
        )
        assert set(per_feat.keys()) == set(target_features)
        # Per-feature stats carry the base 7 keys.
        for stats in per_feat.values():
            for k in ("mean_abs_delta", "max_abs_delta", "mean_delta", "std_delta",
                      "s1_or_s2", "feature_type", "baseline_method"):
                assert k in stats


# ---------------------------------------------------------------------------
# Stream-aware target filter (regression for previously-dropped HRV columns)
# ---------------------------------------------------------------------------
class TestStreamAwareTargetFilter:
    """Validate that `_is_target_feature` retains the canonical ECG/HR/BR/
    Temperature/Centrifuge columns and the ECG-derived ``HRV (SDNN)`` /
    ``HRV (RMSSD)`` columns that lack the ``- Equivital`` device suffix."""

    ALL_FEATURES = [
        # HR sub-stream + ECG + BR + Temperature + Centrifuge (Equivital-suffixed)
        "HR (bpm) - Equivital_v0_mean_s1",
        "ECG Lead 1 - Equivital_v0_mean_s1",
        "BR (rpm) - Equivital_v0_mean_s1",
        "Skin Temperature - IR Thermometer (°C) - Equivital_v0_mean_s1",
        "magnitude - Centrifuge_v0_mean_s1",
        # ECG-derived HRV columns, NOT Equivital-suffixed (the regression case).
        "HRV (SDNN)_v0_mean_s1",
        "HRV (RMSSD)_v0_mean_s1",
        # Non-requested sensor columns that should be DROPPED under the filter.
        "Fz_alpha - EEG_v0_mean_s1",
        "Pupil diameter left [mm] - Tobii_v0_mean_s1",
        # Auto-appended AFE-indicator column (stream-independent, excluded).
        "AFE_indicator_windowed",
    ]

    def test_returns_true_for_all_five_requested_streams(self):
        substrings = ["ecg", "hr", "br", "temp", "centrifuge"]
        kept = [n for n in self.ALL_FEATURES if _is_target_feature(n, substrings)]
        assert "HR (bpm) - Equivital_v0_mean_s1" in kept
        assert "ECG Lead 1 - Equivital_v0_mean_s1" in kept
        assert "BR (rpm) - Equivital_v0_mean_s1" in kept
        assert "Skin Temperature - IR Thermometer (°C) - Equivital_v0_mean_s1" in kept
        assert "magnitude - Centrifuge_v0_mean_s1" in kept

    def test_hrv_derived_columns_retained_under_hr_substring(self):
        """The previously-dropped regression: HRV (SDNN) and HRV (RMSSD) have
        no ``- Equivital`` device suffix, but the union-substring matcher
        catches them because their lowercased names contain ``"hr"``.
        """
        substrings = ["ecg", "hr", "br", "temp", "centrifuge"]
        assert _is_target_feature("HRV (SDNN)_v0_mean_s1", substrings) is True
        assert _is_target_feature("HRV (RMSSD)_v0_mean_s1", substrings) is True

    def test_drops_non_requested_streams_and_afe_indicator(self):
        substrings = ["ecg", "hr", "br", "temp", "centrifuge"]
        assert _is_target_feature("Fz_alpha - EEG_v0_mean_s1", substrings) is False
        assert _is_target_feature("Pupil diameter left [mm] - Tobii_v0_mean_s1", substrings) is False
        # AFE_indicator_windowed is excluded by name even if substrings match.
        assert _is_target_feature("AFE_indicator_windowed", substrings) is False

    def test_none_filter_falls_back_to_include_all_excluding_afe(self):
        """Back-compat: when no streams are requested, every column except the
        AFE-indicator column is in-scope."""
        kept = [n for n in self.ALL_FEATURES if _is_target_feature(n, None)]
        assert "Fz_alpha - EEG_v0_mean_s1" in kept  # non-requested stream included under None
        assert "AFE_indicator_windowed" not in kept


# ---------------------------------------------------------------------------
# _analyse_fold integration against synthetic captures
# ---------------------------------------------------------------------------
class TestAnalyseFoldContextFlags:
    def test_top_features_worst_row_context_populated(self, tmp_path):
        s = _synthetic_setup(n_raw_features=5)
        # 5 raw feature columns × {s1, s2} = 10 target columns; under the
        # new stream-aware filter all 10 are retained (the 5 raw feature
        # names match one of the requested stream keywords). Top-N selection
        # ranks by the per-row maximum abs delta across target columns.
        X_raw = s["std_train_only"].copy()
        impute_mask_per_window = _reduce_impute_mask_per_window(
            impute_mask_per_sample=s["pre_impute_mask"],
            trial_id_per_sample=s["trial_id_per_sample"],
            time_per_sample=s["time_per_sample"],
            time_start=s["time_start"], offset=s["offset"], stride=s["stride"],
            window_size=s["window_size"],
            feature_names=[f"raw_{i}" for i in range(5)],
            trial_id_per_row=s["trial_id_per_row"],
        )
        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir(parents=True)
        report = _analyse_fold(
            X_raw=X_raw,
            trial_id=s["trial_id_per_row"],
            train_mask=s["train_mask"],
            all_features=s["all_features"],
            y_gloc_labels=s["y_gloc_labels"],
            impute_mask_per_window=impute_mask_per_window,
            filter_substrings=TEST_FILTER_SUBSTRINGS,
            fold_dir=fold_dir,
            model_name="TEST",
            fold_id=0,
            generate_plots=True,
        )

        # The orchestrator sets fold_id after _analyse_fold returns; mirror
        # that behavior here to verify it travels through to JSON.
        report["fold_id"] = 0
        assert report["fold_id"] == 0
        assert report["n_train_rows"] == int(s["train_mask"].sum())
        # Under the new stream-aware filter with 5 raw features × {s1, s2}
        # = 10 target columns (HR, Centrifuge, BR, Temperature, HRV all
        # retained), the top features list is bounded by TOP_N_FEATURES.
        assert 1 <= len(report["top_features"]) <= TOP_N_FEATURES
        # Each top-feature entry carries the worst-row context fields.
        for entry in report["top_features"]:
            assert set(["rank", "feature", "max_abs_delta", "mean_abs_delta",
                        "worst_row_context"]).issubset(entry.keys())
            wrc = entry["worst_row_context"]
            for k in ("row_local_idx", "is_beginning_of_trial", "positive_gloc",
                      "had_imputed_input", "abs_delta_value",
                      "std_train_only_value", "std_all_rows_value"):
                assert k in wrc
            # Boolean flags are bools (not numpy types) so they JSON-serialize.
            assert isinstance(wrc["is_beginning_of_trial"], bool)
            assert isinstance(wrc["positive_gloc"], bool)
            assert isinstance(wrc["had_imputed_input"], bool)

        # Aggregated flag counts returned for the per-fold text report.
        counts = report["top_worst_row_flag_counts"]
        assert set(counts.keys()) == {
            "beginning_of_trial", "positive_gloc", "had_imputed_input"
        }
        for v in counts.values():
            assert 0 <= v <= len(report["top_features"])

        # Plots were generated for each top-feature.
        plots_dir = fold_dir / "plots"
        plot_files = sorted(plots_dir.glob("*.png"))
        assert len(plot_files) == len(report["top_features"])
        # Plot path recorded in JSON entries.
        for entry in report["top_features"]:
            assert entry["plot_path"].endswith(".png")

        # JSON round-trip works (no numpy/None types that break json.dump).
        roundtripped = json.loads(json.dumps(report, default=str))
        assert roundtripped["fold_id"] == 0

    def test_no_plots_skipped(self, tmp_path):
        s = _synthetic_setup(n_raw_features=5)
        X_raw = s["std_train_only"].copy()
        fold_dir = tmp_path / "fold_1"
        fold_dir.mkdir()
        report = _analyse_fold(
            X_raw=X_raw,
            trial_id=s["trial_id_per_row"],
            train_mask=s["train_mask"],
            all_features=s["all_features"],
            y_gloc_labels=s["y_gloc_labels"],
            impute_mask_per_window=np.zeros(s["n_rows"], dtype=bool),
            filter_substrings=TEST_FILTER_SUBSTRINGS,
            fold_dir=fold_dir,
            model_name="TEST",
            fold_id=1,
            generate_plots=False,
        )
        assert not (fold_dir / "plots").exists()
        for entry in report.get("top_features", []):
            assert "plot_path" not in entry


# ---------------------------------------------------------------------------
# Plot function (headless)
# ---------------------------------------------------------------------------
class TestPlotTopFeatureOverlay:
    def test_writes_png_for_deterministic_inputs(self, tmp_path):
        rng = np.random.default_rng(99)
        n = 50
        train_only = np.cumsum(rng.standard_normal(n))
        all_rows = train_only + rng.standard_normal(n) * 0.5
        flag = np.zeros(n, dtype=bool)
        flag[0] = True
        positive = np.zeros(n, dtype=bool)
        positive[10] = True
        imputed = np.zeros(n, dtype=bool)
        imputed[5:8] = True
        out = tmp_path / "overlay.png"
        _plot_top_feature_overlay(
            feature_name="HR (bpm) - Equivital_v0_mean_s1",
            train_only_values=train_only,
            all_rows_values=all_rows,
            is_beginning=flag,
            positive_gloc=positive,
            had_imputed=imputed,
            out_path=out,
            title="t",
        )
        assert out.exists()
        assert out.stat().st_size > 200
