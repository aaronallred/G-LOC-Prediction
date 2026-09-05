"""Unit and integration tests for verify_imputation_pipeline module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.real_time.verify_imputation_pipeline import (
    filter_target_stream_features,
    classify_trial_phases,
    profile_imputed_phases,
    verify_imputation_integrity,
    analyze_dropped_rows_and_randomness,
    explain_dropped_rows_distribution,
    plot_single_trial_imputation,
    run_imputation_verification,
)


def test_filter_target_stream_features():
    """Verify that filter_target_stream_features keeps ECG, HR, BR, Temp, Centrifuge and excludes EEG, Tobii, etc."""
    all_features = [
        "HR (bpm) - Equivital",
        "ECG Lead 1 - Equivital",
        "ECG Lead 2 - Equivital",
        "HR_instant - Equivital",
        "BR (rpm) - Equivital",
        "Skin Temperature - IR Thermometer (°C) - Equivital",
        "magnitude - Centrifuge",
        "Fz - EEG",
        "Pupil diameter left [mm] - Tobii",
        "deviation - Cog",
        "participant_HR_seated",
        "participant_HR_stand",
        "participant_weight",
    ]

    filtered = filter_target_stream_features(all_features)

    expected = [
        "HR (bpm) - Equivital",
        "ECG Lead 1 - Equivital",
        "ECG Lead 2 - Equivital",
        "HR_instant - Equivital",
        "BR (rpm) - Equivital",
        "Skin Temperature - IR Thermometer (°C) - Equivital",
        "magnitude - Centrifuge",
    ]

    assert filtered == expected
    assert "Fz - EEG" not in filtered
    assert "Pupil diameter left [mm] - Tobii" not in filtered
    assert "deviation - Cog" not in filtered
    assert "participant_HR_seated" not in filtered


def test_verify_imputation_integrity_passed_target_streams():
    """Verify that verify_imputation_integrity passes on target features when only NaNs are imputed."""
    before = np.array([
        [1.0, np.nan, 3.0, 10.0],
        [4.0, 5.0, np.nan, 20.0],
        [7.0, 8.0, 9.0, 30.0],
    ], dtype=np.float32)

    after = np.array([
        [1.0, 2.5, 3.0, 99.0],  # Note: col 3 (non-target) changed, but target columns (0, 1, 2) are untouched
        [4.0, 5.0, 6.5, 20.0],
        [7.0, 8.0, 9.0, 30.0],
    ], dtype=np.float32)

    feature_names = ["feat1", "feat2", "feat3", "excluded_feat"]
    target_features = ["feat1", "feat2", "feat3"]

    results = verify_imputation_integrity(before, after, feature_names, target_features=target_features)

    assert results["verification_passed"] is True
    assert results["total_imputed_nans"] == 2
    assert results["remaining_nans_after"] == 0
    assert results["modified_non_nans"] == 0
    assert results["max_non_nan_difference"] == 0.0
    assert results["target_features_count"] == 3


def test_verify_imputation_integrity_failed_on_modified_value():
    """Verify that verify_imputation_integrity detects modifications to valid entries in target streams."""
    before = np.array([
        [1.0, np.nan, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float32)

    # after matrix where [0, 0] changed from 1.0 to 1.5
    after = np.array([
        [1.5, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float32)

    feature_names = ["feat1", "feat2", "feat3"]
    results = verify_imputation_integrity(before, after, feature_names)

    assert results["verification_passed"] is False
    assert results["modified_non_nans"] == 1
    assert results["max_non_nan_difference"] == pytest.approx(0.5)


def test_classify_trial_phases():
    """Verify that trial phases are correctly categorized into Baseline, Ramp, Plateau, G-LOC, and Recovery."""
    time_pts = 100
    times = np.linspace(0, 50, time_pts)
    gz = np.ones(time_pts)
    gz[20:40] = np.linspace(1.0, 6.0, 20)  # ramp
    gz[40:70] = 6.0  # plateau
    gz[70:85] = np.linspace(6.0, 1.0, 15)  # deceleration / recovery
    gz[85:] = 1.0  # post-run baseline / recovery

    event = np.zeros(time_pts)
    event[55:65] = 1  # G-LOC event during plateau

    trial_df = pd.DataFrame({
        "Time (s)": times,
        "magnitude - Centrifuge": gz,
        "event": event,
    })

    phases = classify_trial_phases(trial_df)

    assert phases.iloc[5] == "Baseline"
    assert phases.iloc[30] == "Ramp / Onset"
    assert phases.iloc[45] == "Plateau / Peak Gz"
    assert phases.iloc[60] == "G-LOC"
    assert phases.iloc[80] == "Deceleration / Recovery"


def test_analyze_dropped_rows_and_randomness_clustered():
    """Verify that Wald-Wolfowitz runs test identifies contiguous block dropouts as non-random."""
    n_rows = 10000
    df = pd.DataFrame({
        "trial_id": ["01-01"] * n_rows,
        "Time (s)": np.linspace(0, 400, n_rows),
        "magnitude - Centrifuge": np.ones(n_rows) * 1.2,
        "event": np.zeros(n_rows),
    })

    # Continuous block of 300 missing values at the end (burst)
    before = np.ones((n_rows, 2), dtype=np.float32)
    before[9700:, 0] = np.nan

    feature_names = ["HR (bpm) - Equivital", "magnitude - Centrifuge"]
    target_features = ["HR (bpm) - Equivital"]

    analysis, dropped_mask = analyze_dropped_rows_and_randomness(
        gloc_data=df,
        before_matrix=before,
        feature_names=feature_names,
        target_features=target_features,
    )

    assert analysis["dropped_rows_count"] == 300
    assert analysis["randomness_test"]["is_random"] is False
    assert analysis["randomness_test"]["z_score"] < -10.0
    assert analysis["randomness_test"]["p_value"] < 1e-10
    assert analysis["contiguous_bursts"]["burst_count"] == 1
    assert analysis["contiguous_bursts"]["max_burst_seconds"] == pytest.approx(300 / 25.0)

    # Test explanation generator
    explanation = explain_dropped_rows_distribution(analysis, df, dropped_mask, target_features)
    assert "Wald-Wolfowitz" in explanation
    assert "Deceleration / Recovery" in explanation or "Post-Run" in explanation or "burst" in explanation


def test_analyze_dropped_rows_and_randomness_random():
    """Verify that Wald-Wolfowitz runs test recognizes uniformly random Bernoulli dropouts."""
    np.random.seed(42)
    n_rows = 10000
    df = pd.DataFrame({
        "trial_id": ["01-01"] * n_rows,
        "Time (s)": np.linspace(0, 400, n_rows),
        "magnitude - Centrifuge": np.ones(n_rows) * 1.2,
        "event": np.zeros(n_rows),
    })

    # Scattered random missing values (100 scattered points)
    before = np.ones((n_rows, 2), dtype=np.float32)
    random_indices = np.random.choice(n_rows, size=100, replace=False)
    before[random_indices, 0] = np.nan

    feature_names = ["HR (bpm) - Equivital", "magnitude - Centrifuge"]
    target_features = ["HR (bpm) - Equivital"]

    analysis, _ = analyze_dropped_rows_and_randomness(
        gloc_data=df,
        before_matrix=before,
        feature_names=feature_names,
        target_features=target_features,
    )

    assert analysis["dropped_rows_count"] == 100
    assert analysis["randomness_test"]["is_random"] is True
    assert abs(analysis["randomness_test"]["z_score"]) < 1.96
    assert analysis["randomness_test"]["p_value"] >= 0.05


def test_plot_single_trial_imputation(tmp_path: Path):
    """Verify that plot_single_trial_imputation generates and saves a PNG file."""
    n_rows = 50
    df = pd.DataFrame({
        "trial_id": ["trial_01"] * n_rows,
        "Time (s)": np.linspace(0, 50, n_rows),
        "magnitude - Centrifuge": np.ones(n_rows) * 1.5,
        "event": np.zeros(n_rows),
    })

    before = np.ones((n_rows, 2), dtype=np.float32)
    before[10:15, 0] = np.nan
    after = np.ones((n_rows, 2), dtype=np.float32) * 2.0

    feature_names = ["HR (bpm) - Equivital", "magnitude - Centrifuge"]
    plot_file = plot_single_trial_imputation(
        gloc_data=df,
        before_matrix=before,
        after_matrix=after,
        feature_names=feature_names,
        trial_id="trial_01",
        features_to_plot=feature_names,
        output_dir=tmp_path,
    )

    assert plot_file != ""
    assert Path(plot_file).exists()
    assert Path(plot_file).stat().st_size > 0


@pytest.mark.integration
def test_run_imputation_verification_integration(tmp_path: Path):
    """Integration test verifying full execution of run_imputation_verification on data_reduced."""
    results = run_imputation_verification(
        config_path="configs/test.yaml",
        data_path="data_reduced",
        output_dir=str(tmp_path / "results"),
        random_seed=42,
    )

    assert results["verification"]["verification_passed"] is True
    assert results["verification"]["remaining_nans_after"] == 0
    assert results["verification"]["modified_non_nans"] == 0
    assert len(results["target_stream_features"]) > 0
    assert "dropped_rows_analysis" in results
    assert "randomness_test" in results["dropped_rows_analysis"]
    assert "non_randomness_explanation" in results
    assert Path(results["plot_path"]).exists()
    assert (tmp_path / "results" / "imputation_verification_report.json").exists()
    assert (tmp_path / "results" / "imputation_verification_summary.md").exists()
