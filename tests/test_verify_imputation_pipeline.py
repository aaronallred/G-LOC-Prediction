"""Unit and integration tests for verify_imputation_pipeline module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.real_time.verify_imputation_pipeline import (
    classify_trial_phases,
    profile_imputed_phases,
    verify_imputation_integrity,
    plot_single_trial_imputation,
    run_imputation_verification,
)


def test_verify_imputation_integrity_passed():
    """Verify that verify_imputation_integrity passes when only NaNs are imputed."""
    # Create before matrix with NaNs
    before = np.array([
        [1.0, np.nan, 3.0],
        [4.0, 5.0, np.nan],
        [7.0, 8.0, 9.0],
    ], dtype=np.float32)

    # Create after matrix where NaNs are replaced and valid values remain untouched
    after = np.array([
        [1.0, 2.5, 3.0],
        [4.0, 5.0, 6.5],
        [7.0, 8.0, 9.0],
    ], dtype=np.float32)

    feature_names = ["feat1", "feat2", "feat3"]
    results = verify_imputation_integrity(before, after, feature_names)

    assert results["verification_passed"] is True
    assert results["total_imputed_nans"] == 2
    assert results["remaining_nans_after"] == 0
    assert results["modified_non_nans"] == 0
    assert results["max_non_nan_difference"] == 0.0


def test_verify_imputation_integrity_failed_on_modified_value():
    """Verify that verify_imputation_integrity detects modifications to valid entries."""
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
    # Construct synthetic trial
    time_pts = 100
    times = np.linspace(0, 50, time_pts)
    # G profile: 1.0 (baseline) -> rising to 6.0 -> plateau at 6.0 -> dropping to 1.0
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

    # Check phase designations
    assert phases.iloc[5] == "Baseline"
    assert phases.iloc[30] == "Ramp / Onset"
    assert phases.iloc[45] == "Plateau / Peak Gz"
    assert phases.iloc[60] == "G-LOC"
    assert phases.iloc[80] == "Deceleration / Recovery"


def test_profile_imputed_phases():
    """Verify that profile_imputed_phases aggregates missingness across phases."""
    n_rows = 50
    df = pd.DataFrame({
        "trial_id": ["trial_01"] * n_rows,
        "Time (s)": np.linspace(0, 50, n_rows),
        "magnitude - Centrifuge": np.ones(n_rows),
        "event": np.zeros(n_rows),
    })

    before = np.ones((n_rows, 2), dtype=np.float32)
    before[10:15, 0] = np.nan  # 5 NaNs in feat0

    feature_names = ["feat0", "feat1"]
    summary = profile_imputed_phases(df, before, feature_names)

    assert summary["total_imputed_points"] == 5
    assert "trial_01" in summary["per_trial_summary"]
    assert summary["phase_counts"]["Baseline"] == 5


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
    assert Path(results["plot_path"]).exists()
    assert (tmp_path / "results" / "imputation_verification_report.json").exists()
    assert (tmp_path / "results" / "imputation_verification_summary.md").exists()
