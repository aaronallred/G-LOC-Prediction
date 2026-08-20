"""Tests for the top-deltas analysis script.

Builds a small synthetic output tree (2 models × 2 folds) of the three required
files (delta_data.npz, trial_data.npz, fold_metadata.json) and verifies the
loading, ranking, and missing-file handling on synthetic data without
exercising the pipeline.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.real_time.standardization_metrics_analysis.top_deltas_analysis import (
    _collect_entries,
    _fold_entries,
    _load_fold,
    run_top_deltas,
)


def _write_fold(fold_dir, model, fold_id, feature_names, delta_s1, delta_s1_std,
                delta_s2, delta_s2_std, subject, trial, seed=0):
    fold_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    delta_s1 = np.asarray(delta_s1, dtype=np.float64) if delta_s1 is not None else rng.normal(size=(4, len(feature_names)))
    delta_s1_std = np.asarray(delta_s1_std, dtype=np.float64) if delta_s1_std is not None else rng.normal(size=delta_s1.shape)
    delta_s2 = np.asarray(delta_s2, dtype=np.float64) if delta_s2 is not None else rng.normal(size=len(feature_names))
    delta_s2_std = np.asarray(delta_s2_std, dtype=np.float64) if delta_s2_std is not None else rng.normal(size=len(feature_names))
    np.savez_compressed(
        fold_dir / "delta_data.npz",
        delta_s1_mean=delta_s1, delta_s1_std=delta_s1_std,
        delta_s2_mean=delta_s2, delta_s2_std=delta_s2_std,
    )
    np.savez_compressed(
        fold_dir / "trial_data.npz",
        subject=np.array(subject, dtype=str), trial=np.array(trial, dtype=str),
    )
    with open(fold_dir / "fold_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"model_name": model, "fold_id": fold_id, "raw_feature_names": feature_names}, f)


def _mini_tree(tmp_path):
    """Two models × two folds with planted worst rows."""
    features = ["HR_mean", "ECG_mean", "BR_mean"]
    # Model A: fold 0 row 1 / feature 2 is the global worst (delta 9.9).
    _write_fold(
        tmp_path / "Complete" / "RF" / "fold_0", "RF", 0, features,
        delta_s1=[[0.1, 0.2, 0.3], [0.4, 0.5, 9.9], [0.2, 0.1, 0.0], [0.0, 0.0, 0.0]],
        delta_s1_std=[[0.1] * 3, [0.1, 0.1, 0.5], [0.1] * 3, [0.1] * 3],
        delta_s2=[0.2, 0.3, 8.8], delta_s2_std=[0.1, 0.1, 0.4],
        subject=["S1", "S2", "S1", "S3"], trial=["T1", "T2", "T1", "T4"],
    )
    # Model A: fold 1 feature 0 worst (delta 7.5) at row 2.
    _write_fold(
        tmp_path / "Complete" / "RF" / "fold_1", "RF", 1, features,
        delta_s1=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [7.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        delta_s1_std=[[0.1] * 3, [0.1] * 3, [0.3, 0.1, 0.1], [0.1] * 3],
        delta_s2=[0.1, 0.1, 0.1], delta_s2_std=[0.1, 0.1, 0.1],
        subject=["S1", "S1", "S2", "S2"], trial=["T1", "T1", "T3", "T3"],
    )
    # Model B: fold 0 feature 1 worst (delta 6.0).
    _write_fold(
        tmp_path / "Complete" / "SVM" / "fold_0", "SVM", 0, features,
        delta_s1=[[0.0, 6.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        delta_s1_std=[[0.1, 0.2, 0.1], [0.1] * 3, [0.1] * 3, [0.1] * 3],
        delta_s2=[0.1, 0.1, 0.1], delta_s2_std=[0.1, 0.1, 0.1],
        subject=["S1", "S1", "S1", "S1"], trial=["T1", "T1", "T1", "T1"],
    )
    return tmp_path


class TestLoadFold:
    def test_happy_path(self, tmp_path):
        features = ["HR_mean"]
        _write_fold(tmp_path / "fold_0", "RF", 0, features,
                    delta_s1=[[1.0], [2.0]], delta_s1_std=[[0.1], [0.2]],
                    delta_s2=[0.5], delta_s2_std=[0.1],
                    subject=["S1", "S2"], trial=["T1", "T2"])
        data = _load_fold(tmp_path / "fold_0")
        assert data is not None
        assert data["model_name"] == "RF"
        assert data["fold_id"] == 0
        assert data["feature_names"] == features
        assert data["delta_s1_mean"].shape == (2, 1)

    def test_missing_file_returns_none(self, tmp_path):
        assert _load_fold(tmp_path / "fold_0") is None  # nothing written

    def test_feature_column_mismatch_returns_none(self, tmp_path, caplog):
        _write_fold(tmp_path / "fold_0", "RF", 0, ["a", "b"],
                    delta_s1=[[1.0, 2.0, 3.0]], delta_s1_std=[[0.1, 0.1, 0.1]],
                    delta_s2=[0.5, 0.5], delta_s2_std=[0.1, 0.1],
                    subject=["S1"], trial=["T1"])
        assert _load_fold(tmp_path / "fold_0") is None
        assert "raw_feature_names" in caplog.text


class TestFoldEntries:
    def test_pools_s1_and_s2(self):
        data = {
            "model_name": "RF", "fold_id": 3, "feature_names": ["HR_mean", "ECG_mean"],
            "delta_s1_mean": np.array([[1.0, 2.0], [3.0, 0.0]]),
            "delta_s1_std": np.array([[0.1, 0.2], [0.3, 0.1]]),
            "delta_s2_mean": np.array([0.5, -4.0]),
            "delta_s2_std": np.array([0.1, 0.4]),
            "subject": np.array(["S1", "S2"]), "trial": np.array(["T1", "T2"]),
        }
        entries = _fold_entries(data)
        assert len(entries) == 4
        s1_hr, s2_hr, s1_ecg, s2_ecg = entries
        assert s1_hr["abs_delta"] == 3.0 and s1_hr["subject"] == "S2" and s1_hr["trial"] == "T2"
        assert s1_hr["delta_std"] == 0.3 and s1_hr["s1_or_s2"] == "s1"
        assert s1_ecg["abs_delta"] == 2.0 and s1_ecg["subject"] == "S1"
        assert s2_ecg["abs_delta"] == 4.0 and s2_ecg["subject"] == "--" and s2_ecg["s1_or_s2"] == "s2"


class TestRunTopDeltas:
    def test_ranks_per_model_with_subject_trial(self, tmp_path, capsys):
        root = _mini_tree(tmp_path)
        results = run_top_deltas(root, top_n=10)
        assert set(results) == {"RF", "SVM"}
        rf_top = results["RF"][0]
        assert rf_top["feature"] == "BR_mean"
        assert rf_top["abs_delta"] == pytest.approx(9.9)
        assert rf_top["subject"] == "S2" and rf_top["trial"] == "T2"
        # s2 entries pooled into the same ranking.
        assert any(e["s1_or_s2"] == "s2" for e in results["RF"])
        svm_top = results["SVM"][0]
        assert svm_top["feature"] == "ECG_mean"
        assert svm_top["abs_delta"] == pytest.approx(6.0)
        out = capsys.readouterr().out
        assert "RF" in out and "BR_mean" in out

    def test_top_n_limits_entries(self, tmp_path):
        root = _mini_tree(tmp_path)
        results = run_top_deltas(root, top_n=2)
        assert len(results["RF"]) == 2

    def test_writes_output_json(self, tmp_path):
        root = _mini_tree(tmp_path)
        out = tmp_path / "out" / "top.json"
        run_top_deltas(root, top_n=10, output_path=out)
        with open(out, encoding="utf-8") as f:
            loaded = json.load(f)
        assert set(loaded) == {"RF", "SVM"}
        assert loaded["RF"][0]["abs_delta"] == pytest.approx(9.9)

    def test_skips_incomplete_folds(self, tmp_path, caplog):
        root = _mini_tree(tmp_path)
        (root / "Complete" / "RF" / "fold_1" / "delta_data.npz").unlink()
        results = run_top_deltas(root, top_n=10)
        assert "fold_1" in caplog.text and "Skipping" in caplog.text
        # Fold 1 entries gone but fold 0 still present.
        assert any(e["abs_delta"] == pytest.approx(9.9) for e in results["RF"])