"""Tests for the modern sensor-ablation mode (src/modes/sensor_ablation.py).

These tests target the *post-refactor* API surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.model_type import ModelType
from src.modes.sensor_ablation import (
    run_sensor_ablation_training,
    run_sensor_ablation_review,
    _load_sensor_ablation_f1_results_for_model,
    _save_summary,
    _sort_streams_by_median_f1,
)


# ---------------------------------------------------------------------------
# Fixtures & stubs
# ---------------------------------------------------------------------------

class _TinyTraditionalModel:
    """Minimal traditional model that satisfies the sensor-ablation contract."""

    def __init__(self, name: str = "TinyKNN") -> None:
        self.name = name
        self.calls: list[dict] = []
        self.is_traditional_model = True

    def train(self, X, y):
        self.calls.append({"x_shape": X.shape, "y_shape": y.shape})

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def get_model_parameters(self):
        return {}

    def set_model_parameters(self, params):
        pass

    def save_model(self, path: str):
        import joblib
        joblib.dump(self, path)

    def classify_traditional(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


class _FakeModelFactory:
    """Factory that returns _TinyTraditionalModel instances."""

    def create_model(self, model_name: str, **kwargs):
        return _TinyTraditionalModel(model_name)


class _FakePipeline:
    """Pipeline whose get_data() returns fake data regardless of arguments."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def set_random_seed(self, random_seed: int) -> None:
        pass

    def set_model_type(self, model_type) -> None:
        pass

    def get_data(self, model=None, kfold_id=None, num_splits=None, feature_streams=None, 
                 traditional_feature_selection="cache", return_feature_names=False):
        self.calls.append({
            "model": getattr(model, "name", None), 
            "feature_streams": list(feature_streams or []),
            "kfold_id": kfold_id,
            "num_splits": num_splits
        })
        if kfold_id is not None:
            # Simulate advanced pipeline return value
            return np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]), np.array([0, 0]), np.array([1, 1]), ["feat_a", "feat_b"]
        else:
            # Simulate traditional pipeline return value
            if return_feature_names:
                return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]), np.array([0, 1, 0, 1]), ["feat_a", "feat_b"]
            else:
                return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]), np.array([0, 1, 0, 1])


def _make_config(tmp_path: Path) -> dict:
    return {
        "data_path": str(tmp_path / "data"),
        "shared_data_parameters": {
            "subject_to_analyze": None,
            "trial_to_analyze": None,
            "analysis_type": 2,
            "remove_NaN_trials": True,
            "impute_file_name": "imputed.pkl",
            "save_impute": False,
            "load_impute": False,
            "impute_phase": "pre_feature",
            "output_feature_dtype": "float32",
        },
        "advanced_data_parameters": {
            "n_neighbors": 4,
            "baseline_window": 32.5,
            "horizon": 0,
        },
        "traditional_data_parameters": {
            "backstep": 0,
            "data_rate": 25,
            "offset": 0,
            "time_start": 0,
        },
    "sensor_ablation": {
        "training": {
            "enabled": True,
            "save_results_folder": str(tmp_path / "Results" / "Sensor_Ablation"),
            "models": ["KNN"],
            "model_type": ModelType("Complete", "Explicit"),
            "random_seed": 13,
            "num_splits": 2,
            "median_hyperparameters_folder": str(tmp_path / "ModelSave" / "CV"),
            "manual_ablation": False,
            "class_weight": None,
            "streams": [
                ["ECG"],
                ["EEG", "Pupil"],
            ],
        },
            "review": {
                "enabled": True,
                "save_results_folder": str(tmp_path / "Results" / "Sensor_Ablation"),
                "models": ["KNN"],
                "model_type": ModelType("Complete", "Explicit"),
                "stream_groups": [
                    ["ECG"],
                    ["EEG", "Pupil"],
                ],
                "sort_streams_by_median": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

def test_sort_streams_by_median_f1():
    f1_results = {
        ("ECG",): np.array([0.5, 0.6, 0.7]),
        ("EEG", "Pupil"): np.array([0.8, 0.9, 0.7]),
    }
    result = _sort_streams_by_median_f1(f1_results)
    keys = list(result.keys())
    assert keys[0] == ("EEG", "Pupil")
    assert keys[1] == ("ECG",)


def test_save_summary(tmp_path):
    fold_results = [
        {"accuracy": 0.8, "f1": 0.7, "precision": 0.75, "recall": 0.65, "specificity": 0.9, "g_mean": 0.77},
        {"accuracy": 0.85, "f1": 0.75, "precision": 0.8, "recall": 0.7, "specificity": 0.92, "g_mean": 0.8},
    ]
    hyperparams = {"k": 3}
    output_path = _save_summary(tmp_path, hyperparams, fold_results)
    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
    assert data["model_hyperparameters"] == {"k": 3}
    assert data["performance"]["f1"] == [0.7, 0.75]
    assert data["performance"]["accuracy"] == [0.8, 0.85]


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------

def test_sensor_ablation_config_uses_model_type_objects(tmp_path):
    config = _make_config(tmp_path)
    assert config["sensor_ablation"]["training"]["model_type"] == ModelType(
        "Complete", "Explicit"
    )
    assert config["sensor_ablation"]["review"]["model_type"] == ModelType(
        "Complete", "Explicit"
    )


@pytest.mark.integration
def test_run_sensor_ablation_training_saves_stream_scores(tmp_path):
    config = _make_config(tmp_path)
    pipeline = _FakePipeline()

    # Set up median hyperparameters so the code can read them.
    median_dir = tmp_path / "ModelSave" / "CV" / "Complete_Explicit" / "KNN"
    median_dir.mkdir(parents=True, exist_ok=True)
    median_dir.joinpath("median_hyperparameters.json").write_text(
        json.dumps({
            "best_params": {"k": 3},
            "selected_features": ["feat_a", "feat_b"],
            "fold_id": 0,
            "f1_score": 0.8,
        })
    )

    run_sensor_ablation_training(config, pipeline, _FakeModelFactory(), tmp_path)

    results = tmp_path / "Results" / "Sensor_Ablation" / "Complete_Explicit" / "KNN"
    assert (results / "ECG" / "summary.json").exists()
    assert (results / "EEG-Pupil" / "summary.json").exists()
    assert (results / "ECG" / "fold_0.pkl").exists()
    assert (results / "ECG" / "fold_1.pkl").exists()

    config_path = tmp_path / "Results" / "Sensor_Ablation" / "Complete_Explicit" / "run_config.yaml"
    assert config_path.exists()
    with open(config_path) as f:
        saved_config = yaml.safe_load(f)
    assert saved_config["sensor_ablation"]["training"]["models"] == ["KNN"]


@pytest.mark.integration
def test_run_sensor_ablation_review_loads_and_plots(tmp_path):
    config = _make_config(tmp_path)
    # Ensure training is disabled and review is enabled
    config["sensor_ablation"]["training"]["enabled"] = False
    config["sensor_ablation"]["review"]["enabled"] = True

    # Pre-populate results so review has data to load
    results_dir = tmp_path / "Results" / "Sensor_Ablation" / "Complete_Explicit" / "KNN"
    ecg_dir = results_dir / "ECG"
    eeg_pupil_dir = results_dir / "EEG-Pupil"
    ecg_dir.mkdir(parents=True, exist_ok=True)
    eeg_pupil_dir.mkdir(parents=True, exist_ok=True)
    (ecg_dir / "summary.json").write_text(json.dumps(
        {"model_hyperparameters": {}, "performance": {"f1": [0.5, 0.6]}}
    ))
    (eeg_pupil_dir / "summary.json").write_text(json.dumps(
        {"model_hyperparameters": {}, "performance": {"f1": [0.8, 0.7]}}
    ))

    run_sensor_ablation_review(config)


def test_sensor_ablation_training_requires_median_hyperparameters_folder(tmp_path):
    config = _make_config(tmp_path)
    del config["sensor_ablation"]["training"]["median_hyperparameters_folder"]
    pipeline = _FakePipeline()

    with pytest.raises(KeyError):
        run_sensor_ablation_training(config, pipeline, _FakeModelFactory(), tmp_path)


@pytest.mark.integration
def test_run_sensor_ablation_training_uses_parser_median_folder(tmp_path):
    config = _make_config(tmp_path)
    pipeline = _FakePipeline()

    median_dir = tmp_path / "ModelSave" / "CV" / "Complete_Explicit" / "KNN"
    median_dir.mkdir(parents=True, exist_ok=True)
    median_dir.joinpath("median_hyperparameters.json").write_text(
        json.dumps({
            "best_params": {"n_neighbors": 3},
            "selected_features": ["feat_a", "feat_b"],
            "fold_id": 0,
            "f1_score": 0.8,
        })
    )

    run_sensor_ablation_training(config, pipeline, _FakeModelFactory(), tmp_path)

    results = tmp_path / "Results" / "Sensor_Ablation" / "Complete_Explicit" / "KNN"
    assert (results / "ECG" / "summary.json").exists()
    assert (results / "EEG-Pupil" / "summary.json").exists()
