"""Tests for the modern feature-space review mode (src/modes/feature_space_review.py).

These tests target the *post-refactor* API surface.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.model_type import ModelType
from src.models.model_factory import ModelFactory
from src.modes.feature_space_review import run_feature_space_review


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, *, models: list[str]) -> dict:
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
        "feature_space_review": {
            "enabled": True,
            "models": models,
            "model_type": ModelType("Complete", "Explicit"),
            "median_hyperparameters_folder": str(tmp_path / "Results" / "CV"),
        },
    }


# ---------------------------------------------------------------------------
# Helper to create fake median-hyperparameter JSON files
# ---------------------------------------------------------------------------

def _write_median_hyperparameters(path: Path, model_type_str: str, model_name: str, *, features: list[str]) -> None:
    model_dir = path / model_type_str / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "median_hyperparameters.json").write_text(
        json.dumps({
            "best_params": {"alpha": 0.1},
            "selected_features": features,
            "fold_id": 0,
            "f1_score": 0.85,
        })
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_feature_space_review_reads_models_from_config(tmp_path):
    config = _make_config(tmp_path, models=["KNN", "RF"])
    _write_median_hyperparameters(
        Path(config["feature_space_review"]["median_hyperparameters_folder"]),
        "Complete_Explicit",
        "KNN",
        features=["f1", "f2", "f3"],
    )
    _write_median_hyperparameters(
        Path(config["feature_space_review"]["median_hyperparameters_folder"]),
        "Complete_Explicit",
        "RF",
        features=["f2", "f3", "f4"],
    )

    # Patch plt.show so we do not actually display a figure during the test
    run_feature_space_review(config, ModelFactory())


def test_feature_space_review_with_two_models(tmp_path):
    config = _make_config(tmp_path, models=["KNN", "RF"])
    _write_median_hyperparameters(
        Path(config["feature_space_review"]["median_hyperparameters_folder"]),
        "Complete_Explicit",
        "KNN",
        features=["f1", "f2", "f3"],
    )
    _write_median_hyperparameters(
        Path(config["feature_space_review"]["median_hyperparameters_folder"]),
        "Complete_Explicit",
        "RF",
        features=["f2", "f3", "f4"],
    )

    run_feature_space_review(config, ModelFactory())


def test_feature_space_review_with_three_models(tmp_path):
    config = _make_config(tmp_path, models=["KNN", "RF", "LDA"])
    for name, features in [
        ("KNN", ["f1", "f2"]),
        ("RF", ["f2", "f3"]),
        ("LDA", ["f2", "f4"]),
    ]:
        _write_median_hyperparameters(
            Path(config["feature_space_review"]["median_hyperparameters_folder"]),
            "Complete_Explicit",
            name,
            features=features,
        )

    run_feature_space_review(config, ModelFactory())


def test_feature_space_review_with_four_or_more_models(tmp_path):
    config = _make_config(tmp_path, models=["KNN", "RF", "LDA", "SVM"])
    for name, features in [
        ("KNN", ["f1"]),
        ("RF", ["f2"]),
        ("LDA", ["f3"]),
        ("SVM", ["f4"]),
    ]:
        _write_median_hyperparameters(
            Path(config["feature_space_review"]["median_hyperparameters_folder"]),
            "Complete_Explicit",
            name,
            features=features,
        )

    run_feature_space_review(config, ModelFactory())
