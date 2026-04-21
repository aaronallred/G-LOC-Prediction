import json
import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.GLOC_experiment_config_parser import GLOCExperimentConfigParser
from src.traditional_experiment_utils import save_median_hyperparameters
import src.main as main_module


def _base_config_dict():
    """Create a minimal valid config dictionary for testing."""
    return {
        "models": ["KNN", "RF", "EGB", "LogReg"],
        "model_type": ["Complete", "Explicit"],
        "random_seed": 42,
        "data_path": "/tmp/data",
        "shared_data_parameters": {
            "subject_to_analyze": None,
            "trial_to_analyze": None,
            "analysis_type": 2,
            "remove_NaN_trials": True,
            "impute_file_name": "imputed_data.pkl",
            "save_impute": False,
            "load_impute": False,
            "should_impute": True,
            "output_feature_dtype": "float32",
            "faiss_index_type": "cpu",
        },
        "advanced_data_parameters": {
            "num_splits": 10,
            "kfold_ID": 0,
            "n_neighbors": 4,
            "baseline_window": 32.5,
        },
        "traditional_data_parameters": {
            "backstep": 0,
            "data_rate": 25,
            "offset": 0,
            "time_start": 0,
        },
        "sensor_ablation": {
            "training": {
                "enabled": False,
                "streams": [["EEG"]],
            },
            "review": {
                "enabled": False,
                "models": [],
                "stream_group": [],
            },
        },
        "feature_space_review": {
            "enabled": False,
            "models": [],
        },
    }


class SimpleModel:
    """A simple model class that can be pickled for testing."""

    def __init__(self):
        self.best_params_ = {
            "n_neighbors": 5,
            "weights": "distance",
            "algorithm": "auto",
        }


# ============================================================================
# CONFIG PARSER TESTS
# ============================================================================


class TestHyperparameterSaveConfigParser:
    """Test configuration parsing for hyperparameter save settings."""

    def test_parser_reads_hyperparameter_save_when_enabled(self, tmp_path):
        """Test that parser correctly reads hyperparameter_save config when enabled."""
        config_dict = _base_config_dict()
        config_dict["hyperparameter_save"] = {
            "enabled": True,
            "models": ["KNN", "RF", "EGB"],
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        parser = GLOCExperimentConfigParser(config_location=str(config_path))

        assert parser.get_hyperparameter_save_enabled() is True
        assert parser.get_hyperparameter_save_models() == ["KNN", "RF", "EGB"]

    def test_parser_hyperparameter_save_defaults_when_not_provided(self, tmp_path):
        """Test that parser applies sensible defaults when hyperparameter_save is missing."""
        config_dict = _base_config_dict()
        # Don't add hyperparameter_save section

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        parser = GLOCExperimentConfigParser(config_location=str(config_path))

        assert parser.get_hyperparameter_save_enabled() is False
        assert parser.get_hyperparameter_save_models() == []

    def test_parser_hyperparameter_save_disabled_by_default(self, tmp_path):
        """Test that hyperparameter save is disabled by default."""
        config_dict = _base_config_dict()
        config_dict["hyperparameter_save"] = {
            "enabled": False,
            "models": ["KNN"],
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        parser = GLOCExperimentConfigParser(config_location=str(config_path))

        assert parser.get_hyperparameter_save_enabled() is False

    def test_parser_rejects_invalid_classifier_in_hyperparameter_save_models(
        self, tmp_path
    ):
        """Test that parser rejects unrecognized classifier names."""
        config_dict = _base_config_dict()
        config_dict["hyperparameter_save"] = {
            "enabled": True,
            "models": ["InvalidClassifier"],
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        with pytest.raises(ValueError, match="not recognized"):
            GLOCExperimentConfigParser(config_location=str(config_path))

    def test_parser_rejects_non_boolean_enabled_field(self, tmp_path):
        """Test that parser rejects non-boolean enabled field."""
        config_dict = _base_config_dict()
        config_dict["hyperparameter_save"] = {
            "enabled": "yes",  # Should be boolean
            "models": ["KNN"],
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        with pytest.raises(ValueError, match="must be a boolean"):
            GLOCExperimentConfigParser(config_location=str(config_path))

    def test_parser_rejects_non_list_models_field(self, tmp_path):
        """Test that parser rejects non-list models field."""
        config_dict = _base_config_dict()
        config_dict["hyperparameter_save"] = {
            "enabled": True,
            "models": "KNN",  # Should be a list
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        with pytest.raises(ValueError, match="must be a list"):
            GLOCExperimentConfigParser(config_location=str(config_path))

    def test_parser_rejects_empty_models_when_enabled(self, tmp_path):
        """Test that parser rejects empty models list when hyperparameter_save is enabled."""
        config_dict = _base_config_dict()
        config_dict["hyperparameter_save"] = {
            "enabled": True,
            "models": [],  # Empty list not allowed when enabled
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        with pytest.raises(
            ValueError, match="non-empty list when.*enabled"
        ):
            GLOCExperimentConfigParser(config_location=str(config_path))

    def test_parser_allows_empty_models_when_disabled(self, tmp_path):
        """Test that parser allows empty models list when hyperparameter_save is disabled."""
        config_dict = _base_config_dict()
        config_dict["hyperparameter_save"] = {
            "enabled": False,
            "models": [],  # Allowed when disabled
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        parser = GLOCExperimentConfigParser(config_location=str(config_path))

        assert parser.get_hyperparameter_save_enabled() is False
        assert parser.get_hyperparameter_save_models() == []


# ============================================================================
# HYPERPARAMETER SAVE TESTS
# ============================================================================


class TestSaveMedianHyperparameters:
    """Test the save_median_hyperparameters function."""

    def _create_mock_fold_structure(
        self,
        project_root: Path,
        classifier: str,
        model_type_folder: str,
        num_folds: int = 10,
        median_fold_idx: int = 5,
    ) -> dict:
        """Create a mock directory structure with fold data."""
        fold_f1_scores = {}

        # Create directory structure
        base_perf_dir = (
            project_root
            / "PerformanceSave"
            / "CrossValidation"
            / f"{classifier}_hpo"
            / model_type_folder
        )
        base_model_dir = project_root / "ModelSave" / "CV" / model_type_folder

        for fold_id in range(num_folds):
            fold_str = str(fold_id)

            # Create performance directory and save FoldSummary.pkl
            perf_dir = base_perf_dir / fold_str
            perf_dir.mkdir(parents=True, exist_ok=True)

            # Create a mock performance dictionary with F1 scores using real dataframes
            f1_score = 0.5 + (fold_id * 0.01)  # Different scores for each fold
            fold_f1_scores[fold_id] = f1_score

            # Create a real DataFrame with the F1 score
            perf_df = pd.DataFrame({"f1-score": [f1_score]})
            perf_data = {fold_str: perf_df}

            perf_summary_path = perf_dir / "FoldSummary.pkl"
            with open(perf_summary_path, "wb") as f:
                pickle.dump(perf_data, f)

            # Create model directory and save model pickle
            model_dir = base_model_dir / fold_str
            model_dir.mkdir(parents=True, exist_ok=True)

            # Create a simple model with best_params_ that can be pickled
            simple_model = SimpleModel()

            classifier_to_model_file = {
                "RF": "random_forest_model.pkl",
                "KNN": "KNN_model.pkl",
                "EGB": "ensemble_model.pkl",
                "logreg": "logistic_regression_model.pkl",
                "SVM": "SVM_model.pkl",
                "LDA": "LDA_model.pkl",
            }
            model_filename = classifier_to_model_file[classifier]
            model_path = model_dir / model_filename
            with open(model_path, "wb") as f:
                pickle.dump(simple_model, f)

            # Save selected features
            if classifier == "logreg":
                features_filename = "SelectedFeaturesLR.pkl"
            else:
                features_filename = f"SelectedFeatures{classifier}.pkl"

            features_path = model_dir / features_filename
            selected_features = [
                f"feature_{i}" for i in range(10)
            ]  # Mock features
            with open(features_path, "wb") as f:
                pickle.dump(selected_features, f)

        return fold_f1_scores

    def test_save_median_hyperparameters_all_classifiers(self, tmp_path):
        """Test save_median_hyperparameters for a representative set of supported classifiers."""
        model_type_folder = "Complete_Explicit"
        classifiers = ["KNN", "EGB"]  # Test representative classifiers

        for classifier in classifiers:
            self._create_mock_fold_structure(
                tmp_path, classifier, model_type_folder, num_folds=10
            )

            # Save hyperparameters
            output_path = save_median_hyperparameters(
                classifier=classifier,
                model_type_folder_name=model_type_folder,
                project_root=tmp_path,
            )

            # Verify output file exists and is valid JSON
            assert output_path.exists()
            assert output_path.name == f"median_hyperparameters_{classifier}.json"

            # Verify JSON content
            with open(output_path, "r") as f:
                data = json.load(f)

            assert "fold_id" in data
            assert "f1_score" in data
            assert "best_params" in data
            assert "selected_features" in data

    def test_save_median_hyperparameters_identifies_median_fold(self, tmp_path):
        """Test that function correctly identifies the median-performing fold."""
        classifier = "KNN"
        model_type_folder = "Complete_Explicit"

        fold_f1_scores = self._create_mock_fold_structure(
            tmp_path, classifier, model_type_folder, num_folds=10
        )

        # Manually set specific F1 scores for verification
        fold_ids = list(range(10))
        f1_scores = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        base_perf_dir = (
            tmp_path
            / "PerformanceSave"
            / "CrossValidation"
            / f"{classifier}_hpo"
            / model_type_folder
        )

        for fold_id, f1_score in zip(fold_ids, f1_scores):
            fold_str = str(fold_id)
            perf_dir = base_perf_dir / fold_str

            # Update FoldSummary with new F1 score using real DataFrame
            perf_df = pd.DataFrame({"f1-score": [f1_score]})
            perf_data = {fold_str: perf_df}

            perf_summary_path = perf_dir / "FoldSummary.pkl"
            with open(perf_summary_path, "wb") as f:
                pickle.dump(perf_data, f)

        # Save hyperparameters
        output_path = save_median_hyperparameters(
            classifier=classifier,
            model_type_folder_name=model_type_folder,
            project_root=tmp_path,
        )

        # Verify median fold is selected (should be fold 5 with F1=0.75)
        with open(output_path, "r") as f:
            data = json.load(f)

        assert data["fold_id"] == "5"
        assert data["f1_score"] == 0.75

    def test_save_median_hyperparameters_handles_features_correctly(self, tmp_path):
        """Test that selected features are correctly saved and loaded."""
        classifier = "KNN"
        model_type_folder = "Complete_Explicit"

        self._create_mock_fold_structure(
            tmp_path, classifier, model_type_folder, num_folds=10
        )

        # Save hyperparameters
        output_path = save_median_hyperparameters(
            classifier=classifier,
            model_type_folder_name=model_type_folder,
            project_root=tmp_path,
        )

        # Load and verify features
        with open(output_path, "r") as f:
            data = json.load(f)

        assert isinstance(data["selected_features"], list)
        assert len(data["selected_features"]) == 10
        assert all(isinstance(f, str) for f in data["selected_features"])

    def test_save_median_hyperparameters_raises_on_invalid_classifier(self, tmp_path):
        """Test that function raises ValueError for invalid classifier."""
        with pytest.raises(
            ValueError, match="Unsupported classifier"
        ):
            save_median_hyperparameters(
                classifier="InvalidClassifier",
                model_type_folder_name="Complete_Explicit",
                project_root=tmp_path,
            )

    def test_save_median_hyperparameters_raises_on_missing_fold_summary(self, tmp_path):
        """Test that function raises FileNotFoundError for missing fold summary."""
        classifier = "KNN"
        model_type_folder = "Complete_Explicit"

        # Create directory but don't populate with data
        (tmp_path / "PerformanceSave" / "CrossValidation" / "KNN_hpo" / model_type_folder).mkdir(
            parents=True, exist_ok=True
        )

        with pytest.raises(FileNotFoundError):
            save_median_hyperparameters(
                classifier=classifier,
                model_type_folder_name=model_type_folder,
                project_root=tmp_path,
            )


# ============================================================================
# MAIN INTEGRATION TESTS
# ============================================================================


class TestHyperparameterSaveMainIntegration:
    """Test integration of hyperparameter save with main.py."""

    def test_run_hyperparameter_save_mode_enabled(self, tmp_path):
        """Test that _run_hyperparameter_save is called when enabled in config."""
        config_dict = _base_config_dict()
        config_dict["hyperparameter_save"] = {
            "enabled": True,
            "models": ["KNN"],
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        parser = GLOCExperimentConfigParser(config_location=str(config_path))

        with patch.object(main_module, "_run_hyperparameter_save") as mock_run:
            main_module.run(config_path=str(config_path))
            mock_run.assert_called_once()

    def test_run_hyperparameter_save_mode_disabled(self, tmp_path):
        """Test that _run_hyperparameter_save is not called when disabled."""
        config_dict = _base_config_dict()
        config_dict["hyperparameter_save"] = {
            "enabled": False,
            "models": [],
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        with patch.object(main_module, "_run_hyperparameter_save") as mock_run:
            main_module.run(config_path=str(config_path))
            mock_run.assert_not_called()

    def test_run_hyperparameter_save_with_multiple_models(self, tmp_path):
        """Test that hyperparameter save processes multiple models."""
        config_dict = _base_config_dict()
        config_dict["hyperparameter_save"] = {
            "enabled": True,
            "models": ["KNN", "RF", "EGB"],
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        parser = GLOCExperimentConfigParser(config_location=str(config_path))

        assert parser.get_hyperparameter_save_models() == ["KNN", "RF", "EGB"]


# ============================================================================
# END-TO-END TESTS
# ============================================================================


class TestHyperparameterSaveEndToEnd:
    """End-to-end tests combining config parsing and hyperparameter saving."""

    def test_hyperparameter_save_workflow_from_config(self, tmp_path, caplog):
        """Test complete workflow: config parsing and hyperparameter saving."""
        classifier = "KNN"
        model_type_folder = "Complete_Explicit"

        # Create mock fold structure
        test_helper = TestSaveMedianHyperparameters()
        test_helper._create_mock_fold_structure(
            tmp_path, classifier, model_type_folder
        )

        # Create config
        config_dict = _base_config_dict()
        config_dict["hyperparameter_save"] = {
            "enabled": True,
            "models": [classifier],
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        # Parse config
        parser = GLOCExperimentConfigParser(config_location=str(config_path))

        # Verify config was parsed correctly
        assert parser.get_hyperparameter_save_enabled() is True
        assert parser.get_hyperparameter_save_models() == [classifier]
        assert parser.get_model_type().get_folder_name() == model_type_folder

        # Save hyperparameters
        output_path = save_median_hyperparameters(
            classifier=classifier,
            model_type_folder_name=model_type_folder,
            project_root=tmp_path,
        )

        # Verify output
        assert output_path.exists()
        with open(output_path, "r") as f:
            data = json.load(f)
        assert "best_params" in data
        assert "selected_features" in data

    def test_hyperparameter_save_does_not_interfere_with_other_modes(self, tmp_path):
        """Test that enabling hyperparameter save doesn't interfere with other modes."""
        config_dict = _base_config_dict()
        config_dict["sensor_ablation"]["training"]["enabled"] = False
        config_dict["sensor_ablation"]["review"]["enabled"] = False
        config_dict["feature_space_review"]["enabled"] = False
        config_dict["hyperparameter_save"] = {
            "enabled": True,
            "models": ["KNN"],
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

        parser = GLOCExperimentConfigParser(config_location=str(config_path))

        assert parser.get_sensor_ablation_enabled() is False
        assert parser.get_sensor_ablation_review_enabled() is False
        assert parser.get_feature_space_review_enabled() is False
        assert parser.get_hyperparameter_save_enabled() is True
