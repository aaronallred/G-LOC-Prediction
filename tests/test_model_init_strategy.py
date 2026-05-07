"""
Tests for ModelInitStrategy enum and model initialization logic.

Tests the new semantic initialization strategy layer that replaces confusing
retrain/temporal flags with a clear enum-based approach.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest

from src.models.base import ModelInitStrategy
from src.models.k_nearest_neighbors import KNearestNeighborsModel
from src.models.logistic_regression import LogisticRegression
from src.models.random_forest import RandomForestModel
from src.models.support_vector_machine import SupportVectorMachineModel
from src.models.linear_discriminant_analysis import LinearDiscriminantAnalysisModel
from src.models.extreme_gradient_boosting import ExtremeGradientBoostingModel


class TestModelInitStrategy:
    """Test ModelInitStrategy enum values and semantics."""

    def test_enum_values_exist(self):
        """Verify all expected strategy values are defined."""
        assert hasattr(ModelInitStrategy, "RETRAIN_WITH_DEFAULTS")
        assert hasattr(ModelInitStrategy, "RETRAIN_WITH_CONFIG_PARAMS")
        assert hasattr(ModelInitStrategy, "LOAD_SAVED_MODEL")

    def test_enum_values_are_unique(self):
        """Verify enum values are distinct."""
        values = [s.value for s in ModelInitStrategy]
        assert len(values) == len(set(values))

    def test_enum_string_representation(self):
        """Verify enum string values are clear and descriptive."""
        assert ModelInitStrategy.RETRAIN_WITH_DEFAULTS.value == "retrain_with_defaults"
        assert ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS.value == "retrain_with_config_params"
        assert ModelInitStrategy.LOAD_SAVED_MODEL.value == "load_saved_model"


class TestBaseModelInitializeMethod:
    """Test _initialize_model_for_classification() implementation."""

    def test_retrain_with_defaults_strategy(self):
        """Test RETRAIN_WITH_DEFAULTS creates and trains with default params."""
        model = LogisticRegression({})
        X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y_train = np.array([0, 1, 0, 1])

        estimator = model._initialize_model_for_classification(
            strategy=ModelInitStrategy.RETRAIN_WITH_DEFAULTS,
            x_train=X_train,
            y_train=y_train,
            class_weight_imb=None,
            random_state=42,
            save_folder=None,
            model_name=None,
            best_params=None,
        )

        # Verify estimator is trained and can predict
        assert estimator is not None
        predictions = estimator.predict(X_train)
        assert len(predictions) == len(y_train)

    def test_retrain_with_config_params_strategy(self):
        """Test RETRAIN_WITH_CONFIG_PARAMS trains with provided params."""
        model = RandomForestModel({"n_estimators": 10})
        X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y_train = np.array([0, 1, 0, 1])
        config_params = {"n_estimators": 5, "max_depth": 2}

        estimator = model._initialize_model_for_classification(
            strategy=ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS,
            x_train=X_train,
            y_train=y_train,
            class_weight_imb=None,
            random_state=42,
            save_folder=None,
            model_name=None,
            best_params=config_params,
        )

        # Verify estimator is trained
        assert estimator is not None
        predictions = estimator.predict(X_train)
        assert len(predictions) == len(y_train)
        # Verify the params were applied (n_estimators should be 5)
        assert estimator.n_estimators == 5

    def test_retrain_with_config_params_missing_best_params_raises(self):
        """Test RETRAIN_WITH_CONFIG_PARAMS raises when best_params missing."""
        model = LogisticRegression({})
        X_train = np.array([[0, 0], [1, 1]])
        y_train = np.array([0, 1])

        with pytest.raises(ValueError, match="best_params"):
            model._initialize_model_for_classification(
                strategy=ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS,
                x_train=X_train,
                y_train=y_train,
                class_weight_imb=None,
                random_state=42,
                save_folder=None,
                model_name=None,
                best_params=None,
            )

    def test_load_saved_model_strategy(self):
        """Test LOAD_SAVED_MODEL loads model from disk."""
        model = LogisticRegression({})
        X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y_train = np.array([0, 1, 0, 1])

        # First, train and save a model
        with tempfile.TemporaryDirectory() as tmpdir:
            X_train_initial = np.array([[0, 0], [1, 1]])
            y_train_initial = np.array([0, 1])
            initial_estimator = model._initialize_model_for_classification(
                strategy=ModelInitStrategy.RETRAIN_WITH_DEFAULTS,
                x_train=X_train_initial,
                y_train=y_train_initial,
                class_weight_imb=None,
                random_state=42,
                save_folder=None,
                model_name=None,
                best_params=None,
            )
            
            model_path = Path(tmpdir) / "test_model.pkl"
            joblib.dump(initial_estimator, str(model_path))

            # Now load it using LOAD_SAVED_MODEL strategy
            loaded_estimator = model._initialize_model_for_classification(
                strategy=ModelInitStrategy.LOAD_SAVED_MODEL,
                x_train=X_train,
                y_train=y_train,
                class_weight_imb=None,
                random_state=42,
                save_folder=tmpdir,
                model_name="test_model.pkl",
                best_params=None,
            )

            assert loaded_estimator is not None
            # Verify loaded model makes predictions
            predictions = loaded_estimator.predict(X_train)
            assert len(predictions) == len(y_train)

    def test_load_saved_model_missing_folder_raises(self):
        """Test LOAD_SAVED_MODEL raises when save_folder missing."""
        model = LogisticRegression({})
        X_train = np.array([[0, 0], [1, 1]])
        y_train = np.array([0, 1])

        with pytest.raises(ValueError, match="save_folder"):
            model._initialize_model_for_classification(
                strategy=ModelInitStrategy.LOAD_SAVED_MODEL,
                x_train=X_train,
                y_train=y_train,
                class_weight_imb=None,
                random_state=42,
                save_folder=None,
                model_name="test.pkl",
                best_params=None,
            )

    def test_load_saved_model_file_not_found_raises(self):
        """Test LOAD_SAVED_MODEL raises when file doesn't exist."""
        model = LogisticRegression({})
        X_train = np.array([[0, 0], [1, 1]])
        y_train = np.array([0, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                model._initialize_model_for_classification(
                    strategy=ModelInitStrategy.LOAD_SAVED_MODEL,
                    x_train=X_train,
                    y_train=y_train,
                    class_weight_imb=None,
                    random_state=42,
                    save_folder=tmpdir,
                    model_name="nonexistent.pkl",
                    best_params=None,
                )

    def test_unknown_strategy_raises(self):
        """Test that invalid strategy raises ValueError."""
        model = LogisticRegression({})
        X_train = np.array([[0, 0], [1, 1]])
        y_train = np.array([0, 1])

        with pytest.raises(ValueError, match="Unknown initialization strategy"):
            model._initialize_model_for_classification(
                strategy="invalid_strategy",  # type: ignore
                x_train=X_train,
                y_train=y_train,
                class_weight_imb=None,
                random_state=42,
                save_folder=None,
                model_name=None,
                best_params=None,
            )


class TestBuildSklearnEstimator:
    """Test _build_sklearn_estimator() implementations."""

    def test_random_forest_build_estimator_with_defaults(self):
        """Test RandomForest builds estimator with defaults."""
        model = RandomForestModel({})
        estimator = model._build_sklearn_estimator(class_weight=None, random_state=42, params=None)
        assert estimator is not None
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")

    def test_random_forest_build_estimator_with_params(self):
        """Test RandomForest builds with provided params."""
        model = RandomForestModel({})
        params = {"n_estimators": 50, "max_depth": 10}
        estimator = model._build_sklearn_estimator(class_weight=None, random_state=42, params=params)
        assert estimator.n_estimators == 50
        assert estimator.max_depth == 10

    def test_logistic_regression_build_estimator(self):
        """Test LogisticRegression builds estimator."""
        model = LogisticRegression({})
        estimator = model._build_sklearn_estimator(class_weight="balanced", random_state=42, params=None)
        assert estimator is not None
        assert estimator.class_weight == "balanced"
        assert estimator.random_state == 42

    def test_knn_build_estimator(self):
        """Test KNN builds estimator."""
        model = KNearestNeighborsModel({})
        estimator = model._build_sklearn_estimator(class_weight=None, random_state=None, params=None)
        assert estimator is not None
        assert hasattr(estimator, "n_jobs")

    def test_svm_build_estimator(self):
        """Test SVM builds estimator."""
        model = SupportVectorMachineModel({})
        estimator = model._build_sklearn_estimator(class_weight="balanced", random_state=None, params=None)
        assert estimator is not None
        assert estimator.class_weight == "balanced"

    def test_lda_build_estimator(self):
        """Test LDA builds estimator."""
        model = LinearDiscriminantAnalysisModel({})
        estimator = model._build_sklearn_estimator(class_weight=None, random_state=None, params=None)
        assert estimator is not None

    def test_egb_build_estimator(self):
        """Test EGB builds estimator."""
        model = ExtremeGradientBoostingModel({})
        estimator = model._build_sklearn_estimator(class_weight=None, random_state=42, params=None)
        assert estimator is not None
        assert estimator.random_state == 42


class TestClassifyTraditionalWithStrategy:
    """Test classify_traditional() with new strategy parameter."""

    def test_classify_traditional_default_strategy(self):
        """Test classify_traditional with default strategy."""
        model = LogisticRegression({})
        X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        X_test = np.array([[0.5, 0.5], [2.5, 2.5]])
        y_train = np.array([0, 1, 0, 1])
        y_test = np.array([0, 1])

        result = model.classify_traditional(
            x_train=X_train,
            x_test=X_test,
            y_train=y_train,
            y_test=y_test,
            class_weight_imb=None,
            random_state=42,
            save_folder=None,
            model_name=None,
            strategy=ModelInitStrategy.RETRAIN_WITH_DEFAULTS,
            best_params=None,
        )

        # Verify result is a tuple with expected metrics
        assert isinstance(result, tuple)
        assert len(result) >= 6  # At least 6 metrics
        accuracy, precision, recall, f1, specificity, g_mean = result[:6]
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1

    def test_classify_traditional_with_config_params(self):
        """Test classify_traditional with RETRAIN_WITH_CONFIG_PARAMS."""
        model = RandomForestModel({})
        X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        X_test = np.array([[0.5, 0.5], [2.5, 2.5]])
        y_train = np.array([0, 1, 0, 1])
        y_test = np.array([0, 1])
        config_params = {"n_estimators": 10, "max_depth": 5}

        result = model.classify_traditional(
            x_train=X_train,
            x_test=X_test,
            y_train=y_train,
            y_test=y_test,
            class_weight_imb=None,
            random_state=42,
            save_folder=None,
            model_name=None,
            strategy=ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS,
            best_params=config_params,
        )

        assert isinstance(result, tuple)
        # RandomForest returns 7 values (includes tree depths)
        assert len(result) == 7

    def test_all_models_classify_traditional_basic(self):
        """Test classify_traditional works on all model types."""
        models = [
            RandomForestModel({}),
            LogisticRegression({}),
            KNearestNeighborsModel({}),
            SupportVectorMachineModel({}),
            LinearDiscriminantAnalysisModel({}),
            ExtremeGradientBoostingModel({}),
        ]

        # Use larger dataset to accommodate KNN's default n_neighbors=5
        X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [0, 1], [1, 0]])
        X_test = np.array([[0.5, 0.5], [2.5, 2.5]])
        y_train = np.array([0, 1, 0, 1, 0, 1])
        y_test = np.array([0, 1])

        for model in models:
            result = model.classify_traditional(
                x_train=X_train,
                x_test=X_test,
                y_train=y_train,
                y_test=y_test,
                class_weight_imb=None,
                random_state=42,
                save_folder=None,
                model_name=None,
                strategy=ModelInitStrategy.RETRAIN_WITH_DEFAULTS,
                best_params=None,
            )
            assert isinstance(result, tuple)
            assert len(result) >= 6, f"{model.get_name()} returned unexpected tuple length"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
