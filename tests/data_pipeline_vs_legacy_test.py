import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.base import ModelInitStrategy
from src.models.extreme_gradient_boosting import ExtremeGradientBoostingModel
from src.models.k_nearest_neighbors import KNearestNeighborsModel
from src.models.linear_discriminant_analysis import LinearDiscriminantAnalysisModel
from src.models.logistic_regression import LogisticRegression
from src.models.random_forest import RandomForestModel
from src.models.support_vector_machine import SupportVectorMachineModel
from src.scripts import GLOC_classifier_traditional as legacy


@pytest.fixture
def binary_split_data():
    X, y = make_classification(
        n_samples=60,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


@pytest.fixture(autouse=True)
def patch_legacy_confusion_matrix(monkeypatch):
    monkeypatch.setattr(legacy, "create_confusion_matrix", lambda *args, **kwargs: None)


def _assert_float_tuple_close(actual, expected, atol=1e-12):
    assert len(actual) == len(expected)
    assert np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), atol=atol)


@pytest.mark.parametrize(
    "model_cls, expected_hyperparameters",
    [
        (
            LogisticRegression,
            {
                "baseline_window": 5,
                "window_size": 12.5,
                "stride": 0.25,
                "feature_reduction_type": "lasso",
                "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 5,
            },
        ),
        (
            RandomForestModel,
            {
                "baseline_window": 18.75,
                "window_size": 7.5,
                "stride": 0.25,
                "feature_reduction_type": "none",
                "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 3,
            },
        ),
        (
            LinearDiscriminantAnalysisModel,
            {
                "baseline_window": 46.25,
                "window_size": 15,
                "stride": 0.25,
                "feature_reduction_type": "lasso",
                "baseline_methods_to_use": ["v0", "v1", "v2"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 3,
            },
        ),
        (
            SupportVectorMachineModel,
            {
                "baseline_window": 32.5,
                "window_size": 15,
                "stride": 0.25,
                "feature_reduction_type": "ridge",
                "baseline_methods_to_use": ["v0", "v1", "v2"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 3,
            },
        ),
        (
            ExtremeGradientBoostingModel,
            {
                "baseline_window": 46.25,
                "window_size": 12.5,
                "stride": 0.25,
                "feature_reduction_type": "lasso",
                "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 3,
            },
        ),
        (
            KNearestNeighborsModel,
            {
                "baseline_window": 32.5,
                "window_size": 15,
                "stride": 0.25,
                "feature_reduction_type": "performance",
                "baseline_methods_to_use": ["v0", "v1", "v2"],
                "imbalance_type": "ros",
                "impute_type": 1,
                "n_neighbors": 5,
            },
        ),
    ],
)
def test_traditional_models_expose_legacy_hyperparameters(model_cls, expected_hyperparameters):
    model = model_cls(config={})
    assert model.get_traditional_hyperparameters() == expected_hyperparameters


def _call_legacy_classifier(model_cls, legacy_fn, x_train, x_test, y_train, y_test, legacy_model_name, best_params, strategy):
    """Call legacy classifier with appropriate parameters based on strategy."""
    # Convert strategy enum to the old retrain/temporal flags for legacy comparison
    retrain = strategy == ModelInitStrategy.RETRAIN_WITH_DEFAULTS
    temporal = strategy == ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS

    if model_cls in {LogisticRegression, SupportVectorMachineModel}:
        return legacy_fn(
            x_train,
            x_test,
            y_train,
            y_test,
            None,
            42,
            ".",
            legacy_model_name,
            retrain,
            temporal,
            best_params,
        )

    return legacy_fn(
        x_train,
        x_test,
        y_train,
        y_test,
        42,
        ".",
        legacy_model_name,
        retrain,
        temporal,
        best_params,
    )


@pytest.mark.parametrize(
    "model_cls,legacy_fn,best_params,legacy_model_name",
    [
        (
            LogisticRegression,
            legacy.classify_logistic_regression,
            {"solver": "lbfgs", "C": 1.0, "max_iter": 1000},
            "logreg.pkl",
        ),
        (
            KNearestNeighborsModel,
            legacy.classify_knn,
            {"n_neighbors": 7, "weights": "uniform", "metric": "minkowski"},
            "knn.pkl",
        ),
        (
            LinearDiscriminantAnalysisModel,
            legacy.classify_lda,
            {"solver": "svd"},
            "lda.pkl",
        ),
        (
            SupportVectorMachineModel,
            legacy.classify_svm,
            {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
            "svm.pkl",
        ),
        (
            ExtremeGradientBoostingModel,
            legacy.classify_ensemble_with_gradboost,
            {"n_estimators": 30, "learning_rate": 0.05, "max_depth": 3},
            "egb.pkl",
        ),
    ],
)
def test_traditional_model_temporal_eval_matches_legacy(
    binary_split_data,
    model_cls,
    legacy_fn,
    best_params,
    legacy_model_name,
):
    x_train, x_test, y_train, y_test = binary_split_data
    model = model_cls(config={})

    model_output = model.classify_traditional(
        x_train,
        x_test,
        y_train,
        y_test,
        class_weight_imb=None,
        random_state=42,
        save_folder=".",
        model_name=legacy_model_name,
        strategy=ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS,
        best_params=best_params,
    )

    legacy_output = _call_legacy_classifier(
        model_cls,
        legacy_fn,
        x_train,
        x_test,
        y_train,
        y_test,
        legacy_model_name,
        best_params,
        strategy=ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS,
    )

    _assert_float_tuple_close(model_output, legacy_output)

