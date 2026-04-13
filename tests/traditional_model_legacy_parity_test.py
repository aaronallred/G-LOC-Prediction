import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

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
        n_samples=220,
        n_features=14,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


@pytest.fixture(autouse=True)
def patch_legacy_confusion_matrix(monkeypatch):
    # Disable plotting side-effects so parity tests focus on model outputs.
    monkeypatch.setattr(legacy, "create_confusion_matrix", lambda *args, **kwargs: None)


def _assert_float_tuple_close(actual, expected, atol=1e-12):
    assert len(actual) == len(expected)
    assert np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), atol=atol)


def _call_legacy_classifier(model_cls, legacy_fn, x_train, x_test, y_train, y_test, legacy_model_name, best_params, retrain):
    if model_cls is LogisticRegression:
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
            not retrain,
            best_params,
        )

    if model_cls is SupportVectorMachineModel:
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
            not retrain,
            best_params,
        )

    if model_cls is KNearestNeighborsModel:
        return legacy_fn(
            x_train,
            x_test,
            y_train,
            y_test,
            42,
            ".",
            legacy_model_name,
            retrain,
            not retrain,
            best_params,
        )

    if model_cls is LinearDiscriminantAnalysisModel:
        return legacy_fn(
            x_train,
            x_test,
            y_train,
            y_test,
            42,
            ".",
            legacy_model_name,
            retrain,
            not retrain,
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
        not retrain,
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
            {"n_estimators": 60, "learning_rate": 0.05, "max_depth": 3},
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
        retrain=False,
        temporal=True,
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
        retrain=False,
    )

    _assert_float_tuple_close(model_output, legacy_output)


def test_random_forest_temporal_eval_matches_legacy(binary_split_data):
    x_train, x_test, y_train, y_test = binary_split_data
    model = RandomForestModel(config={})
    best_params = {"n_estimators": 40, "max_depth": 5, "min_samples_split": 2}

    model_output = model.classify_traditional(
        x_train,
        x_test,
        y_train,
        y_test,
        class_weight_imb=None,
        random_state=42,
        save_folder=".",
        model_name="rf.pkl",
        retrain=False,
        temporal=True,
        best_params=best_params,
    )

    legacy_output = legacy.classify_random_forest(
        x_train,
        x_test,
        y_train,
        y_test,
        None,
        42,
        ".",
        "rf.pkl",
        False,
        True,
        best_params,
    )

    _assert_float_tuple_close(model_output[:4], legacy_output[:4])
    assert model_output[4] == legacy_output[4]
    _assert_float_tuple_close(model_output[5:], legacy_output[5:])


@pytest.mark.parametrize(
    "model_cls,legacy_fn,legacy_model_name,needs_class_weight",
    [
        (LogisticRegression, legacy.classify_logistic_regression, "logreg.pkl", True),
        (KNearestNeighborsModel, legacy.classify_knn, "knn.pkl", False),
        (LinearDiscriminantAnalysisModel, legacy.classify_lda, "lda.pkl", False),
        (SupportVectorMachineModel, legacy.classify_svm, "svm.pkl", True),
        (ExtremeGradientBoostingModel, legacy.classify_ensemble_with_gradboost, "egb.pkl", False),
    ],
)
def test_traditional_model_retrain_eval_matches_legacy(
    binary_split_data,
    model_cls,
    legacy_fn,
    legacy_model_name,
    needs_class_weight,
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
        retrain=True,
        temporal=False,
        best_params=None,
    )

    legacy_output = _call_legacy_classifier(
        model_cls,
        legacy_fn,
        x_train,
        x_test,
        y_train,
        y_test,
        legacy_model_name,
        best_params=None,
        retrain=True,
    )

    _assert_float_tuple_close(model_output, legacy_output)


def test_random_forest_retrain_eval_matches_legacy(binary_split_data):
    x_train, x_test, y_train, y_test = binary_split_data
    model = RandomForestModel(config={})

    model_output = model.classify_traditional(
        x_train,
        x_test,
        y_train,
        y_test,
        class_weight_imb=None,
        random_state=42,
        save_folder=".",
        model_name="rf.pkl",
        retrain=True,
        temporal=False,
        best_params=None,
    )

    legacy_output = legacy.classify_random_forest(
        x_train,
        x_test,
        y_train,
        y_test,
        None,
        42,
        ".",
        "rf.pkl",
        True,
        False,
        None,
    )

    _assert_float_tuple_close(model_output[:4], legacy_output[:4])
    assert model_output[4] == legacy_output[4]
    _assert_float_tuple_close(model_output[5:], legacy_output[5:])
