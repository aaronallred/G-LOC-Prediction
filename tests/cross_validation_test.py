import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.GLOC_experiment_config_parser import GLOCExperimentConfigParser
from src.model_type import ModelType
from src.models.base import ModelInitStrategy
from src.models.dl_adapter import DLModelAdapter
from src.modes.cross_validation import (
    _aggregate_cv_results,
    _compute_metrics_from_predictions,
    _extract_median_hyperparameters,
    _find_median_fold_idx,
    _is_dl_adapter,
    _is_traditional_model,
    run_cross_validation,
)


class FakeCVConfig:
    def __init__(self, save_median_hyperparameters=True):
        self.save_median_hyperparameters = save_median_hyperparameters

    def get_cross_validation_save_median_hyperparameters(self):
        return self.save_median_hyperparameters

    def get_model_type(self):
        return ModelType("Complete", "Explicit")


class FlexiblePipeline:
    def __init__(self):
        self.calls = []
        self.model_type = None
        self.random_seed = None

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed

    def set_model_type(self, model_type):
        self.model_type = model_type

    def get_data(self, model=None, kfold_id=None):
        self.calls.append({"model": getattr(model, "get_name", lambda: None)(), "kfold_id": kfold_id})
        if kfold_id is None:
            X = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            y = np.asarray([0, 1, 0, 1])
            return X, y

        X_train = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=float)
        X_val = np.asarray([[0.5, 0.5], [0.9, 0.1]], dtype=float)
        y_train = np.asarray([0, 1, 0, 1])
        y_val = np.asarray([0, 1])
        features = ["f1", "f2"]
        return X_train, X_val, y_train, y_val, features


class TinyTraditionalModel:
    def __init__(self, name="TinyTrad"):
        self._name = name
        self.is_traditional = True
        self.best_params = {"C": 1.0}
        self.calls = []

    def get_name(self):
        return self._name

    def classify_traditional(
        self,
        x_train,
        x_test,
        y_train,
        y_test,
        class_weight_imb,
        random_state,
        save_folder,
        model_name,
        strategy=ModelInitStrategy.RETRAIN_WITH_DEFAULTS,
        best_params=None,
    ):
        self.calls.append({"x_train": x_train, "x_test": x_test, "best_params": best_params, "strategy": strategy})
        return (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.txt").write_text("saved", encoding="utf-8")


class TinyAdvancedModel:
    def __init__(self, name="TinyAdv"):
        self._name = name
        self.is_traditional = False
        self.best_params = {"learning_rate": 0.1}
        self.calls = []

    def get_name(self):
        return self._name

    def train(self, X, y, params=None):
        self.calls.append(("train", X.shape, y.shape))

    def evaluate(self, X, y):
        self.calls.append(("evaluate", X.shape, y.shape))
        return {"accuracy": 0.88, "precision": 0.9, "recall": 0.87, "f1": 0.89}

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.txt").write_text("saved", encoding="utf-8")


class TinyDLAdapter(DLModelAdapter):
    def __init__(self):
        self._name = "TinyDL"
        self._bias = 0.7

    def fit(self, X_train, y_train, X_val, y_val, config):
        self._bias = float(np.mean(y_train))

    def predict_proba(self, X):
        p1 = np.full((len(X), 1), self._bias)
        return np.hstack([1.0 - p1, p1])

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.txt").write_text("saved", encoding="utf-8")

    def get_name(self):
        return self._name


def test_cross_validation_parser_reads_mode_specific_models(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data_path": "/tmp/data",
                "shared_data_parameters": {
                    "subject_to_analyze": None,
                    "trial_to_analyze": None,
                    "analysis_type": 2,
                    "remove_NaN_trials": True,
                    "impute_file_name": "imputed.pkl",
                    "save_impute": False,
                    "load_impute": False,
                    "should_impute": True,
                    "output_feature_dtype": "float32",
                },
                "advanced_data_parameters": {"n_neighbors": 4, "baseline_window": 32.5},
                "traditional_data_parameters": {"backstep": 0, "data_rate": 25, "offset": 0, "time_start": 0},
                "cross_validation": {
                    "enabled": True,
                    "models": ["KNN", "RF"],
                    "model_type": ["Complete", "Explicit"],
                    "random_seed": 42,
                    "num_splits": 2,
                    "save_results_folder": "Results/CrossValidation",
                    "class_weight": None,
                    "support_deep_learning": False,
                    "save_median_hyperparameters": True,
                    "impute_handling": {},
                },
            }
        ),
        encoding="utf-8",
    )
    parser = GLOCExperimentConfigParser(config_location=str(config_path))

    assert [model.get_name() for model in parser.get_cross_validation_models()] == ["KNN", "RF"]
    assert parser.get_cross_validation_model_type() == ModelType("Complete", "Explicit")


def test_cross_validation_helper_branches():
    traditional = TinyTraditionalModel()
    advanced = TinyAdvancedModel()
    dl = TinyDLAdapter()

    assert _is_traditional_model(traditional) is True
    assert _is_traditional_model(advanced) is False
    assert _is_dl_adapter(dl) is True

    metrics = _compute_metrics_from_predictions(np.asarray([0, 1, 0, 1]), np.asarray([0, 1, 1, 1]))
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert 0 <= metrics["f1"] <= 1

    fold_idx, median_f1 = _find_median_fold_idx(
        [{"metrics": {"f1": 0.2}}, {"metrics": {"f1": 0.8}}, {"metrics": {"f1": 0.5}}]
    )
    assert fold_idx == 2
    assert median_f1 == 0.5

    aggregated, _, _ = _aggregate_cv_results(
        [{"metrics": {"f1": 0.5, "accuracy": 0.9}}, {"metrics": {"f1": 0.7, "accuracy": 0.8}}]
    )
    assert aggregated["f1_mean"] == pytest.approx(0.6)
    assert aggregated["num_folds"] == 2


def test_traditional_cross_validation_saves_metrics_and_hyperparameters(tmp_path):
    pipeline = FlexiblePipeline()
    model = TinyTraditionalModel()
    results = run_cross_validation(
        FakeCVConfig(save_median_hyperparameters=True),
        pipeline,
        tmp_path,
        [model],
        num_splits=2,
        results_root=tmp_path / "Results",
        model_type=ModelType("Complete", "Explicit"),
    )

    assert len(results["TinyTrad"]) == 2
    assert model.calls and len(model.calls) == 2
    assert (tmp_path / "Results" / "Complete_Explicit" / "TinyTrad" / "fold_0" / "metrics.pkl").exists()
    hyperparams_path = tmp_path / "Results" / "Complete_Explicit" / "TinyTrad" / "median_hyperparameters.json"
    assert hyperparams_path.exists()
    with open(hyperparams_path, "r", encoding="utf-8") as handle:
        hyperparams = json.load(handle)
    assert hyperparams["best_params"] == {"C": 1.0}
    assert hyperparams["selected_features"] == []


def test_advanced_cross_validation_uses_fold_cache_and_saves_models(tmp_path):
    pipeline = FlexiblePipeline()
    model = TinyAdvancedModel()
    results = run_cross_validation(
        FakeCVConfig(save_median_hyperparameters=True),
        pipeline,
        tmp_path,
        [model],
        num_splits=2,
        results_root=tmp_path / "Results",
        model_type=ModelType("Complete", "Explicit"),
    )

    assert len(results["TinyAdv"]) == 2
    assert ("train", (4, 2), (4,)) in model.calls
    assert (tmp_path / "Results" / "Complete_Explicit" / "TinyAdv" / "fold_0" / "metrics.pkl").exists()
    assert (tmp_path / "Results" / "Complete_Explicit" / "TinyAdv" / "fold_0" / "model" / "model.txt").exists()


def test_dl_adapter_cross_validation_runs_with_support_flag(tmp_path):
    pipeline = FlexiblePipeline()
    model = TinyDLAdapter()
    results = run_cross_validation(
        FakeCVConfig(save_median_hyperparameters=False),
        pipeline,
        tmp_path,
        [model],
        num_splits=2,
        support_deep_learning=True,
        results_root=tmp_path / "Results",
        model_type=ModelType("Complete", "Explicit"),
    )

    assert len(results["TinyDL"]) == 2
    assert (tmp_path / "Results" / "Complete_Explicit" / "TinyDL" / "fold_0" / "metrics.pkl").exists()
    assert (tmp_path / "Results" / "Complete_Explicit" / "TinyDL" / "fold_0" / "model" / "model.txt").exists()

