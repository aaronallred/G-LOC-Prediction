import json
import pickle
from pathlib import Path

import numpy as np

from src.model_type import ModelType
from src.modes.cross_validation import (
    run_cross_validation,
    _extract_median_hyperparameters,
    _find_median_fold_idx,
)


class FakeConfig:
    def __init__(self, save_median_hyperparameters=True):
        self.save_median_hyperparameters = save_median_hyperparameters

    def get_num_splits(self):
        return 3

    def get_random_seed(self):
        return 42

    def get_cross_validation_save_median_hyperparameters(self):
        return self.save_median_hyperparameters

    def get_model_type(self):
        return ModelType("Complete", "Explicit")


class FakePipelineAdvanced:
    def __init__(self):
        # create deterministic tiny dataset per fold
        pass

    def get_data(self, model=None, kfold_id=None):
        # Return (x_train,x_test,y_train,y_test,features)
        # For tests we return small arrays; shapes are consistent with expectations
        X_train = np.zeros((4, 2)) + kfold_id
        X_test = np.ones((2, 2)) * kfold_id
        y_train = np.array([0, 1, 0, 1])
        y_test = np.array([0, 1])
        return X_train, X_test, y_train, y_test, ["f1", "f2"]


class FakeModelAdvanced:
    def __init__(self):
        self.trained = 0
        self.saved = 0
        self.is_traditional = False
        self.best_params_ = {"C": 1.0, "kernel": "rbf"}
        self.n_features_in_ = 10

    def get_name(self):
        return "FakeAdv"

    def train(self, X, y, params=None):
        self.trained += 1

    def evaluate(self, X, y):
        return {"f1": 0.5}

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "saved.txt", "w") as fh:
            fh.write("ok")
        self.saved += 1


class FakePipelineTraditional:
    def get_data(self, model=None):
        X = np.arange(20).reshape(10, 2)
        y = np.array([0, 1] * 5)
        return X, y


class FakeModelTraditional:
    def __init__(self):
        self.is_traditional = True
        self.called = 0

    def get_name(self):
        return "FakeTrad"

    def classify_traditional(self, x_train, x_test, y_train, y_test, *args, **kwargs):
        self.called += 1
        # return legacy metrics tuple (accuracy, precision, recall, f1, spec, gmean)
        return (0.9, 0.8, 0.7, 0.6, 0.85, 0.75)


def test_advanced_model_cv(tmp_path):
    config = FakeConfig(save_median_hyperparameters=False)
    pipeline = FakePipelineAdvanced()
    model = FakeModelAdvanced()

    results = run_cross_validation(config, pipeline, tmp_path, [model], num_splits=3, results_root=tmp_path / "Results")

    assert "FakeAdv" in results
    assert len(results["FakeAdv"]) == 3
    # saved artifacts per fold (now nested by model_type)
    for k in range(3):
        assert (tmp_path / "Results" / "Complete_Explicit" / "FakeAdv" / f"fold_{k}" / "metrics.pkl").exists()


def test_traditional_model_cv(tmp_path):
    config = FakeConfig(save_median_hyperparameters=False)
    pipeline = FakePipelineTraditional()
    model = FakeModelTraditional()

    results = run_cross_validation(config, pipeline, tmp_path, [model], num_splits=2, results_root=tmp_path / "Results")

    assert "FakeTrad" in results
    # model.called should equal num_splits
    assert model.called == 2
    # persisted metrics files (now nested by model_type)
    assert (tmp_path / "Results" / "Complete_Explicit" / "FakeTrad" / "metrics_fold_0.pkl").exists()
    assert (tmp_path / "Results" / "Complete_Explicit" / "FakeTrad" / "metrics_fold_1.pkl").exists()


def test_find_median_fold_idx():
    """Test median fold identification by F1 score."""
    fold_results = [
        {"metrics": {"f1": 0.7}},
        {"metrics": {"f1": 0.9}},
        {"metrics": {"f1": 0.8}},
    ]
    idx, f1 = _find_median_fold_idx(fold_results)
    assert idx == 2  # Fold with F1=0.8 is median
    assert f1 == 0.8


def test_extract_median_hyperparameters():
    """Test hyperparameter extraction from model."""
    model = FakeModelAdvanced()
    fold_results = [
        {"metrics": {"f1": 0.7}, "selected_features": ["f1", "f2"], "best_params": {"C": 0.5, "kernel": "linear"}},
        {"metrics": {"f1": 0.85}, "selected_features": ["f1", "f2"], "best_params": {"C": 1.0, "kernel": "rbf"}},
        {"metrics": {"f1": 0.8}, "selected_features": ["f1", "f2"], "best_params": {"C": 2.0, "kernel": "rbf"}},
    ]
    hyperparams = _extract_median_hyperparameters(model, 1, 0.85, "TestModel", fold_results)
    
    assert hyperparams["fold_id"] == 1
    assert hyperparams["f1_score"] == 0.85
    # Should extract from fold_results, not from model
    assert hyperparams["best_params"] == {"C": 1.0, "kernel": "rbf"}
    assert hyperparams["selected_features"] == ["f1", "f2"]


def test_cv_saves_median_hyperparameters_when_enabled(tmp_path):
    """Test that median hyperparameters are saved when flag is enabled."""
    config = FakeConfig(save_median_hyperparameters=True)
    pipeline = FakePipelineAdvanced()
    model = FakeModelAdvanced()

    results = run_cross_validation(
        config, pipeline, tmp_path, [model], num_splits=3, results_root=tmp_path / "Results"
    )

    # Check that median_hyperparameters.json was created (now nested by model_type)
    hyperparams_path = tmp_path / "Results" / "Complete_Explicit" / "FakeAdv" / "median_hyperparameters.json"
    assert hyperparams_path.exists(), f"Hyperparameters file not found at {hyperparams_path}"
    
    # Load and verify content
    with open(hyperparams_path) as f:
        hyperparams = json.load(f)
    
    assert "fold_id" in hyperparams
    assert "f1_score" in hyperparams
    assert "best_params" in hyperparams
    assert "selected_features" in hyperparams
    assert hyperparams["best_params"] == {"C": 1.0, "kernel": "rbf"}


def test_cv_skips_median_hyperparameters_when_disabled(tmp_path):
    """Test that median hyperparameters are NOT saved when flag is disabled."""
    config = FakeConfig(save_median_hyperparameters=False)
    pipeline = FakePipelineAdvanced()
    model = FakeModelAdvanced()

    results = run_cross_validation(
        config, pipeline, tmp_path, [model], num_splits=3, results_root=tmp_path / "Results"
    )

    # Check that median_hyperparameters.json was NOT created
    hyperparams_path = tmp_path / "Results" / "Complete_Explicit" / "FakeAdv" / "median_hyperparameters.json"
    assert not hyperparams_path.exists(), f"Hyperparameters file should not exist at {hyperparams_path}"
