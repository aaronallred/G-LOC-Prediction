import pickle
from pathlib import Path

import numpy as np

from src.modes.cross_validation import run_cross_validation


class FakeConfig:
    def get_num_splits(self):
        return 3

    def get_random_seed(self):
        return 42


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
        self.best_params = {"C": 1}

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
    config = FakeConfig()
    pipeline = FakePipelineAdvanced()
    model = FakeModelAdvanced()

    results = run_cross_validation(config, pipeline, tmp_path, [model], num_splits=3, results_root=tmp_path / "Results")

    assert "FakeAdv" in results
    assert len(results["FakeAdv"]) == 3
    # saved artifacts per fold
    for k in range(3):
        assert (tmp_path / "Results" / "FakeAdv" / f"fold_{k}" / "metrics.pkl").exists()


def test_traditional_model_cv(tmp_path):
    config = FakeConfig()
    pipeline = FakePipelineTraditional()
    model = FakeModelTraditional()

    results = run_cross_validation(config, pipeline, tmp_path, [model], num_splits=2, results_root=tmp_path / "Results")

    assert "FakeTrad" in results
    # model.called should equal num_splits
    assert model.called == 2
    # persisted metrics files
    assert (tmp_path / "Results" / "FakeTrad" / "metrics_fold_0.pkl").exists()
    assert (tmp_path / "Results" / "FakeTrad" / "metrics_fold_1.pkl").exists()
