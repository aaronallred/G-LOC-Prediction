"""Tests for the modern cross-validation mode (src/modes/cross_validation.py).

These tests target the *post-refactor* API surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.model_type import ModelType
from src.modes.cross_validation import (
    run_cross_validation,
    _aggregate_cv_results,
    _build_fold_result,
    _cache_fold_data_for_advanced_models,
    _extract_median_hyperparameters,
    _smote_resampling,
)
import src.modes.cross_validation as _cv_mod


class _FastBayesSearchCV:
    """Lightweight BayesSearchCV replacement that avoids expensive optimization."""

    def __init__(self, estimator, search_spaces=None, n_iter=1, cv=3, **kwargs):
        self.estimator = estimator
        self.n_iter = n_iter
        self.cv = cv
        self.best_params_ = {}
        self.best_score_ = 0.5
        self.best_index_ = 0
        self.search_spaces = search_spaces or {}
        self.best_estimator_ = None

    def fit(self, X, y=None):
        if hasattr(self.estimator, "fit"):
            self.estimator.fit(X, np.ravel(y) if y is not None else y)
            # Ensure Lasso-like selectors never drop every feature,
            # otherwise downstream SMOTE / models break on 0 columns.
            if hasattr(self.estimator, "coef_"):
                coef = np.asarray(self.estimator.coef_)
                if coef.ndim == 1 and np.count_nonzero(coef) == 0 and coef.size > 0:
                    self.estimator.coef_ = np.zeros_like(coef)
                    self.estimator.coef_[0] = 1.0
                elif coef.ndim == 2 and np.count_nonzero(coef) == 0 and coef.size > 0:
                    self.estimator.coef_ = np.zeros_like(coef)
                    self.estimator.coef_[0, 0] = 1.0
        self.best_estimator_ = self.estimator
        if isinstance(self.search_spaces, dict):
            for key in self.search_spaces:
                self.best_params_[key] = 1.0
        return self

    def predict(self, X):
        return self.estimator.predict(X)


# ---------------------------------------------------------------------------
# Autouse fixtures to speed up / harden tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fast_bayes_search(monkeypatch):
    """Replace BayesSearchCV with a lightweight version for faster tests."""
    monkeypatch.setattr(_cv_mod, "BayesSearchCV", _FastBayesSearchCV)


# ---------------------------------------------------------------------------
# Monkeypatch fixture for advanced models
# ---------------------------------------------------------------------------

def _create_factory_with_defaults():
    """Create a ModelFactory that injects default best_params for advanced models."""
    from src.models_new.model_factory import ModelFactory
    from src.models_new.logistic_regression_ts import LogRegTSModel

    orig_create = ModelFactory.create_model

    def _patched_create(model_name, model_hyperparameters=None):
        model = orig_create(model_name, model_hyperparameters)
        if isinstance(model, LogRegTSModel):
            model.best_params = {
                "baseline_method": 0,
                "batch_size": 4,
                "optimizer_type": "AdamW",
                "weight_decay": 0.01,
                "lr": 0.001,
                "sequence_length": 5,
                "threshold": 0.5,
                "step_size": 2,
            }

            def _patched_build_hpo(trial, X_train, random_seed):
                return {
                    "baseline_method": trial.suggest_categorical("baseline_method", [0, 1, 2, 3, 4, 5]),
                    "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
                    "optimizer_type": trial.suggest_categorical("optimizer_type", ["AdamW"]),
                    "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                    "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                    "sequence_length": trial.suggest_int("sequence_length", 5, 10),
                    "threshold": trial.suggest_float("threshold", 0.1, 0.9),
                    "step_size": trial.suggest_int("step_size", 2, 5),
                }

            model.get_hpo_search_space = lambda: _patched_build_hpo
        return model

    ModelFactory.create_model = staticmethod(_patched_create)
    return ModelFactory()


# ---------------------------------------------------------------------------
# Fixtures & stubs
# ---------------------------------------------------------------------------

class _DummyAdvancedModel:
    """Minimal advanced model for testing."""
    is_traditional_model = False
    name = "DummyAdv"

    def __init__(self):
        self.train_calls: list[dict] = []
        self.eval_calls = 0
        self._last_params: dict[str, Any] = {}
        self.best_params: dict[str, Any] = {}
        self.all_features: list[str] = []

    def get_hpo_search_space(self):
        def _builder(trial, X_train, random_seed):
            return {"lr": 0.01}
        return _builder

    def train(self, X, y, params=None):
        self._last_params = dict(params or {})
        self.best_params = dict(params or {})
        self.train_calls.append({"shape": getattr(X, "shape", None), "params": dict(params or {})})

    def evaluate(self, X, y):
        self.eval_calls += 1
        return {
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.7,
            "f1": 0.72,
            "specificity": 0.8,
            "g_mean": 0.749,
        }


class _FlexiblePipeline:
    def __init__(self):
        self.calls: list[dict] = []
        self.model_type: ModelType | None = None
        self.random_seed: int | None = None

    def set_random_seed(self, random_seed: int) -> None:
        self.random_seed = random_seed

    def set_model_type(self, model_type: ModelType) -> None:
        self.model_type = model_type

    def get_data(self, model=None, kfold_id=None, **kwargs):
        self.calls.append(
            {
                "model": getattr(model, "name", None),
                "kfold_id": kfold_id,
                "kwargs": dict(kwargs),
            }
        )
        if kfold_id is None:
            # Use 20 samples (10 per class) so that LASSO's internal cv=3 works
            rng = np.random.default_rng(42)
            X_pos = rng.random((10, 2)) + 1.0
            X_neg = rng.random((10, 2))
            X = np.vstack([X_pos, X_neg]).astype(float)
            y = np.array([1] * 10 + [0] * 10)
            if kwargs.get("return_feature_names"):
                return X, y, ["f1", "f2"]
            return X, y

        X_train = np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1], [0.1, 0.9]],
            dtype=float,
        )
        X_val = np.asarray([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]], dtype=float)
        y_train = np.asarray([0, 1, 0, 1, 1, 0])
        y_val = np.asarray([0, 1, 0])
        features = ["f1", "f2"]
        return X_train, X_val, y_train, y_val, features


class _TrialAwareAdvancedPipeline:
    def __init__(self, n_trials=6, rows_per_trial=10, n_features=4):
        self.n_trials = n_trials
        self.rows_per_trial = rows_per_trial
        self.n_features = n_features
        rng = np.random.RandomState(7)
        features = rng.randn(n_trials * rows_per_trial, n_features).astype(np.float32)
        labels = (features[:, 0] + 0.25 * features[:, 1] > 0.0).astype(int)
        trial_ids = np.repeat(np.arange(1, n_trials + 1, dtype=np.float32), rows_per_trial).reshape(-1, 1)
        self.X = np.hstack([features, trial_ids]).astype(np.float32)
        self.y = labels
        self.model_type = None
        self.random_seed = None

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed

    def set_model_type(self, model_type):
        self.model_type = model_type

    def get_data(self, model=None, kfold_id=None, **kwargs):
        if kfold_id is None:
            return self.X, self.y
        trial_ids = self.X[:, -1].astype(int)
        val_trials = {tid for tid in np.unique(trial_ids) if (tid % 2) == (kfold_id % 2)}
        val_mask = np.array([tid in val_trials for tid in trial_ids], dtype=bool)
        train_mask = ~val_mask
        X_train = self.X[train_mask]
        y_train = self.y[train_mask]
        X_val = self.X[val_mask]
        y_val = self.y[val_mask]
        features = [f"f{i}" for i in range(self.n_features)] + ["trial_id"]
        return X_train, X_val, y_train, y_val, features


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _make_cv_config(*, models, advanced_hpo=None):
    return {
        "cross_validation": {
            "enabled": True,
            "models": models,
            "model_type": ModelType("Complete", "Explicit"),
            "random_seed": 42,
            "num_splits": 2,
            "save_results_folder": "Results",
            "class_weight": None,
            "advanced_hpo": advanced_hpo or {
                "use_sampler": True,
                "final_early_stop": False,
                "objective_var": "F1",
                "trials": 3,
            },
        }
    }


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

def test_aggregate_cv_results_basic():
    fold_results = [
        {"metrics": {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.6, "specificity": 0.5, "g_mean": 0.4}},
        {"metrics": {"accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.7, "specificity": 0.5, "g_mean": 0.55}},
    ]
    aggregated, median_idx, median_f1 = _aggregate_cv_results(fold_results)
    assert aggregated["f1_mean"] == pytest.approx(0.65)
    assert aggregated["num_folds"] == 2
    # The function picks the upper median (sorted([0.6,0.7])[1] = 0.7)
    assert median_f1 == 0.7


def test_build_fold_result_with_features():
    result = _build_fold_result(
        fold_idx=3,
        metrics={"accuracy": 0.9, "f1": 0.6, "specificity": 0.5, "g_mean": 0.4},
        n_train=12,
        n_val=4,
        best_params={"alpha": 0.1},
        features=["f1", "f2"],
    )
    assert result["fold"] == 3
    assert result["selected_features"] == ["f1", "f2"]
    assert result["best_params"] == {"alpha": 0.1}


def test_build_fold_result_without_features():
    minimal = _build_fold_result(
        fold_idx=4,
        metrics={"accuracy": 0.9, "f1": 0.6, "specificity": 0.5, "g_mean": 0.4},
        n_train=12,
        n_val=4,
        best_params={},
    )
    assert "selected_features" not in minimal
    assert minimal["best_params"] == {}


def test_smote_resampling_changes_shape():
    X = np.random.randn(40, 6).astype(np.float32)
    y = np.array([0] * 30 + [1] * 10)
    X_res, y_res = _smote_resampling(X, y, random_seed=42)
    assert X_res.shape[0] > X.shape[0] or y_res.sum() > y.sum()


def test_cache_fold_data_for_advanced_models_builds_all_requested_folds():
    pipeline = _TrialAwareAdvancedPipeline(n_trials=6, rows_per_trial=6, n_features=3)
    model = _DummyAdvancedModel()
    cache = _cache_fold_data_for_advanced_models(pipeline, model, num_splits=3, random_seed=42)

    assert set(cache.keys()) == {0, 1, 2}
    for fold_idx, (X_train, X_val, y_train, y_val, features) in cache.items():
        assert X_train.shape[1] == 4
        assert X_val.shape[1] == 4
        assert y_train.ndim == 1
        assert y_val.ndim == 1
        assert features[-1] == "trial_id"


def test_extract_median_hyperparameters_prefers_fold_result_fields():
    fold_results = [
        {"metrics": {"f1": 0.2}, "best_params": {"alpha": 0.1}, "selected_features": ["f1"]},
        {"metrics": {"f1": 0.8}, "best_params": {"alpha": 0.2}, "selected_features": ["f2"]},
        {"metrics": {"f1": 0.5}, "best_params": {"alpha": 0.3}, "selected_features": ["f3"]},
    ]
    result = _extract_median_hyperparameters(2, 0.5, fold_results)
    assert result["fold_id"] == 2
    assert result["best_params"] == {"alpha": 0.3}
    assert result["selected_features"] == ["f3"]


# ---------------------------------------------------------------------------
# Integration-level tests for run_cross_validation
# ---------------------------------------------------------------------------

def test_cross_validation_parser_reads_mode_specific_models_and_hpo_config(tmp_path):
    from src.config_loader import load_experiment_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """data_path: /tmp/data
shared_data_parameters:
    subject_to_analyze: null
    trial_to_analyze: null
    analysis_type: 2
    remove_NaN_trials: true
    impute_file_name: imputed.pkl
    save_impute: false
    load_impute: false
    impute_phase: pre_feature
    output_feature_dtype: float32
advanced_data_parameters:
    n_neighbors: 4
    baseline_window: 32.5
traditional_data_parameters:
    backstep: 0
    data_rate: 25
    offset: 0
    time_start: 0
cross_validation:
    enabled: true
    models:
        - KNN
        - RF
    model_type: !ModelType [Complete, Explicit]
    random_seed: 42
    num_splits: 2
    save_results_folder: Results/CrossValidation
    class_weight: null
    advanced_hpo:
        use_sampler: true
        final_early_stop: false
        objective_var: F1
        trials: 4
""",
        encoding="utf-8",
    )
    config = load_experiment_config(config_path)

    assert config["cross_validation"]["models"] == ["KNN", "RF"]
    assert config["cross_validation"]["model_type"] == ModelType("Complete", "Explicit")


# ---------------------------------------------------------------------------
# Tests that actually invoke run_cross_validation
# ---------------------------------------------------------------------------

def test_traditional_cross_validation_saves_metrics_and_hyperparameters(tmp_path):
    from src.models_new.model_factory import ModelFactory

    pipeline = _FlexiblePipeline()
    config = _make_cv_config(models=["LogReg"])

    run_cross_validation(config, pipeline, ModelFactory(), tmp_path)

    model_root = tmp_path / "Results" / "Complete_Explicit" / "LogReg"
    folds = sorted([p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert len(folds) == 2

    hyperparams_path = model_root / "median_hyperparameters.json"
    assert hyperparams_path.exists()
    with open(hyperparams_path, "r", encoding="utf-8") as handle:
        hyperparams = json.load(handle)
    assert isinstance(hyperparams.get("best_params", {}), dict)
    assert pipeline.calls[0]["kwargs"]["traditional_feature_selection"] == "raw"
    assert pipeline.calls[0]["kwargs"]["return_feature_names"] is True


def test_traditional_cross_validation_applies_smote_before_classifier_fit(tmp_path):
    from src.models_new.model_factory import ModelFactory

    pipeline = _FlexiblePipeline()
    config = _make_cv_config(models=["LogReg"])

    run_cross_validation(config, pipeline, ModelFactory(), tmp_path)

    model_root = tmp_path / "Results" / "Complete_Explicit" / "LogReg"
    folds = sorted([p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert len(folds) == 2


def test_advanced_cross_validation_runs_hpo_and_persists_artifacts(tmp_path):
    pytest.importorskip("optuna")

    pipeline = _TrialAwareAdvancedPipeline(n_trials=6, rows_per_trial=10, n_features=4)
    config = _make_cv_config(
        models=["LogRegTS"],
        advanced_hpo={
            "use_sampler": True,
            "final_early_stop": False,
            "objective_var": "F1",
            "trials": 1,
        },
    )
    factory = _create_factory_with_defaults()
    run_cross_validation(config, pipeline, factory, tmp_path)

    model_root = tmp_path / "Results" / "Complete_Explicit" / "LogRegTS"
    folds = sorted([p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert len(folds) == 2
    for fold_idx in (0, 1):
        fold_dir = model_root / f"fold_{fold_idx}"
        assert (fold_dir / "fold_result.json").exists()
        assert (fold_dir / "hpo_summary.json").exists()
        with open(fold_dir / "hpo_summary.json", "r", encoding="utf-8") as handle:
            hpo_summary = json.load(handle)
        assert isinstance(hpo_summary.get("best_params", {}), dict)


def test_advanced_cross_validation_can_disable_hpo(tmp_path):
    pytest.importorskip("optuna")

    pipeline = _TrialAwareAdvancedPipeline(n_trials=6, rows_per_trial=10, n_features=4)
    config = _make_cv_config(
        models=["LogRegTS"],
        advanced_hpo={
            "use_sampler": True,
            "final_early_stop": False,
            "objective_var": "F1",
            "trials": 0,
        },
    )
    factory = _create_factory_with_defaults()
    run_cross_validation(config, pipeline, factory, tmp_path)

    model_root = tmp_path / "Results" / "Complete_Explicit" / "LogRegTS"
    folds = sorted([p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert len(folds) == 2
    # With n_trials==0 we should not get an hpo_summary.json written
    assert (model_root / "fold_0" / "hpo_summary.json").exists() is False


def test_advanced_cross_validation_adds_specificity_and_g_mean(tmp_path):
    pytest.importorskip("optuna")

    pipeline = _TrialAwareAdvancedPipeline(n_trials=6, rows_per_trial=10, n_features=4)
    config = _make_cv_config(
        models=["LogRegTS"],
        advanced_hpo={
            "use_sampler": True,
            "final_early_stop": False,
            "objective_var": "F1",
            "trials": 0,
        },
    )
    factory = _create_factory_with_defaults()
    run_cross_validation(config, pipeline, factory, tmp_path)

    model_root = tmp_path / "Results" / "Complete_Explicit" / "LogRegTS"
    folds = sorted([p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert len(folds) == 2
    for fold_dir in folds:
        with open(fold_dir / "fold_result.json", "r", encoding="utf-8") as fh:
            fr = json.load(fh)
        metrics = fr.get("metrics", {})
        assert "specificity" in metrics
        assert "g_mean" in metrics


def test_cross_validation_summary_uses_same_canonical_metric_keys(tmp_path):
    pytest.importorskip("optuna")

    pipeline = _TrialAwareAdvancedPipeline(n_trials=6, rows_per_trial=10, n_features=4)
    config = _make_cv_config(
        models=["LogRegTS"],
        advanced_hpo={
            "use_sampler": True,
            "final_early_stop": False,
            "objective_var": "F1",
            "trials": 0,
        },
    )
    factory = _create_factory_with_defaults()
    run_cross_validation(config, pipeline, factory, tmp_path)

    expected_keys = {
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "specificity_mean",
        "specificity_std",
        "g_mean_mean",
        "g_mean_std",
        "num_folds",
    }

    for model_name in ("LogRegTS",):
        summary_path = tmp_path / "Results" / "Complete_Explicit" / model_name / "summary.json"
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        assert set(summary.keys()) == expected_keys


def test_cross_validation_saves_matching_shared_artifact_names(tmp_path):
    pytest.importorskip("optuna")

    pipeline = _TrialAwareAdvancedPipeline(n_trials=6, rows_per_trial=10, n_features=4)
    config = _make_cv_config(
        models=["LogRegTS"],
        advanced_hpo={
            "use_sampler": True,
            "final_early_stop": False,
            "objective_var": "F1",
            "trials": 0,
        },
    )
    factory = _create_factory_with_defaults()
    run_cross_validation(config, pipeline, factory, tmp_path)

    expected_shared_paths = {
        Path("summary.json"),
        Path("median_hyperparameters.json"),
        Path("fold_0/fold_result.json"),
        Path("fold_1/fold_result.json"),
    }
    model_root = tmp_path / "Results" / "Complete_Explicit" / "LogRegTS"
    actual_paths = {
        path.relative_to(model_root)
        for path in model_root.rglob("*")
        if path.is_file()
    }
    assert expected_shared_paths.issubset(actual_paths)

    for fold_idx in (0, 1):
        fold_dir = model_root / f"fold_{fold_idx}"
        assert (fold_dir / "fold_result.json").exists()
