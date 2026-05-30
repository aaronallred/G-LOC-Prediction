import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.datasets import make_classification

from src.GLOC_experiment_config_parser import GLOCExperimentConfigParser
from src.model_type import ModelType
from src.models.base import ModelInitStrategy
from src.modes.cross_validation import (
    _aggregate_cv_results,
    _find_median_fold_idx,
    _is_traditional_model,
    _run_traditional_smote_resampling,
    run_cross_validation,
)
from src.scripts import imbalance_techniques_traditional as legacy_imbalance


class FakeCVConfig:
    def __init__(self, save_median_hyperparameters=True, hpo_config=None, advanced_hpo_settings=None):
        self.save_median_hyperparameters = save_median_hyperparameters
        self.hpo_config = hpo_config or {
            "enabled": False,
        }
        # Mock output from parser: user-facing params + hardcoded Optuna defaults
        self.advanced_hpo_settings = advanced_hpo_settings or {
            "use_sampler": True,
            "final_early_stop": False,
            "metric": "f1",
            "n_trials": 3,
            "train_fraction": 0.8,  # Hardcoded: _DEFAULT_TRAIN_FRACTION
            "timeout": None,  # Hardcoded: _DEFAULT_HPO_TIMEOUT
            "sampler_seed": None,  # Hardcoded: set to random_seed in CV driver
            "pruner_startup_trials": 3,  # Hardcoded: _DEFAULT_PRUNER_STARTUP_TRIALS
            "pruner_warmup_steps": 0,  # Hardcoded: _DEFAULT_PRUNER_WARMUP_STEPS
        }

    def get_cross_validation_save_median_hyperparameters(self):
        return self.save_median_hyperparameters

    def get_cross_validation_hpo(self):
        return dict(self.hpo_config)

    def get_model_type(self):
        return ModelType("Complete", "Explicit")

    def get_advanced_hpo_settings(self):
        """Return mock advanced HPO settings for testing advanced models."""
        return dict(self.advanced_hpo_settings)


class FlexiblePipeline:
    def __init__(self):
        self.calls = []
        self.model_type = None
        self.random_seed = None

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed

    def set_model_type(self, model_type):
        self.model_type = model_type

    def get_data(self, model=None, kfold_id=None, **kwargs):
        self.calls.append(
            {
                "model": getattr(model, "get_name", lambda: None)(),
                "kfold_id": kfold_id,
                "kwargs": dict(kwargs),
            }
        )
        if kfold_id is None:
            X = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            y = np.asarray([0, 1, 0, 1])
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


class TinyTraditionalModel:
    def __init__(self, name="TinyTrad"):
        self._name = name
        self.is_traditional = True
        self.best_params = {"C": 1.0}
        self.calls = []

    def get_name(self):
        return self._name

    def _build_sklearn_estimator(self, class_weight=None, random_state=None, params=None):
        return _TinyEstimator(**(params or {}))

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


class _TinyEstimator:
    def __init__(self, **params):
        self.params = dict(params)

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class _FakeBayesSearchCV:
    instances = []

    def __init__(
        self,
        estimator,
        search_spaces,
        n_iter,
        cv,
        scoring=None,
        random_state=None,
        n_jobs=None,
        verbose=0,
        error_score=None,
    ):
        self.estimator = estimator
        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.error_score = error_score
        self.best_params_ = {}
        self.best_score_ = 0.0
        self.best_index_ = 0
        self.best_estimator_ = estimator
        self.fit_args = None
        self.estimator_name = estimator.__class__.__name__
        _FakeBayesSearchCV.instances.append(self)

    def fit(self, X, y):
        self.fit_args = (np.asarray(X).shape, np.asarray(y).shape)
        if self.estimator_name == "Lasso":
            self.best_params_ = {"alpha": 0.1}

            class _BestLasso:
                def __init__(self):
                    self.coef_ = np.asarray([1.0, 0.0])

            self.best_estimator_ = _BestLasso()
            self.best_score_ = 0.83
            self.best_index_ = 1
        else:
            self.best_params_ = {
                "penalty": "elasticnet",
                "C": 1.5,
                "solver": "saga",
                "l1_ratio": 0.4,
            }
            self.best_score_ = 0.91
            self.best_index_ = 2
        return self


class _TrackingSMOTE:
    calls = []

    def __init__(self, random_state=None, k_neighbors=None):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        _OrderTrackingBayesSearchCV.events.append("SMOTE.fit_resample")
        _TrackingSMOTE.calls.append(
            {
                "random_state": self.random_state,
                "k_neighbors": self.k_neighbors,
                "x_shape": X.shape,
                "y_shape": y.shape,
            }
        )
        return np.vstack([X, X[:1]]), np.concatenate([y, y[:1]])


class _OrderTrackingBayesSearchCV:
    events = []

    def __init__(self, estimator, search_spaces, n_iter, cv, scoring=None, random_state=None, n_jobs=None, verbose=0, error_score=None):
        self.estimator = estimator
        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.error_score = error_score
        self.best_params_ = {}
        self.best_score_ = 0.0
        self.best_index_ = 0
        self.best_estimator_ = estimator
        self.fit_args = None
        self.estimator_name = estimator.__class__.__name__

    def fit(self, X, y):
        self.fit_args = (np.asarray(X).shape, np.asarray(y).shape)
        _OrderTrackingBayesSearchCV.events.append(f"{self.estimator_name}.fit")
        if self.estimator_name == "Lasso":
            self.best_params_ = {"alpha": 0.1}

            class _BestLasso:
                def __init__(self):
                    self.coef_ = np.asarray([1.0, 0.0])

            self.best_estimator_ = _BestLasso()
            self.best_score_ = 0.83
            self.best_index_ = 1
        else:
            self.best_params_ = {
                "penalty": "elasticnet",
                "C": 1.5,
                "solver": "saga",
                "l1_ratio": 0.4,
            }
            self.best_score_ = 0.91
            self.best_index_ = 2
        return self


class _TraditionalSMOTEPipeline:
    def __init__(self):
        self.calls = []
        X, y = make_classification(
            n_samples=40,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            n_clusters_per_class=1,
            weights=[0.5, 0.5],
            class_sep=1.25,
            random_state=7,
        )
        self.X = X.astype(np.float32)
        self.y = y.astype(int)

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed

    def set_model_type(self, model_type):
        self.model_type = model_type

    def get_data(self, model=None, kfold_id=None, **kwargs):
        self.calls.append(
            {
                "model": getattr(model, "get_name", lambda: None)(),
                "kfold_id": kfold_id,
                "kwargs": dict(kwargs),
            }
        )
        return self.X, self.y, [f"f{i}" for i in range(self.X.shape[1])]


class TinyAdvancedModel:
    def __init__(self, name="LSTM"):
        self._name = name
        self.is_traditional = False
        self.best_params = {}
        self.train_calls = []
        self.eval_calls = 0
        self._last_params = {}

    def get_name(self):
        return self._name

    def hpo_defaults(self):
        # Tests expect 3 trials per fold; models now declare their own HPO defaults
        return {"enabled": True, "n_trials": 3, "timeout": None, "metric": "f1", "train_fraction": 0.8}

    def build_hpo_search_space(self, trial, X_train, random_seed):
        # Minimal per-model search-space to exercise HPO loop in tests
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        if self._name == "LSTM":
            hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
            return {"hidden_dim": hidden_dim, "lr": lr, "random_seed": int(random_seed)}
        if self._name == "TCN":
            num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])
            return {"num_filters": num_filters, "lr": lr, "random_seed": int(random_seed)}
        if self._name == "NAM":
            hidden_dim = trial.suggest_categorical("hidden_dim", [4, 8, 16])
            return {"hidden_dim": hidden_dim, "lr": lr, "random_seed": int(random_seed)}
        if self._name == "Trans":
            d_model = trial.suggest_categorical("d_model", [32, 64, 128])
            nhead = trial.suggest_categorical("nhead", [2, 4, 8])
            return {"d_model": d_model, "nhead": nhead, "learning_rate": lr, "random_seed": int(random_seed)}
        # LogRegTS or default
        return {"lr": lr, "random_seed": int(random_seed)}

    def train(self, X, y, params=None):
        self._last_params = dict(params or {})
        self.best_params = dict(params or {})
        self.train_calls.append({"shape": X.shape, "params": dict(params or {})})

    def evaluate(self, X, y):
        self.eval_calls += 1
        p = self._last_params
        score = 0.60
        if "hidden_dim" in p:
            score += float(p["hidden_dim"]) / 10000.0
        if "d_model" in p:
            score += float(p["d_model"]) / 10000.0
        if "num_filters" in p:
            score += float(p["num_filters"]) / 10000.0
        if "lr" in p:
            score += max(0.0, 0.02 - abs(float(p["lr"]) - 0.01))
        if "learning_rate" in p:
            score += max(0.0, 0.02 - abs(float(p["learning_rate"]) - 0.01))
        score = float(min(score, 0.99))
        # Provide specificity and g_mean to satisfy advanced evaluate() contract
        from sklearn.metrics import confusion_matrix
        # Create deterministic preds from last params influence to keep tests stable
        preds = np.zeros(len(y), dtype=int)
        if "hidden_dim" in p:
            preds = (np.arange(len(y)) % 2).astype(int)
        metrics = {"accuracy": score, "precision": score, "recall": score, "f1": score}
        cm = confusion_matrix(y, preds)
        if cm.shape == (2, 2):
            tn = cm[0, 0]
            fp = cm[0, 1]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0
        metrics["specificity"] = float(specificity)
        metrics["g_mean"] = float(np.sqrt(max(0.0, metrics["recall"] * metrics["specificity"])))
        return metrics

    def predict(self, X):
        X = np.asarray(X)
        return (X[:, 0] > np.median(X[:, 0])).astype(int)

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.txt").write_text("saved", encoding="utf-8")


def test_cross_validation_parser_reads_mode_specific_models_and_hpo_config(tmp_path):
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
                    "impute_phase": "pre_feature",
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
                    "save_median_hyperparameters": True,
                    "hpo": {
                        "enabled": True,
                        "n_trials": 4,
                        "metric": "f1",
                        "timeout": None,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    parser = GLOCExperimentConfigParser(config_location=str(config_path))

    assert [model.get_name() for model in parser.get_cross_validation_models()] == ["KNN", "RF"]
    assert parser.get_cross_validation_model_type() == ModelType("Complete", "Explicit")


def test_legacy_advanced_classifier_names_are_supported_by_hpo():
    # Ensure legacy script references the same model names (sanity check)
    legacy_source = Path("src/scripts/cross_validation_enhanced.py").read_text(encoding="utf-8")
    for model_name in ["LogRegTS", "NAM", "LSTM", "TCN", "Trans"]:
        assert f'"{model_name}"' in legacy_source


def test_cross_validation_helper_branches():
    traditional = TinyTraditionalModel()
    advanced = TinyAdvancedModel(name="LSTM")

    assert _is_traditional_model(traditional) is True
    assert _is_traditional_model(advanced) is False

    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray([0, 1, 0, 1])
    y_pred = np.asarray([0, 1, 1, 1])
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    assert accuracy == pytest.approx(0.75)
    assert 0 <= f1 <= 1

    fold_idx, median_f1 = _find_median_fold_idx(
        [{"metrics": {"f1": 0.2}}, {"metrics": {"f1": 0.8}}, {"metrics": {"f1": 0.5}}]
    )
    assert fold_idx == 2
    assert median_f1 == 0.5

    aggregated, _, _ = _aggregate_cv_results(
        [
            {
                "metrics": {
                    "accuracy": 0.9,
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.5,
                    "specificity": 0.6,
                    "g_mean": 0.65,
                }
            },
            {
                "metrics": {
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.6,
                    "f1": 0.7,
                    "specificity": 0.5,
                    "g_mean": 0.55,
                }
            },
        ]
    )
    assert aggregated["f1_mean"] == pytest.approx(0.6)
    assert aggregated["num_folds"] == 2


def test_aggregate_cv_results_requires_canonical_metrics():
    with pytest.raises(RuntimeError, match="missing required keys"):
        _aggregate_cv_results(
            [
                {
                    "metrics": {
                        "accuracy": 0.9,
                        "precision": 0.8,
                        "recall": 0.7,
                        "f1": 0.6,
                    }
                }
            ]
        )


def test_traditional_cross_validation_saves_metrics_and_hyperparameters(tmp_path, monkeypatch):
    pipeline = FlexiblePipeline()
    model = TinyTraditionalModel(name="LogReg")
    _FakeBayesSearchCV.instances = []
    monkeypatch.setattr("src.modes.cross_validation.BayesSearchCV", _FakeBayesSearchCV)
    results = run_cross_validation(
        FakeCVConfig(save_median_hyperparameters=True),
        pipeline,
        tmp_path,
        [model],
        num_splits=2,
        results_root=tmp_path / "Results",
        model_type=ModelType("Complete", "Explicit"),
    )

    assert len(results["LogReg"]) == 2
    assert len(_FakeBayesSearchCV.instances) == 4
    assert len(model.calls) == 2
    assert (tmp_path / "Results" / "Complete_Explicit" / "LogReg" / "fold_0" / "metrics.pkl").exists()
    hyperparams_path = tmp_path / "Results" / "Complete_Explicit" / "LogReg" / "median_hyperparameters.json"
    assert hyperparams_path.exists()
    with open(hyperparams_path, "r", encoding="utf-8") as handle:
        hyperparams = json.load(handle)
    assert hyperparams["best_params"] == {
        "penalty": "elasticnet",
        "C": 1.5,
        "solver": "saga",
        "l1_ratio": 0.4,
    }
    assert hyperparams["selected_features"] == ["f1"]
    assert pipeline.calls[0]["kwargs"]["traditional_feature_selection"] == "raw"
    assert pipeline.calls[0]["kwargs"]["return_feature_names"] is True


def test_traditional_cross_validation_runs_fold_local_lasso_and_bayessearchcv_hpo(tmp_path, monkeypatch):
    pipeline = FlexiblePipeline()
    model = TinyTraditionalModel(name="LogReg")
    _FakeBayesSearchCV.instances = []
    monkeypatch.setattr("src.modes.cross_validation.BayesSearchCV", _FakeBayesSearchCV)

    results = run_cross_validation(
        FakeCVConfig(save_median_hyperparameters=True, hpo_config={"enabled": False}),
        pipeline,
        tmp_path,
        [model],
        num_splits=2,
        results_root=tmp_path / "Results",
        model_type=ModelType("Complete", "Explicit"),
    )

    assert len(results["LogReg"]) == 2
    assert len(_FakeBayesSearchCV.instances) == 4
    lasso_instances = [instance for instance in _FakeBayesSearchCV.instances if instance.estimator_name == "Lasso"]
    classifier_instances = [instance for instance in _FakeBayesSearchCV.instances if instance.estimator_name != "Lasso"]

    assert len(lasso_instances) == 2
    assert len(classifier_instances) == 2

    for instance in lasso_instances:
        assert instance.n_iter == 50
        assert instance.cv == 3
        assert instance.scoring is None
        assert instance.fit_args == ((2, 2), (2,))
        # Ensure we limited parallelism for memory safety in traditional folds
        assert instance.n_jobs == 1

    for instance in classifier_instances:
        assert instance.n_iter == 30
        assert instance.cv == 3
        assert instance.scoring == "f1"
        assert instance.best_params_ == {
            "penalty": "elasticnet",
            "C": 1.5,
            "solver": "saga",
            "l1_ratio": 0.4,
        }
        # Traditional classifier HPO should also run with constrained workers
        assert instance.n_jobs == 1
        if isinstance(instance.search_spaces, dict):
            assert set(instance.search_spaces.keys()) == {"penalty", "C", "solver", "l1_ratio"}

    assert len(model.calls) == 2
    assert model.calls[0]["strategy"] == ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS
    assert model.calls[0]["best_params"] == {
        "penalty": "elasticnet",
        "C": 1.5,
        "solver": "saga",
        "l1_ratio": 0.4,
    }
    assert model.calls[0]["x_train"].shape[1] == 1
    assert model.calls[0]["x_test"].shape[1] == 1
    assert results["LogReg"][0]["best_params"] == {
        "penalty": "elasticnet",
        "C": 1.5,
        "solver": "saga",
        "l1_ratio": 0.4,
    }
    assert results["LogReg"][0]["selected_features"] == ["f1"]


def test_traditional_smote_helper_matches_legacy_simple_smote():
    X, y = make_classification(
        n_samples=40,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.5, 0.5],
        class_sep=1.25,
        random_state=11,
    )
    X = X.astype(np.float32)
    y = y.astype(int)

    modern_x, modern_y = _run_traditional_smote_resampling(X, y, random_seed=42)
    legacy_x, legacy_y = legacy_imbalance.simple_smote(X, y, random_state=42)

    np.testing.assert_allclose(modern_x, legacy_x)
    np.testing.assert_array_equal(modern_y, legacy_y)


def test_traditional_cross_validation_applies_smote_before_classifier_fit(tmp_path, monkeypatch):
    pipeline = _TraditionalSMOTEPipeline()
    model = TinyTraditionalModel(name="LogReg")

    _TrackingSMOTE.calls = []
    _OrderTrackingBayesSearchCV.events = []

    monkeypatch.setattr("src.modes.cross_validation.SMOTE", _TrackingSMOTE)
    monkeypatch.setattr("src.modes.cross_validation.BayesSearchCV", _OrderTrackingBayesSearchCV)

    results = run_cross_validation(
        FakeCVConfig(save_median_hyperparameters=True, hpo_config={"enabled": False}),
        pipeline,
        tmp_path,
        [model],
        num_splits=2,
        results_root=tmp_path / "Results",
        model_type=ModelType("Complete", "Explicit"),
        save_models=False,
    )

    assert len(results["LogReg"]) == 2
    assert _TrackingSMOTE.calls == [
        {"random_state": 42, "k_neighbors": 7, "x_shape": (20, 1), "y_shape": (20,)},
        {"random_state": 42, "k_neighbors": 7, "x_shape": (20, 1), "y_shape": (20,)},
    ]
    assert _OrderTrackingBayesSearchCV.events == [
        "Lasso.fit",
        "SMOTE.fit_resample",
        "_TinyEstimator.fit",
        "Lasso.fit",
        "SMOTE.fit_resample",
        "_TinyEstimator.fit",
    ]
    assert len(model.calls) == 2
    for call in model.calls:
        assert call["x_train"].shape[0] == 21
        assert call["x_test"].shape[1] == 1


@pytest.mark.parametrize("model_name", ["LogRegTS", "LSTM", "NAM", "TCN", "Trans"])
def test_advanced_cross_validation_runs_hpo_and_persists_artifacts(tmp_path, model_name):
    pytest.importorskip("optuna")

    pipeline = FlexiblePipeline()
    model = TinyAdvancedModel(name=model_name)
    results = run_cross_validation(
        FakeCVConfig(
            save_median_hyperparameters=True,
            hpo_config={"enabled": True, "n_trials": 3, "timeout": None, "metric": "f1"},
        ),
        pipeline,
        tmp_path,
        [model],
        num_splits=2,
        results_root=tmp_path / "Results",
        model_type=ModelType("Complete", "Explicit"),
    )

    assert len(results[model_name]) == 2
    # each fold: n_trials HPO train calls + one final training call
    assert len(model.train_calls) >= 2 * (3 + 1)
    assert model.eval_calls >= 2 * (3 + 1)
    for fold_idx in (0, 1):
        fold_dir = tmp_path / "Results" / "Complete_Explicit" / model_name / f"fold_{fold_idx}"
        assert (fold_dir / "metrics.pkl").exists()
        assert (fold_dir / "hpo_summary.json").exists()
        assert (fold_dir / "model" / "model.txt").exists()
        with open(fold_dir / "hpo_summary.json", "r", encoding="utf-8") as handle:
            hpo_summary = json.load(handle)
        assert hpo_summary["n_trials"] == 3
        assert isinstance(hpo_summary["best_params"], dict)


def test_advanced_cross_validation_can_disable_hpo(tmp_path):
    pipeline = FlexiblePipeline()
    model = TinyAdvancedModel(name="LSTM")
    results = run_cross_validation(
        FakeCVConfig(
            save_median_hyperparameters=True,
            advanced_hpo_settings={
                "use_sampler": True,
                "final_early_stop": False,
                "metric": "f1",
                "n_trials": 0,
                "train_fraction": 0.8,
                "timeout": None,
                "sampler_seed": 42,
                "pruner_startup_trials": 3,
                "pruner_warmup_steps": 0,
            },
        ),
        pipeline,
        tmp_path,
        [model],
        num_splits=2,
        results_root=tmp_path / "Results",
        model_type=ModelType("Complete", "Explicit"),
    )

    assert len(results["LSTM"]) == 2
    # no HPO objective training, only final per-fold train
    assert len(model.train_calls) == 2
    fold_dir = tmp_path / "Results" / "Complete_Explicit" / "LSTM" / "fold_0"
    assert (fold_dir / "hpo_summary.json").exists() is False


def test_advanced_cross_validation_adds_specificity_and_g_mean(tmp_path):
    pipeline = FlexiblePipeline()
    model = TinyAdvancedModel(name="LSTM")
    results = run_cross_validation(
        FakeCVConfig(save_median_hyperparameters=False),
        pipeline,
        tmp_path,
        [model],
        num_splits=2,
        results_root=tmp_path / "Results",
        model_type=ModelType("Complete", "Explicit"),
    )

    assert len(results["LSTM"]) == 2
    for fold in results["LSTM"]:
        metrics = fold["metrics"]
        assert "specificity" in metrics
        assert "g_mean" in metrics
        assert 0.0 <= metrics["specificity"] <= 1.0
        assert 0.0 <= metrics["g_mean"] <= 1.0


def test_cross_validation_summary_uses_same_canonical_metric_keys(tmp_path, monkeypatch):
    pipeline = FlexiblePipeline()
    traditional_model = TinyTraditionalModel(name="LogReg")
    advanced_model = TinyAdvancedModel(name="LSTM")

    _FakeBayesSearchCV.instances = []
    monkeypatch.setattr("src.modes.cross_validation.BayesSearchCV", _FakeBayesSearchCV)

    run_cross_validation(
        FakeCVConfig(
            save_median_hyperparameters=False,
            advanced_hpo_settings={
                "use_sampler": True,
                "final_early_stop": False,
                "metric": "f1",
                "n_trials": 0,
                "train_fraction": 0.8,
                "timeout": None,
                "sampler_seed": 42,
                "pruner_startup_trials": 3,
                "pruner_warmup_steps": 0,
            },
        ),
        pipeline,
        tmp_path,
        [traditional_model, advanced_model],
        num_splits=2,
        results_root=tmp_path / "Results",
        model_type=ModelType("Complete", "Explicit"),
    )

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

    for model_name in ("LogReg", "LSTM"):
        summary_path = (
            tmp_path
            / "Results"
            / "Complete_Explicit"
            / model_name
            / "summary.json"
        )
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        assert set(summary.keys()) == expected_keys


class TrialAwareAdvancedPipeline:
    def __init__(self, n_trials=8, rows_per_trial=12, n_features=4):
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
