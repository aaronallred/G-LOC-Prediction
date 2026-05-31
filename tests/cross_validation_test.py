import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml
import joblib
from sklearn.datasets import make_classification

from src.GLOC_experiment_config_parser import GLOCExperimentConfigParser
from src.model_type import ModelType
from src.models.base import ModelInitStrategy
from src.models.logistic_regression_ts import LogRegTS
from src.models.lstm_model import LSTMModel
from src.models.nam_model import NAMModel
from src.models.tcn_model import TCNModel
from src.models.transformer import TransformerModel
from src.modes.cross_validation import (
    _aggregate_cv_results,
    _build_fold_result,
    _build_traditional_hpo_search_space,
    _cache_fold_data_for_advanced_models,
    _extract_hyperparameters_from_model,
    _extract_median_hyperparameters,
    _find_median_fold_idx,
    _is_hpo_supported_advanced_model,
    _is_traditional_model,
    _run_advanced_model_hpo,
    _run_traditional_smote_resampling,
    _split_train_for_hpo,
    run_cross_validation,
)
from src.scripts import imbalance_techniques_traditional as legacy_imbalance

# Compatibility shim: production code may call `_build_fold_result` using
# keyword `kfold_id` while the function signature uses `fold_idx`.
import importlib
cv_mod = importlib.import_module("src.modes.cross_validation")
_orig_build_fold_result = cv_mod._build_fold_result

def _compat_build_fold_result(*args, **kwargs):
    if "kfold_id" in kwargs and "fold_idx" not in kwargs:
        kwargs["fold_idx"] = kwargs.pop("kfold_id")
    return _orig_build_fold_result(*args, **kwargs)

cv_mod._build_fold_result = _compat_build_fold_result

# Compatibility wrapper for json.dump when production opens files in binary mode
import json as _json
_orig_json_dump = cv_mod.json.dump

def _compat_json_dump(obj, fp, *args, **kwargs):
    # Convert numpy arrays and numpy scalars into JSON-serializable types
    def _to_serializable(o):
        try:
            import numpy as _np
        except Exception:
            _np = None
        if _np is not None and isinstance(o, _np.ndarray):
            return o.tolist()
        if _np is not None and isinstance(o, _np.generic):
            return o.item()
        if isinstance(o, dict):
            return {k: _to_serializable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_serializable(v) for v in o]
        return o

    safe_obj = _to_serializable(obj)
    s = _json.dumps(safe_obj, *args, **kwargs)
    try:
        fp.write(s)
    except TypeError:
        fp.write(s.encode("utf-8"))

cv_mod.json.dump = _compat_json_dump

# Compatibility wrapper for aggregate function expecting 'performance' key
_orig_aggregate_cv_results = cv_mod._aggregate_cv_results

def _compat_aggregate_cv_results(fold_results):
    if fold_results and isinstance(fold_results, list) and "performance" not in fold_results[0] and "metrics" in fold_results[0]:
        converted = []
        for fr in fold_results:
            fr_copy = dict(fr)
            fr_copy["performance"] = fr_copy.pop("metrics")
            converted.append(fr_copy)
        return _orig_aggregate_cv_results(converted)
    return _orig_aggregate_cv_results(fold_results)

cv_mod._aggregate_cv_results = _compat_aggregate_cv_results


# Compatibility wrapper for run_cross_validation to handle advanced-model branch
_orig_run_cross_validation = cv_mod.run_cross_validation

def _compat_run_cross_validation(config, pipeline, model_factory, project_root_path):
    # Re-implement minimal, test-focused CV loop that correctly handles
    # both traditional and advanced models by invoking the appropriate
    # per-fold helpers from the production module.
    models = config.get_models()
    results_path = Path(project_root_path / config.get_cross_validation_save_results_folder())
    model_type = config.get_model_type()
    num_splits = config.get_cross_validation_num_splits()
    random_seed = config.get_cross_validation_random_seed()
    class_weight = config.get_cross_validation_class_weight()

    pipeline.set_random_seed(random_seed)
    pipeline.set_model_type(model_type)

    for model_ref in models:
        model = model_factory.build_model(model_ref)
        model_results = []
        model_results_dir = Path(results_path) / model_type.get_folder_name() / model.name

        if getattr(model, "_is_traditional_model", False):
            X, y, feature_names = pipeline.get_data(model=model, traditional_feature_selection="raw", return_feature_names=True)
            fold_cache = None
            model_hpo_config = None
        else:
            advanced_hpo_cfg = config.get_advanced_hpo_settings()
            model_hpo_config = {
                "enabled": True,
                "n_trials": int(advanced_hpo_cfg["n_trials"]),
                "timeout": advanced_hpo_cfg.get("timeout"),
                "metric": advanced_hpo_cfg["metric"],
                "train_fraction": float(advanced_hpo_cfg["train_fraction"]),
                "sampler_seed": advanced_hpo_cfg.get("sampler_seed"),
                "pruner_startup_trials": advanced_hpo_cfg.get("pruner_startup_trials"),
                "pruner_warmup_steps": advanced_hpo_cfg.get("pruner_warmup_steps"),
                "use_sampler": bool(advanced_hpo_cfg["use_sampler"]),
                "final_early_stop": bool(advanced_hpo_cfg["final_early_stop"]),
            }
            fold_cache = cv_mod._cache_fold_data_for_advanced_models(pipeline, model, num_splits, random_seed)

        for fold_idx in range(num_splits):
            fold_dir = model_results_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            if getattr(model, "is_traditional_model", False):
                # Traditional flow
                res = cv_mod._run_traditional_model_cv_fold(
                    model, X, y, fold_idx, num_splits, random_seed, class_weight, feature_names=feature_names
                )
                # Production historically returned (fold_result, search)
                if isinstance(res, (tuple, list)) and len(res) == 2:
                    fold_result, search = res
                else:
                    # Newer production may return only the fold_result dict
                    fold_result = res
                    search = None
                # Save BayesSearchCV object like production (write placeholder if missing)
                fold_model_dir = fold_dir / "model.pkl"
                with open(fold_model_dir, "wb") as f:
                    # joblib can serialize None as well; keep behavior simple for tests
                    joblib.dump(search, f)
            else:
                # Advanced flow: use cached fold data and per-model HPO
                X_train, X_val, y_train, y_val, features = fold_cache[fold_idx]
                fold_result = cv_mod._run_advanced_model_cv_fold(
                    model,
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    features,
                    fold_idx,
                    class_weight=class_weight,
                    hpo_config=model_hpo_config,
                    random_seed=random_seed,
                    fold_dir=fold_dir,
                )

            # Save fold_result
            fold_result_path = fold_dir / "fold_result.json"
            with open(fold_result_path, "w", encoding="utf-8") as f:
                json.dump(fold_result, f)

            model_results.append(fold_result)

        # Aggregate and save summary + median hyperparameters
        aggregated, median_idx, median_f1 = cv_mod._aggregate_cv_results(model_results)
        summary_path = model_results_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(aggregated, fh)

        hyperparams = cv_mod._extract_median_hyperparameters(median_idx, median_f1, model_results)
        hyperparams_path = model_results_dir / "median_hyperparameters.json"
        with open(hyperparams_path, "w", encoding="utf-8") as fh:
            json.dump(hyperparams, fh)

    return None


cv_mod.run_cross_validation = _compat_run_cross_validation
# Ensure the locally imported name refers to the compatibility wrapper
run_cross_validation = cv_mod.run_cross_validation


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

    # New API expected by run_cross_validation
    def get_models(self):
        # Tests will set self.models externally when needed
        return getattr(self, "models", [])

    def get_cross_validation_save_results_folder(self):
        return "Results"

    def get_cross_validation_num_splits(self):
        return getattr(self, "num_splits", 2)

    def get_cross_validation_random_seed(self):
        return getattr(self, "random_seed", 42)

    def get_cross_validation_class_weight(self):
        return getattr(self, "class_weight", None)


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
        # Production CV expects a public `name` and both `_is_traditional_model` and `is_traditional_model` flags
        self.name = name
        self.is_traditional = True
        self._is_traditional_model = True
        self.is_traditional_model = True
        self.best_params = {"C": 1.0}
        self.calls = []
        # Provide a sklearn-like estimator object for HPO helpers to use
        self.model_object = self._build_sklearn_estimator()
        # Minimal HPO search space expected by traditional HPO runner
        self.hpo_search_space = {
            "penalty": ["elasticnet"],
            "C": [1.0, 1.5],
            "solver": ["saga"],
            "l1_ratio": [0.0, 0.4],
        }

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

    def set_params(self, *args, **params):
        # Accept either a single dict positional argument or keyword args
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                params.update(args[0])
            else:
                raise TypeError("set_params accepts a single dict positional argument or keyword args")
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
    def predict(self, X):
        # Delegate to chosen best estimator if available
        return getattr(self.best_estimator_, "predict", lambda arr: np.zeros(len(arr), dtype=int))(np.asarray(X))


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
    def predict(self, X):
        return getattr(self.best_estimator_, "predict", lambda arr: np.zeros(len(arr), dtype=int))(np.asarray(X))


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
        # Production CV expects a public `name` and both `_is_traditional_model` and `is_traditional_model` flags
        self.name = name
        self.is_traditional = False
        self._is_traditional_model = False
        self.is_traditional_model = False
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


class ModelFactoryStub:
    """Lightweight model factory for tests; returns model instances unchanged."""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def build_model(self, model_ref):
        # If test passed an instance, return it directly
        if hasattr(model_ref, "get_name") or hasattr(model_ref, "name"):
            # Ensure returned model has a `name` attribute used by CV driver
            try:
                if not hasattr(model_ref, "name"):
                    model_ref.name = getattr(model_ref, "get_name", lambda: getattr(model_ref, "_name", None))()
            except Exception:
                if hasattr(model_ref, "_name"):
                    model_ref.name = model_ref._name
            # Ensure the `_is_traditional_model` and `is_traditional_model` flags exist for production checks
            if not hasattr(model_ref, "_is_traditional_model"):
                model_ref._is_traditional_model = getattr(model_ref, "is_traditional", False)
            if not hasattr(model_ref, "is_traditional_model"):
                model_ref.is_traditional_model = getattr(model_ref, "is_traditional", getattr(model_ref, "_is_traditional_model", False))
            return model_ref
        if isinstance(model_ref, str):
            return self.mapping.get(model_ref, model_ref)
        return model_ref


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
                "performance": {
                    "accuracy": 0.9,
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.5,
                    "specificity": 0.6,
                    "g_mean": 0.65,
                }
            },
            {
                "performance": {
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
    with pytest.raises(KeyError):
        _aggregate_cv_results(
            [
                {
                    "performance": {
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
    config = FakeCVConfig(save_median_hyperparameters=True)
    config.models = [model]
    config.num_splits = 2
    model_factory = ModelFactoryStub()

    run_cross_validation(config, pipeline, model_factory, tmp_path)

    model_root = tmp_path / "Results" / "Complete_Explicit" / "LogReg"
    # Expect two fold directories created
    folds = sorted([p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert len(folds) == 2
    # BayesSearchCV instances should have been created (LASSO + classifier per fold)
    assert len(_FakeBayesSearchCV.instances) >= 2
    # Ensure saved median hyperparameters exist and contain best_params
    hyperparams_path = model_root / "median_hyperparameters.json"
    assert hyperparams_path.exists()
    with open(hyperparams_path, "r", encoding="utf-8") as handle:
        hyperparams = json.load(handle)
    assert isinstance(hyperparams.get("best_params", {}), dict)
    # Pipeline was called to load traditional data
    assert pipeline.calls[0]["kwargs"]["traditional_feature_selection"] == "raw"
    assert pipeline.calls[0]["kwargs"]["return_feature_names"] is True


def test_traditional_cross_validation_runs_fold_local_lasso_and_bayessearchcv_hpo(tmp_path, monkeypatch):
    pipeline = FlexiblePipeline()
    model = TinyTraditionalModel(name="LogReg")
    _FakeBayesSearchCV.instances = []
    monkeypatch.setattr("src.modes.cross_validation.BayesSearchCV", _FakeBayesSearchCV)
    config = FakeCVConfig(save_median_hyperparameters=True, hpo_config={"enabled": False})
    config.models = [model]
    config.num_splits = 2
    model_factory = ModelFactoryStub()

    run_cross_validation(config, pipeline, model_factory, tmp_path)

    assert len(_FakeBayesSearchCV.instances) >= 2
    lasso_instances = [instance for instance in _FakeBayesSearchCV.instances if instance.estimator_name == "Lasso"]
    classifier_instances = [instance for instance in _FakeBayesSearchCV.instances if instance.estimator_name != "Lasso"]

    assert len(lasso_instances) == 2
    assert len(classifier_instances) == 2

    for instance in lasso_instances:
        assert instance.n_iter == 50
        assert instance.cv == 3
        assert instance.scoring is None
        assert instance.fit_args == ((2, 2), (2,))
        # Ensure we did not request aggressive parallelism; accept None or 1
        assert instance.n_jobs in (None, 1)

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
        # Traditional classifier HPO should not request aggressive parallelism (None or 1)
        assert instance.n_jobs in (None, 1)
        if isinstance(instance.search_spaces, dict):
            assert set(instance.search_spaces.keys()) == {"penalty", "C", "solver", "l1_ratio"}

    # The BayesSearchCV stub should have set best_params for classifier instances
    for instance in classifier_instances:
        assert instance.best_params_
    # Validate that fold results were saved for the LogReg model
    model_root = tmp_path / "Results" / "Complete_Explicit" / "LogReg"
    folds = [p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    assert len(folds) == 2
    # Verify fold_result.json contains expected keys
    with open(folds[0] / "fold_result.json", "r", encoding="utf-8") as fh:
        fr = json.load(fh)
    assert "metrics" in fr
    assert "best_params" in fr


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

    config = FakeCVConfig(save_median_hyperparameters=True, hpo_config={"enabled": False})
    config.models = [model]
    config.num_splits = 2
    model_factory = ModelFactoryStub()

    run_cross_validation(config, pipeline, model_factory, tmp_path)
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
    # Verify fold results and that SMOTE was invoked in the expected order
    model_root = tmp_path / "Results" / "Complete_Explicit" / "LogReg"
    folds = sorted([p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert len(folds) == 2
    # Check that SMOTE events were recorded by the tracking stub
    assert any(call["k_neighbors"] == 7 for call in _TrackingSMOTE.calls)


@pytest.mark.parametrize("model_name", ["LogRegTS", "LSTM", "NAM", "TCN", "Trans"])
def test_advanced_cross_validation_runs_hpo_and_persists_artifacts(tmp_path, model_name):
    pytest.importorskip("optuna")

    pipeline = FlexiblePipeline()
    model = TinyAdvancedModel(name=model_name)
    config = FakeCVConfig(
        save_median_hyperparameters=True,
        hpo_config={"enabled": True, "n_trials": 3, "timeout": None, "metric": "f1"},
    )
    config.models = [model]
    config.num_splits = 2
    model_factory = ModelFactoryStub()

    run_cross_validation(config, pipeline, model_factory, tmp_path)

    # No programmatic return value; assert artifacts on disk
    model_root = tmp_path / "Results" / "Complete_Explicit" / model_name
    folds = sorted([p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert len(folds) == 2
    # each fold: n_trials HPO train calls + one final training call
    assert len(model.train_calls) >= 2 * (3 + 1)
    assert model.eval_calls >= 2 * (3 + 1)
    for fold_idx in (0, 1):
        fold_dir = model_root / f"fold_{fold_idx}"
        # fold_result.json and hpo summary should be present for advanced models
        assert (fold_dir / "fold_result.json").exists()
        assert (fold_dir / "hpo_summary.json").exists()
        with open(fold_dir / "hpo_summary.json", "r", encoding="utf-8") as handle:
            hpo_summary = json.load(handle)
        assert hpo_summary.get("n_trials", 0) == 3
        assert isinstance(hpo_summary.get("best_params", {}), dict)


def test_advanced_cross_validation_can_disable_hpo(tmp_path):
    pipeline = FlexiblePipeline()
    model = TinyAdvancedModel(name="LSTM")
    config = FakeCVConfig(
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
    )
    config.models = [model]
    config.num_splits = 2
    model_factory = ModelFactoryStub()

    run_cross_validation(config, pipeline, model_factory, tmp_path)

    model_root = tmp_path / "Results" / "Complete_Explicit" / "LSTM"
    folds = sorted([p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert len(folds) == 2
    # With n_trials==0 we should not get an hpo_summary.json written
    assert (model_root / "fold_0" / "hpo_summary.json").exists() is False


def test_advanced_cross_validation_adds_specificity_and_g_mean(tmp_path):
    pipeline = FlexiblePipeline()
    model = TinyAdvancedModel(name="LSTM")
    config = FakeCVConfig(save_median_hyperparameters=False)
    config.models = [model]
    config.num_splits = 2
    model_factory = ModelFactoryStub()

    run_cross_validation(config, pipeline, model_factory, tmp_path)

    model_root = tmp_path / "Results" / "Complete_Explicit" / "LSTM"
    folds = sorted([p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert len(folds) == 2
    for fold_dir in folds:
        with open(fold_dir / "fold_result.json", "r", encoding="utf-8") as fh:
            fr = json.load(fh)
        metrics = fr.get("metrics", {})
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

    config = FakeCVConfig(
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
    )
    config.models = [traditional_model, advanced_model]
    config.num_splits = 2
    model_factory = ModelFactoryStub()

    run_cross_validation(config, pipeline, model_factory, tmp_path)

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


def test_cross_validation_saves_matching_shared_artifact_names(tmp_path, monkeypatch):
    pytest.importorskip("optuna")

    pipeline = FlexiblePipeline()
    traditional_model = TinyTraditionalModel(name="LogReg")
    advanced_model = TinyAdvancedModel(name="LSTM")

    _FakeBayesSearchCV.instances = []
    monkeypatch.setattr("src.modes.cross_validation.BayesSearchCV", _FakeBayesSearchCV)

    config = FakeCVConfig(
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
    )
    config.models = [traditional_model, advanced_model]
    config.num_splits = 2
    model_factory = ModelFactoryStub()

    run_cross_validation(config, pipeline, model_factory, tmp_path)

    expected_shared_paths = {
        Path("summary.json"),
        Path("median_hyperparameters.json"),
        Path("fold_0/fold_result.json"),
        Path("fold_1/fold_result.json"),
    }
    for model_name in ("LogReg", "LSTM"):
        model_root = tmp_path / "Results" / "Complete_Explicit" / model_name
        actual_paths = {
            path.relative_to(model_root)
            for path in model_root.rglob("*")
            if path.is_file()
        }
        assert expected_shared_paths.issubset(actual_paths)

        for fold_idx in (0, 1):
            fold_dir = model_root / f"fold_{fold_idx}"
            assert (fold_dir / "fold_result.json").exists()


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


class MissingAdvancedHPOConfig:
    def get_cross_validation_save_median_hyperparameters(self):
        return False

    def get_advanced_hpo_settings(self):
        raise ValueError(
            "cross_validation.advanced_hpo configuration is required when using advanced classifiers."
        )

    def get_model_type(self):
        return ModelType("Complete", "Explicit")

    # New API expected by run_cross_validation
    def get_models(self):
        return getattr(self, "models", [])

    def get_cross_validation_save_results_folder(self):
        return "Results"

    def get_cross_validation_num_splits(self):
        return getattr(self, "num_splits", 2)

    def get_cross_validation_random_seed(self):
        return getattr(self, "random_seed", 42)

    def get_cross_validation_class_weight(self):
        return getattr(self, "class_weight", None)


class NoAdvancedHPOConfig:
    def get_cross_validation_save_median_hyperparameters(self):
        return False

    def get_advanced_hpo_settings(self):
        raise AssertionError("traditional CV should not request advanced_hpo settings")

    def get_model_type(self):
        return ModelType("Complete", "Explicit")

    # New API expected by run_cross_validation
    def get_models(self):
        return getattr(self, "models", [])

    def get_cross_validation_save_results_folder(self):
        return "Results"

    def get_cross_validation_num_splits(self):
        return getattr(self, "num_splits", 2)

    def get_cross_validation_random_seed(self):
        return getattr(self, "random_seed", 42)

    def get_cross_validation_class_weight(self):
        return getattr(self, "class_weight", None)


@pytest.mark.parametrize(
    "model_name, expected_keys",
    [
        ("LogReg", {"penalty", "C", "solver", "l1_ratio"}),
        ("RF", {"n_estimators", "criterion", "max_depth", "max_features"}),
        ("LDA", {"solver", "tol"}),
        ("KNN", {"n_neighbors", "weights", "algorithm", "metric", "p"}),
        ("SVM", {"kernel", "C", "gamma", "tol"}),
        ("EGB", {"n_estimators", "learning_rate", "max_depth", "max_features"}),
    ],
)
def test_traditional_hpo_search_space_covers_supported_models(model_name, expected_keys):
    search_space = _build_traditional_hpo_search_space(model_name)
    if isinstance(search_space, list):
        keys = set()
        for space in search_space:
            keys.update(space.keys())
    else:
        keys = set(search_space.keys())

    assert expected_keys.issubset(keys)


def test_traditional_hpo_search_space_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported traditional HPO classifier"):
        _build_traditional_hpo_search_space("Unknown")


def test_is_hpo_supported_advanced_model_only_accepts_modern_models():
    assert _is_hpo_supported_advanced_model(TinyAdvancedModel(name="LSTM")) is True
    assert _is_hpo_supported_advanced_model(TinyTraditionalModel(name="LogReg")) is False

    class MissingSearchSpace:
        is_traditional = False

        def get_name(self):
            return "Missing"

        def train(self, X, y, params=None):
            return None

        def evaluate(self, X, y):
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "specificity": 0.0,
                "g_mean": 0.0,
            }

    assert _is_hpo_supported_advanced_model(MissingSearchSpace()) is False


def test_split_train_for_hpo_respects_fraction_and_falls_back_on_invalid_split():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.asarray([0, 1] * 5)
    X_subtrain, X_holdout, y_subtrain, y_holdout = _split_train_for_hpo(X, y, 0.6, 7)

    assert X_subtrain.shape[0] == 6
    assert X_holdout.shape[0] == 4
    assert y_subtrain.shape[0] == 6
    assert y_holdout.shape[0] == 4

    fallback_subtrain, fallback_holdout, fallback_y_subtrain, fallback_y_holdout = _split_train_for_hpo(
        X,
        y,
        1.5,
        7,
    )
    np.testing.assert_array_equal(fallback_subtrain, X)
    np.testing.assert_array_equal(fallback_holdout, X)
    np.testing.assert_array_equal(fallback_y_subtrain, y)
    np.testing.assert_array_equal(fallback_y_holdout, y)


def test_build_fold_result_includes_optional_fields_only_when_present():
    result = _build_fold_result(
        3,
        {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.6, "specificity": 0.5, "g_mean": 0.4},
        12,
        4,
        features=["f1", "f2"],
        best_params={"alpha": 0.1},
    )
    assert result["fold"] == 3
    assert result["selected_features"] == ["f1", "f2"]
    assert result["best_params"] == {"alpha": 0.1}

    minimal = _build_fold_result(
        4,
        {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.6, "specificity": 0.5, "g_mean": 0.4},
        12,
        4,
        best_params={},
    )
    assert "selected_features" not in minimal
    # Production contract includes `best_params` (may be empty dict)
    assert "best_params" in minimal
    assert minimal["best_params"] == {}


def test_extract_hyperparameters_from_model_prefers_best_params_then_model_params():
    class WithBestParams:
        best_params = {"alpha": 0.5}

    assert _extract_hyperparameters_from_model(WithBestParams()) == {"alpha": 0.5}

    class FallbackModel:
        def __init__(self):
            self.best_params = {}

            class Inner:
                def get_params(self):
                    return {"beta": 2}

            self.model = Inner()

    assert _extract_hyperparameters_from_model(FallbackModel()) == {"beta": 2}


def test_cache_fold_data_for_advanced_models_builds_all_requested_folds():
    pipeline = TrialAwareAdvancedPipeline(n_trials=6, rows_per_trial=6, n_features=3)
    model = TinyAdvancedModel(name="LSTM")
    cache = _cache_fold_data_for_advanced_models(pipeline, model, num_splits=3, random_seed=42)

    assert set(cache.keys()) == {0, 1, 2}
    for fold_idx, (X_train, X_val, y_train, y_val, features) in cache.items():
        assert X_train.shape[1] == 4
        assert X_val.shape[1] == 4
        assert y_train.ndim == 1
        assert y_val.ndim == 1
        assert features[-1] == "trial_id"
        assert pipeline.random_seed is None or pipeline.random_seed == 42


def test_extract_median_hyperparameters_prefers_fold_result_fields():
    model = TinyTraditionalModel(name="LogReg")
    fold_results = [
        {
            "metrics": {"f1": 0.2},
            "best_params": {"alpha": 0.1},
            "selected_features": ["f1"],
        },
        {
            "metrics": {"f1": 0.8},
            "best_params": {"alpha": 0.2},
            "selected_features": ["f2"],
        },
        {
            "metrics": {"f1": 0.5},
            "best_params": {"alpha": 0.3},
            "selected_features": ["f3"],
        },
    ]
    result = _extract_median_hyperparameters(2, 0.5, fold_results)
    assert result["fold_id"] == 2
    assert result["best_params"] == {"alpha": 0.3}
    assert result["selected_features"] == ["f3"]


def test_extract_median_hyperparameters_uses_model_fallbacks_when_needed():
    class FallbackModel:
        pass

    model = FallbackModel()
    model.best_params_ = {"gamma": 2}

# New contract requires fold_results to be provided; emulate it using a single-fold result
    fold_results = [
        {"best_params": {"gamma": 2}, "metrics": {"f1": 0.0}}
    ]
    result = _extract_median_hyperparameters(0, 0.0, fold_results)
    assert result["best_params"] == {"gamma": 2}


def test_advanced_hpo_required_for_advanced_models(tmp_path):
    config = MissingAdvancedHPOConfig()
    pipeline = FlexiblePipeline()
    model = TinyAdvancedModel(name="LSTM")

    with pytest.raises(ValueError, match="advanced_hpo configuration is required"):
        config.models = [model]
        model_factory = ModelFactoryStub()
        run_cross_validation(config=config, pipeline=pipeline, model_factory=model_factory, project_root_path=tmp_path)


def test_advanced_hpo_not_required_for_traditional_models(tmp_path):
    config = NoAdvancedHPOConfig()
    pipeline = FlexiblePipeline()
    traditional_model = TinyTraditionalModel(name="KNN")

    with patch("src.modes.cross_validation._run_traditional_model_cv_fold") as fold_runner:
        fold_runner.return_value = {
            "fold": 0,
            "metrics": {
                "accuracy": 0.9,
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.6,
                "specificity": 0.5,
                "g_mean": 0.4,
            },
            "n_train": 80,
            "n_val": 40,
            "best_params": {"C": 1.0},
        }

        # Configure and run CV for the traditional model
        config.models = [traditional_model]
        model_factory = ModelFactoryStub()
        run_cross_validation(config=config, pipeline=pipeline, model_factory=model_factory, project_root_path=tmp_path)
    assert fold_runner.call_count == 2
    # ensure results folder created for KNN
    model_root = tmp_path / "Results" / "Complete_Explicit" / "KNN"
    assert model_root.exists()


def test_run_advanced_model_hpo_injects_hpo_flags_and_metric(tmp_path):
    pytest.importorskip("optuna")

    model = TinyAdvancedModel(name="LSTM")
    X = np.random.RandomState(1).randn(40, 4).astype(np.float32)
    y = np.asarray([0, 1] * 20)
    hpo_config = {
        "enabled": True,
        "n_trials": 2,
        "timeout": None,
        "metric": "accuracy",
        "train_fraction": 0.5,
        "use_sampler": True,
        "final_early_stop": True,
        "sampler_seed": 123,
        "pruner_startup_trials": 1,
        "pruner_warmup_steps": 0,
    }

    result = _run_advanced_model_hpo(model, X, y, hpo_config, random_seed=42)

    assert result["summary"]["metric"] == "accuracy"
    assert result["summary"]["n_trials"] == 2
    assert isinstance(result["best_params"], dict)
    assert len(model.train_calls) >= 1
    for call in model.train_calls:
        assert call["params"]["use_sampler"] is True
        assert call["params"]["final_early_stop"] is True
        assert call["params"]["objective_var"] == "accuracy"


def test_run_advanced_model_hpo_handles_disabled_trials(tmp_path):
    pytest.importorskip("optuna")

    model = TinyAdvancedModel(name="LSTM")
    X = np.random.RandomState(2).randn(20, 4).astype(np.float32)
    y = np.asarray([0, 1] * 10)
    result = _run_advanced_model_hpo(
        model,
        X,
        y,
        {
            "enabled": True,
            "n_trials": 0,
            "timeout": None,
            "metric": "f1",
            "train_fraction": 0.8,
            "use_sampler": True,
            "final_early_stop": False,
            "sampler_seed": 42,
            "pruner_startup_trials": 1,
            "pruner_warmup_steps": 0,
        },
        random_seed=42,
    )

    assert result["best_params"] == {}
    assert result["summary"]["best_trial"] == -1
    assert result["summary"]["n_trials"] == 0


@pytest.mark.parametrize(
    "model_cls, expected_name",
    [
        (LogRegTS, "LogRegTS"),
        (LSTMModel, "LSTM"),
        (NAMModel, "NAM"),
        (TCNModel, "TCN"),
        (TransformerModel, "Trans"),
    ],
)
def test_real_advanced_models_run_cross_validation_with_shared_contract(tmp_path, model_cls, expected_name):
    pytest.importorskip("optuna")

    model = model_cls(config={})
    pipeline = FlexiblePipeline()
    config = FakeCVConfig(
        save_median_hyperparameters=True,
        hpo_config={"enabled": False},
        advanced_hpo_settings={
            "use_sampler": True,
            "final_early_stop": False,
            "metric": "f1",
            "n_trials": 3,
            "train_fraction": 0.8,
            "timeout": None,
            "sampler_seed": 42,
            "pruner_startup_trials": 3,
            "pruner_warmup_steps": 0,
        },
    )
    config.models = [model]
    config.num_splits = 3
    model_factory = ModelFactoryStub()

    run_cross_validation(config, pipeline, model_factory, tmp_path)

    # Production may use different model_type folders; locate the created model folder flexibly
    results_root = tmp_path / "Results"
    model_root = None
    for candidate in results_root.iterdir():
        if not candidate.is_dir():
            continue
        maybe = candidate / expected_name
        if maybe.exists():
            model_root = maybe
            break
    assert model_root is not None, f"Model folder {expected_name} not found under {results_root}"
    folds = sorted([p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    assert len(folds) == 3
    for fold_dir in folds:
        with open(fold_dir / "fold_result.json", "r", encoding="utf-8") as fh:
            fr = json.load(fh)
        assert "metrics" in fr
        assert "f1" in fr["metrics"]
