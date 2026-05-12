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
    _find_median_fold_idx,
    _is_dl_adapter,
    _is_traditional_model,
    run_cross_validation,
)


class FakeCVConfig:
    def __init__(self, save_median_hyperparameters=True, hpo_config=None, advanced_hpo_settings=None):
        self.save_median_hyperparameters = save_median_hyperparameters
        self.hpo_config = hpo_config or {
            "enabled": True,
            "n_trials": 3,
            "timeout": None,
            "metric": "f1",
        }
        self.advanced_hpo_settings = advanced_hpo_settings or {
            "use_sampler": True,
            "final_early_stop": False,
            "metric": "f1",
            "n_trials": 3,
            "train_fraction": 0.8,
            "timeout": None,
            "sampler_seed": 42,
            "pruner_startup_trials": 3,
            "pruner_warmup_steps": 0,
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
        self.calls.append({"model": getattr(model, "get_name", lambda: None)(), "kfold_id": kfold_id})
        if kfold_id is None:
            X = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            y = np.asarray([0, 1, 0, 1])
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
        return {"accuracy": score, "precision": score, "recall": score, "f1": score}

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


def test_dglm_and_gam_have_model_implementations():
    # Ensure modern codebase contains DGLM and GAM model files
    assert Path("src/models/dglm_model.py").exists()
    assert Path("src/models/gam_model.py").exists()


def test_cross_validation_parser_supports_dglm_and_gam_models(tmp_path):
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
                    "models": ["DGLM", "GAM"],
                    "model_type": ["Complete", "Explicit"],
                    "random_seed": 42,
                    "num_splits": 2,
                    "save_results_folder": "Results/CrossValidation",
                    "class_weight": None,
                    "support_deep_learning": True,
                    "save_median_hyperparameters": False,
                    "impute_handling": {},
                    "hpo": {"enabled": False, "n_trials": 1, "metric": "f1"},
                },
            }
        ),
        encoding="utf-8",
    )
    parser = GLOCExperimentConfigParser(config_location=str(config_path))
    assert [model.get_name() for model in parser.get_cross_validation_models()] == ["DGLM", "GAM"]


def test_cross_validation_helper_branches():
    traditional = TinyTraditionalModel()
    advanced = TinyAdvancedModel(name="LSTM")
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
    assert len(model.calls) == 2
    assert (tmp_path / "Results" / "Complete_Explicit" / "TinyTrad" / "fold_0" / "metrics.pkl").exists()
    hyperparams_path = tmp_path / "Results" / "Complete_Explicit" / "TinyTrad" / "median_hyperparameters.json"
    assert hyperparams_path.exists()
    with open(hyperparams_path, "r", encoding="utf-8") as handle:
        hyperparams = json.load(handle)
    assert hyperparams["best_params"] == {"C": 1.0}
    assert hyperparams["selected_features"] == []


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


@pytest.mark.parametrize("model_name", ["DGLM", "GAM"])
def test_dglm_gam_cross_validation_runs_end_to_end(tmp_path, model_name):
    if model_name == "DGLM":
        pytest.importorskip("pyro")
        from src.models.dglm_model import DGLMModel

        model = DGLMModel(config={})
    else:
        pytest.importorskip("pygam")
        from src.models.gam_model import GAMModel

        model = GAMModel(config={})

    pipeline = TrialAwareAdvancedPipeline()
    results = run_cross_validation(
        FakeCVConfig(
            save_median_hyperparameters=False,
            hpo_config={"enabled": False, "n_trials": 1, "timeout": None, "metric": "f1"},
        ),
        pipeline,
        tmp_path,
        [model],
        num_splits=2,
        support_deep_learning=True,
        results_root=tmp_path / "Results",
        model_type=ModelType("Complete", "Explicit"),
    )

    assert model.get_name() in results
    assert len(results[model.get_name()]) == 2
    fold_dir = tmp_path / "Results" / "Complete_Explicit" / model.get_name() / "fold_0"
    assert (fold_dir / "metrics.pkl").exists()
    assert (fold_dir / "model").exists()
