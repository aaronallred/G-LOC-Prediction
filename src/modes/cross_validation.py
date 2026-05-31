"""Cross-validation and advanced-model hyperparameter optimization mode for G-LOC.

This module keeps the existing traditional cross-validation contract while adding Optuna
HPO for advanced models (LogRegTS, LSTM, NAM, TCN, Trans), aligned with the legacy
`src/scripts/cross_validation_enhanced.py` orchestration pattern.
"""

import json
import joblib
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.metrics import geometric_mean_score
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.GLOC_experiment_config_parser import GLOCExperimentConfigParser
from src.model_type import ModelType
from src.models_new.base import BaseModel, ModelInitStrategy
from src.models.sequence_window_utils import max_trial_sequence_length
from src.models_new.model_factory import ModelFactory
from src.traditional_experiment_utils import stratified_kfold_split

logger = logging.getLogger(__name__)


# Per-model HPO handled via model.hpo_defaults() and model.build_hpo_search_space()


def _build_traditional_hpo_search_space(model_name: str) -> Any:
    """Return the legacy BayesSearchCV search space for a traditional classifier."""
    if Categorical is None or Integer is None or Real is None:
        raise ImportError(
            "BayesSearchCV support requires scikit-optimize (skopt) to be installed."
        )

    if model_name == "LogReg":
        return {
            "penalty": Categorical(["elasticnet"]),
            "C": Real(0.01, 100, prior="log-uniform"),
            "solver": Categorical(["saga"]),
            "l1_ratio": Real(0.0, 1.0),
        }

    if model_name == "RF":
        return {
            "n_estimators": Integer(10, 1000),
            "criterion": Categorical(["gini", "entropy", "log_loss"]),
            "max_depth": Integer(3, 100),
            "max_features": Categorical(["sqrt", "log2"]),
            "min_samples_leaf": Integer(1, 4),
            "min_samples_split": Integer(2, 10),
            "min_weight_fraction_leaf": Real(0.0, 0.5),
        }

    if model_name == "LDA":
        return [
            {
                "solver": Categorical(["svd"]),
                "tol": Real(1e-10, 1e-2, prior="log-uniform"),
            },
            {
                "solver": Categorical(["lsqr", "eigen"]),
                "shrinkage": Categorical([0.1, 0.2, 0.4, 0.6, 0.8, 1, "auto"]),
                "tol": Real(1e-10, 1e-2, prior="log-uniform"),
            },
        ]

    if model_name == "KNN":
        return {
            "n_neighbors": Integer(3, 30),
            "weights": Categorical(["uniform", "distance"]),
            "algorithm": Categorical(["auto", "brute"]),
            "metric": Categorical(["minkowski"]),
            "p": Integer(1, 2),
        }

    if model_name == "SVM":
        return [
            {
                "kernel": Categorical(["linear", "rbf", "sigmoid"]),
                "C": Real(0.1, 1000, prior="log-uniform"),
                "gamma": Categorical(["scale", 0.1, 0.01, 0.001, 0.0001]),
                "tol": Real(1e-6, 1e-2, prior="log-uniform"),
            },
            {
                "kernel": Categorical(["poly"]),
                "C": Real(0.1, 1000, prior="log-uniform"),
                "gamma": Categorical(["scale", 0.1, 0.01, 0.001]),
                "degree": Integer(2, 5),
                "tol": Real(1e-6, 1e-2, prior="log-uniform"),
            },
        ]

    if model_name == "EGB":
        return {
            "n_estimators": Integer(50, 1000),
            "learning_rate": Real(0.01, 1.0, prior="log-uniform"),
            "max_depth": Integer(3, 20),
            "max_features": Categorical(["sqrt", "log2", None]),
            "min_samples_leaf": Integer(1, 4),
            "min_samples_split": Integer(2, 4),
            "loss": Categorical(["log_loss"]),
            "min_weight_fraction_leaf": Real(0.0, 0.5),
        }

    raise ValueError(f"Unsupported traditional HPO classifier '{model_name}'.")


def _build_traditional_hpo_estimator(
    model: BaseModel,
    random_seed: int,
    class_weight: Optional[str],
) -> Any:
    """Build the unfitted estimator used by BayesSearchCV for traditional HPO."""
    if not hasattr(model, "_build_sklearn_estimator"):
        raise RuntimeError(
            f"Model {model.get_name()} does not implement _build_sklearn_estimator()."
        )

    estimator_builder = getattr(model, "_build_sklearn_estimator")
    return estimator_builder(
        class_weight=class_weight,
        random_state=random_seed,
        params=None,
    )



def _is_hpo_supported_advanced_model(model: BaseModel) -> bool:
    """Return True if model exposes the new per-model HPO API.

    The refactor removes the old global search-space mapping and requires models to
    implement build_hpo_search_space(trial, X_train, random_seed).
    """
    return (
        not _is_traditional_model(model)
        and hasattr(model, "train")
        and hasattr(model, "evaluate")
        and hasattr(model, "build_hpo_search_space")
        and callable(getattr(model, "build_hpo_search_space"))
    )


def _split_train_for_hpo(
    X_train: np.ndarray, y_train: np.ndarray, train_fraction: float, random_seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stratify = y_train if len(np.unique(y_train)) > 1 else None
    try:
        X_subtrain, X_holdout, y_subtrain, y_holdout = train_test_split(
            X_train,
            y_train,
            train_size=train_fraction,
            random_state=random_seed,
            stratify=stratify,
        )
    except ValueError:
        X_subtrain, X_holdout, y_subtrain, y_holdout = X_train, X_train, y_train, y_train
    return X_subtrain, X_holdout, y_subtrain, y_holdout


def _run_advanced_model_hpo(
    model: BaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    hpo_config: Dict[str, Any],
    random_seed: int,
) -> Dict[str, Any]:
    import optuna

    # Require the model to provide a search-space builder
    if not hasattr(model, "build_hpo_search_space") or not callable(getattr(model, "build_hpo_search_space")):
        raise RuntimeError(
            f"Model {model.get_name()} does not implement build_hpo_search_space(trial, X_train, random_seed)"
        )
    search_space_builder = getattr(model, "build_hpo_search_space")
    X_subtrain, X_holdout, y_subtrain, y_holdout = _split_train_for_hpo(
        X_train, y_train, hpo_config["train_fraction"], random_seed
    )
    metric_name = hpo_config["metric"]

    sampler_seed = (
        random_seed if hpo_config.get("sampler_seed") is None else int(hpo_config["sampler_seed"])
    )
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(hpo_config.get("pruner_startup_trials", 3)),
        n_warmup_steps=int(hpo_config.get("pruner_warmup_steps", 0)),
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: Any) -> float:
        params = search_space_builder(trial, X_subtrain, random_seed)
        params = dict(params) if isinstance(params, dict) else {}
        # Inject flags from provided hpo_config
        params["use_sampler"] = bool(hpo_config["use_sampler"])
        params["final_early_stop"] = bool(hpo_config["final_early_stop"])
        params["objective_var"] = hpo_config.get("metric")

        model.train(X_subtrain, y_subtrain, params=params)
        metrics = model.evaluate(X_holdout, y_holdout)
        if metric_name in metrics:
            return float(metrics[metric_name])
        if "f1" in metrics:
            return float(metrics["f1"])
        return float(metrics.get("accuracy", 0.0))

    study.optimize(
        objective,
        n_trials=int(hpo_config["n_trials"]),
        timeout=hpo_config["timeout"],
        catch=(RuntimeError, ValueError),
        show_progress_bar=False,
    )

    if not study.trials:
        return {
            "best_params": {},
            "summary": {
                "best_trial": -1,
                "best_value": 0.0,
                "metric": metric_name,
                "n_trials": 0,
                "best_params": {},
            },
        }

    try:
        best_params = dict(study.best_params) if study.best_params else {}
        best_trial_number = int(study.best_trial.number)
        best_value = float(study.best_value)
    except Exception:
        best_params = {}
        best_trial_number = -1
        best_value = 0.0
    summary = {
        "best_trial": best_trial_number,
        "best_value": best_value,
        "metric": metric_name,
        "n_trials": int(hpo_config["n_trials"]),
        "best_params": best_params,
    }
    return {"best_params": best_params, "summary": summary}


def _is_traditional_model(model: Any) -> bool:
    """Detect if a model follows the legacy 'classify_traditional' contract.

    Prefer the explicit is_traditional flag on the instance. Relying on hasattr
   ('classify_traditional') is unreliable because BaseModel exposes that method.
    """
    return bool(getattr(model, "is_traditional", False))


def _extract_hyperparameters_from_model(model: BaseModel) -> Dict[str, Any]:
    """Extract best hyperparameters from a trained model.
    
    Tries multiple strategies to get params:
    1. model.best_params attribute
    2. model.model.get_params() (sklearn models)
    3. Empty dict if neither available
    
    Returns:
        Dict of hyperparameters or empty dict if not found
    """
    best_params_to_save = {}
    
    if hasattr(model, "best_params"):
        best_params = getattr(model, "best_params", {})
        if best_params:
            best_params_to_save = dict(best_params) if isinstance(best_params, dict) else {}
    
    # If best_params empty, try to extract from the trained model object
    if not best_params_to_save and hasattr(model, "model") and model.model is not None:
        try:
            best_params_to_save = model.model.get_params()
        except Exception as e:
            logger.debug(f"Could not extract params from trained model: {e}")
    
    return best_params_to_save


def _run_advanced_model_cv_fold(
    model: BaseModel,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    features: List[str],
    kfold_id: int,
    class_weight: Optional[str] = None,
    hpo_config: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
    fold_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a single fold of cross-validation for an advanced (modern) model.
    
    Uses model.train() and model.evaluate() interface.
    Accepts pre-split data instead of calling pipeline.get_data() to avoid redundant I/O.
    """
    logger.info(f"Running advanced CV fold {kfold_id} for model {model.get_name()}")

    best_params: Dict[str, Any] = {}
    if (
        hpo_config
        and hpo_config.get("enabled", False)
        and int(hpo_config.get("n_trials", 0)) > 0
        and _is_hpo_supported_advanced_model(model)
    ):
        hpo_result = _run_advanced_model_hpo(
            model=model,
            X_train=X_train,
            y_train=y_train,
            hpo_config=hpo_config,
            random_seed=random_seed,
        )
        best_params = hpo_result["best_params"]
        if fold_dir is not None:
            hpo_summary_path = fold_dir / "hpo_summary.json"
            with open(hpo_summary_path, "w", encoding="utf-8") as f:
                json.dump(hpo_result["summary"], f, indent=2)
            logger.info(f"Saved fold {kfold_id} HPO summary to {hpo_summary_path}")

    # Train model with tuned params when available
    model.train(X_train, y_train, params=best_params or None)

    # Evaluate model - REQUIRE that advanced models return specificity and g_mean
    raw_metrics = model.evaluate(X_val, y_val)
    if not isinstance(raw_metrics, dict):
        raise RuntimeError(
            f"Advanced model {model.get_name()} must return a dict from evaluate() (fold {kfold_id})."
        )
    metrics = dict(raw_metrics)

    # Strict contract: advanced models must include specificity and g_mean
    if "specificity" not in metrics or "g_mean" not in metrics:
        raise RuntimeError(
            f"Advanced model {model.get_name()} must include 'specificity' and 'g_mean' in evaluate() result (fold {kfold_id})."
        )

    # Extract hyperparameters and build result
    if not best_params:
        best_params = _extract_hyperparameters_from_model(model)
    fold_result = _build_fold_result(
        kfold_id, metrics, len(X_train), len(X_val), features, best_params
    )
    
    return fold_result


def _run_traditional_model_cv_fold(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    kfold_id: int,
    num_splits: int,
    random_seed: int,
    class_weight: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a single fold of cross-validation for a traditional model to obtain
    accuracy, precision, recall, f1, specificity, and g_mean metrics.
    
    Args:
        model: The traditional model to evaluate
        X: Full feature matrix (pre-loaded)
        y: Full label vector (pre-loaded)
        kfold_id: Which fold to extract
        num_splits: Total number of splits
        random_seed: Seed for reproducibility
        class_weight: Class weighting strategy
        feature_names: Raw feature names aligned with X columns
    """
    logger.info(
        f"Running traditional CV fold {kfold_id} for model {model.name}"
    )

    # Error checking
    if feature_names is None or len(feature_names) == 0:
        raise ValueError("feature_names must be provided for traditional CV fold.")

    # Do fold splitting using stratified_kfold_split on pre-loaded data
    X_train, X_test, y_train, y_test = stratified_kfold_split(
        X, y, num_splits, kfold_id, random_state = random_seed
    )

    logger.info(
        "Running traditional fold-local LASSO feature selection for fold %s on %s features",
        kfold_id,
        X_train.shape[1],
    )

    X_train, X_test, selected_features = _run_traditional_lasso_feature_selection(
        X_train = X_train,
        X_test = X_test,
        y_train = y_train,
        feature_names = feature_names,
        random_seed = random_seed,
    )

    logger.info(
        "Running traditional SMOTE resampling for fold %s after LASSO reduced features to %s",
        kfold_id,
        len(selected_features),
    )

    X_train, y_train = _run_traditional_smote_resampling(
        X_train=X_train,
        y_train=y_train,
        random_seed=random_seed,
    )

    logger.info(
        "Running traditional model HPO for fold %s with class_weight = %s",
        kfold_id,
        class_weight
    )

    hpo_result, search = _run_traditional_model_hpo(
        model=model,
        X_train=X_train,
        y_train=y_train,
        random_seed=random_seed,
        class_weight=class_weight,
    )

    logger.info(
        "Running traditional model evaluation for fold %s",
        kfold_id
    )
    
    fold_performance_summary = _run_traditional_model_evaluation(
        search = search,
        X_test = X_test,
        y_test = y_test
    )

    # Build fold result dictionary
    fold_result = {
        "fold": kfold_id,
        "performance": fold_performance_summary,
        "n_train": len(X_train),
        "n_val": len(X_test),
        "selected_features": selected_features,
        "best_params": search.best_params_,
    }

    return fold_result, search

def _run_traditional_lasso_feature_selection(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    random_seed: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Find optimal lasso alpha parameter and fits a lasso model to determine
    most important features. This should only see the 'training' data.
    """

    search = BayesSearchCV(
        estimator = Lasso(random_state = random_seed),
        search_spaces = {"alpha": Real(1e-5, 100, prior = "log-uniform")},
        cv = 3,
        n_iter = 50,
        random_state = random_seed,
        verbose = 1,
    )
    search.fit(X_train, np.ravel(y_train))

    best_lasso = search.best_estimator_
    lasso_optimal_coef = np.abs(best_lasso.coef_)
    selected_features_indices = np.where(lasso_optimal_coef != 0)[0]
    selected_features = np.array(feature_names)[selected_features_indices]

    return X_train[:, selected_features_indices], X_test[:, selected_features_indices], selected_features

def _run_traditional_smote_resampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE resampling to the training data to address class imbalance."""
    smote_model = SMOTE(random_state = random_seed, k_neighbors = 7)
    return smote_model.fit_resample(X_train, y_train)

def _run_traditional_model_hpo(
    model: BaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int,
    class_weight: Optional[str] = None,
) -> Dict[str, Any]:
    """Run hyperparameter optimization for a traditional model using BayesSearchCV."""

    estimator = model.model_object
    estimator.set_params({ "random_state": random_seed, "class_weight": class_weight })
    search = BayesSearchCV(
        estimator = estimator,
        search_spaces = model.hpo_search_space,
        n_iter = 30,
        cv = 3,
        scoring = "f1",
        random_state = random_seed,
        verbose = 1,
        error_score = np.nan,
    )
    search.fit(X_train, np.ravel(y_train))

    best_params = dict(search.best_params_)
    summary = {
        "best_params": best_params,
        "best_score": float(getattr(search, "best_score_", 0.0)),
        "best_index": int(getattr(search, "best_index_", -1)),
        "n_iter": 30,
        "cv": 3,
        "scoring": "f1",
    }
    return {"best_params": best_params, "summary": summary}, search

def _run_traditional_model_evaluation(
    search: Any, # Should be a BayesSearchCV fitted search object
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    label_predictions = search.predict(X_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label = 0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "g_mean": g_mean
    }

def _build_fold_result(
    fold_idx: int,
    metrics: Dict[str, float],
    n_train: int,
    n_val: int,
    features: List[str],
    best_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a fold result dict with standard fields.
    
    Args:
        fold_idx: Fold index
        metrics: Dict of metric names to values
        n_train: Number of training samples
        n_val: Number of validation samples
        features: Selected features (optional)
        best_params: Best hyperparameters (optional)
        
    Returns:
        Fold result dict with standard structure
    """
    fold_result = {
        "fold": fold_idx,
        "metrics": metrics,
        "n_train": n_train,
        "n_val": n_val,
        "selected_features": features,
        "best_params": best_params,
    }
    
    return fold_result


def run_cross_validation(
    config: GLOCExperimentConfigParser,
    pipeline: DataPipeline,
    model_factory: ModelFactory,
    project_root_path: Path,
    # models: Optional[List[BaseModel]] = None,
    # num_splits: int = 10,
    # random_seed: int = 42,
    # class_weight: Optional[str] = None,
    # save_models: bool = True,
    # results_root: Optional[Path] = None,
    # model_type: Optional[ModelType] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run cross-validation for a list of models.
    
    Detects model type (traditional or advanced) and routes to the appropriate
    training loop. Saves per-fold metrics and aggregated summaries.
    
    Args:
        config: GLOCExperimentConfigParser instance
        pipeline: DataPipeline instance for data loading
        unused_path: Deprecated parameter, kept for backward compatibility
        models: List of BaseModel instances
        num_splits: Number of cross-validation folds
        random_seed: Random seed for deterministic splits
        class_weight: Class weighting strategy (e.g., 'balanced')
        save_models: Whether to save model artifacts per fold
        results_root: Root directory for CV results (keyword argument)
        model_type: ModelType instance for nesting results by model type (keyword argument)
        
    Returns:
        Dict mapping model names to lists of per-fold result dicts.
    """
    models = config.get_models()
    results_path = Path(project_root_path / config.get_cross_validation_save_results_folder())
    model_type = config.get_model_type()
    num_splits = config.get_cross_validation_num_splits()
    random_seed = config.get_cross_validation_random_seed()
    class_weight = config.get_cross_validation_class_weight()
    
    logger.info(
        "Starting cross-validation with %s folds and %s model(s)",
        num_splits,
        len(models),
    )

    # Validate model type and results path
    model_type_folder = model_type.get_folder_name()
    results_root_with_type = results_path / model_type_folder
    all_results = {}
    
    # Set random seed and model type for pipeline operations
    pipeline.set_random_seed(random_seed)
    pipeline.set_model_type(model_type)

    for model in models:
        # Use a dummy model to get its metadata
        model = model_factory.build_model(model)

        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-validating model: {model.name}")
        logger.info(f"{'='*60}")

        model_results = []
        model_results_dir = results_root_with_type / model.name

        logger.info(f"Loading data for model {model.name}...")

        fold_cache = None  # For advanced models: cache fold data upfront

        if model._is_traditional_model:
            # Traditional models can load the whole dataset first and then do fold splits
            X, y, feature_names = pipeline.get_data(
                model = model,
                traditional_feature_selection = "raw",
                return_feature_names = True,
            )

            logger.info(f"Loaded traditional data: X shape {X.shape}, y shape {y.shape}")
        else:
            # Advanced models: require advanced_hpo config and pre-cache fold data
            # This will raise ValueError if advanced_hpo is missing or invalid
            advanced_hpo_cfg = config.get_advanced_hpo_settings()
            # Convert normalized advanced_hpo into model-level hpo_config expected by _run_advanced_model_hpo
            # Note: Parser provides both user-facing parameters (use_sampler, final_early_stop, objective_var, trials)
            # and hardcoded Optuna-level defaults (train_fraction, timeout, sampler_seed, pruner_startup_trials, pruner_warmup_steps)
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

            fold_cache = _cache_fold_data_for_advanced_models(
                pipeline, model, num_splits, random_seed
            )
            X, y = None, None

        for fold_idx in range(num_splits):
            fold_dir = model_results_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents = True, exist_ok = True)

            if model.is_traditional_model:
                fold_result, search = _run_traditional_model_cv_fold(
                    model,
                    X,
                    y,
                    fold_idx,
                    num_splits,
                    random_seed,
                    class_weight,
                    feature_names = feature_names,
                )

                # Save fold metrics in fold folder for all model types
                fold_dir.mkdir(parents = True, exist_ok = True)
                fold_result_path = fold_dir / "fold_result.json"
                with open(fold_result_path, "wb") as f:
                    json.dump(fold_result, f)
                logger.info(f"Saved fold {fold_idx} metrics to {fold_result_path}")

                # Save BayesSearchCV model in fold folder
                fold_model_dir = fold_dir / "model.pkl"
                with open(fold_model_dir, "wb") as f:
                    joblib.dump(search, f)
                logger.info(f"Saved fold {fold_idx} BayesSearchCV object to {fold_model_dir}")


            # try:
            #     # Route based on model type
            #     if _is_traditional_model(model):
            #         # Use pre-loaded data for traditional models
            #         fold_result = _run_traditional_model_cv_fold(
            #             model,
            #             X,
            #             y,
            #             fold_idx,
            #             num_splits,
            #             random_seed,
            #             class_weight,
            #             feature_names=feature_names,
            #         )

            #     else:
            #         # Advanced model: use pre-cached fold data and required advanced_hpo config
            #         X_train, X_val, y_train, y_val, features = fold_cache[fold_idx]
            #         fold_result = _run_advanced_model_cv_fold(
            #             model,
            #             X_train,
            #             X_val,
            #             y_train,
            #             y_val,
            #             features,
            #             fold_idx,
            #             class_weight,
            #             hpo_config=model_hpo_config,
            #             random_seed=random_seed,
            #             fold_dir=fold_dir,
            #         )



            model_results.append(fold_result)

        # Compute and save aggregated metrics
        aggregated, median_fold_idx, median_f1 = _aggregate_cv_results(model_results)
        summary_path = model_results_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(aggregated, f)
        logger.info(f"Saved aggregated CV summary to {summary_path}")
        logger.info(f"Model {model.name} CV results: {aggregated}")

        # Extract and save median fold hyperparameters and selected features
        hyperparams = _extract_median_hyperparameters(median_fold_idx, median_f1, model_results)
        hyperparams_path = model_results_dir / "median_hyperparameters.json"
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f)
        logger.info(
            f"Saved median hyperparameters (fold {median_fold_idx}, F1={median_f1:.4f}) "
            f"to {hyperparams_path}"
        )

    logger.info(f"\nCross-validation complete. Results saved to {results_root_with_type}")

def _aggregate_cv_results(
    fold_results: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int, float]:
    """
    Compute mean/std summary from all metrics and compute median fold based on F1 score.

    Parameters:
        fold_results: List of dicts, each containing 'performance' dict with metrics for a fold
    
    Returns:
        Tuple of (aggregated_metrics, median_fold_idx, median_f1)
    """
    metric_keys = ("accuracy", "precision", "recall", "f1", "specificity", "g_mean")
    metric_values = {key: [] for key in metric_keys}

    # Extract metrics for each fold
    for fold_result in fold_results:
        fold_performance = fold_result["performance"]
        for key in metric_keys:
            metric_values[key].append(float(fold_performance[key]))
    
    # Find median F1 score
    f1_sorted = sorted(metric_values["f1"])
    median_idx = len(f1_sorted) // 2
    median_f1 = f1_sorted[median_idx]

    # Find fold index with median F1 (use first if tie)
    median_fold_idx = 0
    for i, fold_result in enumerate(fold_results):
        fold_f1 = float(fold_result["performance"]["f1"])
        if fold_f1 == median_f1:
            median_fold_idx = i
            break

    # Compute mean and std for each metric
    aggregated = {}
    for key in metric_keys:
        values = metric_values[key]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    # Add fold count
    aggregated["num_folds"] = len(fold_results)

    return aggregated, median_fold_idx, median_f1

def _extract_median_hyperparameters(
    median_fold_idx: int,
    median_f1: float,
    fold_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Extract hyperparameters and feature info from the trained model.
    
    Args:
        median_fold_idx: Index of the median fold
        median_f1: F1 score of the median fold
        fold_results: List of fold result dicts for extracting selected_features and best_params
        
    Returns:
        Dict with fold_id, f1_score, best_params, selected_features (if traditional model)
    """
    result = {
        "fold_id": median_fold_idx,
        "f1_score": median_f1,
        "best_params": {}
    }

    median_fold = fold_results[median_fold_idx]
    result["best_params"] = median_fold["best_params"]
    if "selected_features" in median_fold: # For traditional models
        result["selected_features"] = median_fold["selected_features"]
    
    return result


def _find_median_fold_idx(
    fold_results: List[Dict[str, Any]],
) -> Tuple[int, float]:
    """Find the fold with median F1 score.
    
    Returns:
        Tuple of (fold_index, f1_score) for the median fold.
    """
    if not fold_results:
        return 0, 0.0
    
    # Get F1 scores for each fold
    f1_scores = []
    for result in fold_results:
        metrics = result.get("metrics", {})
        f1 = metrics.get("f1", 0.0)
        f1_scores.append(f1)
    
    # Find median F1
    f1_sorted = sorted(f1_scores)
    median_idx = len(f1_sorted) // 2
    median_f1 = f1_sorted[median_idx]
    
    # Find fold index with median F1 (use first if tie)
    for i, result in enumerate(fold_results):
        metrics = result.get("metrics", {})
        if metrics.get("f1", 0.0) == median_f1:
            return i, median_f1
    
    # Fallback to first fold if no match
    return 0, f1_scores[0] if f1_scores else 0.0

def _cache_fold_data_for_advanced_models(
    pipeline: DataPipeline,
    model: BaseModel,
    num_splits: int,
    random_seed: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    """Pre-cache all fold data for an advanced model to avoid repeated pipeline calls.
    
    Instead of calling pipeline.get_data() per fold, we cache all folds upfront.
    This is more efficient as it avoids redundant data loading/splitting.
    
    Args:
        pipeline: DataPipeline instance
        model: Advanced model instance
        num_splits: Number of folds
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict mapping fold_idx to (X_train, X_val, y_train, y_val, features) tuples
    """
    fold_cache = {}
    for fold_idx in range(num_splits):
        X_train, X_val, y_train, y_val, features = pipeline.get_data(
            model=model, kfold_id=fold_idx, num_splits=num_splits
        )
        fold_cache[fold_idx] = (X_train, X_val, y_train, y_val, features)
    
    logger.info(f"Cached fold data for {model.get_name()}: {num_splits} folds")
    return fold_cache


def _get_selected_features_for_model(
    config: GLOCExperimentConfigParser,
    model: BaseModel,
    model_type: "ModelType",
) -> List[str]:
    """Try to retrieve selected_features for a model from cached median hyperparameters.
    
    Args:
        config: GLOCExperimentConfigParser instance
        model: The model to get features for
        model_type: ModelType instance for constructing cache paths
        
    Returns:
        List of selected feature names, or empty list if not available
    """
    model_name = None
    try:
        from pathlib import Path
        model_name = model.get_name()
        model_type_folder = model_type.get_folder_name()
        
        # Look for cached median_hyperparameters from previous HPO runs
        # This is an optional optimization; it's OK if they don't exist
        project_root = Path(__file__).resolve().parent.parent.parent
        cache_path = (
            project_root / "ModelSave" / "CV" / model_type_folder / 
            f"median_hyperparameters_{model_name}.json"
        )
        
        if cache_path.exists():
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
                if "selected_features" in cached_data:
                    logger.debug(f"Using cached selected_features for {model_name}")
                    return cached_data["selected_features"]
    except Exception as e:
        if model_name:
            logger.debug(f"Could not retrieve cached selected_features for {model_name}: {e}")
        else:
            logger.debug(f"Could not retrieve cached selected_features: {e}")
    
    # Return empty list if not available - this is OK, we'll use an empty list for this CV run
    return []
