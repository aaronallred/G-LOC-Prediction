"""Cross-validation and advanced-model hyperparameter optimization mode for G-LOC.

This module keeps the existing traditional cross-validation contract while adding Optuna
HPO for advanced models (LogRegTS, LSTM, NAM, TCN, Trans), aligned with the legacy
`src/scripts/cross_validation_enhanced.py` orchestration pattern.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import torch
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.utils.class_weight import compute_class_weight
from skopt import BayesSearchCV
from skopt.space import Real
from torch.utils.data import DataLoader

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.advanced_experiment_utils import (
    baseline_down_select,
    train_test_split_trials,
    build_training_components,
    build_sampler,
    train_with_early_stopping,
    get_advanced_predictions_and_targets,
)
from src.models.base import BaseModel, TraditionalModel, AdvancedModel
from src.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


def _run_advanced_model_cv_fold(
        model: AdvancedModel,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        features: List[str],
        kfold_id: int,
        class_weight: Optional[str] = None,
        hpo_config: Optional[Dict[str, Any]] = None,
        random_seed: int = 42,
        fold_dir: Optional[Path] = None
) -> Dict[str, Any]:
    logger.info(f"Running advanced CV fold {kfold_id} for model {model.name}")

    model.all_features = features
    best_params: Dict[str, Any] = {}

    # 1. Hyperparameter Optimization Tuning Phase
    if hpo_config and hpo_config.get("enabled", False) and int(hpo_config.get("n_trials", 0)) > 0:
        hpo_result = _run_advanced_hpo(model, X_train, y_train, hpo_config, random_seed)
        best_params = hpo_result["best_params"]

        if fold_dir is not None:
            with open(fold_dir / "hpo_summary.json", "w", encoding="utf-8") as f:
                json.dump(hpo_result["summary"], f, indent=4)

    # 2. Fit Final Model weights
    model.train(X_train, y_train, params=best_params or None)

    # Save the trained advanced model
    if fold_dir is not None:
        model_path = fold_dir / "model.pt"
        model.save_model(str(model_path))
        logger.info(
            f"Saved fold {kfold_id} advanced model to {model_path}"
        )

    # 3. Alignment and Evaluation Outside of Class Structure
    X_ds_val, _ = baseline_down_select(X_val, model.all_features, model.best_params["baseline_method"])

    actual_labels, predicted_labels = get_advanced_predictions_and_targets(
        model=model,
        X=X_ds_val,
        y=y_val,
        sequence_length=model.best_params["sequence_length"],
        step_size=10,  # Explicit legacy execution constraint match
        batch_size=model.best_params["batch_size"]
    )

    # Calculate performance metrics via the external pure function
    metrics_out = _evaluate_model(actual_labels, predicted_labels)

    if "specificity" not in metrics_out or "g_mean" not in metrics_out:
        raise RuntimeError(f"Advanced model execution output missing contract elements.")

    return _build_fold_result(kfold_id, metrics_out, len(X_train), len(X_val), best_params)


def _run_advanced_hpo(
        model: AdvancedModel,
        X: np.ndarray,
        y: np.ndarray,
        hpo_config: Dict[str, Any],
        random_seed: int,
) -> Dict[str, Any]:
    """Run Optuna HPO for an advanced model and return best params + summary."""
    search_space_builder = model.get_hpo_search_space()
    metric_name = hpo_config.get("metric", "f1")

    class_weight_strategy = hpo_config.get("class_weight", "balanced")

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = search_space_builder(trial, X, random_seed)
        params["objective_var"] = hpo_config.get("metric")

        X_ds, _ = baseline_down_select(X, model.all_features, params["baseline_method"])
        train_dataset, val_dataset, train_windows_tensor, _, _, val_windows_labels = train_test_split_trials(
            X=X_ds,
            Y=y,
            window_size=params["sequence_length"],
            step_size=params["step_size"],
            test_ratio=0.2,
            random_state=random_seed,
            end_label=True,
        )

        trial_net = model._build_model(params, input_dim=train_windows_tensor.shape[2]).to(model.device)

        class_weights = torch.tensor(
            compute_class_weight(
                class_weight_strategy, classes=np.array([0, 1]), y=y
            ),
            dtype=torch.float,
        )

        criterion, optimizer = build_training_components(trial_net, class_weights, params, model.device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            sampler=(
                build_sampler(train_dataset.tensors[1], class_weights)
                if hpo_config["use_sampler"]
                else None
            ),
        )
        val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

        best_state, _, _ = train_with_early_stopping(
            model=trial_net,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=model.device,
            threshold=params["threshold"],
            objective_var=params["objective_var"],
        )
        if best_state:
            trial_net.load_state_dict(best_state)

        trial_net.eval()
        trial_preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                out = trial_net(xb.to(model.device))
                trial_preds.extend((out.reshape(-1) >= params["threshold"]).float().cpu().numpy())

        return float(metrics.f1_score(val_windows_labels.numpy(), np.array(trial_preds)))

    study.optimize(objective, n_trials=int(hpo_config["n_trials"]), catch=(RuntimeError, ValueError))
    model.best_params = dict(study.best_params)
    return {
        "best_params": model.best_params,
        "summary": {
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "metric": metric_name,
        },
    }



def _run_traditional_model_cv_fold(
        model: TraditionalModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fold_idx: int,
        random_seed: int,
        class_weight: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a single fold of cross-validation for a traditional model.

    X_train, X_test, y_train, y_test are already split and fold-aware
    standardized by the pipeline — no stratified_kfold_split call needed.
    """
    logger.info(
        f"Running traditional CV fold {fold_idx} for model {model.name}"
    )

    # Error checking
    if feature_names is None or len(feature_names) == 0:
        raise ValueError("feature_names must be provided for traditional CV fold.")
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError(f"Training fold {fold_idx} is empty")

    logger.info(
        "Running traditional fold-local LASSO feature selection for fold %s on %s features",
        fold_idx,
        X_train.shape[1],
    )

    X_train, X_test, selected_features = _lasso_feature_selection(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        feature_names=feature_names,
        random_seed=random_seed,
    )

    logger.info(
        "Running traditional SMOTE resampling for fold %s after LASSO reduced features to %s",
        fold_idx,
        len(selected_features),
    )

    X_train, y_train = _smote_resampling(
        X_train=X_train,
        y_train=y_train,
        random_seed=random_seed,
    )

    logger.info(
        "Running traditional model HPO for fold %s with class_weight = %s",
        fold_idx,
        class_weight
    )

    hpo_result, search = _run_traditional_hpo(model, X_train, y_train, random_seed, class_weight)

    logger.info(
        "Running traditional model evaluation for fold %s",
        fold_idx
    )

    preds = search.predict(X_test)
    fold_performance_summary = _evaluate_model(y_test, preds)

    # Build fold result dictionary
    fold_result = _build_fold_result(
        fold_idx=fold_idx,
        metrics=fold_performance_summary,
        n_train=len(X_train),
        n_val=len(X_test),
        best_params=hpo_result["best_params"],
        features=selected_features
    )

    return fold_result, search


def _lasso_feature_selection(
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
        estimator=Lasso(random_state=random_seed),
        search_spaces={"alpha": Real(1e-5, 100, prior="log-uniform")},
        cv=3,
        n_iter=50,
        random_state=random_seed,
        verbose=1,
    )
    search.fit(X_train, np.ravel(y_train))

    best_lasso = search.best_estimator_
    lasso_optimal_coef = np.abs(best_lasso.coef_)
    selected_features_indices = np.where(lasso_optimal_coef != 0)[0]
    selected_features = np.array(feature_names)[selected_features_indices].tolist()

    return X_train[:, selected_features_indices], X_test[:, selected_features_indices], selected_features


def _smote_resampling(
        X_train: np.ndarray,
        y_train: np.ndarray,
        random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE resampling to the training data to address class imbalance."""
    smote_model = SMOTE(random_state=random_seed, k_neighbors=7)
    return smote_model.fit_resample(X_train, y_train)


def _run_traditional_hpo(
        model: TraditionalModel,
        X: np.ndarray,
        y: np.ndarray,
        random_seed: int,
        class_weight: Optional[str] = None,
) -> Tuple[Dict[str, Any], Any]:
    """Run BayesSearchCV for a traditional model.

    Returns
    -------
    tuple
        (``{"best_params": ..., "summary": ...}``, fitted_searcher)
    """
    # Inject random_state / class_weight into the estimator if supported
    valid_params = model.model.get_params()
    extra = {k: v for k, v in {"random_state": random_seed, "class_weight": class_weight}.items() if k in valid_params}
    model.model.set_params(**extra)

    searcher = BayesSearchCV(
        estimator=model.model,
        search_spaces=model.get_hpo_search_space(),
        n_iter=30,
        cv=3,
        scoring="f1",
        random_state=random_seed,
        verbose=1,
        error_score=np.nan,
    )
    searcher.fit(X, np.ravel(y))
    model.searcher = searcher

    best_params = dict(searcher.best_params_)
    summary = {
        "best_params": best_params,
        "best_score": float(searcher.best_score_),
        "best_index": int(searcher.best_index_),
        "n_iter": 30,
        "cv": 3,
        "scoring": "f1",
    }
    return {"best_params": best_params, "summary": summary}, searcher


def _evaluate_model(
        actual: np.ndarray,
        preds: np.ndarray
) -> Dict[str, Any]:
    accuracy = metrics.accuracy_score(actual, preds)
    precision = metrics.precision_score(actual, preds)
    recall = metrics.recall_score(actual, preds)
    f1 = metrics.f1_score(actual, preds)
    specificity = metrics.recall_score(actual, preds, pos_label=0)
    g_mean = geometric_mean_score(actual, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "g_mean": g_mean
    }


def run_cross_validation(
        config: dict[str, Any],
        pipeline: DataPipeline,
        model_factory: ModelFactory,
        project_root_path: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run cross-validation for a list of models.
    
    Detects model type (traditional or advanced) and routes to the appropriate
    training loop. Saves per-fold metrics and aggregated summaries.
    
    Args:
        config: Loaded YAML experiment configuration mapping
        pipeline: DataPipeline instance for data loading
        model_factory: ModelFactory instance for creating models
        project_root_path: Root path of the project, used for saving results
        
    Returns:
        Dict mapping model names to lists of per-fold result dicts.
    """
    cross_validation_config = config["cross_validation"]
    models = cross_validation_config.get("models")
    results_path = Path(project_root_path / cross_validation_config["save_results_folder"])
    model_type = cross_validation_config["model_type"]
    num_splits = cross_validation_config["num_splits"]
    random_seed = cross_validation_config["random_seed"]
    class_weight = cross_validation_config.get("class_weight")

    logger.info(
        "Starting cross-validation with %s folds and %s model(s)",
        num_splits,
        len(models),
    )

    # Validate model type and results path
    model_type_folder = model_type.get_folder_name()
    results_root_with_type = results_path / model_type_folder

    # Set random seed and model type for pipeline operations
    pipeline.set_random_seed(random_seed)
    pipeline.set_model_type(model_type)

    for model in models:
        # Use a dummy model to get its metadata
        model = model_factory.create_model(model)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Cross-validating model: {model.name}")
        logger.info(f"{'=' * 60}")

        model_results = []
        model_results_dir = results_root_with_type / model.name

        logger.info(f"Loading data for model {model.name}...")

        fold_cache = None  # For advanced models: cache fold data upfront

        if model.is_traditional_model:
            # Traditional models now call get_data per fold for fold-aware standardization.
            pass  # Dispatch inside fold loop below
        else:
            # Advanced models: require advanced_hpo config and pre-cache fold data
            # This will raise ValueError if advanced_hpo is missing or invalid
            advanced_hpo_cfg = cross_validation_config["advanced_hpo"]
            # Convert normalized advanced_hpo into model-level hpo_config expected by _run_advanced_model_hpo
            # Note: Parser provides both user-facing parameters (use_sampler, final_early_stop, objective_var, trials)
            # and hardcoded Optuna-level defaults (train_fraction, timeout, sampler_seed, pruner_startup_trials, pruner_warmup_steps)
            model_hpo_config = {
                "enabled": True,
                "n_trials": int(advanced_hpo_cfg["trials"]),
                "timeout": advanced_hpo_cfg.get("timeout"),
                "metric": advanced_hpo_cfg.get("metric", str(advanced_hpo_cfg.get("objective_var", "f1")).lower()),
                "train_fraction": float(advanced_hpo_cfg.get("train_fraction", 0.8)),
                "sampler_seed": advanced_hpo_cfg.get("sampler_seed"),
                "pruner_startup_trials": advanced_hpo_cfg.get("pruner_startup_trials"),
                "pruner_warmup_steps": advanced_hpo_cfg.get("pruner_warmup_steps"),
                "use_sampler": bool(advanced_hpo_cfg["use_sampler"]),
                "final_early_stop": bool(advanced_hpo_cfg["final_early_stop"]),
                "class_weight": class_weight or "balanced",
            }

            fold_cache = _cache_fold_data_for_advanced_models(
                pipeline, model, num_splits, random_seed
            )
            X, y = None, None

        for fold_idx in range(num_splits):
            fold_dir = model_results_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            if model.is_traditional_model:
                X_train, X_test, y_train, y_test, feature_names = pipeline.get_data(
                    model=model,
                    kfold_id=fold_idx,
                    num_splits=num_splits,
                    traditional_feature_selection="raw",
                    return_feature_names=True,
                )

                fold_result, search = _run_traditional_model_cv_fold(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    fold_idx,
                    random_seed,
                    class_weight,
                    feature_names=feature_names,
                )

                # Save BayesSearchCV model in fold folder
                fold_model_dir = fold_dir / "model.pkl"
                with open(fold_model_dir, "wb") as f:
                    joblib.dump(search, f)
                logger.info(f"Saved fold {fold_idx} BayesSearchCV object to {fold_model_dir}")
            else:
                X_train, X_val, y_train, y_val, features = fold_cache[fold_idx]

                # Expose feature list to the wrapper for baseline down-selection
                model.all_features = features
                model.hpo_config = model_hpo_config

                fold_result = _run_advanced_model_cv_fold(
                    model=model,
                    X_train=X_train,
                    X_val=X_val,
                    y_train=y_train,
                    y_val=y_val,
                    features=features,
                    kfold_id=fold_idx,
                    class_weight=class_weight,
                    hpo_config=model_hpo_config,
                    random_seed=random_seed,
                    fold_dir=fold_dir,
                )

            # Save fold metrics in fold folder
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_result_path = fold_dir / "fold_result.json"
            with open(fold_result_path, "w") as f:
                json.dump(fold_result, f, indent=4)
            logger.info(f"Saved fold {fold_idx} metrics to {fold_result_path}")

            model_results.append(fold_result)

        # Compute and save aggregated metrics
        aggregated, median_fold_idx, median_f1 = _aggregate_cv_results(model_results)
        summary_path = model_results_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(aggregated, f, indent=4)
        logger.info(f"Saved aggregated CV summary to {summary_path}")
        logger.info(f"Model {model.name} CV results: {aggregated}")

        # Extract and save median fold hyperparameters and selected features
        hyperparams = _extract_median_hyperparameters(median_fold_idx, median_f1, model_results)
        hyperparams_path = model_results_dir / "median_hyperparameters.json"
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=4)
        logger.info(
            f"Saved median hyperparameters (fold {median_fold_idx}, F1={median_f1:.4f}) "
            f"to {hyperparams_path}"
        )

    logger.info(f"\nCross-validation complete. Results saved to {results_root_with_type}")


def _build_fold_result(
        fold_idx: int,
        metrics: Dict[str, float],
        n_train: int,
        n_val: int,
        best_params: Dict[str, Any],
        features: Optional[List[str]] = None
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
        "best_params": best_params,
    }

    if features is not None:
        fold_result["selected_features"] = features

    return fold_result


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
        fold_performance = fold_result["metrics"]
        for key in metric_keys:
            metric_values[key].append(float(fold_performance[key]))

    # Find median F1 score
    f1_sorted = sorted(metric_values["f1"])
    median_idx = len(f1_sorted) // 2
    median_f1 = f1_sorted[median_idx]

    # Find fold index with median F1 (use first if tie)
    median_fold_idx = 0
    for i, fold_result in enumerate(fold_results):
        fold_f1 = float(fold_result["metrics"]["f1"])
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
    if "selected_features" in median_fold:  # For traditional models
        result["selected_features"] = median_fold["selected_features"]

    return result


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

    logger.info(f"Cached fold data for {model.name}: {num_splits} folds")
    return fold_cache
