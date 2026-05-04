"""
Cross-validation mode for G-LOC Prediction.

Supports both traditional (legacy `classify_traditional` contract) and advanced (modern
train/evaluate contract) models, as well as deep-learning models via an adapter pattern.
Does not depend on src/scripts/; instead uses DataPipeline and config-driven workflows.

Architecture:
- CrossValidator: main orchestrator that handles fold generation, model detection, and
  per-fold training/evaluation loops.
- Model type detection: at runtime, determines if a model is traditional, advanced, or
  DL-adapted based on attributes and methods.
- Result storage: saves per-fold metrics to Results/CrossValidation/<model_name>/fold_<i>/
  and aggregated summaries.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.GLOC_experiment_config_parser import GLOCExperimentConfigParser
from src.model_type import ModelType
from src.models.base import BaseModel
from src.models.dl_adapter import DLModelAdapter
from src.traditional_experiment_utils import stratified_kfold_split

logger = logging.getLogger(__name__)


class SyntheticDLAdapter(DLModelAdapter):
    """Synthetic DL adapter for testing (no heavy framework dependency).
    
    Trains a simple logistic regression model to simulate DL training.
    Used for testing and demonstration purposes only.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> None:
        from sklearn.linear_model import LogisticRegression

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = LogisticRegression(
            max_iter=config.get("max_iter", 1000),
            random_state=config.get("random_seed", 42),
        )
        self.model.fit(X_train_scaled, y_train)
        logger.info(
            f"SyntheticDLAdapter trained on {len(X_train)} samples; val accuracy: "
            f"{self.model.score(self.scaler.transform(X_val), y_val):.3f}"
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained; call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        model_path = Path(path) / "synthetic_dl_adapter.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"SyntheticDLAdapter saved to {path}")

    def get_name(self) -> str:
        return "SyntheticDL"


def _is_traditional_model(model: Any) -> bool:
    """Detect if a model follows the legacy 'classify_traditional' contract."""
    return (
        getattr(model, "is_traditional", False)
        or hasattr(model, "classify_traditional")
    )


def _is_dl_adapter(model: Any) -> bool:
    """Detect if a model is a DL adapter."""
    return isinstance(model, DLModelAdapter)


def _compute_metrics_from_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute standard metrics (accuracy, precision, recall, F1, etc.)."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        specificity_score,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity_score(y_true, y_pred)),
    }

    if y_pred_proba is not None:
        # G-mean as sqrt(sensitivity * specificity)
        sensitivity = metrics["recall"]
        specificity = metrics["specificity"]
        metrics["g_mean"] = float(np.sqrt(max(0, sensitivity * specificity)))

    return metrics


def _run_advanced_model_cv_fold(
    model: BaseModel,
    pipeline: DataPipeline,
    kfold_id: int,
    random_seed: int,
    class_weight: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single fold of cross-validation for an advanced (modern) model.
    
    Uses model.train() and model.evaluate() interface.
    """
    logger.info(f"Running advanced CV fold {kfold_id} for model {model.get_name()}")

    # Get fold data from pipeline
    X_train, X_val, y_train, y_val, features = pipeline.get_data(
        model=model, kfold_id=kfold_id
    )

    # Train model
    model.train(X_train, y_train, params=None)

    # Evaluate model
    metrics = model.evaluate(X_val, y_val)

    fold_result = {
        "fold": kfold_id,
        "metrics": metrics,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "selected_features": features if features else [],
    }
    
    # Store best_params from model if available
    best_params_to_save = {}
    
    if hasattr(model, "best_params"):
        best_params = getattr(model, "best_params", {})
        if best_params:
            best_params_to_save = dict(best_params) if isinstance(best_params, dict) else {}
    
    # If best_params empty, try to extract from the trained model object
    if not best_params_to_save and hasattr(model, "model") and model.model is not None:
        # Extract hyperparameters from the trained sklearn model
        try:
            best_params_to_save = model.model.get_params()
        except Exception as e:
            logger.debug(f"Could not extract params from trained model: {e}")
    
    if best_params_to_save:
        fold_result["best_params"] = best_params_to_save
    
    return fold_result


def _run_traditional_model_cv_fold(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    kfold_id: int,
    num_splits: int,
    random_seed: int,
    class_weight: Optional[str] = None,
    selected_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run a single fold of cross-validation for a traditional model.
    
    Uses model.classify_traditional() interface which returns a legacy tuple:
    (accuracy, precision, recall, f1, specificity, g_mean).
    
    Unlike advanced models, traditional models expect the full dataset and do their own
    fold splitting via stratified_kfold_split.
    
    Args:
        model: The traditional model to evaluate
        X: Full feature matrix (pre-loaded)
        y: Full label vector (pre-loaded)
        kfold_id: Which fold to extract
        num_splits: Total number of splits
        random_seed: Seed for reproducibility
        class_weight: Class weighting strategy
        selected_features: List of selected feature names
    """
    logger.info(
        f"Running traditional CV fold {kfold_id} for model {model.get_name()}"
    )

    # Do fold splitting using stratified_kfold_split on pre-loaded data
    X_train, X_test, y_train, y_test = stratified_kfold_split(
        y, X, num_splits, kfold_id, random_state=random_seed
    )

    # Call legacy classify_traditional interface
    # Legacy contract: returns (accuracy, precision, recall, f1, specificity, g_mean)
    # Signature: classify_traditional(x_train, x_test, y_train, y_test, class_weight_imb, 
    #                                  random_state, save_folder, model_name, retrain,
    #                                  temporal=False, best_params=None)
    result = model.classify_traditional(
        x_train=X_train,
        x_test=X_test,
        y_train=y_train,
        y_test=y_test,
        class_weight_imb=class_weight,
        random_state=random_seed,
        save_folder=None,
        model_name=None,
        retrain=True,
        temporal=False,
        best_params=None,
    )

    # Parse legacy tuple - handle variable-length returns
    if isinstance(result, tuple):
        # Legacy tuple contract (variable length, but first 6 are always the same):
        # (accuracy, precision, recall, f1, [optional: tree_depth/other], specificity, g_mean)
        if len(result) == 6:
            # Standard 6-tuple: (accuracy, precision, recall, f1, specificity, g_mean)
            accuracy, precision, recall, f1, specificity, g_mean = result
        elif len(result) == 7:
            # 7-tuple with extra field: (accuracy, precision, recall, f1, extra, specificity, g_mean)
            accuracy, precision, recall, f1, _, specificity, g_mean = result
        else:
            # Unknown tuple format; try to extract first 4 mandatory + last 2 metrics
            accuracy = result[0] if len(result) > 0 else 0
            precision = result[1] if len(result) > 1 else 0
            recall = result[2] if len(result) > 2 else 0
            f1 = result[3] if len(result) > 3 else 0
            specificity = result[-2] if len(result) > 1 else 0
            g_mean = result[-1] if len(result) > 0 else 0
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "g_mean": float(g_mean),
        }
    else:
        # If result is already a dict, use as-is
        metrics = result

    fold_result = {
        "fold": kfold_id,
        "metrics": metrics,
        "n_train": len(X_train),
        "n_val": len(X_test),
    }
    
    # Store best_params from model if available
    # Try to extract actual parameters used during training
    best_params_to_save = {}
    
    if hasattr(model, "best_params"):
        best_params = getattr(model, "best_params", {})
        if best_params:
            best_params_to_save = dict(best_params) if isinstance(best_params, dict) else {}
    
    # If best_params empty, try to extract from the trained model object
    if not best_params_to_save and hasattr(model, "model") and model.model is not None:
        # Extract hyperparameters from the trained sklearn model
        try:
            best_params_to_save = model.model.get_params()
        except Exception as e:
            logger.debug(f"Could not extract params from trained model: {e}")
    
    if best_params_to_save:
        fold_result["best_params"] = best_params_to_save
    
    if selected_features is not None:
        fold_result["selected_features"] = selected_features
    
    return fold_result


def _run_dl_model_cv_fold(
    model: DLModelAdapter,
    pipeline: DataPipeline,
    kfold_id: int,
    random_seed: int,
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a single fold of cross-validation for a DL-adapted model."""
    logger.info(f"Running DL CV fold {kfold_id} for model {model.get_name()}")

    # Get fold data from pipeline
    X_train, X_val, y_train, y_val, features = pipeline.get_data(
        model=None, kfold_id=kfold_id
    )

    # Train DL model
    model.fit(X_train, y_train, X_val, y_val, model_config)

    # Get predictions and compute metrics
    y_pred_proba = model.predict_proba(X_val)
    y_pred = np.argmax(y_pred_proba, axis=1)

    metrics = _compute_metrics_from_predictions(y_val, y_pred, y_pred_proba)

    return {
        "fold": kfold_id,
        "metrics": metrics,
        "n_train": len(X_train),
        "n_val": len(X_val),
    }


def run_cross_validation(
    config: GLOCExperimentConfigParser,
    pipeline: DataPipeline,
    unused_path: Optional[Path] = None,
    models: Optional[List[BaseModel]] = None,
    num_splits: int = 5,
    random_seed: int = 42,
    class_weight: Optional[str] = None,
    support_deep_learning: bool = False,
    save_models: bool = True,
    results_root: Optional[Path] = None,
    model_type: Optional[ModelType] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run cross-validation for a list of models.
    
    Detects model type (traditional, advanced, or DL-adapted) and routes to appropriate
    training loop. Saves per-fold metrics and aggregated summaries.
    
    Args:
        config: GLOCExperimentConfigParser instance
        pipeline: DataPipeline instance for data loading
        unused_path: Deprecated parameter, kept for backward compatibility
        models: List of BaseModel instances or DLModelAdapter instances
        num_splits: Number of cross-validation folds
        random_seed: Random seed for deterministic splits
        class_weight: Class weighting strategy (e.g., 'balanced')
        support_deep_learning: Whether to support DL models
        save_models: Whether to save model artifacts per fold
        results_root: Root directory for CV results (keyword argument)
        model_type: ModelType instance for nesting results by model type (keyword argument)
        
    Returns:
        Dict mapping model names to lists of per-fold result dicts.
    """
    if models is None:
        models = []
    if results_root is None:
        results_root = Path("Results/CrossValidation")
    if model_type is None:
        model_type = config.get_model_type()
    
    logger.info(
        f"Starting cross-validation with {num_splits} folds and {len(models)} model(s)"
    )

    results_root = Path(results_root)
    model_type_folder = model_type.get_folder_name()
    results_root_with_type = results_root / model_type_folder
    all_results = {}

    for model in models:
        model_name = model.get_name() if hasattr(model, "get_name") else str(model)
        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-validating model: {model_name}")
        logger.info(f"{'='*60}")

        model_results = []
        model_results_dir = results_root_with_type / model_name

        # IMPORTANT: Load data ONCE per model to avoid reprocessing for each fold
        logger.info(f"Loading data for model {model_name}...")
        
        if _is_traditional_model(model):
            # Load full dataset once for traditional models
            X, y = pipeline.get_data(model=model)
            logger.info(f"Loaded traditional data: X shape {X.shape}, y shape {y.shape}")
        elif _is_dl_adapter(model):
            # For DL models, load data appropriately (may need adaptation based on DL framework)
            X, y = pipeline.get_data(model=model)
            logger.info(f"Loaded DL data: X shape {X.shape}, y shape {y.shape}")
        else:
            # Advanced models: data will be loaded per-fold by pipeline
            X, y = None, None

        for fold_idx in range(num_splits):
            fold_dir = model_results_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Route based on model type
                if _is_dl_adapter(model):
                    if not support_deep_learning:
                        logger.warning(
                            f"DL model {model_name} requires support_deep_learning=True; skipping"
                        )
                        continue

                    fold_result = _run_dl_model_cv_fold(
                        model, pipeline, fold_idx, random_seed, {}
                    )

                elif _is_traditional_model(model):
                    # Use pre-loaded data for traditional models
                    # Get selected_features from config/cache
                    selected_features = _get_selected_features_for_model(config, model)
                    fold_result = _run_traditional_model_cv_fold(
                        model, X, y, fold_idx, num_splits, random_seed, class_weight,
                        selected_features=selected_features
                    )

                else:
                    # Default to advanced model
                    fold_result = _run_advanced_model_cv_fold(
                        model, pipeline, fold_idx, random_seed, class_weight
                    )

                # Save fold metrics (legacy format for traditional models)
                if _is_traditional_model(model):
                    metrics_path = model_results_dir / f"metrics_fold_{fold_idx}.pkl"
                else:
                    # Advanced/DL models use nested folder structure
                    fold_dir.mkdir(parents=True, exist_ok=True)
                    metrics_path = fold_dir / "metrics.pkl"
                
                with open(metrics_path, "wb") as f:
                    pickle.dump(fold_result, f)
                logger.info(f"Saved fold {fold_idx} metrics to {metrics_path}")

                # Optionally save model
                if save_models and hasattr(model, "save"):
                    model_save_dir = fold_dir / "model"
                    model_save_dir.mkdir(parents=True, exist_ok=True)
                    model.save(str(model_save_dir))
                    logger.info(f"Saved fold {fold_idx} model to {model_save_dir}")

                model_results.append(fold_result)

            except Exception as e:
                logger.error(f"Error in fold {fold_idx} for model {model_name}: {e}")
                raise

        # Compute and save aggregated metrics
        if model_results:
            aggregated, median_fold_idx, median_f1 = _aggregate_cv_results(model_results)
            summary_path = model_results_dir / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(aggregated, f, indent=2)
            logger.info(f"Saved aggregated CV summary to {summary_path}")
            logger.info(f"Model {model_name} CV results: {aggregated}")

            # Save median hyperparameters if enabled
            if config.get_cross_validation_save_median_hyperparameters():
                try:
                    hyperparams = _extract_median_hyperparameters(
                        model, median_fold_idx, median_f1, model_name, model_results
                    )
                    hyperparams_path = model_results_dir / "median_hyperparameters.json"
                    with open(hyperparams_path, "w") as f:
                        json.dump(hyperparams, f, indent=2)
                    logger.info(
                        f"Saved median hyperparameters (fold {median_fold_idx}, F1={median_f1:.4f}) "
                        f"to {hyperparams_path}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to extract median hyperparameters for {model_name}: {e}"
                    )

            all_results[model_name] = model_results

    logger.info(f"\nCross-validation complete. Results saved to {results_root_with_type}")
    return all_results


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


def _extract_median_hyperparameters(
    model: Any,
    median_fold_idx: int,
    median_f1: float,
    classifier_name: str,
    fold_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Extract hyperparameters and feature info from the trained model.
    
    Args:
        model: Trained model instance
        median_fold_idx: Index of the median fold
        median_f1: F1 score of the median fold
        classifier_name: Name of the classifier
        fold_results: List of fold result dicts for extracting selected_features and best_params
        
    Returns:
        Dict with fold_id, f1_score, best_params, selected_features
    """
    result = {
        "fold_id": median_fold_idx,
        "f1_score": float(median_f1),
        "best_params": {},
        "selected_features": [],
    }
    
    # Extract best_params and selected_features from fold results if available
    if fold_results and len(fold_results) > median_fold_idx:
        median_fold = fold_results[median_fold_idx]
        
        # Extract best_params from fold results
        if "best_params" in median_fold:
            result["best_params"] = median_fold["best_params"]
        
        # Extract selected_features from fold results
        if "selected_features" in median_fold:
            result["selected_features"] = median_fold["selected_features"]
    
    # Fallback: try to extract best_params from model if not in fold results
    if not result["best_params"]:
        # Try custom attribute (G-LOC models use "best_params", not "best_params_")
        if hasattr(model, "best_params"):
            best_params = getattr(model, "best_params", {})
            if best_params:
                result["best_params"] = dict(best_params) if isinstance(best_params, dict) else {}
        # Fallback to scikit-learn convention
        elif hasattr(model, "best_params_"):
            result["best_params"] = dict(model.best_params_)
    
    return result


def _aggregate_cv_results(
    fold_results: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int, float]:
    """Compute mean and std of metrics across folds.
    
    Returns:
        Tuple of (aggregated_metrics, median_fold_idx, median_f1)
    """
    aggregated = {}

    if not fold_results:
        return aggregated, 0, 0.0

    # Find median fold
    median_fold_idx, median_f1 = _find_median_fold_idx(fold_results)

    # Extract all metric keys from first fold
    first_metrics = fold_results[0].get("metrics", {})
    metric_keys = set(first_metrics.keys())

    # Compute mean and std for each metric
    for key in metric_keys:
        values = [fold["metrics"].get(key) for fold in fold_results if key in fold["metrics"]]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))

    # Add fold count
    aggregated["num_folds"] = len(fold_results)

    return aggregated, median_fold_idx, median_f1



def _get_selected_features_for_model(
    config: GLOCExperimentConfigParser,
    model: BaseModel,
) -> List[str]:
    """Try to retrieve selected_features for a model from cached median hyperparameters.
    
    Args:
        config: GLOCExperimentConfigParser instance
        model: The model to get features for
        
    Returns:
        List of selected feature names, or empty list if not available
    """
    try:
        from pathlib import Path
        model_type = config.get_model_type()
        model_type_folder = model_type.get_folder_name()
        model_name = model.get_name()
        
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
        logger.debug(f"Could not retrieve cached selected_features for {model_name}: {e}")
    
    # Return empty list if not available - this is OK, we'll use an empty list for this CV run
    return []
