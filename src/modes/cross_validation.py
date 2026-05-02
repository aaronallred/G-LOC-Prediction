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
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.GLOC_experiment_config_parser import GLOCExperimentConfigParser
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

    return {
        "fold": kfold_id,
        "metrics": metrics,
        "n_train": len(X_train),
        "n_val": len(X_val),
    }


def _run_traditional_model_cv_fold(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    kfold_id: int,
    num_splits: int,
    random_seed: int,
    class_weight: Optional[str] = None,
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

    return {
        "fold": kfold_id,
        "metrics": metrics,
        "n_train": len(X_train),
        "n_val": len(X_test),
    }


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
        
    Returns:
        Dict mapping model names to lists of per-fold result dicts.
    """
    if models is None:
        models = []
    if results_root is None:
        results_root = Path("Results/CrossValidation")
    logger.info(
        f"Starting cross-validation with {num_splits} folds and {len(models)} model(s)"
    )

    results_root = Path(results_root)
    all_results = {}

    for model in models:
        model_name = model.get_name() if hasattr(model, "get_name") else str(model)
        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-validating model: {model_name}")
        logger.info(f"{'='*60}")

        model_results = []
        model_results_dir = results_root / model_name

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
                    fold_result = _run_traditional_model_cv_fold(
                        model, X, y, fold_idx, num_splits, random_seed, class_weight
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
            aggregated = _aggregate_cv_results(model_results)
            summary_path = model_results_dir / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(aggregated, f, indent=2)
            logger.info(f"Saved aggregated CV summary to {summary_path}")
            logger.info(f"Model {model_name} CV results: {aggregated}")

            all_results[model_name] = model_results

    logger.info(f"\nCross-validation complete. Results saved to {results_root}")
    return all_results


def _aggregate_cv_results(
    fold_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute mean and std of metrics across folds."""
    aggregated = {}

    if not fold_results:
        return aggregated

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

    return aggregated
