import joblib
import logging
import os
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.base import BaseModel, ModelInitStrategy

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest classifier wrapper used by the traditional pipeline."""

    TRADITIONAL_HYPERPARAMETERS = {
        "baseline_window": 18.75,
        "window_size": 7.5,
        "stride": 0.25,
        "feature_reduction_type": "none",
        "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
        "imbalance_type": "none",
        "impute_type": 1,
        "n_neighbors": 3,
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.is_traditional = True

    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] | None = None) -> None:
        """Train a random forest model with provided or default parameters."""
        if params is None:
            params = self.best_params if self.best_params else self._get_default_params()
        params = dict(params)
        params.setdefault("n_jobs", -1)
        self.model = RandomForestClassifier(**params)
        self.model.fit(X, y)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute standard binary classification metrics."""
        if self.model is None:
            return {}

        predictions = self.model.predict(X)
        return {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, zero_division=0),
            "recall": recall_score(y, predictions, zero_division=0),
            "f1": f1_score(y, predictions, zero_division=0),
        }

    def save(self, path: str) -> None:
        """Persist model and metadata artifacts."""
        if self.model is not None:
            joblib.dump(self.model, f"{path}/random_forest_model.pkl")

        metadata = {
            "best_params": self.best_params,
            "split_info": self.split_info,
            "config": self.config,
        }
        joblib.dump(metadata, f"{path}/random_forest_metadata.pkl")

    def load(self, path: str) -> None:
        """Load model and metadata artifacts from disk."""
        self.model = joblib.load(f"{path}/random_forest_model.pkl")
        metadata = joblib.load(f"{path}/random_forest_metadata.pkl")
        self.best_params = metadata.get("best_params", {})
        self.split_info = metadata.get("split_info", {})

    def _get_default_params(self) -> Dict[str, Any]:
        """Return default parameters for random forest training."""
        return {
            "random_state": self.config.get("random_state", 42),
            "n_estimators": self.config.get("n_estimators", 200),
            "class_weight": self.config.get("class_weight", None),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input samples."""
        if self.model is None:
            return np.array([])
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for input samples."""
        if self.model is None:
            return np.array([])
        return self.model.predict_proba(X)

    def get_name(self) -> str:
        """Return classifier key used by traditional hyperparameter lookup."""
        return "RF"

    def _build_sklearn_estimator(
        self,
        class_weight: str = None,
        random_state: int = None,
        params: Dict[str, Any] = None,
    ):
        """Build an unfitted RandomForestClassifier with the given parameters."""
        if params is not None:
            params_copy = dict(params)
            # Ensure n_jobs is set for parallel processing
            params_copy.setdefault("n_jobs", -1)
            if random_state is not None and "random_state" not in params_copy:
                params_copy["random_state"] = random_state
            if class_weight is not None and "class_weight" not in params_copy:
                params_copy["class_weight"] = class_weight
            return RandomForestClassifier(**params_copy)
        
        # Build from class_weight and random_state
        return RandomForestClassifier(
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

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
        strategy: ModelInitStrategy = ModelInitStrategy.RETRAIN_WITH_DEFAULTS,
        best_params=None,
    ):
        """Return legacy-compatible metric tuple including RF tree depths.
        
        Args:
            strategy: ModelInitStrategy enum specifying initialization behavior.
            best_params: Hyperparameters dict (required for RETRAIN_WITH_CONFIG_PARAMS).
        """
        estimator = self._initialize_model_for_classification(
            strategy=strategy,
            x_train=x_train,
            y_train=y_train,
            class_weight_imb=class_weight_imb,
            random_state=random_state,
            save_folder=save_folder,
            model_name=model_name,
            best_params=best_params,
        )

        self.model = estimator
        predictions = estimator.predict(x_test)
        accuracy, precision, recall, f1, specificity, g_mean = self._legacy_binary_metrics(
            y_test,
            predictions,
        )
        self._print_legacy_metrics("Random Forest Performance Metrics:", (accuracy, precision, recall, f1, specificity, g_mean))
        tree_depth = [tree.get_depth() for tree in estimator.estimators_]
        return accuracy, precision, recall, f1, tree_depth, specificity, g_mean
