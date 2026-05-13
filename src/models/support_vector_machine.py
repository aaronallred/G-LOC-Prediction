import joblib
import os
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC

from src.models.base import BaseModel, ModelInitStrategy


class SupportVectorMachineModel(BaseModel):
    """Support Vector Machine classifier wrapper."""

    TRADITIONAL_HYPERPARAMETERS = {
        "baseline_window": 32.5,
        "window_size": 15,
        "stride": 0.25,
        "feature_reduction_type": "ridge",
        "baseline_methods_to_use": ["v0", "v1", "v2"],
        "imbalance_type": "none",
        "impute_type": 1,
        "n_neighbors": 3,
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.is_traditional = True


    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] | None = None) -> None:
        """Train SVM with provided or default parameters."""
        if params is None:
            params = self.best_params if self.best_params else self._get_default_params()
        self.model = SVC(**params)
        self.model.fit(X, y)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model with basic classification metrics."""
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
        """Persist model and metadata."""
        if self.model is not None:
            joblib.dump(self.model, f"{path}/svm_model.pkl")
        metadata = {"best_params": self.best_params, "split_info": self.split_info, "config": self.config}
        joblib.dump(metadata, f"{path}/svm_metadata.pkl")

    def load(self, path: str) -> None:
        """Load model and metadata."""
        self.model = joblib.load(f"{path}/svm_model.pkl")
        metadata = joblib.load(f"{path}/svm_metadata.pkl")
        self.best_params = metadata.get("best_params", {})
        self.split_info = metadata.get("split_info", {})

    def _get_default_params(self) -> Dict[str, Any]:
        """Return default SVM parameters."""
        return {
            "kernel": self.config.get("kernel", "rbf"),
            "C": self.config.get("C", 1.0),
            "gamma": self.config.get("gamma", "scale"),
            "probability": self.config.get("probability", True),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if self.model is None:
            return np.array([])
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None or not hasattr(self.model, "predict_proba"):
            return np.array([])
        return self.model.predict_proba(X)

    def get_name(self) -> str:
        """Return classifier key for hyperparameter lookup."""
        return "SVM"

    def _build_sklearn_estimator(
        self,
        class_weight: str = None,
        random_state: int = None,
        params: Dict[str, Any] = None,
    ):
        """Build an unfitted SVC with the given parameters."""
        if params is not None:
            params_copy = dict(params)
            if class_weight is not None and "class_weight" not in params_copy:
                params_copy["class_weight"] = class_weight
            return SVC(**params_copy)
        return SVC(class_weight=class_weight)

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
        """Return legacy-compatible metric tuple for SVM traditional evaluation.
        
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
        metrics = self._legacy_binary_metrics(y_test, predictions)
        self._print_legacy_metrics("Support Vector Machine Performance Metrics:", metrics)
        return metrics
