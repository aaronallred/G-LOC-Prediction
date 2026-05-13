import joblib
import os
from typing import Any, Dict

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.base import BaseModel, ModelInitStrategy


class LinearDiscriminantAnalysisModel(BaseModel):
    """Linear Discriminant Analysis classifier wrapper."""

    TRADITIONAL_HYPERPARAMETERS = {
        "baseline_window": 46.25,
        "window_size": 15,
        "stride": 0.25,
        "feature_reduction_type": "lasso",
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
        """Train LDA using provided or default parameters."""
        if params is None:
            params = self.best_params if self.best_params else self._get_default_params()
        self.model = LinearDiscriminantAnalysis(**params)
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
            joblib.dump(self.model, f"{path}/lda_model.pkl")
        metadata = {"best_params": self.best_params, "split_info": self.split_info, "config": self.config}
        joblib.dump(metadata, f"{path}/lda_metadata.pkl")

    def load(self, path: str) -> None:
        """Load model and metadata."""
        self.model = joblib.load(f"{path}/lda_model.pkl")
        metadata = joblib.load(f"{path}/lda_metadata.pkl")
        self.best_params = metadata.get("best_params", {})
        self.split_info = metadata.get("split_info", {})

    def _get_default_params(self) -> Dict[str, Any]:
        """Return default LDA parameters."""
        return {"solver": self.config.get("solver", "svd")}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if self.model is None:
            return np.array([])
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities when available."""
        if self.model is None or not hasattr(self.model, "predict_proba"):
            return np.array([])
        return self.model.predict_proba(X)

    def get_name(self) -> str:
        """Return classifier key for hyperparameter lookup."""
        return "LDA"

    def _build_sklearn_estimator(
        self,
        class_weight: str = None,
        random_state: int = None,
        params: Dict[str, Any] = None,
    ):
        """Build an unfitted LinearDiscriminantAnalysis with the given parameters."""
        if params is not None:
            params_copy = dict(params)
            # Note: LinearDiscriminantAnalysis does not accept random_state
            return LinearDiscriminantAnalysis(**params_copy)
        return LinearDiscriminantAnalysis()

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
        """Return legacy-compatible metric tuple for LDA traditional evaluation.
        
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
        self._print_legacy_metrics("Linear Discriminant Analysis Performance Metrics:", metrics)
        return metrics
