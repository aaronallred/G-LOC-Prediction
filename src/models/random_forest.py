import joblib
import logging
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest classifier wrapper used by the traditional pipeline."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.is_traditional = True

    def tune(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> None:
        """Placeholder for future hyperparameter search."""
        logger.info("Starting hyperparameter tuning for RandomForestModel")

    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] | None = None) -> None:
        """Train a random forest model with provided or default parameters."""
        if params is None:
            params = self.best_params if self.best_params else self._get_default_params()
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
