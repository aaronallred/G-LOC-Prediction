import joblib
import os
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.base import BaseModel


class ExtremeGradientBoostingModel(BaseModel):
    """Gradient boosting classifier wrapper for EGB pipeline key."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.is_traditional = True

    def tune(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> None:
        """Placeholder for future hyperparameter tuning."""

    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] | None = None) -> None:
        """Train gradient boosting with provided or default parameters."""
        if params is None:
            params = self.best_params if self.best_params else self._get_default_params()
        self.model = GradientBoostingClassifier(**params)
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
            joblib.dump(self.model, f"{path}/egb_model.pkl")
        metadata = {"best_params": self.best_params, "split_info": self.split_info, "config": self.config}
        joblib.dump(metadata, f"{path}/egb_metadata.pkl")

    def load(self, path: str) -> None:
        """Load model and metadata."""
        self.model = joblib.load(f"{path}/egb_model.pkl")
        metadata = joblib.load(f"{path}/egb_metadata.pkl")
        self.best_params = metadata.get("best_params", {})
        self.split_info = metadata.get("split_info", {})

    def _get_default_params(self) -> Dict[str, Any]:
        """Return default gradient boosting parameters."""
        return {
            "n_estimators": self.config.get("n_estimators", 200),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "max_depth": self.config.get("max_depth", 3),
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
        return "EGB"

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
        retrain,
        temporal=False,
        best_params=None,
    ):
        """Return legacy-compatible metric tuple for gradient boosting evaluation."""
        del class_weight_imb

        if retrain:
            estimator = GradientBoostingClassifier(random_state=random_state).fit(
                x_train,
                np.ravel(y_train),
            )
        else:
            if temporal:
                estimator = GradientBoostingClassifier(
                    **(best_params or {}),
                    random_state=random_state,
                ).fit(x_train, np.ravel(y_train))
            else:
                model_path = os.path.join(save_folder, model_name)
                estimator = joblib.load(model_path)

        self.model = estimator
        predictions = estimator.predict(x_test)
        return self._legacy_binary_metrics(y_test, predictions)
