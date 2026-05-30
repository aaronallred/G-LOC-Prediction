from typing import Any, Dict
from sklearn.linear_model import LogisticRegression

from .base import TraditionalModel

class LogisticRegressionModel(TraditionalModel):
    """Logistic regression classifier wrapper."""

    def __init__(self, model_hyperparameters: Dict[str, Any] = None):
        super().__init__(model_hyperparameters)

    @property
    def data_pipeline_hyperparameters(self) -> Dict[str, Any]:
        return {
            "baseline_window": 5,
            "window_size": 12.5,
            "stride": 0.25,
            "feature_reduction_type": "lasso",
            "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 5,
        }

    @property
    def name(self) -> str:
        return "LogReg"

    def _build_model(self, model_hyperparameters: Dict[str, Any]) -> Any:
        """Initializes the LogisticRegression with provided hyperparameters."""
        return LogisticRegression(**(model_hyperparameters or {}))