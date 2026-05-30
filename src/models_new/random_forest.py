from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier

from .base import TraditionalModel

class RandomForestModel(TraditionalModel):
    """Random Forest classifier wrapper used by the traditional pipeline."""

    def __init__(self, model_hyperparameters: Dict[str, Any] = None):
        super().__init__(model_hyperparameters)

    @property
    def data_pipeline_hyperparameters(self) -> Dict[str, Any]:
        return {
            "baseline_window": 18.75,
            "window_size": 7.5,
            "stride": 0.25,
            "feature_reduction_type": "none",
            "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        }

    @property
    def name(self) -> str:
        return "RF"

    def _build_model(self, model_hyperparameters: Dict[str, Any]) -> Any:
        """Initializes the RandomForestClassifier with provided hyperparameters."""
        return RandomForestClassifier(**(model_hyperparameters or {}))