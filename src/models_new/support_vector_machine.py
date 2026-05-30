from typing import Any, Dict
from sklearn.svm import SVC

from .base import TraditionalModel

class SupportVectorMachineModel(TraditionalModel):
    """Support Vector Machine classifier wrapper."""

    def __init__(self, model_hyperparameters: Dict[str, Any] = None):
        super().__init__(model_hyperparameters)

    @property
    def data_pipeline_hyperparameters(self) -> Dict[str, Any]:
        return {
            "baseline_window": 32.5,
            "window_size": 15,
            "stride": 0.25,
            "feature_reduction_type": "ridge",
            "baseline_methods_to_use": ["v0", "v1", "v2"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        }

    @property
    def name(self) -> str:
        return "SVM"

    def _build_model(self, model_hyperparameters: Dict[str, Any]) -> Any:
        """Initializes the SVC with provided hyperparameters."""
        return SVC(**(model_hyperparameters or {}))