from typing import Any, Dict
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Categorical, Integer

from .base import TraditionalModel

class KNearestNeighborsModel(TraditionalModel):
    """K-nearest neighbors classifier wrapper."""

    def __init__(self, model_hyperparameters: Dict[str, Any] = None):
        super().__init__(model_hyperparameters)

    @property
    def data_pipeline_hyperparameters(self) -> Dict[str, Any]:
        return {
            "baseline_window": 32.5,
            "window_size": 15,
            "stride": 0.25,
            "feature_reduction_type": "performance",
            "baseline_methods_to_use": ["v0", "v1", "v2"],
            "imbalance_type": "ros",
            "impute_type": 1,
            "n_neighbors": 5,
        }

    @property
    def name(self) -> str:
        return "KNN"

    @property
    def hpo_search_space(self) -> Dict[str, Any]:
        return {
            "n_neighbors": Integer(3, 30),
            "weights": Categorical(["uniform", "distance"]),
            "algorithm": Categorical(["auto", "brute"]),
            "metric": Categorical(["minkowski"]),
            "p": Integer(1, 2),
        }

    def _build_model(self, model_hyperparameters: Dict[str, Any]) -> Any:
        """Initializes the KNeighborsClassifier with provided hyperparameters."""
        return KNeighborsClassifier(**(model_hyperparameters or {}))