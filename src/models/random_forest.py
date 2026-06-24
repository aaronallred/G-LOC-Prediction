from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
from skopt.space import Categorical, Integer, Real

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

    @property
    def hpo_search_space(self) -> Dict[str, Any]:
        return {
            "n_estimators": Integer(10, 1000),
            "criterion": Categorical(["gini", "entropy", "log_loss"]),
            "max_depth": Integer(3, 100),
            "max_features": Categorical(["sqrt", "log2"]),
            "min_samples_leaf": Integer(1, 4),
            "min_samples_split": Integer(2, 10),
            "min_weight_fraction_leaf": Real(0.0, 0.5),
        }

    def _build_model(self, model_hyperparameters: Dict[str, Any]) -> Any:
        """Initializes the RandomForestClassifier with provided hyperparameters."""
        return RandomForestClassifier(**(model_hyperparameters or {}))