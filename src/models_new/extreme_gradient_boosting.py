from typing import Any, Dict
from sklearn.ensemble import GradientBoostingClassifier
from skopt.space import Categorical, Integer, Real

from .base import TraditionalModel

class ExtremeGradientBoostingModel(TraditionalModel):
    """Gradient boosting classifier wrapper for EGB pipeline key."""

    def __init__(self, model_hyperparameters: Dict[str, Any] = None):
        super().__init__(model_hyperparameters)

    @property
    def data_pipeline_hyperparameters(self) -> Dict[str, Any]:
        return {
            "baseline_window": 46.25,
            "window_size": 12.5,
            "stride": 0.25,
            "feature_reduction_type": "lasso",
            "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        }
    @property
    def name(self) -> str:
        return "EGB"

    @property
    def hpo_search_space(self) -> Dict[str, Any]:
        return {
            "n_estimators": Integer(50, 1000),
            "learning_rate": Real(0.01, 1.0, prior="log-uniform"),
            "max_depth": Integer(3, 20),
            "max_features": Categorical(["sqrt", "log2", None]),
            "min_samples_leaf": Integer(1, 4),
            "min_samples_split": Integer(2, 4),
            "loss": Categorical(["log_loss"]),
            "min_weight_fraction_leaf": Real(0.0, 0.5),
        }

    def _build_model(self, model_hyperparameters: Dict[str, Any]) -> Any:
        """Initializes the GradientBoostingClassifier with provided hyperparameters."""
        return GradientBoostingClassifier(**(model_hyperparameters or {}))