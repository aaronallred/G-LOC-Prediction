from typing import Any, Dict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skopt.space import Categorical, Real

from .base import TraditionalModel

class LinearDiscriminantAnalysisModel(TraditionalModel):
    """Linear Discriminant Analysis classifier wrapper."""

    def __init__(self, model_hyperparameters: Dict[str, Any] = None):
        super().__init__(model_hyperparameters)

    @property
    def data_pipeline_hyperparameters(self) -> Dict[str, Any]:
        return {
            "baseline_window": 46.25,
            "window_size": 15,
            "stride": 0.25,
            "feature_reduction_type": "lasso",
            "baseline_methods_to_use": ["v0", "v1", "v2"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        }

    @property
    def name(self) -> str:
        return "LDA"

    @property
    def hpo_search_space(self) -> Any:
        return [
            {
                "solver": Categorical(["svd"]),
                "tol": Real(1e-10, 1e-2, prior="log-uniform"),
            },
            {
                "solver": Categorical(["lsqr", "eigen"]),
                "shrinkage": Categorical([0.1, 0.2, 0.4, 0.6, 0.8, 1, "auto"]),
                "tol": Real(1e-10, 1e-2, prior="log-uniform"),
            },
        ]

    def _build_model(self, model_hyperparameters: Dict[str, Any]) -> Any:
        """Initializes the LinearDiscriminantAnalysis with provided hyperparameters."""
        return LinearDiscriminantAnalysis(**(model_hyperparameters or {}))