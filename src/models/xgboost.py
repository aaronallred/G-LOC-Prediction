from typing import Any, Dict

from skopt.space import Integer, Real
from xgboost import XGBClassifier

from .base import TraditionalModel
from src.runtime import xgboost_device


class XGBoostModel(TraditionalModel):
    """XGBoost binary classifier wrapper."""

    @property
    def name(self) -> str:
        return "XGB"

    @property
    def data_pipeline_hyperparameters(self) -> Dict[str, Any]:
        return {
            "baseline_window": 46.25,
            "window_size": 12.5,
            "stride": 0.25,
            "feature_reduction_type": "lasso",
            "baseline_methods_to_use": [
                "v0", "v1", "v2", "v5", "v6", "v7", "v8"
            ],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        }

    @property
    def hpo_search_space(self) -> Dict[str, Any]:
        return {
            "n_estimators": Integer(100, 500),
            "max_depth": Integer(2, 8),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.6, 1.0),
            "min_child_weight": Integer(1, 10),
        }

    def _build_model(
        self,
        model_hyperparameters: Dict[str, Any] | None,
    ) -> XGBClassifier:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": xgboost_device(),
            "n_jobs": -1,
        }
        params.update(model_hyperparameters or {})
        return XGBClassifier(**params)
