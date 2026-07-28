import pytest

from src.models.extreme_gradient_boosting import ExtremeGradientBoostingModel
from src.models.k_nearest_neighbors import KNearestNeighborsModel
from src.models.linear_discriminant_analysis import LinearDiscriminantAnalysisModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.support_vector_machine import SupportVectorMachineModel


@pytest.mark.parametrize(
    "model_cls, expected_hyperparameters",
    [
        (
            LogisticRegressionModel,
            {
                "baseline_window": 5,
                "window_size": 12.5,
                "stride": 0.25,
                "feature_reduction_type": "lasso",
                "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 5,
            },
        ),
        (
            RandomForestModel,
            {
                "baseline_window": 18.75,
                "window_size": 7.5,
                "stride": 0.25,
                "feature_reduction_type": "none",
                "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 3,
            },
        ),
        (
            LinearDiscriminantAnalysisModel,
            {
                "baseline_window": 46.25,
                "window_size": 15,
                "stride": 0.25,
                "feature_reduction_type": "lasso",
                "baseline_methods_to_use": ["v0", "v1", "v2"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 3,
            },
        ),
        (
            SupportVectorMachineModel,
            {
                "baseline_window": 32.5,
                "window_size": 15,
                "stride": 0.25,
                "feature_reduction_type": "ridge",
                "baseline_methods_to_use": ["v0", "v1", "v2"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 3,
            },
        ),
        (
            ExtremeGradientBoostingModel,
            {
                "baseline_window": 46.25,
                "window_size": 12.5,
                "stride": 0.25,
                "feature_reduction_type": "lasso",
                "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 3,
            },
        ),
        (
            KNearestNeighborsModel,
            {
                "baseline_window": 32.5,
                "window_size": 15,
                "stride": 0.25,
                "feature_reduction_type": "performance",
                "baseline_methods_to_use": ["v0", "v1", "v2"],
                "imbalance_type": "ros",
                "impute_type": 1,
                "n_neighbors": 5,
            },
        ),
    ],
)
def test_traditional_models_expose_legacy_hyperparameters(model_cls, expected_hyperparameters):
    model = model_cls()
    assert model.data_pipeline_hyperparameters == expected_hyperparameters
