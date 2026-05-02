from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import copy
import yaml

from src.model_type import ModelType
from src.models.base import BaseModel
from src.models.extreme_gradient_boosting import ExtremeGradientBoostingModel
from src.models.k_nearest_neighbors import KNearestNeighborsModel
from src.models.linear_discriminant_analysis import LinearDiscriminantAnalysisModel
from src.models.logistic_regression import LogisticRegression
from src.models.random_forest import RandomForestModel
from src.models.support_vector_machine import SupportVectorMachineModel
from src.models.transformer import TransformerModel


class GLOCExperimentConfigParser:
    MODEL_FACTORIES_BY_NAME: Dict[str, Type[BaseModel]] = {
        "Logistic Regression": LogisticRegression,
        "LogReg": LogisticRegression,
        "Random Forest": RandomForestModel,
        "RF": RandomForestModel,
        "Linear Discriminant Analysis": LinearDiscriminantAnalysisModel,
        "LDA": LinearDiscriminantAnalysisModel,
        "Support Vector Machine": SupportVectorMachineModel,
        "SVM": SupportVectorMachineModel,
        "Extreme Gradient Boosting": ExtremeGradientBoostingModel,
        "EGB": ExtremeGradientBoostingModel,
        "K Nearest Neighbors": KNearestNeighborsModel,
        "KNN": KNearestNeighborsModel,
        "Transformer": TransformerModel,
        "Trans": TransformerModel,
    }

    def __init__(self, config_location: Optional[str] = None) -> None:
        if config_location is None:
            config_path = Path(__file__).resolve().parent.parent / "GLOC_experiment_config.yaml"
        else:
            config_path = Path(config_location).expanduser().resolve()

        if config_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                f"Unsupported config file extension '{config_path.suffix}'. Expected .yaml or .yml."
            )

        with open(config_path, "r") as handle:
            loaded_config = yaml.safe_load(handle)

        if not isinstance(loaded_config, dict):
            raise ValueError("Config file must parse into a YAML mapping/object.")

        self.config = loaded_config
        self.models = self._build_models()
        self.model_type = self._build_model_type()

    def _build_models(self) -> List[BaseModel]:
        model_names = self.config.get("models")
        if not isinstance(model_names, list):
            raise ValueError("models must be a list of model names.")
        
        model_config = self.config.get("model_config", {})

        built_models: List[BaseModel] = []
        for model_name in model_names:
            model_factory = self.MODEL_FACTORIES_BY_NAME.get(model_name)
            if model_factory is None:
                raise ValueError(
                    f"Model '{model_name}' is not recognized. "
                    f"Available models: {list(self.MODEL_FACTORIES_BY_NAME.keys())}"
                )
            built_models.append(model_factory(config = model_config))

        return built_models

    def _build_model_type(self) -> ModelType:
        model_type_config = self.config.get("model_type")
        if not isinstance(model_type_config, list) or len(model_type_config) != 2:
            raise ValueError("model_type must be a two-item list: [afe_filter, feature_set].")

        return ModelType(afe_filter=model_type_config[0], feature_set=model_type_config[1])

    def _get_nested(self, *keys: str) -> Any:
        current: Any = self.config
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _get_enabled(self, *keys: str) -> bool:
        value = self._get_nested(*keys)
        if value is None:
            return False
        return bool(value)

    def get_models(self) -> List[BaseModel]:
        return self.models.copy()

    def get_model_type(self) -> ModelType:
        return self.model_type

    def get_random_seed(self) -> Any:
        return self.config.get("random_seed")

    def get_data_path(self) -> Any:
        return self.config.get("data_path")

    def get_subject_to_analyze(self) -> Any:
        return self._get_nested("shared_data_parameters", "subject_to_analyze")

    def get_trial_to_analyze(self) -> Any:
        return self._get_nested("shared_data_parameters", "trial_to_analyze")

    def get_analysis_type(self) -> Any:
        return self._get_nested("shared_data_parameters", "analysis_type")

    def get_remove_NaN_trials(self) -> Any:
        return self._get_nested("shared_data_parameters", "remove_NaN_trials")

    def get_impute_file_name(self) -> Any:
        return self._get_nested("shared_data_parameters", "impute_file_name")

    def get_should_impute(self) -> Any:
        return self._get_nested("shared_data_parameters", "should_impute")

    def get_output_feature_dtype(self) -> Any:
        return self._get_nested("shared_data_parameters", "output_feature_dtype")

    def get_save_impute(self) -> Any:
        return self._get_nested("shared_data_parameters", "save_impute")

    def get_load_impute(self) -> Any:
        return self._get_nested("shared_data_parameters", "load_impute")

    def get_num_splits(self) -> Any:
        return self._get_nested("advanced_data_parameters", "num_splits")

    def get_kfold_ID(self) -> Any:
        return self._get_nested("advanced_data_parameters", "kfold_ID")

    def get_n_neighbors(self) -> Any:
        return self._get_nested("advanced_data_parameters", "n_neighbors")

    def get_baseline_window(self) -> Any:
        return self._get_nested("advanced_data_parameters", "baseline_window")

    def get_backstep(self) -> Any:
        return self._get_nested("traditional_data_parameters", "backstep")

    def get_data_rate(self) -> Any:
        return self._get_nested("traditional_data_parameters", "data_rate")

    def get_offset(self) -> Any:
        return self._get_nested("traditional_data_parameters", "offset")

    def get_time_start(self) -> Any:
        return self._get_nested("traditional_data_parameters", "time_start")

    def get_sensor_ablation_enabled(self) -> bool:
        return self._get_enabled("sensor_ablation", "training", "enabled")

    def get_sensor_ablation_streams(self) -> Any:
        return copy.deepcopy(self._get_nested("sensor_ablation", "training", "streams"))

    def get_sensor_ablation_review_enabled(self) -> bool:
        return self._get_enabled("sensor_ablation", "review", "enabled")

    def get_sensor_ablation_review_models(self) -> Any:
        return copy.deepcopy(self._get_nested("sensor_ablation", "review", "models"))

    def get_sensor_ablation_review_stream_group(self) -> Any:
        return copy.deepcopy(self._get_nested("sensor_ablation", "review", "stream_group"))

    def get_sensor_ablation_review_sort_streams_by_median(self) -> Any:
        return self._get_nested("sensor_ablation", "review", "sort_streams_by_median")

    def get_feature_space_review_enabled(self) -> bool:
        return self._get_enabled("feature_space_review", "enabled")

    def get_feature_space_review_models(self) -> Any:
        return copy.deepcopy(self._get_nested("feature_space_review", "models"))

    def get_hyperparameter_save_enabled(self) -> bool:
        return self._get_enabled("hyperparameter_save", "enabled")

    def get_hyperparameter_save_models(self) -> Any:
        return copy.deepcopy(self._get_nested("hyperparameter_save", "models"))

    def get_cross_validation_enabled(self) -> bool:
        return self._get_enabled("cross_validation", "enabled")

    def get_cross_validation_num_splits(self) -> int:
        return self._get_nested("cross_validation", "num_splits")

    def get_cross_validation_kfold_ID(self) -> int:
        return self._get_nested("cross_validation", "kfold_ID")

    def get_cross_validation_classifiers(self) -> list:
        return copy.deepcopy(self._get_nested("cross_validation", "classifiers"))

    def get_cross_validation_save_results_folder(self) -> str:
        return self._get_nested("cross_validation", "save_results_folder")

    def get_cross_validation_random_seed(self) -> int:
        return self._get_nested("cross_validation", "random_seed")

    def get_cross_validation_class_weight(self) -> Any:
        return self._get_nested("cross_validation", "class_weight")

    def get_cross_validation_support_deep_learning(self) -> bool:
        return self._get_nested("cross_validation", "support_deep_learning")

    def get_cross_validation_impute_handling(self) -> dict:
        return copy.deepcopy(self._get_nested("cross_validation", "impute_handling"))
