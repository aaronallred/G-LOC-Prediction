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
        if model_names is None:
            # No root-level models; this is OK, modes should specify their own
            # Return empty list - modes will provide their own models
            import warnings
            warnings.warn(
                "No 'models' parameter defined at root level. "
                "Modes must specify their own models in their config sections. "
                "For backward compatibility, if a mode doesn't specify models, "
                "the code may fail or use unexpected defaults.",
                UserWarning,
                stacklevel=2
            )
            return []
        
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

    def _build_model_type(self) -> Optional[ModelType]:
        model_type_config = self.config.get("model_type")
        if model_type_config is None:
            # Root-level model_type not defined; modes must provide their own
            return None
        
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

    def get_models_for_cross_validation(self) -> List[BaseModel]:
        """Get models for cross-validation.
        
        Returns models from cross_validation.models section.
        """
        return self.get_cross_validation_models()

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
        """DEPRECATED: num_splits is now mode-specific. Use get_sensor_ablation_training_num_splits() or get_cross_validation_num_splits() instead."""
        import warnings
        warnings.warn(
            "get_num_splits() is deprecated. Use mode-specific getters: "
            "get_sensor_ablation_training_num_splits() for sensor ablation, "
            "or get_cross_validation_num_splits() for cross-validation.",
            DeprecationWarning,
            stacklevel=2
        )
        # Fallback for backward compatibility: try sensor ablation first, then cross-validation
        try:
            return self._get_nested("sensor_ablation", "training", "num_splits")
        except (KeyError, TypeError):
            try:
                return self._get_nested("cross_validation", "num_splits")
            except (KeyError, TypeError):
                # Last resort: old location
                return self._get_nested("advanced_data_parameters", "num_splits")

    def get_kfold_ID(self) -> Any:
        """DEPRECATED: kfold_ID is no longer used and should not be in advanced_data_parameters.
        
        Use get_cross_validation_kfold_ID() if needed (also deprecated).
        """
        import warnings
        warnings.warn(
            "get_kfold_ID() from advanced_data_parameters is deprecated. "
            "kfold_ID should only be in cross_validation section (also deprecated there).",
            DeprecationWarning,
            stacklevel=2
        )
        try:
            return self._get_nested("advanced_data_parameters", "kfold_ID")
        except (KeyError, TypeError):
            return 0  # Default if not found

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

    def get_sensor_ablation_training_save_results_folder(self) -> str:
        """Get save results folder for sensor ablation training mode.
        
        Returns:
            Path where sensor ablation training results should be saved
        """
        folder = self._get_nested("sensor_ablation", "training", "save_results_folder")
        if folder is None:
            return "Results/Sensor_Ablation"
        return str(folder)

    def get_sensor_ablation_review_save_results_folder(self) -> str:
        """Get save results folder for sensor ablation review mode.
        
        Returns:
            Path where sensor ablation review loads results from (must match training path)
        """
        folder = self._get_nested("sensor_ablation", "review", "save_results_folder")
        if folder is None:
            return "Results/Sensor_Ablation"
        return str(folder)

    def get_sensor_ablation_training_num_splits(self) -> int:
        """Get the number of folds for sensor ablation k-fold cross-validation.
        
        Returns:
            Number of splits for sensor ablation training
        """
        return self._get_nested("sensor_ablation", "training", "num_splits")

    def get_feature_space_review_enabled(self) -> bool:
        return self._get_enabled("feature_space_review", "enabled")

    def get_feature_space_review_models(self) -> Any:
        return copy.deepcopy(self._get_nested("feature_space_review", "models"))

    def get_cross_validation_enabled(self) -> bool:
        return self._get_enabled("cross_validation", "enabled")

    def get_cross_validation_num_splits(self) -> int:
        return self._get_nested("cross_validation", "num_splits")

    def get_cross_validation_kfold_ID(self) -> int:
        """DEPRECATED: kfold_ID is no longer used. All folds are executed by default.
        
        This parameter is kept for backward compatibility only. Remove from config.
        """
        try:
            value = self._get_nested("cross_validation", "kfold_ID")
            import warnings
            warnings.warn(
                "cross_validation.kfold_ID is deprecated and ignored. "
                "All folds are always executed. Remove this parameter from your config. "
                "In future versions, partial fold execution should be done via external orchestration.",
                DeprecationWarning,
                stacklevel=2
            )
            return value
        except KeyError:
            return 0  # Default for backward compatibility

    def get_cross_validation_model_type(self) -> "ModelType":
        """Get model type for cross-validation mode."""
        cv_model_type = self._get_nested("cross_validation", "model_type")
        if cv_model_type and isinstance(cv_model_type, list) and len(cv_model_type) == 2:
            return ModelType(afe_filter=cv_model_type[0], feature_set=cv_model_type[1])
        raise ValueError("cross_validation.model_type is required")

    def get_cross_validation_save_results_folder(self) -> str:
        return self._get_nested("cross_validation", "save_results_folder")

    def get_cross_validation_class_weight(self) -> Any:
        return self._get_nested("cross_validation", "class_weight")

    def get_cross_validation_support_deep_learning(self) -> bool:
        return self._get_nested("cross_validation", "support_deep_learning")

    def get_cross_validation_impute_handling(self) -> dict:
        return copy.deepcopy(self._get_nested("cross_validation", "impute_handling"))

    def get_cross_validation_save_median_hyperparameters(self) -> bool:
        return self._get_nested("cross_validation", "save_median_hyperparameters")

    def get_sensor_ablation_training_models(self) -> List[BaseModel]:
        """Get models for sensor ablation training mode.
        
        Returns models from sensor_ablation.training.models, falls back to global models.
        """
        try:
            model_names = self._get_nested("sensor_ablation", "training", "models")
            if model_names and isinstance(model_names, list):
                return self._build_models_from_names(model_names)
        except (KeyError, TypeError, ValueError):
            pass
        # Fallback to global models
        return self.models

    def get_sensor_ablation_training_model_type(self) -> ModelType:
        """Get model type for sensor ablation training mode."""
        model_type_config = self._get_nested("sensor_ablation", "training", "model_type")
        if model_type_config and isinstance(model_type_config, list) and len(model_type_config) == 2:
            return ModelType(afe_filter=model_type_config[0], feature_set=model_type_config[1])
        raise ValueError("sensor_ablation.training.model_type is required")

    def get_sensor_ablation_training_random_seed(self) -> Any:
        """Get random seed for sensor ablation training mode."""
        seed = self._get_nested("sensor_ablation", "training", "random_seed")
        if seed is not None:
            return seed
        raise ValueError("sensor_ablation.training.random_seed is required")

    def get_sensor_ablation_review_models(self) -> List[BaseModel]:
        """Get models for sensor ablation review mode."""
        try:
            model_names = self._get_nested("sensor_ablation", "review", "models")
            if model_names and isinstance(model_names, list):
                return self._build_models_from_names(model_names)
        except (KeyError, TypeError, ValueError):
            pass
        return self.models

    def get_sensor_ablation_review_model_type(self) -> ModelType:
        """Get model type for sensor ablation review mode."""
        model_type_config = self._get_nested("sensor_ablation", "review", "model_type")
        if model_type_config and isinstance(model_type_config, list) and len(model_type_config) == 2:
            return ModelType(afe_filter=model_type_config[0], feature_set=model_type_config[1])
        raise ValueError("sensor_ablation.review.model_type is required")

    def get_feature_space_review_models(self) -> List[BaseModel]:
        """Get models for feature space review mode."""
        try:
            model_names = self._get_nested("feature_space_review", "models")
            if model_names and isinstance(model_names, list):
                return self._build_models_from_names(model_names)
        except (KeyError, TypeError, ValueError):
            pass
        return self.models

    def get_feature_space_review_model_type(self) -> ModelType:
        """Get model type for feature space review mode."""
        model_type_config = self._get_nested("feature_space_review", "model_type")
        if model_type_config and isinstance(model_type_config, list) and len(model_type_config) == 2:
            return ModelType(afe_filter=model_type_config[0], feature_set=model_type_config[1])
        raise ValueError("feature_space_review.model_type is required")

    def get_cross_validation_models(self) -> List[BaseModel]:
        """Get models for cross-validation mode.
        
        Returns models from cross_validation.models, falls back to global models.
        """
        try:
            model_names = self._get_nested("cross_validation", "models")
            if model_names and isinstance(model_names, list):
                return self._build_models_from_names(model_names)
        except (KeyError, TypeError, ValueError):
            pass
        # Fallback to global models
        return self.models

    def get_cross_validation_random_seed(self) -> Any:
        """Get random seed for cross-validation mode."""
        seed = self._get_nested("cross_validation", "random_seed")
        if seed is not None:
            return seed
        raise ValueError("cross_validation.random_seed is required")

    def _build_models_from_names(self, model_names: List[str]) -> List[BaseModel]:
        """Build model instances from a list of model names.
        
        Args:
            model_names: List of model name strings
            
        Returns:
            List of instantiated BaseModel objects
        """
        if not isinstance(model_names, list):
            raise ValueError("model_names must be a list.")
        
        model_config = self.config.get("model_config", {})
        built_models: List[BaseModel] = []
        
        for model_name in model_names:
            model_factory = self.MODEL_FACTORIES_BY_NAME.get(model_name)
            if model_factory is None:
                raise ValueError(
                    f"Model '{model_name}' is not recognized. "
                    f"Available models: {list(self.MODEL_FACTORIES_BY_NAME.keys())}"
                )
            built_models.append(model_factory(config=model_config))
        
        return built_models
