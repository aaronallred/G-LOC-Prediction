from src.model_type import ModelType
from src.models.base import BaseModel
from src.models.logistic_regression import LogisticRegression
from src.models.random_forest import RandomForestModel
from src.models.linear_discriminant_analysis import LinearDiscriminantAnalysisModel
from src.models.support_vector_machine import SupportVectorMachineModel
from src.models.extreme_gradient_boosting import ExtremeGradientBoostingModel
from src.models.k_nearest_neighbors import KNearestNeighborsModel
from src.models.transformer import TransformerModel
from typing import Any, Optional, Dict, List, Type
from pathlib import Path

import numpy as np
import copy
import yaml

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
    SENSOR_ABLATION_STREAM_LABELS = {
        "ECG", "HR", "BR", "Temperature", "Pupil", "Centrifuge", "EEG", "Strain", "Participant",
        "temp", "eyetracking", "G", "rawEEG", "processedEEG", "strain", "demographics",
    }

    def __init__(self, config_location: Optional[str] = None) -> None:
        if config_location is None:
            config_path = Path(__file__).resolve().parent.parent / "GLOC_experiment_config.yaml"
        else:
            config_path = Path(config_location).expanduser().resolve()

        with open(config_path, "r") as f:
            if config_path.suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError(
                    f"Unsupported config file extension '{config_path.suffix}'. "
                    "Expected .yaml or .yml."
                )

            self.config = yaml.safe_load(f)

        if not isinstance(self.config, dict):
            raise ValueError("Config file must parse into a YAML mapping/object.")

        self._parse_general_configs()
        self._parse_shared_data_configs()
        self._parse_advanced_data_configs()
        self._parse_traditional_data_configs()
        self._parse_sensor_ablation_configs()
        self._parse_sensor_ablation_review_configs()
        self._parse_feature_space_review_configs()
        self._parse_hyperparameter_save_configs()

    # General Configurations
    def _parse_general_configs(self) -> None:
        self.models = self._parse_models()
        self.model_type = self._parse_model_type()
        self.random_seed = self._parse_random_seed()
        self.data_path = self._parse_data_path()

    def _parse_shared_data_configs(self) -> None:
        shared_data_parameters = self._get_shared_data_parameters()
        self.subject_to_analyze = self._parse_subject_to_analyze(shared_data_parameters)
        self.trial_to_analyze = self._parse_trial_to_analyze(shared_data_parameters)
        self.analysis_type = self._parse_analysis_type(shared_data_parameters)
        self.remove_NaN_trials = self._parse_remove_NaN_trials(shared_data_parameters)
        self.impute_file_name = self._parse_impute_file_name(shared_data_parameters)
        self.save_impute = self._parse_save_impute(shared_data_parameters)
        self.load_impute = self._parse_load_impute(shared_data_parameters)
        self.should_impute = self._parse_should_impute(shared_data_parameters)
        self.output_feature_dtype = self._parse_output_feature_dtype(shared_data_parameters)

    def _parse_advanced_data_configs(self) -> None:
        advanced_data_parameters = self._get_advanced_data_parameters()
        self.num_splits = self._parse_num_splits(advanced_data_parameters)
        self.kfold_ID = self._parse_kfold_ID(advanced_data_parameters)
        self.n_neighbors = self._parse_n_neighbors(advanced_data_parameters)
        self.baseline_window = self._parse_baseline_window(advanced_data_parameters)

    def _parse_traditional_data_configs(self) -> None:
        traditional_data_parameters = self._get_traditional_data_parameters()
        self.backstep = self._parse_backstep(traditional_data_parameters)
        self.data_rate = self._parse_data_rate(traditional_data_parameters)
        self.offset = self._parse_offset(traditional_data_parameters)
        self.time_start = self._parse_time_start(traditional_data_parameters)

    def _parse_sensor_ablation_configs(self) -> None:
        """Parse sensor ablation settings into a validated internal structure."""
        sensor_ablation_root = self._get_sensor_ablation_root()
        training_parameters = sensor_ablation_root.get("training", {})
        enabled = self._parse_sensor_ablation_enabled(training_parameters)
        streams = self._parse_sensor_ablation_streams(training_parameters)

        if enabled and len(streams) == 0:
            raise ValueError(
                "sensor_ablation.training.streams cannot be empty when sensor_ablation.training.enabled is true."
            )

        self.sensor_ablation = {
            "enabled": enabled,
            "streams": streams,
        }

    def _parse_sensor_ablation_review_configs(self) -> None:
        """Parse sensor ablation review settings used to inspect precomputed F1 results."""
        sensor_ablation_root = self._get_sensor_ablation_root()
        review_parameters = sensor_ablation_root.get("review", {})
        enabled = self._parse_sensor_ablation_review_enabled(review_parameters)
        models = self._parse_sensor_ablation_review_models(review_parameters)
        stream_group = self._parse_sensor_ablation_review_stream_group(review_parameters)

        if enabled and len(models) == 0:
            raise ValueError(
                "sensor_ablation.review.models must be a non-empty list when "
                "sensor_ablation.review.enabled is true."
            )

        if enabled and len(stream_group) == 0:
            raise ValueError(
                "sensor_ablation.review.stream_group must be a non-empty list when "
                "sensor_ablation.review.enabled is true."
            )

        self.sensor_ablation_review = {
            "enabled": enabled,
            "models": models,
            "stream_group": stream_group,
        }

    def _parse_feature_space_review_configs(self) -> None:
        """Parse optional feature-space review settings for classifier feature overlap inspection."""
        feature_space_review_root = self.config.get("feature_space_review", {})
        if feature_space_review_root is None:
            feature_space_review_root = {}
        if not isinstance(feature_space_review_root, dict):
            raise ValueError("feature_space_review must be a JSON object.")

        enabled = self._parse_feature_space_review_enabled(feature_space_review_root)
        models = self._parse_feature_space_review_models(feature_space_review_root)

        if enabled and len(models) == 0:
            raise ValueError(
                "feature_space_review.models must be a non-empty list when feature_space_review.enabled is true."
            )

        self.feature_space_review = {
            "enabled": enabled,
            "models": models,
        }

    def _parse_hyperparameter_save_configs(self) -> None:
        """Parse optional hyperparameter save settings for extracting median fold hyperparameters."""
        hyperparameter_save_root = self.config.get("hyperparameter_save", {})
        if hyperparameter_save_root is None:
            hyperparameter_save_root = {}
        if not isinstance(hyperparameter_save_root, dict):
            raise ValueError("hyperparameter_save must be a JSON object.")

        enabled = self._parse_hyperparameter_save_enabled(hyperparameter_save_root)
        models = self._parse_hyperparameter_save_models(hyperparameter_save_root)

        if enabled and len(models) == 0:
            raise ValueError(
                "hyperparameter_save.models must be a non-empty list when hyperparameter_save.enabled is true."
            )

        self.hyperparameter_save = {
            "enabled": enabled,
            "models": models,
        }
            
    def _parse_models(self) -> List[BaseModel]:
        if "models" not in self.config:
            raise ValueError("models is missing from config. It should be a list of model names.")

        model_names = self.config.get("models", [])
        if not isinstance(model_names, list) or len(model_names) == 0:
            raise ValueError("models must be a non-empty list of model names.")

        for model_name in model_names:
            if not isinstance(model_name, str):
                raise ValueError("Each item in models must be a string model name.")

            if model_name not in self.MODEL_FACTORIES_BY_NAME:
                raise ValueError(
                    f"Model '{model_name}' is not recognized. Available models: {list(self.MODEL_FACTORIES_BY_NAME.keys())}"
                )

        model_parameters = self.config.get("model_parameters", {})
        if not isinstance(model_parameters, dict):
            raise ValueError("model_parameters must be a JSON object when provided.")

        return [
            self.MODEL_FACTORIES_BY_NAME[model_name](config=model_parameters)
            for model_name in model_names
        ]
            
    def _parse_model_type(self) -> ModelType:
        model_type_config = self.config.get("model_type", [])

        if len(model_type_config) != 2:
            raise ValueError("model_type must be [<afe_filter>, <feature_set>] where afe_filter is 'Complete' or 'noAFE' and feature_set is 'Explicit' or 'Implicit'.")

        afe_filter = model_type_config[0]
        feature_set = model_type_config[1]

        if afe_filter not in ["Complete", "noAFE"] or feature_set not in ["Explicit", "Implicit"]:
            raise ValueError("model_type must be [<afe_filter>, <feature_set>] where afe_filter is 'Complete' or 'noAFE' and feature_set is 'Explicit' or 'Implicit'.")

        return ModelType(afe_filter = afe_filter, feature_set = feature_set)
    
    def _parse_random_seed(self) -> int:
        if "random_seed" not in self.config:
            raise ValueError("random_seed is missing from config. It should be an integer indicating the random seed to use for reproducibility.")

        return self.config.get("random_seed")
    
    def _parse_data_path(self) -> str:
        if "data_path" not in self.config:
            raise ValueError("data_path is missing from config. It should be a string indicating the path to the data directory.")

        return self.config.get("data_path")
    
    def get_models(self) -> List[BaseModel]:
        return self.models.copy()
    
    def get_model_type(self) -> ModelType:
        return self.model_type
    
    def get_random_seed(self) -> int:
        return self.random_seed
    
    def get_data_path(self) -> str:
        return self.data_path
    

    
    # Shared Data Configurations
    def _get_shared_data_parameters(self) -> Dict:
        if "shared_data_parameters" not in self.config:
            raise ValueError("shared_data_parameters is missing from config. It should be an object containing shared data settings.")

        shared_data_parameters = self.config.get("shared_data_parameters")
        if not isinstance(shared_data_parameters, dict):
            raise ValueError("shared_data_parameters must be a JSON object.")

        return shared_data_parameters

    def _parse_subject_to_analyze(self, shared_data_parameters: Dict) -> Optional[int]:
        if "subject_to_analyze" not in shared_data_parameters:
            raise ValueError("subject_to_analyze is missing from config. It should be an integer subject ID or null to analyze all subjects.")

        return shared_data_parameters.get("subject_to_analyze")

    def _parse_trial_to_analyze(self, shared_data_parameters: Dict) -> Optional[int]:
        if "trial_to_analyze" not in shared_data_parameters:
            raise ValueError("trial_to_analyze is missing from config. It should be an integer trial ID or null to analyze all trials.")

        return shared_data_parameters.get("trial_to_analyze")

    def _parse_analysis_type(self, shared_data_parameters: Dict) -> int:
        if "analysis_type" not in shared_data_parameters:
            raise ValueError("analysis_type is missing from config. It should be an integer (1, 2, or 3) indicating the type of analysis to perform.")

        return shared_data_parameters.get("analysis_type")

    def _parse_remove_NaN_trials(self, shared_data_parameters: Dict) -> bool:
        if "remove_NaN_trials" not in shared_data_parameters:
            raise ValueError("remove_NaN_trials is missing from config. It should be a boolean indicating whether to remove trials with NaN values.")

        return shared_data_parameters.get("remove_NaN_trials")

    def _parse_impute_file_name(self, shared_data_parameters: Dict) -> str:
        if "impute_file_name" not in shared_data_parameters:
            raise ValueError("impute_file_name is missing from config. It should be a string indicating the filename to save/load imputed data.")

        return shared_data_parameters.get("impute_file_name")

    def _parse_save_impute(self, shared_data_parameters: Dict) -> bool:
        if "save_impute" not in shared_data_parameters:
            raise ValueError("save_impute is missing from config. It should be a boolean indicating whether to save imputed data to disk.")

        return shared_data_parameters.get("save_impute")

    def _parse_load_impute(self, shared_data_parameters: Dict) -> bool:
        if "load_impute" not in shared_data_parameters:
            raise ValueError("load_impute is missing from config. It should be a boolean indicating whether to load imputed data from disk if available.")

        return shared_data_parameters.get("load_impute")

    def _parse_should_impute(self, shared_data_parameters: Dict) -> bool:
        if "should_impute" not in shared_data_parameters:
            raise ValueError("should_impute is missing from config. It should be a boolean indicating whether to perform KNN imputation on missing data.")

        return shared_data_parameters.get("should_impute")

    def _parse_output_feature_dtype(self, shared_data_parameters: Dict) -> np.dtype:
        if "output_feature_dtype" not in shared_data_parameters:
            raise ValueError("output_feature_dtype is missing from config. It should be a string like 'float32', 'float64', etc. indicating the numpy dtype for the output feature matrix.")

        dtype = shared_data_parameters.get("output_feature_dtype")

        valid_dtypes = ["float16", "float32", "float64", "int8", "int16", "int32", "int64"]
        if dtype not in valid_dtypes:
            raise ValueError(f"output_feature_dtype must be one of {valid_dtypes}.")
        
        return np.dtype(dtype)

    def get_subject_to_analyze(self) -> Optional[int]:
        return self.subject_to_analyze
    
    def get_trial_to_analyze(self) -> Optional[int]:
        return self.trial_to_analyze
    
    def get_analysis_type(self) -> int:
        return self.analysis_type
    
    def get_remove_NaN_trials(self) -> bool:
        return self.remove_NaN_trials
    


    # Advanced Classifier Configurations
    def _get_advanced_data_parameters(self) -> Dict:
        if "advanced_data_parameters" not in self.config:
            raise ValueError("advanced_data_parameters is missing from config. It should be an object containing advanced classifier settings.")

        advanced_data_parameters = self.config.get("advanced_data_parameters")
        if not isinstance(advanced_data_parameters, dict):
            raise ValueError("advanced_data_parameters must be a JSON object.")

        return advanced_data_parameters

    def _parse_num_splits(self, advanced_data_parameters: Dict) -> int:
        if "num_splits" not in advanced_data_parameters:
            raise ValueError("num_splits is missing from config. It should be an integer indicating the number of splits for train/test.")

        return advanced_data_parameters.get("num_splits")

    def _parse_kfold_ID(self, advanced_data_parameters: Dict) -> int:
        if "kfold_ID" not in advanced_data_parameters:
            raise ValueError("kfold_ID is missing from config. It should be an integer indicating the k-fold ID for train/test splitting.")

        return advanced_data_parameters.get("kfold_ID")

    def _parse_n_neighbors(self, advanced_data_parameters: Dict) -> int:
        if "n_neighbors" not in advanced_data_parameters:
            raise ValueError("n_neighbors is missing from config. It should be an integer indicating the number of neighbors to use for KNN imputation.")

        return advanced_data_parameters.get("n_neighbors")

    def _parse_baseline_window(self, advanced_data_parameters: Dict) -> float:
        if "baseline_window" not in advanced_data_parameters:
            raise ValueError("baseline_window is missing from config. It should be a float indicating the time window (in seconds) to use for baseline calculation.")

        return advanced_data_parameters.get("baseline_window")

    def get_num_splits(self) -> int:
        return self.num_splits
    
    def get_kfold_ID(self) -> int:
        return self.kfold_ID
    
    def get_impute_file_name(self) -> str:
        return self.impute_file_name
    
    def get_should_impute(self) -> bool:
        return self.should_impute
    
    def get_output_feature_dtype(self) -> np.dtype:
        return self.output_feature_dtype
    
    def get_n_neighbors(self) -> int:
        return self.n_neighbors
    
    def get_baseline_window(self) -> float:
        return self.baseline_window
    
    def get_save_impute(self) -> bool:
        return self.save_impute
    
    def get_load_impute(self) -> bool:
        return self.load_impute



    # Traditional Classifier Configurations
    def _get_traditional_data_parameters(self) -> Dict:
        if "traditional_data_parameters" not in self.config:
            raise ValueError("traditional_data_parameters is missing from config. It should be an object containing traditional classifier settings.")

        traditional_data_parameters = self.config.get("traditional_data_parameters")
        if not isinstance(traditional_data_parameters, dict):
            raise ValueError("traditional_data_parameters must be a JSON object.")

        return traditional_data_parameters

    def _parse_backstep(self, traditional_data_parameters: Dict) -> float:
        if "backstep" not in traditional_data_parameters:
            raise ValueError("backstep is missing from config. It should be a float indicating the backstep (in seconds) to use for traditional pipeline.")

        return traditional_data_parameters.get("backstep")

    def _parse_data_rate(self, traditional_data_parameters: Dict) -> int:
        if "data_rate" not in traditional_data_parameters:
            raise ValueError("data_rate is missing from config. It should be an integer indicating the data rate (in Hz) to use for traditional pipeline.")

        return traditional_data_parameters.get("data_rate")

    def _parse_offset(self, traditional_data_parameters: Dict) -> float:
        if "offset" not in traditional_data_parameters:
            raise ValueError("offset is missing from config. It should be a float indicating the offset (in seconds) to use for traditional pipeline.")

        return traditional_data_parameters.get("offset")

    def _parse_time_start(self, traditional_data_parameters: Dict) -> float:
        if "time_start" not in traditional_data_parameters:
            raise ValueError("time_start is missing from config. It should be a float indicating the time start (in seconds) to use for traditional pipeline.")

        return traditional_data_parameters.get("time_start")

    def get_backstep(self) -> float:
        return self.backstep
    
    def get_data_rate(self) -> int:
        return self.data_rate
    
    def get_offset(self) -> float:
        return self.offset
    
    def get_time_start(self) -> float:
        return self.time_start
    

    
    # Sensor Ablation Configurations
    def _get_sensor_ablation_root(self) -> Dict:
        """Get the root sensor_ablation configuration object."""
        sensor_ablation_root = self.config.get(
            "sensor_ablation",
            {"training": {"enabled": False, "streams": []}, "review": {"enabled": False, "models": [], "stream_group": []}},
        )
        if not isinstance(sensor_ablation_root, dict):
            raise ValueError("sensor_ablation must be a YAML object.")
        
        return sensor_ablation_root

    def _parse_sensor_ablation_enabled(self, training_parameters: Dict) -> bool:
        if "enabled" not in training_parameters:
            raise ValueError("sensor_ablation.training.enabled is missing from config. It should be a boolean indicating whether to perform sensor ablation.")

        return training_parameters.get("enabled", False)
    
    def _parse_sensor_ablation_streams(self, training_parameters: Dict) -> List[List[str]]:
        if "streams" not in training_parameters:
            raise ValueError(
                "sensor_ablation.training.streams is missing from config. "
                "It should be a list of stream groups, where each stream group is a list of strings."
            )

        streams = training_parameters.get("streams", [])
        if not isinstance(streams, list):
            raise ValueError(
                "sensor_ablation.streams must be a list of stream groups (list[list[str]])."
            )
        
        # Validate stream names against a known set of valid streams to catch typos and ensure correctness.
        VALID_STREAMS = {
            "ECG", "HR", "BR", "Temperature", "Pupil", "Centrifuge", "EEG", "Strain", "Participant", "Demographics",
            "temp", "eyetracking", "G", "rawEEG", "processedEEG", "strain", "demographics",
        }
        for group_of_streams in streams:
            if not isinstance(group_of_streams, list):
                raise ValueError(
                    "Each item in sensor_ablation.streams must be a list of stream names."
                )
            
            for stream_name in group_of_streams:
                if not isinstance(stream_name, str):
                    raise ValueError(
                        "Each stream name in sensor_ablation.streams must be a string."
                    )
                
                if stream_name not in VALID_STREAMS:
                    raise ValueError(
                        f"Invalid stream name '{stream_name}' in sensor_ablation.streams. Valid streams are: {VALID_STREAMS}."
                    )

        return streams

    def _parse_sensor_ablation_review_enabled(self, review_parameters: Dict) -> bool:
        enabled = review_parameters.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("sensor_ablation.review.enabled must be a boolean.")

        return enabled

    def _parse_sensor_ablation_review_models(self, review_parameters: Dict) -> List[str]:
        models = review_parameters.get("models", [])
        if models is None:
            return []

        if not isinstance(models, list):
            raise ValueError("sensor_ablation.review.models must be a list of model names.")

        for model_name in models:
            if not isinstance(model_name, str):
                raise ValueError("Each model in sensor_ablation.review.models must be a string.")

            if model_name not in self.MODEL_FACTORIES_BY_NAME:
                raise ValueError(
                    f"Model '{model_name}' is not recognized for sensor_ablation.review.models. "
                    f"Available models: {list(self.MODEL_FACTORIES_BY_NAME.keys())}"
                )

        return models

    def _parse_sensor_ablation_review_stream_group(self, review_parameters: Dict) -> List[str]:
        stream_group = review_parameters.get("stream_group", [])
        if stream_group is None:
            return []

        if not isinstance(stream_group, list):
            raise ValueError("sensor_ablation.review.stream_group must be a list of stream labels.")

        validated_stream_group: List[str] = []
        for stream_name in stream_group:
            if not isinstance(stream_name, str):
                raise ValueError(
                    "Each stream in sensor_ablation.review.stream_group must be a string."
                )

            if stream_name not in self.SENSOR_ABLATION_STREAM_LABELS:
                raise ValueError(
                    f"Stream '{stream_name}' is not recognized for sensor_ablation.review.stream_group. "
                    f"Available streams: {sorted(self.SENSOR_ABLATION_STREAM_LABELS)}"
                )
            validated_stream_group.append(stream_name)

        return validated_stream_group

    def _parse_feature_space_review_enabled(self, feature_space_review_root: Dict) -> bool:
        enabled = feature_space_review_root.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("feature_space_review.enabled must be a boolean.")

        return enabled

    def _parse_feature_space_review_models(self, feature_space_review_root: Dict) -> List[str]:
        models = feature_space_review_root.get("models", ["KNN", "EGB", "RF"])
        if not isinstance(models, list):
            raise ValueError("feature_space_review.models must be a list of model names.")

        validated_models: List[str] = []
        for model_name in models:
            if not isinstance(model_name, str):
                raise ValueError("Each item in feature_space_review.models must be a string model name.")

            if model_name not in self.MODEL_FACTORIES_BY_NAME:
                raise ValueError(
                    f"Model '{model_name}' is not recognized. Available models: {list(self.MODEL_FACTORIES_BY_NAME.keys())}"
                )

            validated_models.append(model_name)

        return validated_models

    def get_sensor_ablation_enabled(self) -> bool:
        return self.sensor_ablation["enabled"]

    def get_sensor_ablation_streams(self) -> List[List[str]]:
        return copy.deepcopy(self.sensor_ablation["streams"])

    def get_sensor_ablation_review_enabled(self) -> bool:
        return self.sensor_ablation_review["enabled"]

    def get_sensor_ablation_review_models(self) -> List[str]:
        return copy.deepcopy(self.sensor_ablation_review["models"])

    def get_sensor_ablation_review_stream_group(self) -> List[str]:
        return copy.deepcopy(self.sensor_ablation_review["stream_group"])

    def get_feature_space_review_enabled(self) -> bool:
        return self.feature_space_review["enabled"]

    def get_feature_space_review_models(self) -> List[str]:
        return copy.deepcopy(self.feature_space_review["models"])

    def _parse_hyperparameter_save_enabled(self, hyperparameter_save_root: Dict) -> bool:
        enabled = hyperparameter_save_root.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("hyperparameter_save.enabled must be a boolean.")
        return enabled

    def _parse_hyperparameter_save_models(self, hyperparameter_save_root: Dict) -> List[str]:
        models = hyperparameter_save_root.get("models", [])
        if not isinstance(models, list):
            raise ValueError("hyperparameter_save.models must be a list of model names.")

        validated_models: List[str] = []
        for model_name in models:
            if not isinstance(model_name, str):
                raise ValueError("Each item in hyperparameter_save.models must be a string model name.")

            if model_name not in self.MODEL_FACTORIES_BY_NAME:
                raise ValueError(
                    f"Model '{model_name}' is not recognized. Available models: {list(self.MODEL_FACTORIES_BY_NAME.keys())}"
                )

            validated_models.append(model_name)

        return validated_models

    def get_hyperparameter_save_enabled(self) -> bool:
        return self.hyperparameter_save["enabled"]

    def get_hyperparameter_save_models(self) -> List[str]:
        return copy.deepcopy(self.hyperparameter_save["models"])