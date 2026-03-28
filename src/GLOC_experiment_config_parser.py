from model_type import ModelType
from models.base import BaseModel
from models.logistic_regression import LogisticRegression
from models.transformer import TransformerModel
from typing import Optional, Dict, List
from pathlib import Path

import json

class GLOCExperimentConfigParser:
    MODELS_BY_NAME: Dict[str, BaseModel] = {
        "Logistic Regression": LogisticRegression(config = {}),
        "Transformer": TransformerModel(config = {}),
    }

    def __init__(self, config_location: Optional[str] = None) -> None:
        if config_location is None:
            config_path = Path(__file__).resolve().parent.parent / "GLOC_experiment_config.json"
        else:
            config_path = Path(config_location).expanduser().resolve()

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self._parse_general_configs()
        self._parse_shared_data_configs()
        self._parse_advanced_data_configs()
        self._parse_traditional_data_configs()

    # General Configurations
    def _parse_general_configs(self) -> None:
        self.model = self._parse_model()
        self.model_type = self._parse_model_type()
        self.random_seed = self._parse_random_seed()
        self.data_path = self._parse_data_path()

    def _parse_shared_data_configs(self) -> None:
        shared_data_parameters = self._get_shared_data_parameters()
        self.subject_to_analyze = self._parse_subject_to_analyze(shared_data_parameters)
        self.trial_to_analyze = self._parse_trial_to_analyze(shared_data_parameters)
        self.analysis_type = self._parse_analysis_type(shared_data_parameters)
        self.remove_NaN_trials = self._parse_remove_NaN_trials(shared_data_parameters)

    def _parse_advanced_data_configs(self) -> None:
        advanced_data_parameters = self._get_advanced_data_parameters()
        self.num_splits = self._parse_num_splits(advanced_data_parameters)
        self.kfold_ID = self._parse_kfold_ID(advanced_data_parameters)
        self.impute_path = self._parse_impute_path(advanced_data_parameters)
        self.impute_type = self._parse_impute_type(advanced_data_parameters)
        self.n_neighbors = self._parse_n_neighbors(advanced_data_parameters)
        self.baseline_window = self._parse_baseline_window(advanced_data_parameters)
        self.save_impute = self._parse_save_impute(advanced_data_parameters)
        self.load_impute = self._parse_load_impute(advanced_data_parameters)

    def _parse_traditional_data_configs(self) -> None:
        traditional_data_parameters = self._get_traditional_data_parameters()
        self.backstep = self._parse_backstep(traditional_data_parameters)
        self.data_rate = self._parse_data_rate(traditional_data_parameters)
        self.classifier_type = self._parse_classifier_type(traditional_data_parameters)
        self.offset = self._parse_offset(traditional_data_parameters)
        self.time_start = self._parse_time_start(traditional_data_parameters)
            
    def _parse_model(self) -> BaseModel:
        if "model" not in self.config:
            raise ValueError("model is missing from config. It should be a string of the name of the model.")

        model_config = self.config.get("model", "")
        if model_config not in self.MODELS_BY_NAME:
            raise ValueError(f"Model '{model_config}' is not recognized. Available models: {list(self.MODELS_BY_NAME.keys())}")

        return self.MODELS_BY_NAME[model_config]
            
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
    
    def get_model(self) -> BaseModel:
        return self.model
    
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

    def _parse_impute_path(self, advanced_data_parameters: Dict) -> str:
        if "impute_path" not in advanced_data_parameters:
            raise ValueError("impute_path is missing from config. It should be a string indicating the path to save/load imputed data.")

        return advanced_data_parameters.get("impute_path")

    def _parse_impute_type(self, advanced_data_parameters: Dict) -> int:
        if "impute_type" not in advanced_data_parameters:
            raise ValueError("impute_type is missing from config. It should be an integer indicating the type of imputation to perform (e.g., 0 for no imputation, 1 for KNN imputation).")

        return advanced_data_parameters.get("impute_type")

    def _parse_n_neighbors(self, advanced_data_parameters: Dict) -> int:
        if "n_neighbors" not in advanced_data_parameters:
            raise ValueError("n_neighbors is missing from config. It should be an integer indicating the number of neighbors to use for KNN imputation.")

        return advanced_data_parameters.get("n_neighbors")

    def _parse_baseline_window(self, advanced_data_parameters: Dict) -> float:
        if "baseline_window" not in advanced_data_parameters:
            raise ValueError("baseline_window is missing from config. It should be a float indicating the time window (in seconds) to use for baseline calculation.")

        return advanced_data_parameters.get("baseline_window")

    def _parse_save_impute(self, advanced_data_parameters: Dict) -> bool:
        if "save_impute" not in advanced_data_parameters:
            raise ValueError("save_impute is missing from config. It should be a boolean indicating whether to save imputed data to disk.")

        return advanced_data_parameters.get("save_impute")

    def _parse_load_impute(self, advanced_data_parameters: Dict) -> bool:
        if "load_impute" not in advanced_data_parameters:
            raise ValueError("load_impute is missing from config. It should be a boolean indicating whether to load imputed data from disk if available.")

        return advanced_data_parameters.get("load_impute")

    def get_num_splits(self) -> int:
        return self.num_splits
    
    def get_kfold_ID(self) -> int:
        return self.kfold_ID
    
    def get_impute_path(self) -> str:
        return self.impute_path
    
    def get_impute_type(self) -> int:
        return self.impute_type
    
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

    def _parse_classifier_type(self, traditional_data_parameters: Dict) -> str:
        if "classifier_type" not in traditional_data_parameters:
            raise ValueError("classifier_type is missing from config. It should be a string indicating the type of classifier to use for traditional pipeline (e.g., 'SVM', 'KNN').")

        return traditional_data_parameters.get("classifier_type")

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
    
    def get_classifier_type(self) -> str:
        return self.classifier_type
    
    def get_offset(self) -> float:
        return self.offset
    
    def get_time_start(self) -> float:
        return self.time_start