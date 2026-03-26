from src.model_type import ModelType
from typing import Optional, Dict, List

import json

class GLOCExperimentConfigParser:
    def __init__(self, config_Location: str = "./../GLOC_experiment_config.json") -> None:
        with open(config_Location, "r") as f:
            self.config = json.load(f)

    def get_config(self) -> Dict:
        return self.config
    


    # General Configurations
    def get_model_type(self) -> ModelType:
        model_type_config = self.config.get("model_type", [])

        if len(model_type_config) != 2:
            raise ValueError("model_type must be [<afe_filter>, <feature_set>] where afe_filter is 'Complete' or 'noAFE' and feature_set is 'Explicit' or 'Implicit'.")

        afe_filter = model_type_config[0]
        feature_set = model_type_config[1]

        if afe_filter not in ["Complete", "noAFE"] or feature_set not in ["Explicit", "Implicit"]:
            raise ValueError("model_type must be [<afe_filter>, <feature_set>] where afe_filter is 'Complete' or 'noAFE' and feature_set is 'Explicit' or 'Implicit'.")

        return ModelType(afe_filter = afe_filter, feature_set = feature_set)
    
    def get_random_seed(self) -> int:
        if "random_seed" not in self.config:
            raise ValueError("random_seed is missing from config. It should be an integer indicating the random seed to use for reproducibility.")

        return self.config.get("random_seed")
    
    def get_data_path(self) -> str:
        if "data_path" not in self.config:
            raise ValueError("data_path is missing from config. It should be a string indicating the path to the data directory.")

        return self.config.get("data_path")
    

    
    # Shared Data Configurations
    def get_subject_to_analyze(self) -> Optional[int]:
        if "subject_to_analyze" not in self.config:
            raise ValueError("subject_to_analyze is missing from config. It should be an integer subject ID or null to analyze all subjects.")

        return self.config.get("subject_to_analyze")
    
    def get_trial_to_analyze(self) -> Optional[int]:
        if "trial_to_analyze" not in self.config:
            raise ValueError("trial_to_analyze is missing from config. It should be an integer trial ID or null to analyze all trials.")

        return self.config.get("trial_to_analyze")
    
    def get_analysis_type(self) -> int:
        if "analysis_type" not in self.config:
            raise ValueError("analysis_type is missing from config. It should be an integer (1, 2, or 3) indicating the type of analysis to perform.")

        return self.config.get("analysis_type")
    
    def get_remove_NaN_trials(self) -> bool:
        if "remove_NaN_trials" not in self.config:
            raise ValueError("remove_NaN_trials is missing from config. It should be a boolean indicating whether to remove trials with NaN values.")

        return self.config.get("remove_NaN_trials")
    


    # Advanced Classifier Configurations
    def get_num_splits(self) -> int:
        if "num_splits" not in self.config:
            raise ValueError("num_splits is missing from config. It should be an integer indicating the number of splits for train/test.")

        return self.config.get("num_splits")
    
    def get_kfold_ID(self) -> int:
        if "kfold_ID" not in self.config:
            raise ValueError("kfold_ID is missing from config. It should be an integer indicating the k-fold ID for train/test splitting.")

        return self.config.get("kfold_ID")
    
    def get_impute_path(self) -> str:
        if "impute_path" not in self.config:
            raise ValueError("impute_path is missing from config. It should be a string indicating the path to save/load imputed data.")

        return self.config.get("impute_path")
    
    def get_impute_type(self) -> int:
        if "impute_type" not in self.config:
            raise ValueError("impute_type is missing from config. It should be an integer indicating the type of imputation to perform (e.g., 0 for no imputation, 1 for KNN imputation).")

        return self.config.get("impute_type")
    
    def get_n_neighbors(self) -> int:
        if "n_neighbors" not in self.config:
            raise ValueError("n_neighbors is missing from config. It should be an integer indicating the number of neighbors to use for KNN imputation.")

        return self.config.get("n_neighbors")
    
    def get_baseline_window(self) -> float:
        if "baseline_window" not in self.config:
            raise ValueError("baseline_window is missing from config. It should be a float indicating the time window (in seconds) to use for baseline calculation.")

        return self.config.get("baseline_window")
    
    def get_save_impute(self) -> bool:
        if "save_impute" not in self.config:
            raise ValueError("save_impute is missing from config. It should be a boolean indicating whether to save imputed data to disk.")

        return self.config.get("save_impute")
    
    def get_load_impute(self) -> bool:
        if "load_impute" not in self.config:
            raise ValueError("load_impute is missing from config. It should be a boolean indicating whether to load imputed data from disk if available.")

        return self.config.get("load_impute")
    


    # Traditional Classifier Configurations
    def get_backstep(self) -> float:
        if "backstep" not in self.config:
            raise ValueError("backstep is missing from config. It should be a float indicating the backstep (in seconds) to use for traditional pipeline.")

        return self.config.get("backstep")
    
    def get_data_rate(self) -> int:
        if "data_rate" not in self.config:
            raise ValueError("data_rate is missing from config. It should be an integer indicating the data rate (in Hz) to use for traditional pipeline.")

        return self.config.get("data_rate")
    
    def get_classifier_type(self) -> str:
        if "classifier_type" not in self.config:
            raise ValueError("classifier_type is missing from config. It should be a string indicating the type of classifier to use for traditional pipeline (e.g., 'SVM', 'KNN').")

        return self.config.get("classifier_type")
    
    def get_offset(self) -> float:
        if "offset" not in self.config:
            raise ValueError("offset is missing from config. It should be a float indicating the offset (in seconds) to use for traditional pipeline.")

        return self.config.get("offset")
    
    def get_time_start(self) -> float:
        if "time_start" not in self.config:
            raise ValueError("time_start is missing from config. It should be a float indicating the time start (in seconds) to use for traditional pipeline.")

        return self.config.get("time_start")