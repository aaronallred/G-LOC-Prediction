from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import os
import json
from collections import OrderedDict

from advanced_data_pipeline import AdvancedDataPipeline
from traditional_data_pipeline import TraditionalDataPipeline
from GLOC_experiment_config_parser import GLOCExperimentConfigParser
from model_type import ModelType
from models.base import BaseModel

PipelineKind = Literal["auto", "advanced", "traditional"]


@dataclass(frozen=True)
class ExperimentMetadata:
    trial_id: np.ndarray
    trial_ints: np.ndarray
    time_s: np.ndarray
    event_validated: np.ndarray
    subject: np.ndarray
    afe_indicator: np.ndarray

class DataPipeline:
    """Facade that routes data loading to the advanced or traditional backend.

    This class is the single entry point for data preparation. It selects a backend
    either explicitly based on the model type and routes to the appropriate implementation.
    
    Configuration is sourced from GLOCExperimentConfigParser for reproducibility and
    centralized configuration management.
    
    Args:
        config_parser: GLOCExperimentConfigParser instance containing experiment configuration
        model: Optional BaseModel instance for type resolution
        data_path_override: Optional override for data_path from config (for testing)
    """

    def __init__(
            self,
            config_parser: GLOCExperimentConfigParser,
    ) -> None:
        """Initialize DataPipeline with configuration from parser.
        
        Args:
            config_parser: Parser instance that reads experiment configuration
            model: Optional model for backend resolution. If not provided, model_type
                   from config will be used for backend routing.
            data_path_override: Optional override for data path (useful for testing)
            
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        self._config_parser = config_parser

    def get_data(self) -> Any:
        """Execute the selected backend data pipeline.

        For advanced pipelines this returns:
        ``x_train, x_test, y_train, y_test, all_features``

        For traditional pipelines this returns:
        ``x_feature_matrix, y_gloc_labels``
        
        Returns:
            Tuple or data from the backend pipeline
        """
        backend_type = self._resolve_pipeline_kind()
        backend_data_pipeline = self._build_backend()
        request_kwargs: dict[str, Any] = {
            "model_type": self._config_parser.get_model_type(),
            "remove_NaN_trials": self._config_parser.get_remove_NaN_trials(),
            "subject_to_analyze": self._config_parser.get_subject_to_analyze(),
            "trial_to_analyze": self._config_parser.get_trial_to_analyze(),
            "analysis_type": self._config_parser.get_analysis_type(),
        }

        if backend_type == "advanced":
            request_kwargs["num_splits"] = self._config_parser.get_num_splits()
            request_kwargs["kfold_ID"] = self._config_parser.get_kfold_ID()
            request_kwargs["impute_path"] = self._config_parser.get_impute_path()
            request_kwargs["impute_type"] = self._config_parser.get_impute_type()
            request_kwargs["n_neighbors"] = self._config_parser.get_n_neighbors()
            request_kwargs["baseline_window"] = self._config_parser.get_baseline_window()
            request_kwargs["save_impute"] = self._config_parser.get_save_impute()
            request_kwargs["load_impute"] = self._config_parser.get_load_impute()
        else:
            request_kwargs["classifier_type"] = self._resolve_classifier_name()
            request_kwargs["select_features"] = self._resolve_select_features(request_kwargs)
            request_kwargs["backstep"] = self._config_parser.get_backstep()
            request_kwargs["data_rate"] = self._config_parser.get_data_rate()
            request_kwargs["offset"] = self._config_parser.get_offset()
            request_kwargs["time_start"] = self._config_parser.get_time_start()

        return backend_data_pipeline.get_data(**request_kwargs)

    def _build_backend(self) -> Any:
        """Build the appropriate backend pipeline based on model type.
        
        Returns:
            Initialized backend pipeline (Advanced or Traditional)
            
        Raises:
            ValueError: If backend type cannot be determined
        """
        pipeline_kind = self._resolve_pipeline_kind()
        
        if pipeline_kind == "traditional":
            return TraditionalDataPipeline(
                data_path=self._config_parser.get_data_path(),
                random_seed=self._config_parser.get_random_seed()
            )
        else:
            return AdvancedDataPipeline(
                data_path=self._config_parser.get_data_path(),
                random_seed=self._config_parser.get_random_seed()
            )

    def _resolve_pipeline_kind(self) -> Literal["advanced", "traditional"]:
        """Determine which backend pipeline to use.
        
        Returns:
            "traditional" if model is explicitly traditional, otherwise "advanced"
            
        Raises:
            ValueError: If backend type cannot be determined
        """
        model = self._config_parser.get_model()

        if model is None or not hasattr(model, "is_traditional"):
            raise ValueError("Model does not have 'is_traditional' attribute. Unable to determine pipeline kind.")

        return "traditional" if model.is_traditional else "advanced"
    

    def _resolve_classifier_name(self) -> str:
        """Get classifier name from model or config.
        
        Returns:
            Classifier name/type
            
        Raises:
            ValueError: If classifier name cannot be determined
        """
        model = self._config_parser.get_model()
        if model is None or not hasattr(model, "get_name"):
            raise ValueError("Unable to determine classifier name.")

        return model.get_name()

    def _resolve_select_features(self, current_kwargs: dict[str, Any]) -> list[str]:
        """Resolve which features to select for the pipeline.
        
        Priority order:
        1. select_features kwarg passed to get_data()
        2. model.config['select_features'] if model provided
        3. select_features from experiment config
        
        Args:
            current_kwargs: Keyword arguments passed to get_data()
            
        Returns:
            List of feature names to use
            
        Raises:
            ValueError: If select_features cannot be determined
        """

        # Function to load in median hyperparameters from a simple JSON
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_type_string = f"{current_kwargs['model_type'].afe_filter}_{current_kwargs['model_type'].feature_set}"
        json_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type_string, f'median_hyperparameters_{current_kwargs["classifier_type"]}.json')

        with open(json_path, 'r') as f:
            data = json.load(f)

        return data['selected_features']