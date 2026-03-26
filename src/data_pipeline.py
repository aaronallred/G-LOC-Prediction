from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from .advanced_data_pipeline import AdvancedDataPipeline
from .traditional_data_pipeline import TraditionalDataPipeline
from .GLOC_experiment_config_parser import GLOCExperimentConfigParser
from .model_type import ModelType
from .models.base import BaseModel

PipelineKind = Literal["auto", "advanced", "traditional"]


@dataclass(frozen=True)
class ExperimentMetadata:
    trial_id: np.ndarray
    trial_ints: np.ndarray
    time_s: np.ndarray
    event_validated: np.ndarray
    subject: np.ndarray
    afe_indicator: np.ndarray


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration object extracted from experiment config.
    
    This dataclass holds all configuration needed for the data pipeline,
    extracted at initialization time for clarity and reproducibility.
    """
    data_path: str
    model_type: ModelType
    random_seed: int
    num_splits: int
    kfold_ID: int
    impute_path: str
    impute_type: int
    n_neighbors: int
    baseline_window: float
    save_impute: bool
    load_impute: bool
    backstep: float
    data_rate: int
    offset: float
    time_start: float
    remove_NaN_trials: bool
    subject_to_analyze: Optional[int]
    trial_to_analyze: Optional[int]
    analysis_type: int
    classifier_type: str
    select_features: list[str]

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
            model: Optional[BaseModel] = None,
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
        self.model = model
        self._backend: Optional[Any] = None
        
        # Extract and validate configuration early
        self._config = self._extract_config()

    def _extract_config(self) -> PipelineConfig:
        """Extract and validate configuration from parser.
        
        Configuration is extracted once at initialization to ensure consistency
        and catch issues early.
        
        Args:
            data_path_override: Optional override for data_path
            
        Returns:
            PipelineConfig: Immutable configuration object
            
        Raises:
            ValueError: If configuration is invalid or missing required fields
        """
        try:            
            return PipelineConfig(
                data_path=self._config_parser.get_data_path(),
                model_type=self._config_parser.get_model_type(),
                random_seed=self._config_parser.get_random_seed(),
                num_splits=self._config_parser.get_num_splits(),
                kfold_ID=self._config_parser.get_kfold_ID(),
                impute_path=self._config_parser.get_impute_path(),
                impute_type=self._config_parser.get_impute_type(),
                n_neighbors=self._config_parser.get_n_neighbors(),
                baseline_window=self._config_parser.get_baseline_window(),
                save_impute=self._config_parser.get_save_impute(),
                load_impute=self._config_parser.get_load_impute(),
                backstep=self._config_parser.get_backstep(),
                data_rate=self._config_parser.get_data_rate(),
                offset=self._config_parser.get_offset(),
                time_start=self._config_parser.get_time_start(),
                remove_NaN_trials=self._config_parser.get_remove_NaN_trials(),
                subject_to_analyze=self._config_parser.get_subject_to_analyze(),
                trial_to_analyze=self._config_parser.get_trial_to_analyze(),
                analysis_type=self._config_parser.get_analysis_type(),
                classifier_type=self._config_parser.get_classifier_type(),
                select_features=self._config_parser.get_select_features(),
            )
        except Exception as e:
            raise ValueError(f"Failed to extract configuration: {e}") from e

    @property
    def model_type(self) -> ModelType:
        """Get the model type from configuration."""
        return self._config.model_type

    @property
    def config(self) -> PipelineConfig:
        """Get the extracted configuration object."""
        return self._config

    @property
    def backend(self) -> Any:
        """Lazily build and return the appropriate backend pipeline."""
        if self._backend is None:
            self._backend = self._build_backend()
        return self._backend

    def get_data(self, **kwargs: Any) -> Any:
        """Execute the selected backend data pipeline.

        For advanced pipelines this returns:
        ``x_train, x_test, y_train, y_test, all_features``

        For traditional pipelines this returns:
        ``x_feature_matrix, y_gloc_labels``
        
        Args:
            **kwargs: Additional arguments to pass to the backend pipeline,
                     overriding defaults from configuration
                     
        Returns:
            Tuple or data from the backend pipeline
        """
        backend_type = self._resolve_pipeline_kind()
        request_kwargs = dict(kwargs)

        if "model_type" not in request_kwargs:
            request_kwargs["model_type"] = self._config.model_type

        if backend_type == "advanced":
            request_kwargs.setdefault("num_splits", self._config_parser.get_num_splits())
            request_kwargs.setdefault("kfold_ID", self._config_parser.get_kfold_ID())
            request_kwargs.setdefault("impute_path", self._config_parser.get_impute_path())
        else:
            request_kwargs.setdefault("classifier_type", self._resolve_classifier_name())
            request_kwargs.setdefault("select_features", self._resolve_select_features(request_kwargs))
            request_kwargs.setdefault("backstep", self._config_parser.get_backstep())
            request_kwargs.setdefault("data_rate", self._config_parser.get_data_rate())
            request_kwargs.setdefault("offset", self._config_parser.get_offset())
            request_kwargs.setdefault("time_start", self._config_parser.get_time_start())

        request_kwargs.setdefault("remove_NaN_trials", self._config_parser.get_remove_NaN_trials())
        request_kwargs.setdefault("subject_to_analyze", self._config_parser.get_subject_to_analyze())
        request_kwargs.setdefault("trial_to_analyze", self._config_parser.get_trial_to_analyze())
        request_kwargs.setdefault("analysis_type", self._config_parser.get_analysis_type())

        return self.backend.get_data(**request_kwargs)

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
        if self.model is not None and hasattr(self.model, "is_traditional"):
            return "traditional" if self.model.is_traditional else "advanced"
        
    

    def _resolve_classifier_name(self) -> str:
        """Get classifier name from model or config.
        
        Returns:
            Classifier name/type
            
        Raises:
            ValueError: If classifier name cannot be determined
        """
        if self.model is not None and hasattr(self.model, "get_name"):
            return str(self.model.get_name())
        
        # Fallback to classifier_type from config
        return self._config_parser.get_classifier_type()

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
        if "select_features" in current_kwargs and current_kwargs["select_features"] is not None:
            return list(current_kwargs["select_features"])

        if self.model is not None:
            config = getattr(self.model, "config", {}) or {}
            features = config.get("select_features")
            if features is not None:
                return list(features)

        # Use from experiment config
        features = self._config_parser.get_select_features()
        if not features:
            raise ValueError(
                "select_features not found in model config, kwargs, or experiment config. "
                "Provide select_features in one of these locations."
            )
        return features