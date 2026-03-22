from dataclasses import dataclass
import logging
import os
import pickle
import re
from itertools import islice
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import faiss
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from baseline import BaselineContext, baseline_data
from features import FEATURE_REGISTRY, RawEEGGroup, ProcessedEEGGroup
from src.models.base import BaseModel

logger = logging.getLogger(__name__)

@dataclass(frozen = True)
class ModelType:
    afe_filter: str  # "noAFE" or "Complete"
    feature_set: str # "Explicit" or "Implicit"
    
class DataPipeline:
    """ Unified data pipeline for Traditional and Advanced Classifiers """

    FEATURE_GROUPS_BY_MODEL_TYPE = {
        ModelType("noAFE", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "AFE", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ModelType("noAFE", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "rawEEG"),
        ModelType("Complete", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "AFE", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ModelType("Complete", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE"),
    }
    BASELINING_CHARACTERISTICS_BY_MODEL_TYPE = {
        "noAFE": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
        "Complete": ["v0", "v1", "v2", "v5", "v6"],
    }

    _CLASSIFIER_HYPERPARAMETERS: Dict[BaseModel, Dict[str, Any]] = {
        "logreg": {
            "baseline_window": 5,
            "window_size": 12.5,
            "stride": 0.25,
            "feature_reduction_type": "lasso",
            "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 5,
        },
        "RF": {
            "baseline_window": 18.75,
            "window_size": 7.5,
            "stride": 0.25,
            "feature_reduction_type": "none",
            "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        },
        "LDA": {
            "baseline_window": 46.25,
            "window_size": 15,
            "stride": 0.25,
            "feature_reduction_type": "lasso",
            "baseline_methods_to_use": ["v0", "v1", "v2"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        },
        "SVM": {
            "baseline_window": 32.5,
            "window_size": 15,
            "stride": 0.25,
            "feature_reduction_type": "ridge",
            "baseline_methods_to_use": ["v0", "v1", "v2"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        },
        "EGB": {
            "baseline_window": 46.25,
            "window_size": 12.5,
            "stride": 0.25,
            "feature_reduction_type": "lasso",
            "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        },
        "KNN": {
            "baseline_window": 32.5,
            "window_size": 15,
            "stride": 0.25,
            "feature_reduction_type": "performance",
            "baseline_methods_to_use": ["v0", "v1", "v2"],
            "imbalance_type": "ros",
            "impute_type": 1,
            "n_neighbors": 5,
        },
    }

    def __init__(
            self, 
            datafolder: str, 
            model: BaseModel, 
            model_type: ModelType,
            baseline_window: float = 32.5,
            window_size: Optional[float] = None,
            stride: Optional[float] = None,
            feature_reduction_type: Optional[str] = None,
            imbalance_type: str = "none",
            should_impute: bool = True,
            n_neighbors: int = 4,
            num_splits: int = 10,
            kfold_ID: int = 0,
            subject_to_analyze: Optional[str] = None,
            trial_to_analyze: Optional[str] = None,
            analysis_type: int = 2,
            remove_NaN_trials: bool = True,
            impute_file_name: Optional[str] = None,
            save_impute: bool = True,
            load_impute: bool = True
        ):
        self.datafolder = datafolder
        self.model = model
        self.model_type = model_type

        # Data Processing Hyperparameters
        self.baseline_window = baseline_window
        self.window_size = window_size
        self.stride = stride
        self.feature_reduction_type = feature_reduction_type
        self.feature_groups_to_analyze = None
        self.baseline_methods_to_use = None
        self.imbalance_type = imbalance_type
        self.should_impute = should_impute
        self.n_neighbors = n_neighbors

        # Dataset Splitting Parameters
        self.num_splits = num_splits
        self.kfold_ID = kfold_ID
        self.subject_to_analyze = subject_to_analyze
        self.trial_to_analyze = trial_to_analyze
        self.analysis_type = analysis_type
        self.remove_NaN_trials = remove_NaN_trials

        # Data Saving Parameters
        self.impute_file_name = impute_file_name
        self.save_impute = save_impute
        self.load_impute = load_impute

    def get_data(self):
        self._assign_hyperparameters_by_classifier(self.model.get_name())
        self._assign_feature_groups_and_baseline_methods()

        return None
    
    def _assign_hyperparameters_by_classifier(self, classifier_type: str) -> None:
        """Set hyperparameters for a given classifier type."""

        params = self._CLASSIFIER_HYPERPARAMETERS.get(classifier_type)
        if params is None:
            # Using advanced classifier which should have provided parameters
            logger.warning(f"Classifier type '{classifier_type}' not found in predefined hyperparameters. Using initialization values.")

        self.baseline_window = params.get("baseline_window", self.baseline_window)
        self.window_size = params.get("window_size", self.window_size)
        self.stride = params.get("stride", self.stride)
        self.feature_reduction_type = params.get("feature_reduction_type", self.feature_reduction_type)
        self.baseline_methods_to_use = params.get("baseline_methods_to_use", self.baseline_methods_to_use)
        self.imbalance_type = params.get("imbalance_type", self.imbalance_type)
        self.impute_type = params.get("impute_type", self.impute_type)
        self.n_neighbors = params.get("n_neighbors", self.n_neighbors)

    def _assign_feature_groups_and_baseline_methods(self) -> None:
        """Set feature groups and baseline methods based on model type."""
        self.feature_groups_to_analyze = self.FEATURE_GROUPS_BY_MODEL_TYPE[self.model_type]
        self.baseline_methods_to_use = self.BASELINING_CHARACTERISTICS_BY_MODEL_TYPE[self.model_type.afe_filter]

        # NOTE:
        # AFE indicator is required for EEG imputation in Complete models,
        # but is only included as a predictive feature for Explicit models.