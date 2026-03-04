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

logger = logging.getLogger(__name__)

class TraditionalDataManager:
    FEATURE_GROUPS_BY_MODEL_TYPE = {
        ("noAFE", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "AFE", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ("noAFE", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "rawEEG"),
        ("Complete", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "AFE", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ("Complete", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE"),
    }
    # Mapping of participant -> DC trial numbers for GOR EEG data files
    _EEG_PARTICIPANT_TRIALS = {
        1: [1, 2, 3],  2: [1, 2, 3],  3: [1, 2, 3],  4: [1, 2, 3],  5: [1, 2, 3],
        6: [1, 4, 6],  7: [2, 4, 6],  8: [1, 3],     9: [2, 5, 6],
        10: [2, 4, 5], 11: [1],       12: [1, 5],     13: [1, 3, 6],
    }

    _EEG_BASELINE_BANDS = ["delta", "theta", "alpha", "beta"]

    """Data manager for traditional data pipeline."""
    def __init__(self, data_path: str = "../data/", testing: bool = False, random_seed: int = 42, use_reduced_dataset: bool = False) -> None:
        self.data_path = data_path
        self._data_locations = None
        self.testing = testing
        self.random_seed = random_seed
        self.use_reduced_dataset = use_reduced_dataset

    def get_data(self, backstep, data_rate, classifier_type, model_type, select_features):
        """Return data for a given set of parameters."""

        baseline_window, window_size, stride, feature_reduction_type, baseline_methods_to_use, imbalance_type, impute_type, n_neighbors = self._get_hyperparameters_by_classifier(classifier_type)
        feature_groups_to_analyze, baseline_methods_to_use = self._get_feature_groups_and_baseline_methods(model_type, baseline_methods_to_use)

        # Code for loading txt and processing data would go here

        return None  # Placeholder for actual data return
    
    def _get_hyperparameters_by_classifier(self, classifier_type):
        """Return hyperparameters for a given classifier type."""
        if classifier_type == 'logreg':
            # Specifying Methods from Sequential optimization
            baseline_window = 5  # seconds - PULLED FROM NIKKI PAPER
            window_size = 12.5 # seconds - PULLED FROM NIKKI PAPER
            stride = 0.25 # seconds - PULLED FROM NIKKI PAPER
            imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
            feature_reduction_type = 'lasso' #- PULLED FROM NIKKI PAPER
            baseline_methods_to_use = ['v0', 'v1', 'v2','v5','v6','v7','v8'] #- PULLED FROM NIKKI PAPER
            impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
            n_neighbors = 5  # -For imputation PULLED FROM NIKKI PAPER

            ## Investigating different windows per EVAN ANDERSON
            #window_size = 8 # ~ 0.1 hit to f1 score



        if classifier_type == 'RF':
            # Specifying Methods from Sequential optimization
            baseline_window = 18.75  # seconds - PULLED FROM NIKKI PAPER
            window_size = 7.5  # seconds - PULLED FROM NIKKI PAPER
            stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
            feature_reduction_type = 'none'  # - PULLED FROM NIKKI PAPER
            threshold = 30  # - PULLED FROM NIKKI PAPER
            baseline_methods_to_use = ['v0', 'v1', 'v2','v5','v6','v7','v8']  # - PULLED FROM NIKKI PAPER
            imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
            impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
            n_neighbors = 3  # -For imputation PULLED FROM NIKKI PAPER
            # Code for loading txt

            ## Investigating different windows per EVAN ANDERSON
            #window_size = 5 # ~ 0.1 hit to f1 score


        if classifier_type == 'LDA':
            # Specifying Methods from Sequential optimization
            baseline_window = 46.25  # seconds - PULLED FROM NIKKI PAPER
            window_size = 15  # seconds - PULLED FROM NIKKI PAPER
            stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
            feature_reduction_type = 'lasso'  # - PULLED FROM NIKKI PAPER
            baseline_methods_to_use = ['v0', 'v1', 'v2']  # - PULLED FROM NIKKI PAPER
            imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
            impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
            n_neighbors = 3  # -For imputation PULLED FROM NIKKI PAPER
            # Code for loading txt

            ## Investigating different windows per EVAN ANDERSON
            #window_size = 10 # ~ 0.3 hit to f1 score


        if classifier_type == 'SVM':
            # Specifying Methods from Sequential optimization
            baseline_window = 32.5  # seconds - PULLED FROM NIKKI PAPER
            window_size = 15  # seconds - PULLED FROM NIKKI PAPER
            stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
            feature_reduction_type = 'ridge'  # - PULLED FROM NIKKI PAPER
            threshold = 10  # - PULLED FROM NIKKI PAPER
            baseline_methods_to_use = ['v0', 'v1', 'v2']  # - PULLED FROM NIKKI PAPER
            impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
            n_neighbors = 3  # - For imputation PULLED FROM NIKKI PAPER
            imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER

            ## Investigating different windows per EVAN ANDERSON
            #window_size = 8 # ~ 0.2 hit to f1 score


        if classifier_type == 'EGB':
            # Specifying Methods from Sequential optimization
            baseline_window = 46.25  # seconds - PULLED FROM NIKKI PAPER
            window_size = 12.5  # seconds - PULLED FROM NIKKI PAPER
            stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
            feature_reduction_type = 'lasso'  # - PULLED FROM NIKKI PAPER
            baseline_methods_to_use = ['v0', 'v1', 'v2','v5','v6','v7','v8']  # - PULLED FROM NIKKI PAPER
            imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
            impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
            n_neighbors = 3  # -For imputation PULLED FROM NIKKI PAPER
            # Code for loading txt

            ## Investigating different windows per EVAN ANDERSON
            #window_size = 8 # ~ 0.1 hit to f1 score


        if classifier_type == 'KNN':
            # Specifying Methods from Sequential optimization
            baseline_window = 32.5  # seconds - PULLED FROM NIKKI PAPER
            window_size = 15  # seconds - PULLED FROM NIKKI PAPER
            stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
            feature_reduction_type = 'performance'  # - PULLED FROM NIKKI PAPER
            baseline_methods_to_use = ['v0', 'v1', 'v2']  # - PULLED FROM NIKKI PAPER
            imbalance_type = 'ros' # - PULLED FROM NIKKI PAPER
            impute_type = 1 # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
            n_neighbors = 5 # -For imputation PULLED FROM NIKKI PAPER
            # Code for loading txt

            ## Investigating different windows per EVAN ANDERSON
            # window_size = 12 # ~ 0.1 hit to f1 score

        return baseline_window, window_size, stride, feature_reduction_type, baseline_methods_to_use, imbalance_type, impute_type, n_neighbors
    
    def _get_feature_groups_and_baseline_methods(self, model_type: Tuple[str, str], baseline_methods_to_use: List[str]) -> Tuple[Sequence[str], List[str]]:
        feature_groups_to_analyze = self.FEATURE_GROUPS_BY_MODEL_TYPE[model_type]
        baseline_methods_to_use = ["v0", "v1", "v2", "v5", "v6"] if model_type[0] == "Complete" else baseline_methods_to_use

        # NOTE:
        # AFE indicator is required for EEG imputation in complete models,
        # but is only included as a predictive feature for explicit models.

        return feature_groups_to_analyze, baseline_methods_to_use

    def _get_data_locations(self) -> Dict[str, Any]:
        """
        Get file locations for all relevant data files.

        Returns:
            dict with keys: "main", "baseline", "demographic", "eeg_list", "baseline_eeg_processed_list"
        """
        if self._data_locations is not None:
            return self._data_locations

        eeg_dir = "GLOC_GOR_EEG_data_participants_1-13"
        list_of_eeg_data_file_paths = [
            os.path.join(self.data_path, eeg_dir, f"GLOC_{p:02d}_DC{t}_25Hz_EEG_power_wMAR.xlsx")
            for p, trials in self._EEG_PARTICIPANT_TRIALS.items()
            for t in trials
        ]

        list_of_baseline_eeg_processed_file_paths = [
            os.path.join(self.data_path, f"GLOC_EEG_baseline_{band}_noAFE1.csv")
            for band in self._EEG_BASELINE_BANDS
        ]

        if self.use_reduced_dataset:
            logger.info("!!!!!!!!!!!!!!!!!!! USING REDUCED DATASET !!!!!!!!!!!!!!!!!!!!!!")
            main_csv = "all_trials_25_hz_stacked_null_str_filled_reduced.csv"
        else:
            logger.info("!!!!!!!!!!!!!!!!!!! USING FULL DATASET - ALL STRAIN DATA FILLED !!!!!!!!!!!!!!!!!!!!!!")
            main_csv = "all_trials_25_hz_stacked_null_str_filled.csv"

        self._data_locations = {
            "main": os.path.join(self.data_path, main_csv),
            "baseline": os.path.join(self.data_path, "ParticipantBaseline.csv"),
            "demographic": os.path.join(self.data_path, "GLOC_Effectiveness_Final.csv"),
            "eeg_list": list_of_eeg_data_file_paths,
            "baseline_eeg_processed_list": list_of_baseline_eeg_processed_file_paths,
        }

        return self._data_locations