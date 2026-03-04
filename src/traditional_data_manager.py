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

        ############################################# LOAD AND PROCESS DATA #############################################
        # "Grabs GLOC event and predictor data, depending on 'analysis_type' and 'feature_groups_to_analyze"


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
    
    def _load_data(self, file_paths: Dict[str, Any]) -> pd.DataFrame:
        """Load data from CSV or pickle files. If pickle does not exist, create it from CSV."""

        main_data_pickle_file = file_paths["main"].replace(".csv", ".pkl")

        # Check if pickle exists, if not create it then save it
        if not os.path.isfile(main_data_pickle_file):
            logger.info("Pickle not found at %s. Loading from CSV and caching.", main_data_pickle_file)
            gloc_data = pd.read_csv(file_paths["main"])
            gloc_data.to_pickle(main_data_pickle_file)
        else:
            logger.info("Loading data from pickle at %s.", main_data_pickle_file)
            gloc_data = pd.read_pickle(main_data_pickle_file)
        
        # Add GOR and EEG data from other files
        gloc_data = self._process_EEG_GOR(file_paths["eeg_list"], gloc_data)

        # Adjust AFE condition column always
        gloc_data["condition"] = gloc_data["condition"].map({"N": 0, "AFE": 1})
        gloc_data = gloc_data.rename(columns = {"condition": "AFE_indicator"})

        # Convert float64 to float32 to save memory, and copy to defragment the DataFrame
        float64_cols = gloc_data.select_dtypes(include="float64").columns
        if len(float64_cols) > 0:
            gloc_data = gloc_data.astype({col: "float32" for col in float64_cols}).copy()
        
        # Extracting subject and trial into separate columns
        trial_ids = gloc_data["trial_id"].to_numpy().astype("str")
        trial_ids = np.array(np.char.split(trial_ids, "-").tolist())
        gloc_data["subject"] = trial_ids[:, 0]
        gloc_data["trial"] = trial_ids[:, 1]

        # Decouple from original dataframe to prevent unwanted modifications later on
        return gloc_data
    
    def _process_EEG_GOR(self, list_of_eeg_data_files: List[str], gloc_data: pd.DataFrame) -> pd.DataFrame:
        """Slot in GOR EEG band power data from xlsx files, replacing NaNs in the main CSV."""
        trial_indices_map = gloc_data.groupby("trial_id", sort=False).indices
        event_validated = gloc_data["event_validated"].to_numpy()
        trial_ids = gloc_data["trial_id"].to_numpy()
        begin_mask = event_validated == "begin GOR"
        begin_idx_map = (
            pd.Series(np.flatnonzero(begin_mask), index=trial_ids[begin_mask])
            .groupby(level = 0, sort=False)
            .first()
            .to_dict()
        )

        band_names = ["delta", "theta", "alpha", "beta"]

        for current_file in list_of_eeg_data_files:
            # Parse trial ID from filename: e.g. "GLOC_01_DC1_..." -> "01-01"
            match = re.search(r'GLOC_(\d{2})_DC(\d+)', os.path.basename(current_file))
            if not match:
                logger.warning("Could not parse trial ID from filename: %s", current_file)
                continue
            corresponding_trial = f"{match.group(1)}-0{match.group(2)}"

            # Read all band sheets and drop the time column
            band_dfs = {
                band: pd.read_excel(current_file, sheet_name=band).iloc[:, :-1]
                for band in band_names
            }

            trial_indices = trial_indices_map.get(corresponding_trial)
            if trial_indices is None:
                logger.warning("Could not find trial %s in data.", corresponding_trial)
                continue

            index_begin_GOR = begin_idx_map.get(corresponding_trial)
            if index_begin_GOR is None:
                logger.warning("Could not find 'begin GOR' for trial %s.", corresponding_trial)
                continue

            start_pos = np.searchsorted(trial_indices, index_begin_GOR)
            n_rows = len(band_dfs["delta"])
            trial_indexer = trial_indices[start_pos : start_pos + n_rows]

            # Build column names and assign values for each band
            column_names = band_dfs["delta"].columns
            for band in band_names:
                cols = [f"{c}_{band} - EEG" for c in column_names]
                gloc_data.loc[trial_indexer, cols] = band_dfs[band].to_numpy(dtype=np.float32)

        return gloc_data
    
    def _filter_data_by_analysis_type(
            self,
            analysis_type: int,
            gloc_data: pd.DataFrame,
            subject_to_analyze: Optional[str] = None,
            trial_to_analyze: Optional[str] = None,
    ) -> pd.DataFrame:
        """Analyze only section of gloc_data specified using analysis_type."""
        if analysis_type == 0: # One Trial / One Subject
            mask = (gloc_data["subject"] == subject_to_analyze) & (gloc_data["trial"] == trial_to_analyze)
        elif analysis_type == 1: # All Trials for One Subject
            mask = (gloc_data["subject"] == subject_to_analyze)
        else: # All Trials for All Subjects
            return gloc_data
        
        return gloc_data[mask]
    
    def _process_and_get_feature_names(
            self,
            gloc_data: pd.DataFrame,
            feature_groups_to_analyze: Sequence[str],
            model_type: Tuple[str, str],
            file_names: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Process data and extract feature names based on specified feature groups."""
        # Defining which features go into which group of feature groups
        GROUPS_OF_FEATURE_GROUPS = {
            "Phys": {"ECG", "BR", "temp", "fnirs", "eyetracking", "rawEEG", "processedEEG"},
            "ECG": {"ECG"},
            "EEG": {"processedEEG"} # Adding rawEEG does not change anything (rawEEG ignored during baseline v7 and v8 calculations)
        }

        features = {
            "All": [],
            "Phys": [],
            "ECG": [],
            "EEG": []
        }
        features_all = features["All"]
        features_phys = features["Phys"]
        features_ecg = features["ECG"]
        features_eeg = features["EEG"]

        for group_name in feature_groups_to_analyze:
            if group_name not in FEATURE_REGISTRY:
                logger.warning("Feature group '%s' not recognized. Skipping.", group_name)
                continue

            processor = FEATURE_REGISTRY[group_name]

            # Process data for the feature group
            gloc_data = processor.process(gloc_data, file_names)
            feature_names = processor.get_feature_names(model_type)

            # Adding features to relevant groups
            if group_name in GROUPS_OF_FEATURE_GROUPS["Phys"]:
                features_phys.extend(feature_names)

            if group_name in GROUPS_OF_FEATURE_GROUPS["ECG"]:
                features_ecg.extend(feature_names)

            if group_name in GROUPS_OF_FEATURE_GROUPS["EEG"]:
                features_eeg.extend(feature_names)

            features_all.extend(feature_names)

        return gloc_data, features