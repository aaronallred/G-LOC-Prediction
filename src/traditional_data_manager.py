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
        ("noAFE", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "rawEEG", "processedEEG"),
        ("Complete", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "AFE", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ("Complete", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "AFE", "rawEEG", "processedEEG"),
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

    def get_data(self, backstep, data_rate, classifier_type, model_type, select_features, remove_NaN_trials, offset, time_start, subject_to_analyze, trial_to_analyze, analysis_type):
        """Return data for a given set of parameters."""
        baseline_window, window_size, stride, feature_reduction_type, baseline_methods_to_use, imbalance_type, impute_type, n_neighbors = self._get_hyperparameters_by_classifier(classifier_type)
        feature_groups_to_analyze, baseline_methods_to_use = self._get_feature_groups_and_baseline_methods(model_type, baseline_methods_to_use)

        ############################################# LOAD AND PROCESS DATA #############################################
        # "Grabs GLOC event and predictor data, depending on 'analysis_type' and 'feature_groups_to_analyze"
        file_paths = self._get_data_locations()

        # Load data and slot in GOR EEG features from xlsx files, then filter to specified analysis type and process features based on specified feature groups
        gloc_data = self._load_data(file_paths)
        gloc_data = self._filter_data_by_analysis_type(analysis_type, gloc_data, subject_to_analyze, trial_to_analyze)
        gloc_data, features = self._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, model_type, file_paths)
        
        # Create GLOC categorical vector
        gloc_labels = self._label_gloc_events(gloc_data)

        if model_type[0] == "Complete" and model_type[1] == "Explicit":
            # Impute raw (using mean) value of the missing channels for each AFE condition
            gloc_data = self._eeg_specific_imputation(gloc_data, features)

        pipeline_features = {name: feature_names.copy() for name, feature_names in features.items()}
        if model_type[0] == "Complete" and model_type[1] == "Explicit":
            # Match legacy: keep AFE indicator out of the working predictor matrix,
            # then add it back only as windowed indicator later.
            pipeline_features["All"] = [
                feature_name for feature_name in pipeline_features["All"] if feature_name != "AFE_indicator"
            ]

        if model_type[0] == "noAFE":
            # Reduce dataset based on AFE/noAFE condition
            gloc_data, gloc_labels = self._afe_subset(gloc_data, gloc_labels)

        ############################################### DATA CLEAN AND Some Imputation ###############################################
        if remove_NaN_trials:
            gloc_data, gloc_labels, _ = self._remove_all_nan_trials(gloc_data, pipeline_features, gloc_labels)

        # Legacy baselines use group arrays captured before global KNN imputation.
        baseline_group_data = {
            "Phys": gloc_data[pipeline_features["Phys"]].to_numpy(dtype=np.float32, copy=True),
            "ECG": gloc_data[pipeline_features["ECG"]].to_numpy(dtype=np.float32, copy=True),
            "EEG": gloc_data[pipeline_features["EEG"]].to_numpy(dtype=np.float32, copy=True),
        }

        if impute_type == 1:
            # Imputes missing row data - extract as float64 to match legacy numeric precision
            # (legacy features array is float64 by the time KNN runs, due to mixed-dtype DataFrame).
            imputed_features = self._faster_knn_impute(gloc_data[pipeline_features["All"]].to_numpy(dtype=np.float64), k=n_neighbors)
            gloc_data[pipeline_features["All"]] = imputed_features
        
        ################################################## REDUCE MEMORY ##################################################
        # Extract out columns from gloc_data into experiment_metadata
        gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata = self._reduce_memory(gloc_data, gloc_labels, pipeline_features, model_type)

        ###################################################### Prediction Offset ###############################################
        gloc_labels_numpy = self._y_prediction_offset(gloc_labels_numpy, backstep, data_rate, experiment_metadata["trial_id"])

        ################################################ BASELINE ################################################
        combined_baseline, combined_baseline_names, baseline_v0, baseline_names_v0 = self._get_combined_baseline_data(
            gloc_data_all_features_numpy,
            experiment_metadata,
            baseline_window,
            baseline_methods_to_use,
            pipeline_features,
            file_paths,
            model_type,
            baseline_group_data,
            gloc_labels_numpy,
        )

        ################################# FEATURE GENERATION ########################################
        # Feature generation must run for each offset to window GLOC labels
        raw_gloc_labels_numpy = gloc_labels_numpy.copy()
        gloc_labels_numpy, gloc_data_all_features_numpy, pipeline_features["All"] = self._feature_generation(
            time_start,
            offset,
            stride,
            window_size,
            combined_baseline,
            gloc_labels_numpy,
            experiment_metadata["trial_id"],
            experiment_metadata["Time (s)"],
            combined_baseline_names,
            baseline_names_v0,
            baseline_v0,
            feature_groups_to_analyze
        )

        ################################################ Feature Reduction ################################################
        # Add windowed AFE indicator if required by model type
        if model_type[0] == "Complete" and model_type[1] == "Explicit":
            experiment_metadata["AFE_indicator_windowed"], _, _ = self._sliding_window_max(
                experiment_metadata["AFE_indicator"],
                experiment_metadata["trial_id"],
                experiment_metadata["Time (s)"],
                raw_gloc_labels_numpy,
                offset,
                stride,
                window_size,
                time_start
            )

            gloc_data_all_features_numpy = np.hstack([gloc_data_all_features_numpy, experiment_metadata["AFE_indicator_windowed"]])
            pipeline_features["All"].append("AFE_indicator_windowed")

        # Convert feature matrix into a DataFrame for column selection
        gloc_data_all_features_numpy = pd.DataFrame(gloc_data_all_features_numpy, columns = pipeline_features["All"])
        gloc_data_all_features_numpy = gloc_data_all_features_numpy[select_features]
        gloc_data_all_features_numpy = gloc_data_all_features_numpy.to_numpy()

        gloc_data_all_features_numpy, select_features = self._remove_constant_columns(gloc_data_all_features_numpy, select_features)

        ################################################ NaN Processing ################################################
        gloc_labels_numpy, gloc_data_all_features_numpy, pipeline_features["All"], removed_ind = self._process_NaN_temporal(gloc_labels_numpy, gloc_data_all_features_numpy, select_features)

        ################################################ Get Outputs Ready ############################################
        gloc_data_all_features_numpy, gloc_labels_numpy = self._ready_outputs(gloc_data_all_features_numpy, gloc_labels_numpy)

        return gloc_data_all_features_numpy, gloc_labels_numpy
    
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
    
    def _label_gloc_events(self, gloc_data: pd.DataFrame) -> np.ndarray:
        """
        This function creates a g-loc label for the data based on the event_validated column. The event
        is labeled as 1 between GLOC and Return to Consciousness.
        """
        event_validated = gloc_data["event_validated"]

        # Find all GLOC and RTC indices, pair them in order, and label between each pair
        gloc_indices = np.where(event_validated.to_numpy() == "GLOC")[0]
        rtc_indices = np.where(event_validated.to_numpy() == "return to consciousness")[0]

        trial_ids = gloc_data["trial_id"].to_numpy()
        gloc_labels = np.zeros(len(gloc_data))

        for i in range(len(gloc_indices)):
            start = gloc_indices[i]
            end = rtc_indices[i]
            if trial_ids[start] == trial_ids[end]:
                gloc_labels[start:end] = 1

        return gloc_labels
    
    def _eeg_specific_imputation(self, gloc_data: pd.DataFrame, features: Dict[str, List[str]], verbose: bool = False) -> pd.DataFrame:
        """Mean-impute EEG channels that are exclusive to one AFE condition."""
        self._eeg_condition_impute(gloc_data, features, gloc_data["AFE_indicator"], verbose = verbose)
        return gloc_data
    
    def _eeg_condition_impute(self, gloc_data: pd.DataFrame, features: Dict[str, List[str]], afe_indicator_column: pd.Series, verbose: bool = False) -> None:
        """Mean-impute condition-specific EEG columns so both AFE/non-AFE have all features. Modifies gloc_data in-place."""
        # Create masks for each condition
        afe_mask = afe_indicator_column == 1
        nonafe_mask = afe_indicator_column == 0

        # Pull columns that need to be imputed for each type
        raw_eeg_feature_names = RawEEGGroup.get_separated_feature_names()
        processed_eeg_feature_names = ProcessedEEGGroup.get_separated_feature_names()
        afe_only_cols = raw_eeg_feature_names["AFE Only"] + processed_eeg_feature_names["AFE Only"]
        nonafe_only_cols = raw_eeg_feature_names["Non-AFE Only"] + processed_eeg_feature_names["Non-AFE Only"]

        # Mean imputation processing
        if afe_only_cols:
            means = gloc_data.loc[afe_mask, afe_only_cols].mean(skipna = True)
            if verbose:
                missing_counts = gloc_data.loc[nonafe_mask, afe_only_cols].isna().sum()
            gloc_data.loc[nonafe_mask, afe_only_cols] = gloc_data.loc[nonafe_mask, afe_only_cols].fillna(means)

            # Show columns imputed and how many rows were imputed for each column
            if verbose:
                for col, n in missing_counts.items():
                    logger.debug("Imputed %d values in '%s' for non-AFE rows.", n, col)

        if nonafe_only_cols:
            means = gloc_data.loc[nonafe_mask, nonafe_only_cols].mean(skipna = True)
            if verbose:
                missing_counts = gloc_data.loc[afe_mask, nonafe_only_cols].isna().sum()
            gloc_data.loc[afe_mask, nonafe_only_cols] = gloc_data.loc[afe_mask, nonafe_only_cols].fillna(means)

            # Show columns imputed and how many rows were imputed for each column
            if verbose:
                for col, n in missing_counts.items():
                    logger.debug("Imputed %d values in '%s' for AFE rows.", n, col)

    def _afe_subset(self, gloc_data: pd.DataFrame, gloc_labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
            Remove any trial that contains AFE condition (condition == 1).
        """
        trial_has_afe = gloc_data.groupby(["subject", "trial"])["AFE_indicator"].transform("max") # Mark trial as all 1 if any are 1, otherwise 0
        keep_mask = trial_has_afe != 1

        gloc_data = gloc_data.loc[keep_mask].reset_index(drop = True)
        gloc_labels = gloc_labels[keep_mask]

        return gloc_data, gloc_labels
    
    def _remove_all_nan_trials(
            self,
            gloc_data: pd.DataFrame,
            features: Dict[str, List[str]],
            gloc_labels: np.ndarray,
            verbose: bool = False,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Remove trials where at least one feature is entirely NaN. Returns NaN proportion table."""
        # All features and subject trial info to be put into a reduced dataframe from gloc_data
        all_features = features["All"]
        all_features_with_ids = all_features + ["subject", "trial"]
        reduced_data_frame = gloc_data[all_features_with_ids]

        nan_flags = reduced_data_frame[all_features].isna()
        group_keys = [reduced_data_frame["subject"], reduced_data_frame["trial"]]
        grouped = nan_flags.groupby(group_keys, sort=False)

        nan_proportion_df = grouped.mean()
        all_nan_cols_df = grouped.all()
        bad_trials = all_nan_cols_df.any(axis=1)

        if verbose and bad_trials.any():
            for (subject, trial), is_bad in bad_trials.items():
                if is_bad:
                    nan_features = all_nan_cols_df.columns[all_nan_cols_df.loc[(subject, trial)]].tolist()
                    logger.info("Subject %s, Trial %s: features entirely NaN → %s", subject, trial, nan_features)

        nan_proportion_df.insert(
            0,
            "subject-trial",
            [f"{subject}-{trial}" for subject, trial in nan_proportion_df.index],
        )
        nan_proportion_df.reset_index(drop=True, inplace=True)

        group_ids = reduced_data_frame.groupby(["subject", "trial"], sort=False).ngroup().to_numpy()
        keep_mask = ~bad_trials.to_numpy()[group_ids]

        rows_to_remove = gloc_data.index[~keep_mask]
        gloc_data.drop(rows_to_remove, inplace=True)
        gloc_data.reset_index(drop=True, inplace=True)

        kept_labels = gloc_labels[keep_mask]
        gloc_labels.resize(kept_labels.shape, refcheck=False)
        gloc_labels[:] = kept_labels

        N = int(bad_trials.shape[0])
        M = int(bad_trials.sum())

        # Print NaN findings
        logger.info("%d trials with all NaNs for at least one feature out of %d trials. %d remaining.", M, N, N - M)

        return gloc_data, gloc_labels, nan_proportion_df
    
    def _reduce_memory(
            self,
            gloc_data: pd.DataFrame,
            gloc_labels: np.ndarray,
            features: Dict[str, List[str]],
            model_type: Tuple[str, str],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Extract numpy arrays from DataFrame and free the DataFrame to reduce memory usage.
        
        Returns:
            gloc_data_all_features_numpy: feature matrix with legacy-equivalent dtype by model
            gloc_labels_numpy: boolean label array
            experiment_metadata: dict of metadata arrays
        """
        trial_id_arr = gloc_data["trial_id"].to_numpy()
        experiment_metadata = {
            "trial_id": trial_id_arr,
            "trial_ints": self._convert_to_unique_ordered_integers(trial_id_arr),
            "Time (s)": gloc_data["Time (s)"].to_numpy(),  # Keep float64 to match legacy precision in np.gradient
            "event_validated": gloc_data["event_validated"].to_numpy(),
            "subject": gloc_data["subject"].to_numpy(),
            "AFE_indicator": gloc_data["AFE_indicator"].to_numpy(dtype = np.float32).reshape(-1, 1),
        }
        # Legacy precision differs by model path:
        # - noAFE models keep features in float32 at this stage.
        # - Complete models can be promoted to float64 during the EEG condition
        #   imputation path because the condition column participates in array
        #   reconstruction. Match that behavior for strict parity tests.
        feature_dtype = np.float64 if model_type[0] == "Complete" else np.float32 # TODO: Make this a user input parameter
        gloc_data_all_features_numpy = gloc_data[features["All"]].to_numpy(dtype=feature_dtype)
        gloc_labels_numpy = gloc_labels

        del gloc_data, gloc_labels
        return gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata

    def _convert_to_unique_ordered_integers(self, strings: np.ndarray) -> np.ndarray:
        """Convert strings to 1-based integers preserving first-appearance order."""
        codes, _ = pd.factorize(strings, sort=False)
        return (codes + 1).astype(np.float32)

    def _faster_knn_impute(self, X, k = 5, M = 32, efSearch = 64):
        """
        Perform KNN imputation using FAISS HNSW index.
        Parameters:
        - X: (n_samples, n_features) matrix with missing values as np.nan
        - k: Number of neighbors for imputation
        - M: Number of neighbors in the HNSW graph (higher = more accurate, slower)
        - efSearch: Number of candidates to consider during search (higher = better recall)
        Returns:
        - X_imputed: Matrix with missing values imputed
        """
        mask = np.isnan(X)
        X_imputed = X.copy()
        
        # Temporarily mean impute missing values
        X_temp = np.where(mask, np.nanmean(X, axis=0), X)
        
        if self.testing:
            faiss.omp_set_num_threads(1)  # Use single thread for deterministic behavior in testing

        # Build FAISS index (HNSW)
        d = X.shape[1] # dimension
        index = faiss.IndexHNSWFlat(d, M)
        index.hnsw.efSearch = efSearch
        
        rng = faiss.RandomGenerator(42)
        index.hnsw.rng = rng
        
        index.add(X_temp.astype(np.float32))
        
        # Find k nearest neighbors
        distances, indices = index.search(X_temp.astype(np.float32), k + 1)
        
        # Impute missing values (skip self, which is always the first neighbor)
        for i in range(X.shape[0]):
            neighbors = indices[i, 1:] # skip self
            for j in range(X.shape[1]):
                if mask[i, j]: # Only impute missing values
                    neighbor_values = X_temp[neighbors, j]
                    X_imputed[i, j] = np.nanmean(neighbor_values)

        return X_imputed
    
    def _y_prediction_offset(self, y, backstep, data_rate, trial_set):
        """
        Shifts GLOC flags to the left by 'backstep' frames.
        Truncates the beginning and pads the end with zeros.
        """
        y = np.array(y)
        offset = int(backstep * data_rate) # the actual number of indices to offset.
        # if backstep is given as seconds and data rate as hz
        # the result would be something like 5 seconds back * 25hz so 125 indices shift

        # y is passed as every single subject and trial in one so we have to break out the indices.

        unique_trials = np.unique(trial_set) # finds the unique trials within the set. Gives an array of name of each unique

        for trial in unique_trials:
            # Clearing temporary variables if they exist
            trial_indices = None
            current_y = None
            gloc_indices = None
            y_shifted = None

            # Only make corrections within this trial
            trial_indices = np.nonzero(trial_set == trial) # find indices within trial set where this unique trial was
            current_y = y[trial_indices] # the range of y we are interested in (this trial set)
            gloc_indices = np.nonzero(current_y)[0] # find gloc indices within trial. These are the locations of nonzero values in array

            if len(gloc_indices) == 0:
                # No GLOC events present, return as is
                y[trial_indices] = current_y # no change

            else:
                y_shifted = current_y[offset:] # Remove the backstep from the start
                current_y = np.append(y_shifted, [0] * offset)[:len(current_y)] # add zeros to the back
                y[trial_indices] = current_y # reassign the indices of y to what has been edited

        return y
    
    def _get_combined_baseline_data(
            self,
            gloc_data_all_features_imputed_numpy: np.ndarray,
            experiment_metadata: Dict[str, Any],
            baseline_window: float,
            baseline_methods_to_use: List[str],
            features: Dict[str, List[str]],
            file_paths: Dict[str, Any],
            model_type: Tuple[str, str],
            baseline_group_data: Optional[Dict[str, np.ndarray]] = None,
            gloc_labels_numpy: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, np.ndarray], List[str]]:
        """Compute baselines and return combined outputs plus v0 baseline data/names."""
        participant_baseline = pd.read_csv(file_paths["baseline"])
        participant_baseline_rhr = participant_baseline["resting HR [seated]"][:-1]
        participant_baseline_rhr.index = [f"{i:02d}" for i in range(1, 14)]

        eeg_baseline_data = {}
        for filepath in file_paths["baseline_eeg_processed_list"]:
            df = pd.read_csv(filepath)
            df.index = [f"{i:02d}" for i in range(1, 14)]
            # Extract band name from filename pattern: GLOC_EEG_baseline_{band}_noAFE1.csv
            band = os.path.basename(filepath).split("_")[3]
            eeg_baseline_data[band] = df

        if baseline_group_data is None:
            # Fallback for callers that only provide the imputed full matrix.
            feature_index = {feature_name: i for i, feature_name in enumerate(features["All"])}
            baseline_group_data = {
                "Phys": gloc_data_all_features_imputed_numpy[:, [feature_index[name] for name in features["Phys"]]],
                "ECG": gloc_data_all_features_imputed_numpy[:, [feature_index[name] for name in features["ECG"]]],
                "EEG": gloc_data_all_features_imputed_numpy[:, [feature_index[name] for name in features["EEG"]]],
            }

        context = BaselineContext(
            trial_column=experiment_metadata["trial_id"],
            time_column=experiment_metadata["Time (s)"],
            event_validated_column=experiment_metadata["event_validated"],
            subject_column=experiment_metadata["subject"],
            data_by_features={
                "All": gloc_data_all_features_imputed_numpy,
                "Phys": baseline_group_data["Phys"],
                "ECG": baseline_group_data["ECG"],
                "EEG": baseline_group_data["EEG"],
            },
            features=features,
            baseline_window=baseline_window,
            model_type=model_type,
            participant_baseline_data=participant_baseline_rhr,
            eeg_baseline_data=eeg_baseline_data,
        )
        combined_baseline, combined_names, baseline_v0, baseline_names_v0, trial_order = baseline_data(baseline_methods_to_use, context)
        experiment_metadata["trial_order"] = trial_order

        return combined_baseline, combined_names, baseline_v0, baseline_names_v0

    def _feature_generation(self, time_start, offset, stride, window_size, combined_baseline, gloc, trial_column, time_column,
                        combined_baseline_names,baseline_names_v0, baseline_v0, feature_groups_to_analyze):

        """
        Generates Features from Baseline Data
        :return:
        """

        # Sliding Window Mean
        gloc_window, sliding_window_mean_s1, number_windows, all_features_mean_s1, sliding_window_mean_s2, all_features_mean_s2 = (
            self._sliding_window_mean_calc(time_start, offset, stride, window_size, combined_baseline, gloc, trial_column,
            time_column, combined_baseline_names))

        # Sliding Window Standard Deviation, Max, Range
        (sliding_window_stddev_s1, sliding_window_max_s1, sliding_window_range_s1, all_features_stddev_s1,
        all_features_max_s1,
        all_features_range_s1, sliding_window_stddev_s2, sliding_window_max_s2, sliding_window_range_s2,
        all_features_stddev_s2, all_features_max_s2,
        all_features_range_s2) = (
            self._sliding_window_calc(time_start, stride, window_size, combined_baseline, trial_column, time_column,
                                      number_windows, combined_baseline_names))

        # Additional Features
        (all_features_additional_s1, sliding_window_integral_left_pupil_s1, sliding_window_integral_right_pupil_s1,
        sliding_window_consecutive_elements_mean_left_pupil_s1, sliding_window_consecutive_elements_mean_right_pupil_s1,
        sliding_window_consecutive_elements_max_left_pupil_s1, sliding_window_consecutive_elements_max_right_pupil_s1,
        sliding_window_consecutive_elements_sum_left_pupil_s1, sliding_window_consecutive_elements_sum_right_pupil_s1,
        sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1,
        sliding_window_cognitive_ies_s1,
        all_features_additional_s2, sliding_window_integral_left_pupil_s2, sliding_window_integral_right_pupil_s2,
        sliding_window_consecutive_elements_mean_left_pupil_s2, sliding_window_consecutive_elements_mean_right_pupil_s2,
        sliding_window_consecutive_elements_max_left_pupil_s2, sliding_window_consecutive_elements_max_right_pupil_s2,
        sliding_window_consecutive_elements_sum_left_pupil_s2, sliding_window_consecutive_elements_sum_right_pupil_s2,
        sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2,
        sliding_window_cognitive_ies_s2) = \
            (self._sliding_window_other_features(time_start, stride, window_size, trial_column, time_column,
                                            number_windows,
                                            baseline_names_v0, baseline_v0, feature_groups_to_analyze))

        # Unpack Dictionary into Array & combine features into one feature array
        y_gloc_labels, x_feature_matrix = self._unpack_dict(gloc_window, sliding_window_mean_s1, number_windows,
                                                            sliding_window_stddev_s1,
                                                            sliding_window_max_s1, sliding_window_range_s1,
                                                            sliding_window_integral_left_pupil_s1,
                                                            sliding_window_integral_right_pupil_s1,
                                                            sliding_window_consecutive_elements_mean_left_pupil_s1,
                                                            sliding_window_consecutive_elements_mean_right_pupil_s1,
                                                            sliding_window_consecutive_elements_max_left_pupil_s1,
                                                            sliding_window_consecutive_elements_max_right_pupil_s1,
                                                            sliding_window_consecutive_elements_sum_left_pupil_s1,
                                                            sliding_window_consecutive_elements_sum_right_pupil_s1,
                                                            sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1,
                                                            sliding_window_cognitive_ies_s1,
                                                            sliding_window_mean_s2, sliding_window_stddev_s2,
                                                            sliding_window_max_s2, sliding_window_range_s2,
                                                            sliding_window_integral_left_pupil_s2,
                                                            sliding_window_integral_right_pupil_s2,
                                                            sliding_window_consecutive_elements_mean_left_pupil_s2,
                                                            sliding_window_consecutive_elements_mean_right_pupil_s2,
                                                            sliding_window_consecutive_elements_max_left_pupil_s2,
                                                            sliding_window_consecutive_elements_max_right_pupil_s2,
                                                            sliding_window_consecutive_elements_sum_left_pupil_s2,
                                                            sliding_window_consecutive_elements_sum_right_pupil_s2,
                                                            sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2,
                                                            sliding_window_cognitive_ies_s2)

        # Combine all features into array
        all_features = (all_features_mean_s1 + all_features_stddev_s1 + all_features_max_s1 + all_features_range_s1 +
                        all_features_additional_s1 + all_features_mean_s2 + all_features_stddev_s2 + all_features_max_s2 +
                        all_features_range_s2 + all_features_additional_s2)

        return y_gloc_labels.astype(np.float32), x_feature_matrix.astype(np.float32), all_features

    def _inter_trial_standardization(self, feature_dictionary):
        """
        This function takes the input of a feature dictionary and finds the inter-trial z-score by
        first unpacking the dictionary, then taking the mean and standard deviation of every
        column. The output is the inter-trial standardized feature dictionary.
        """

        # Find Unique Trial ID
        trial_id_in_data = list(feature_dictionary.keys())

        ## FIND INTER TRIAL MEAN AND STD. DEVIATION TO USE FOR INTER TRIAL STANDARDIZATION ##
        # To do this, I first unpack the combined_baseline dictionary &
        # Determine total length of new unpacked dictionary items
        total_rows = 0
        for i in range(np.size(trial_id_in_data)):
            total_rows += np.shape(feature_dictionary[trial_id_in_data[i]])[0]

        # Find number of columns (using non-empty dictionaries)
        num_cols = np.shape(feature_dictionary[trial_id_in_data[0]])[1]

        # Pre-allocate
        all_data = np.zeros((total_rows, num_cols))

        # Iterate through unique trial_id
        current_index = 0
        for i in range(np.size(trial_id_in_data)):

            # Find number of rows in trial
            num_rows = np.shape(feature_dictionary[trial_id_in_data[i]])[0]

            # Set rows and columns in x_feature_matrix equal to current dictionary
            all_data[current_index:num_rows + current_index,:] = feature_dictionary[trial_id_in_data[i]]

            # Increment row index
            current_index += num_rows

        # Find mean and stand deviation of all data
        inter_trial_mean = np.nanmean(all_data, axis = 0, keepdims=True)
        inter_trial_standard_deviation = np.nanstd(all_data, axis = 0, keepdims=True)

        # Build Dictionary for each trial_id
        sliding_window_s2 = dict()

        # Iterate through all unique trial_id
        for i in range(np.size(trial_id_in_data)):
            # Get data from current trial key
            current_trial_data = feature_dictionary[trial_id_in_data[i]]

            # Find inter-trial z-score
            inter_trial_z_score = ((current_trial_data - inter_trial_mean)/inter_trial_standard_deviation)

            # Define dictionary item for trial_id
            sliding_window_s2[trial_id_in_data[i]] = inter_trial_z_score

        return sliding_window_s2

    def _sliding_window_mean_calc(self, time_start, offset, stride, window_size, combined_baseline, gloc, trial_column, time_column, combined_baseline_names):
        """
        This function creates the engineered features and gloc labels for the data. This includes a
        sliding window mean for each of the features for each trial_id. The number of windows is
        determined from the specified stride, window size, and offset. The gloc label is determined
        by finding if there are any 1 GLOC labels within the window (at some offset from the engineered
        feature window). A dictionary for the engineered feature, engineered label, and number of windows
        is returned. These dictionaries are sorted by trial_id.
        """

        # Find Unique Trial ID
        trial_id_in_data = pd.unique(trial_column)  # order-preserving, matching legacy script behavior

        # Build Dictionary for each trial_id
        sliding_window_mean = dict()
        sliding_window_mean_s1 = dict()
        gloc_window = dict()
        number_windows = dict()

        # Iterate through all unique trial_id
        for i in range(np.size(trial_id_in_data)):

            # Determine index from current trial_id
            current_index = (trial_column == trial_id_in_data[i])

            # Create time array based on current_index
            current_time = np.array(time_column)
            time_trimmed = current_time[current_index]

            # Find end time for specific trial
            time_end = np.max(time_trimmed)

            # Determine number of windows
            number_windows_current = np.int32(((time_end - offset) // stride) - (window_size // stride - 1))

            # Pre-allocate arrays
            sliding_window_mean_current = np.zeros((number_windows_current, np.shape(combined_baseline[trial_id_in_data[i]])[1]))
            gloc_window_current = np.zeros((number_windows_current, 1))

            # Create trimmed gloc data for the specific
            gloc_trimmed = gloc[(trial_column == trial_id_in_data[i])]

            # Define iteration time
            time_iteration = time_start

            # Iterate through all windows to compute relevant parameters
            for j in range(number_windows_current):

                # Find index for current window
                time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))
                current_combined_baseline = combined_baseline[trial_id_in_data[i]][time_period_feature]

                # Take nanmean for the window (one value per column (feature))
                sliding_window_mean_current[j,:] = np.nanmean(current_combined_baseline, axis = 0, keepdims=True)

                # Find the offset time for G-LOC label
                time_period_gloc = (((time_iteration + offset) <= time_trimmed) &
                                    (time_trimmed < (time_iteration + offset + window_size)))

                # Create engineered label set to 1 if any values in window are 1
                gloc_window_current[j] = np.any(gloc_trimmed[time_period_gloc])

                # Adjust iteration_time
                time_iteration = stride + time_iteration

            # Compute z-score to standardize (intra-trial standardization)
            # This was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
            # should be removed in separate code or during feature selection.
            sliding_window_mean_current_z_score = np.zeros(np.shape(sliding_window_mean_current))
            if np.any(np.nanstd(sliding_window_mean_current, axis = 0, keepdims=True) == 0):
                # Z-score columns that don't have zero standard deviation
                for col in range(np.shape(sliding_window_mean_current)[1]):
                    if np.nanstd(sliding_window_mean_current[:,col]) != 0:
                        sliding_window_mean_current_z_score[:,col] = ((sliding_window_mean_current[:,col] - np.nanmean(
                            sliding_window_mean_current[:,col])) / np.nanstd(sliding_window_mean_current[:,col]))
                    else:
                        sliding_window_mean_current_z_score[:, col] = np.zeros(np.shape(sliding_window_mean_current)[0])
            else:
                sliding_window_mean_current_z_score = ((sliding_window_mean_current - np.nanmean(sliding_window_mean_current, axis = 0, keepdims=True))
                                                / np.nanstd(sliding_window_mean_current, axis = 0, keepdims=True))

            # Define dictionary item for trial_id
            sliding_window_mean_s1[trial_id_in_data[i]] = sliding_window_mean_current_z_score
            sliding_window_mean[trial_id_in_data[i]] = sliding_window_mean_current
            gloc_window[trial_id_in_data[i]] = gloc_window_current
            number_windows[trial_id_in_data[i]] = number_windows_current

            # Name all features (s1 (intra-trial) standardization)
            all_features_mean_s1 = [s + '_mean_s1' for s in combined_baseline_names]

        # Compute inter-trial standardization
        sliding_window_mean_s2 = self._inter_trial_standardization(sliding_window_mean)

        # Name all features (s1 (intra-trial) standardization)
        all_features_mean_s2 = [s + '_mean_s2' for s in combined_baseline_names]

        return gloc_window, sliding_window_mean_s1, number_windows, all_features_mean_s1, sliding_window_mean_s2, all_features_mean_s2

    def _sliding_window_calc(self, time_start, stride, window_size, combined_baseline, trial_column, time_column,
                            number_windows, combined_baseline_names):
        """
        This function creates the engineered features and gloc labels for the data. This includes a
        sliding window standard deviation for each of the features for each trial_id. A dictionary
        sorted by trial_id for the engineered feature is returned.
        """

        # Find Unique Trial ID
        trial_id_in_data = pd.unique(trial_column)  # order-preserving, matching legacy script behavior

        # Build Dictionary for each trial_id
        # Windowed data (no standardization)
        sliding_window_stddev = dict()
        sliding_window_max = dict()
        sliding_window_range = dict()

        # s1 = Intra Trial Standardization
        sliding_window_stddev_s1 = dict()
        sliding_window_max_s1 = dict()
        sliding_window_range_s1 = dict()

        # s2 = Intra Trial Standardization
        sliding_window_stddev_s2 = dict()
        sliding_window_max_s2 = dict()
        sliding_window_range_s2 = dict()

        # Iterate through all unique trial_id
        for i in range(np.size(trial_id_in_data)):

            # Determine index from current trial_id
            current_index = (trial_column == trial_id_in_data[i])

            # Create time array based on current_index
            current_time = np.array(time_column)
            time_trimmed = current_time[current_index]

            # Determine number of windows
            number_windows_current = number_windows[trial_id_in_data[i]]

            # Pre-allocate arrays
            sliding_window_stddev_current = np.zeros((number_windows_current, np.shape(combined_baseline[trial_id_in_data[i]])[1]))
            sliding_window_max_current = np.zeros((number_windows_current, np.shape(combined_baseline[trial_id_in_data[i]])[1]))
            sliding_window_range_current = np.zeros((number_windows_current, np.shape(combined_baseline[trial_id_in_data[i]])[1]))

            # Define iteration time
            time_iteration = time_start

            # Iterate through all windows to compute relevant parameters
            for j in range(number_windows_current):

                # Find index for current window
                time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))

                # Find feature for current window
                current_combined_baseline = combined_baseline[trial_id_in_data[i]][time_period_feature]

                # Take nan stddev for the window (one value per column (feature))
                sliding_window_stddev_current[j,:] = np.nanstd(current_combined_baseline, axis = 0, keepdims=True)

                # Take nan max for the window (one value per column (feature))
                sliding_window_max_current[j, :] = np.nanmax(current_combined_baseline, axis=0, keepdims=True)

                # Take nan range for the window (one value per column (feature))
                sliding_window_range_current[j, :] = np.nanmax(current_combined_baseline, axis=0, keepdims=True) - np.nanmin(current_combined_baseline, axis=0, keepdims=True)

                # Adjust iteration_time
                time_iteration = stride + time_iteration

            # Compute z-score to standardize
            # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
            # should be removed in separate code or during feature selection
            # Standard Deviation
            sliding_window_stddev_current_z_score_s1 = np.zeros(np.shape(sliding_window_stddev_current))
            if np.any(np.nanstd(sliding_window_stddev_current, axis=0, keepdims=True) == 0):
                # Z-score columns that don't have zero standard deviation
                for col in range(np.shape(sliding_window_stddev_current)[1]):
                    if np.nanstd(sliding_window_stddev_current[:, col]) != 0:
                        sliding_window_stddev_current_z_score_s1[:, col] = ((sliding_window_stddev_current[:, col] - np.nanmean(sliding_window_stddev_current[:, col])) /
                                                                np.nanstd(sliding_window_stddev_current[:, col]))
                    else:
                        sliding_window_stddev_current_z_score_s1[:, col] = np.zeros(np.shape(sliding_window_stddev_current)[0])
            else:
                sliding_window_stddev_current_z_score_s1 = ((sliding_window_stddev_current - np.nanmean(sliding_window_stddev_current, axis = 0, keepdims=True))
                                                / np.nanstd(sliding_window_stddev_current, axis = 0, keepdims=True))

            # Max
            sliding_window_max_current_z_score_s1 = np.zeros(np.shape(sliding_window_max_current))
            if np.any(np.nanstd(sliding_window_max_current, axis=0, keepdims=True) == 0):
                # Find columns with zero standard deviation
                for col in range(np.shape(sliding_window_max_current)[1]):
                    if np.nanstd(sliding_window_max_current[:, col]) != 0:
                        sliding_window_max_current_z_score_s1[:, col] = ((sliding_window_max_current[:, col] - np.nanmean(
                            sliding_window_max_current[:, col])) / np.nanstd(sliding_window_max_current[:, col]))
                    else:
                        sliding_window_max_current_z_score_s1[:, col] = np.zeros(np.shape(sliding_window_max_current)[0])
            else:
                sliding_window_max_current_z_score_s1 = ((sliding_window_max_current - np.nanmean(sliding_window_max_current, axis = 0, keepdims=True))
                                                / np.nanstd(sliding_window_max_current, axis = 0, keepdims=True))
            # Range
            sliding_window_range_current_z_score_s1 = np.zeros(np.shape(sliding_window_range_current))
            if np.any(np.nanstd(sliding_window_range_current, axis=0, keepdims=True) == 0):
                # Find columns with zero standard deviation
                for col in range(np.shape(sliding_window_range_current)[1]):
                    if np.nanstd(sliding_window_range_current[:, col]) != 0:
                        sliding_window_range_current_z_score_s1[:, col] = ((sliding_window_range_current[:, col] - np.nanmean(
                            sliding_window_range_current[:, col])) / np.nanstd(sliding_window_range_current[:, col]))
                    else:
                        sliding_window_range_current_z_score_s1[:, col] = np.zeros(np.shape(sliding_window_range_current)[0])
            else:
                sliding_window_range_current_z_score_s1 = ((sliding_window_range_current - np.nanmean(sliding_window_range_current, axis = 0, keepdims=True))
                                                / np.nanstd(sliding_window_range_current, axis = 0, keepdims=True))


            # Define dictionary item for trial_id
            # No standardization
            sliding_window_stddev[trial_id_in_data[i]] = sliding_window_stddev_current
            sliding_window_max[trial_id_in_data[i]] = sliding_window_max_current
            sliding_window_range[trial_id_in_data[i]] = sliding_window_range_current

            # Intra-trial standardization
            sliding_window_stddev_s1[trial_id_in_data[i]] = sliding_window_stddev_current_z_score_s1
            sliding_window_max_s1[trial_id_in_data[i]] = sliding_window_max_current_z_score_s1
            sliding_window_range_s1[trial_id_in_data[i]] = sliding_window_range_current_z_score_s1

            # Name features
            all_features_stddev_s1 = [s + '_stddev_s1' for s in combined_baseline_names]
            all_features_max_s1 = [s + '_max_s1' for s in combined_baseline_names]
            all_features_range_s1 = [s + '_range_s1' for s in combined_baseline_names]

        # Inter trial standardization
        sliding_window_stddev_s2 = self._inter_trial_standardization(sliding_window_stddev)
        sliding_window_max_s2 = self._inter_trial_standardization(sliding_window_max)
        sliding_window_range_s2 = self._inter_trial_standardization(sliding_window_range)

        all_features_stddev_s2 = [s + '_stddev_s2' for s in combined_baseline_names]
        all_features_max_s2 = [s + '_max_s2' for s in combined_baseline_names]
        all_features_range_s2 = [s + '_range_s2' for s in combined_baseline_names]

        return (sliding_window_stddev_s1, sliding_window_max_s1, sliding_window_range_s1, all_features_stddev_s1, all_features_max_s1,
                all_features_range_s1, sliding_window_stddev_s2, sliding_window_max_s2, sliding_window_range_s2, all_features_stddev_s2,
                all_features_max_s2, all_features_range_s2)

    def _sliding_window_other_features(self, time_start, stride, window_size, trial_column, time_column, number_windows,
                                    baseline_names_v0, baseline_v0, feature_groups_to_analyze):
        """
        This function creates the engineered features and gloc labels for the data. This includes a
        sliding window mean of the difference between left and right pupil and HbO/Hbd ratio.
        """

        # Find Unique Trial ID
        trial_id_in_data = pd.unique(trial_column)  # order-preserving, matching legacy script behavior

        # Accept either a direct v0 name list or the full baseline-name dict.
        if isinstance(baseline_names_v0, dict):
            baseline_names_v0 = baseline_names_v0.get("v0", [])

        if 'eyetracking' in feature_groups_to_analyze:
            # Find indices of left and right pupil
            index_left_pupil = baseline_names_v0.index('Pupil diameter left [mm] - Tobii_v0')
            index_right_pupil = baseline_names_v0.index('Pupil diameter right [mm] - Tobii_v0')

            # Define eyetracking feature names
            eye_tracking_features = ['Left Pupil Integral (Non-Baseline)', 'Right Pupil Integral (Non-Baseline)',
                                    'Left Pupil Mean of Consecutive Difference (Non-Baseline)',
                                    'Right Pupil Mean of Consecutive Difference (Non-Baseline)',
                                    'Left Pupil Max of Consecutive Difference (Non-Baseline)',
                                    'Right Pupil Max of Consecutive Difference (Non-Baseline)',
                                    'Left Pupil Sum of Consecutive Difference (Non-Baseline)',
                                    'Right Pupil Sum of Consecutive Difference (Non-Baseline)']
        else:
            eye_tracking_features = []

        if 'ECG' in feature_groups_to_analyze:
            # Find indices of HR
            index_hr = baseline_names_v0.index('HR (bpm) - Equivital_v0')

            # Define ECG feature names
            ecg_features = ['HRV (SDNN)', 'HRV (RMSSD)']# , 'HRV (PNN50)']. Removed PNN50 due to interpolation
        else:
            ecg_features = []

        if 'cognitive' in feature_groups_to_analyze:
            # Find indices of Cognitive Response Time and Correct
            index_response_time = baseline_names_v0.index('RespTime - Cog_v0')
            index_correct = baseline_names_v0.index('Correct - Cog_v0')

            # Define ECG feature names
            cognitive_features = ['Cognitive IES']
        else:
            cognitive_features = []

        # Build Dictionary for each trial_id
        # No Standardization
        sliding_window_integral_left_pupil = dict()
        sliding_window_integral_right_pupil = dict()
        sliding_window_consecutive_elements_mean_left_pupil = dict()
        sliding_window_consecutive_elements_mean_right_pupil = dict()
        sliding_window_consecutive_elements_max_left_pupil = dict()
        sliding_window_consecutive_elements_max_right_pupil = dict()
        sliding_window_consecutive_elements_sum_left_pupil = dict()
        sliding_window_consecutive_elements_sum_right_pupil = dict()
        sliding_window_hrv_sdnn = dict()
        sliding_window_hrv_rmssd = dict()
        # sliding_window_hrv_pnn50 = dict()
        sliding_window_cognitive_ies = dict()

        # Intra-trial standardization (s1)
        sliding_window_integral_left_pupil_s1 = dict()
        sliding_window_integral_right_pupil_s1 = dict()
        sliding_window_consecutive_elements_mean_left_pupil_s1 = dict()
        sliding_window_consecutive_elements_mean_right_pupil_s1 = dict()
        sliding_window_consecutive_elements_max_left_pupil_s1 = dict()
        sliding_window_consecutive_elements_max_right_pupil_s1 = dict()
        sliding_window_consecutive_elements_sum_left_pupil_s1 = dict()
        sliding_window_consecutive_elements_sum_right_pupil_s1 = dict()
        sliding_window_hrv_sdnn_s1 = dict()
        sliding_window_hrv_rmssd_s1 = dict()
        # sliding_window_hrv_pnn50_s1 = dict()
        sliding_window_cognitive_ies_s1 = dict()

        # Inter-trial standardization (s2)
        sliding_window_integral_left_pupil_s2 = dict()
        sliding_window_integral_right_pupil_s2 = dict()
        sliding_window_consecutive_elements_mean_left_pupil_s2 = dict()
        sliding_window_consecutive_elements_mean_right_pupil_s2 = dict()
        sliding_window_consecutive_elements_max_left_pupil_s2 = dict()
        sliding_window_consecutive_elements_max_right_pupil_s2 = dict()
        sliding_window_consecutive_elements_sum_left_pupil_s2 = dict()
        sliding_window_consecutive_elements_sum_right_pupil_s2 = dict()
        sliding_window_hrv_sdnn_s2 = dict()
        sliding_window_hrv_rmssd_s2 = dict()
        # sliding_window_hrv_pnn50_s2 = dict()
        sliding_window_cognitive_ies_s2 = dict()

        # Iterate through all unique trial_id
        for i in range(np.size(trial_id_in_data)):

            # Determine index from current trial_id
            current_index = (trial_column == trial_id_in_data[i])

            # Create time array based on current_index
            current_time = np.array(time_column)
            time_trimmed = current_time[current_index]

            # Determine number of windows
            number_windows_current = number_windows[trial_id_in_data[i]]

            # Pre-allocate arrays
            if 'eyetracking' in feature_groups_to_analyze:
                sliding_window_integral_left_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_integral_right_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_consecutive_elements_mean_left_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_consecutive_elements_mean_right_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_consecutive_elements_max_left_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_consecutive_elements_max_right_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_consecutive_elements_sum_left_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_consecutive_elements_sum_right_pupil_current = np.zeros((number_windows_current, 1))
            if 'ECG' in feature_groups_to_analyze:
                sliding_window_hrv_sdnn_current = np.zeros((number_windows_current, 1))
                sliding_window_hrv_rmssd_current = np.zeros((number_windows_current, 1))
                # sliding_window_hrv_pnn50_current = np.zeros((number_windows_current, 1))
            if 'cognitive' in feature_groups_to_analyze:
                sliding_window_cognitive_ies_current = np.zeros((number_windows_current, 1))

            # Define iteration time
            time_iteration = time_start

            # Iterate through all windows to compute relevant parameters
            for j in range(number_windows_current):

                # Find index for current window
                time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))

                # Find non-baseline feature for current window
                feature_window_no_baseline = baseline_v0[trial_id_in_data[i]][time_period_feature]

                if 'ECG' in feature_groups_to_analyze:
                    # Compute HRV
                    rr_interval = 60000 / feature_window_no_baseline[:,index_hr]
                    sliding_window_hrv_sdnn_current[j] = np.nanstd(rr_interval)

                    successive_difference = np.diff(rr_interval)
                    sliding_window_hrv_rmssd_current[j] = np.sqrt(np.nanmean(successive_difference**2))

                    # Compute PNN50
                    # count_50ms_diff_current = np.sum(np.abs(successive_difference) > 50 * 0.04) # 50 times (1/sampling freqeuncy)
                    # sliding_window_hrv_pnn50_current[j] = (count_50ms_diff_current / len(successive_difference)) * 100

                if 'cognitive' in feature_groups_to_analyze:
                    # Compute IES (Inverse Efficiency Score)
                    sliding_window_cognitive_ies_current[j] = np.nanmean(feature_window_no_baseline[:,index_response_time]) / (np.nanmean(feature_window_no_baseline[:,index_correct]))

                if 'eyetracking' in feature_groups_to_analyze:
                    # Compute non-baseline pupil features
                    left_pupil_no_baseline = feature_window_no_baseline[:, index_left_pupil]
                    right_pupil_no_baseline = feature_window_no_baseline[:, index_right_pupil]

                    # Integral (using Trapezoid rule)
                    sliding_window_integral_left_pupil_current[j] = (window_size / 2) * (left_pupil_no_baseline[-1] + left_pupil_no_baseline[0])
                    sliding_window_integral_right_pupil_current[j] = (window_size / 2) * (right_pupil_no_baseline[-1] + right_pupil_no_baseline[0])

                    # Compute average difference between consecutive elements
                    left_pupil_consecutive_difference = np.diff(left_pupil_no_baseline)
                    left_pupil_consecutive_difference_full = np.append(left_pupil_consecutive_difference, np.nan)

                    right_pupil_consecutive_difference = np.diff(right_pupil_no_baseline)
                    right_pupil_consecutive_difference_full = np.append(right_pupil_consecutive_difference, np.nan)

                    sliding_window_consecutive_elements_mean_left_pupil_current[j] = np.nanmean(left_pupil_consecutive_difference_full)
                    sliding_window_consecutive_elements_mean_right_pupil_current[j] = np.nanmean(right_pupil_consecutive_difference_full)

                    # Compute max difference between consecutive elements
                    sliding_window_consecutive_elements_max_left_pupil_current[j] = np.nanmax(left_pupil_consecutive_difference_full)
                    sliding_window_consecutive_elements_max_right_pupil_current[j] = np.nanmax(right_pupil_consecutive_difference_full)

                    # Compute sum of difference between consecutive elements
                    sliding_window_consecutive_elements_sum_left_pupil_current[j] = np.nansum(left_pupil_consecutive_difference_full)
                    sliding_window_consecutive_elements_sum_right_pupil_current[j] = np.nansum(right_pupil_consecutive_difference_full)

                # Adjust iteration_time
                time_iteration = stride + time_iteration

            # Compute Z-score
            if 'eyetracking' in feature_groups_to_analyze:
                # Compute z-score to standardize integral left pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_integral_left_pupil_current_z_score = np.zeros(np.shape(sliding_window_integral_left_pupil_current))
                if np.any(np.nanstd(sliding_window_integral_left_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_integral_left_pupil_current)[1]):
                        if np.nanstd(sliding_window_integral_left_pupil_current[:, col]) != 0:
                            sliding_window_integral_left_pupil_current_z_score[:, col] = ((sliding_window_integral_left_pupil_current[:, col] - np.nanmean(
                                            sliding_window_integral_left_pupil_current[:, col])) / np.nanstd(sliding_window_integral_left_pupil_current[:, col]))
                        else:
                            sliding_window_integral_left_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_integral_left_pupil_current)[0])
                else:
                    sliding_window_integral_left_pupil_current_z_score = ((sliding_window_integral_left_pupil_current - np.nanmean(sliding_window_integral_left_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_integral_left_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize integral right pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_integral_right_pupil_current_z_score = np.zeros(np.shape(sliding_window_integral_right_pupil_current))
                if np.any(np.nanstd(sliding_window_integral_right_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_integral_right_pupil_current)[1]):
                        if np.nanstd(sliding_window_integral_right_pupil_current[:, col]) != 0:
                            sliding_window_integral_right_pupil_current_z_score[:, col] = ((sliding_window_integral_right_pupil_current[:, col] - np.nanmean(
                                            sliding_window_integral_right_pupil_current[:, col])) / np.nanstd(sliding_window_integral_right_pupil_current[:, col]))
                        else:
                            sliding_window_integral_right_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_integral_right_pupil_current)[0])
                else:
                    sliding_window_integral_right_pupil_current_z_score = ((sliding_window_integral_right_pupil_current - np.nanmean(sliding_window_integral_right_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_integral_right_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize mean of difference of consecutive elements-left pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_consecutive_elements_mean_left_pupil_current_z_score = np.zeros(np.shape(sliding_window_consecutive_elements_mean_left_pupil_current))
                if np.any(np.nanstd(sliding_window_consecutive_elements_mean_left_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_consecutive_elements_mean_left_pupil_current)[1]):
                        if np.nanstd(sliding_window_consecutive_elements_mean_left_pupil_current[:, col]) != 0:
                            sliding_window_consecutive_elements_mean_left_pupil_current_z_score[:, col] = ((sliding_window_consecutive_elements_mean_left_pupil_current[:, col] - np.nanmean(
                                            sliding_window_consecutive_elements_mean_left_pupil_current[:, col])) / np.nanstd(sliding_window_consecutive_elements_mean_left_pupil_current[:, col]))
                        else:
                            sliding_window_consecutive_elements_mean_left_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_consecutive_elements_mean_left_pupil_current)[0])
                else:
                    sliding_window_consecutive_elements_mean_left_pupil_current_z_score = ((sliding_window_consecutive_elements_mean_left_pupil_current - np.nanmean(sliding_window_consecutive_elements_mean_left_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_consecutive_elements_mean_left_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize mean of difference of consecutive elements-right pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_consecutive_elements_mean_right_pupil_current_z_score = np.zeros(np.shape(sliding_window_consecutive_elements_mean_right_pupil_current))
                if np.any(np.nanstd(sliding_window_consecutive_elements_mean_right_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_consecutive_elements_mean_right_pupil_current)[1]):
                        if np.nanstd(sliding_window_consecutive_elements_mean_right_pupil_current[:, col]) != 0:
                            sliding_window_consecutive_elements_mean_right_pupil_current_z_score[:, col] = ((sliding_window_consecutive_elements_mean_right_pupil_current[:, col] - np.nanmean(
                                            sliding_window_consecutive_elements_mean_right_pupil_current[:, col])) / np.nanstd(sliding_window_consecutive_elements_mean_right_pupil_current[:, col]))
                        else:
                            sliding_window_consecutive_elements_mean_right_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_consecutive_elements_mean_right_pupil_current)[0])
                else:
                    sliding_window_consecutive_elements_mean_right_pupil_current_z_score = ((sliding_window_consecutive_elements_mean_right_pupil_current - np.nanmean(sliding_window_consecutive_elements_mean_right_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_consecutive_elements_mean_right_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize max of difference of consecutive elements-left pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_consecutive_elements_max_left_pupil_current_z_score = np.zeros(np.shape(sliding_window_consecutive_elements_max_left_pupil_current))
                if np.any(np.nanstd(sliding_window_consecutive_elements_max_left_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_consecutive_elements_max_left_pupil_current)[1]):
                        if np.nanstd(sliding_window_consecutive_elements_max_left_pupil_current[:, col]) != 0:
                            sliding_window_consecutive_elements_max_left_pupil_current_z_score[:, col] = ((sliding_window_consecutive_elements_max_left_pupil_current[:, col] - np.nanmean(
                                            sliding_window_consecutive_elements_max_left_pupil_current[:, col])) / np.nanstd(sliding_window_consecutive_elements_max_left_pupil_current[:, col]))
                        else:
                            sliding_window_consecutive_elements_max_left_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_consecutive_elements_max_left_pupil_current)[0])
                else:
                    sliding_window_consecutive_elements_max_left_pupil_current_z_score = ((sliding_window_consecutive_elements_max_left_pupil_current - np.nanmean(sliding_window_consecutive_elements_max_left_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_consecutive_elements_max_left_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize max of difference of consecutive elements-right pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature
                sliding_window_consecutive_elements_max_right_pupil_current_z_score = np.zeros(np.shape(sliding_window_consecutive_elements_max_right_pupil_current))
                if np.any(np.nanstd(sliding_window_consecutive_elements_max_right_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_consecutive_elements_max_right_pupil_current)[1]):
                        if np.nanstd(sliding_window_consecutive_elements_max_right_pupil_current[:, col]) != 0:
                            sliding_window_consecutive_elements_max_right_pupil_current_z_score[:, col] = ((sliding_window_consecutive_elements_max_right_pupil_current[:, col] - np.nanmean(
                                            sliding_window_consecutive_elements_max_right_pupil_current[:, col])) / np.nanstd(sliding_window_consecutive_elements_max_right_pupil_current[:, col]))
                        else:
                            sliding_window_consecutive_elements_max_right_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_consecutive_elements_max_right_pupil_current)[0])
                else:
                    sliding_window_consecutive_elements_max_right_pupil_current_z_score = ((sliding_window_consecutive_elements_max_right_pupil_current - np.nanmean(sliding_window_consecutive_elements_max_right_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_consecutive_elements_max_right_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize sum of difference of consecutive elements-left pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_consecutive_elements_sum_left_pupil_current_z_score = np.zeros(np.shape(sliding_window_consecutive_elements_sum_left_pupil_current))
                if np.any(np.nanstd(sliding_window_consecutive_elements_sum_left_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_consecutive_elements_sum_left_pupil_current)[1]):
                        if np.nanstd(sliding_window_consecutive_elements_sum_left_pupil_current[:, col]) != 0:
                            sliding_window_consecutive_elements_sum_left_pupil_current_z_score[:, col] = ((sliding_window_consecutive_elements_sum_left_pupil_current[:, col] - np.nanmean(
                                            sliding_window_consecutive_elements_sum_left_pupil_current[:, col])) / np.nanstd(sliding_window_consecutive_elements_sum_left_pupil_current[:, col]))
                        else:
                            sliding_window_consecutive_elements_sum_left_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_consecutive_elements_sum_left_pupil_current)[0])
                else:
                    sliding_window_consecutive_elements_sum_left_pupil_current_z_score = ((sliding_window_consecutive_elements_sum_left_pupil_current - np.nanmean(sliding_window_consecutive_elements_sum_left_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_consecutive_elements_sum_left_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize sum of difference of consecutive elements-right pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_consecutive_elements_sum_right_pupil_current_z_score = np.zeros(np.shape(sliding_window_consecutive_elements_sum_right_pupil_current))
                if np.any(np.nanstd(sliding_window_consecutive_elements_sum_right_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_consecutive_elements_sum_right_pupil_current)[1]):
                        if np.nanstd(sliding_window_consecutive_elements_sum_right_pupil_current[:, col]) != 0:
                            sliding_window_consecutive_elements_sum_right_pupil_current_z_score[:, col] = ((sliding_window_consecutive_elements_sum_right_pupil_current[:, col] - np.nanmean(
                                            sliding_window_consecutive_elements_sum_right_pupil_current[:, col])) / np.nanstd(sliding_window_consecutive_elements_sum_right_pupil_current[:, col]))
                        else:
                            sliding_window_consecutive_elements_sum_right_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_consecutive_elements_sum_right_pupil_current)[0])
                else:
                    sliding_window_consecutive_elements_sum_right_pupil_current_z_score = ((sliding_window_consecutive_elements_sum_right_pupil_current - np.nanmean(sliding_window_consecutive_elements_sum_right_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_consecutive_elements_sum_right_pupil_current, axis = 0, keepdims=True))
            if 'ECG' in feature_groups_to_analyze:
                # Compute z-score to standardize hrv sdnn
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_hrv_sdnn_current_z_score = np.zeros(np.shape(sliding_window_hrv_sdnn_current))
                if np.any(np.nanstd(sliding_window_hrv_sdnn_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_hrv_sdnn_current)[1]):
                        if np.nanstd(sliding_window_hrv_sdnn_current[:, col]) != 0:
                            sliding_window_hrv_sdnn_current_z_score[:, col] = ((sliding_window_hrv_sdnn_current[:, col] - np.nanmean(
                                            sliding_window_hrv_sdnn_current[:, col])) / np.nanstd(sliding_window_hrv_sdnn_current[:, col]))
                        else:
                            sliding_window_hrv_sdnn_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_hrv_sdnn_current)[0])
                else:
                    sliding_window_hrv_sdnn_current_z_score = ((sliding_window_hrv_sdnn_current - np.nanmean(sliding_window_hrv_sdnn_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_hrv_sdnn_current, axis = 0, keepdims=True))

                # Compute z-score to standardize hrv rmssd
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_hrv_rmssd_current_z_score = np.zeros(np.shape(sliding_window_hrv_rmssd_current))
                if np.any(np.nanstd(sliding_window_hrv_rmssd_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_hrv_rmssd_current)[1]):
                        if np.nanstd(sliding_window_hrv_rmssd_current[:, col]) != 0:
                            sliding_window_hrv_rmssd_current_z_score[:, col] = ((sliding_window_hrv_rmssd_current[:, col] - np.nanmean(
                                            sliding_window_hrv_rmssd_current[:, col])) / np.nanstd(sliding_window_hrv_rmssd_current[:, col]))
                        else:
                            sliding_window_hrv_rmssd_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_hrv_rmssd_current)[0])
                else:
                    sliding_window_hrv_rmssd_current_z_score = ((sliding_window_hrv_rmssd_current - np.nanmean(sliding_window_hrv_rmssd_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_hrv_rmssd_current, axis = 0, keepdims=True))

                # # Compute z-score to standardize hrv pnn50
                # # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # # should be removed in separate code or during feature selection
                # sliding_window_hrv_pnn50_current_z_score = np.zeros(np.shape(sliding_window_hrv_pnn50_current))
                # if np.any(np.nanstd(sliding_window_hrv_pnn50_current, axis=0, keepdims=True) == 0):
                #     # Find columns with zero standard deviation
                #     for col in range(np.shape(sliding_window_hrv_pnn50_current)[1]):
                #         if np.nanstd(sliding_window_hrv_pnn50_current[:, col]) != 0:
                #             sliding_window_hrv_pnn50_current_z_score[:, col] = ((sliding_window_hrv_pnn50_current[:, col] - np.nanmean(
                #                             sliding_window_hrv_pnn50_current[:, col])) / np.nanstd(sliding_window_hrv_pnn50_current[:, col]))
                #         else:
                #             sliding_window_hrv_pnn50_current_z_score[:, col] = np.zeros(
                #                 np.shape(sliding_window_hrv_pnn50_current)[0])
                # else:
                #     sliding_window_hrv_pnn50_current_z_score = ((sliding_window_hrv_pnn50_current - np.nanmean(sliding_window_hrv_pnn50_current, axis=0, keepdims=True))
                #                                        / np.nanstd(sliding_window_hrv_pnn50_current, axis=0, keepdims=True))

            if 'cognitive' in feature_groups_to_analyze:
                # Compute z-score to standardize cognitive IES
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_cognitive_IES_current_z_score = np.zeros(np.shape(sliding_window_cognitive_ies_current))
                if np.any(np.nanstd(sliding_window_cognitive_ies_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_cognitive_ies_current)[1]):
                        if np.nanstd(sliding_window_cognitive_ies_current[:, col]) != 0:
                            sliding_window_cognitive_IES_current_z_score[:, col] = ((sliding_window_cognitive_ies_current[:, col] - np.nanmean(
                                            sliding_window_cognitive_ies_current[:, col])) / np.nanstd(sliding_window_cognitive_ies_current[:, col]))
                        else:
                            sliding_window_cognitive_IES_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_cognitive_ies_current)[0])
                else:
                    sliding_window_cognitive_IES_current_z_score = ((sliding_window_cognitive_ies_current - np.nanmean(sliding_window_cognitive_ies_current, axis=0, keepdims=True))
                                                    / np.nanstd(sliding_window_cognitive_ies_current, axis=0, keepdims=True))

            # Define dictionary item for trial_id
            if 'eyetracking' in feature_groups_to_analyze:
                # No standardization
                sliding_window_integral_left_pupil[trial_id_in_data[i]] = sliding_window_integral_left_pupil_current
                sliding_window_integral_right_pupil[trial_id_in_data[i]] = sliding_window_integral_right_pupil_current
                sliding_window_consecutive_elements_mean_left_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_mean_left_pupil_current
                sliding_window_consecutive_elements_mean_right_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_mean_right_pupil_current
                sliding_window_consecutive_elements_max_left_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_max_left_pupil_current
                sliding_window_consecutive_elements_max_right_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_max_right_pupil_current
                sliding_window_consecutive_elements_sum_left_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_sum_left_pupil_current
                sliding_window_consecutive_elements_sum_right_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_sum_right_pupil_current

                # Intra-Trial Standardization (s1)
                sliding_window_integral_left_pupil_s1[trial_id_in_data[i]] = sliding_window_integral_left_pupil_current_z_score
                sliding_window_integral_right_pupil_s1[trial_id_in_data[i]] = sliding_window_integral_right_pupil_current_z_score
                sliding_window_consecutive_elements_mean_left_pupil_s1[trial_id_in_data[i]] = sliding_window_consecutive_elements_mean_left_pupil_current_z_score
                sliding_window_consecutive_elements_mean_right_pupil_s1[trial_id_in_data[i]] = sliding_window_consecutive_elements_mean_right_pupil_current_z_score
                sliding_window_consecutive_elements_max_left_pupil_s1[trial_id_in_data[i]] = sliding_window_consecutive_elements_max_left_pupil_current_z_score
                sliding_window_consecutive_elements_max_right_pupil_s1[trial_id_in_data[i]] = sliding_window_consecutive_elements_max_right_pupil_current_z_score
                sliding_window_consecutive_elements_sum_left_pupil_s1[trial_id_in_data[i]] = sliding_window_consecutive_elements_sum_left_pupil_current_z_score
                sliding_window_consecutive_elements_sum_right_pupil_s1[trial_id_in_data[i]] = sliding_window_consecutive_elements_sum_right_pupil_current_z_score
            if 'ECG' in feature_groups_to_analyze:
                # No standardization
                sliding_window_hrv_sdnn[trial_id_in_data[i]] = sliding_window_hrv_sdnn_current
                sliding_window_hrv_rmssd[trial_id_in_data[i]] = sliding_window_hrv_rmssd_current
                # sliding_window_hrv_pnn50[trial_id_in_data[i]] = sliding_window_hrv_pnn50_current

                # Intra-Trial Standardization (s1)
                sliding_window_hrv_sdnn_s1[trial_id_in_data[i]] = sliding_window_hrv_sdnn_current_z_score
                sliding_window_hrv_rmssd_s1[trial_id_in_data[i]] = sliding_window_hrv_rmssd_current_z_score
                # sliding_window_hrv_pnn50_s1[trial_id_in_data[i]] = sliding_window_hrv_pnn50_current_z_score
            if 'cognitive' in feature_groups_to_analyze:
                # No standardization
                sliding_window_cognitive_ies[trial_id_in_data[i]] = sliding_window_cognitive_ies_current

                # Intra-Trial Standardization (s1)
                sliding_window_cognitive_ies_s1[trial_id_in_data[i]] = sliding_window_cognitive_IES_current_z_score

            # Name all features
            all_features_additional = eye_tracking_features + ecg_features + cognitive_features
            all_features_additional_s1 = [s + '_s1' for s in all_features_additional]

        # Inter-trial standardization (s2)
        if 'eyetracking' in feature_groups_to_analyze:
            sliding_window_integral_left_pupil_s2 = self._inter_trial_standardization(sliding_window_integral_left_pupil)
            sliding_window_integral_right_pupil_s2 = self._inter_trial_standardization(sliding_window_integral_right_pupil)
            sliding_window_consecutive_elements_mean_left_pupil_s2 = self._inter_trial_standardization(sliding_window_consecutive_elements_mean_left_pupil)
            sliding_window_consecutive_elements_mean_right_pupil_s2 = self._inter_trial_standardization(sliding_window_consecutive_elements_mean_right_pupil)
            sliding_window_consecutive_elements_max_left_pupil_s2 = self._inter_trial_standardization(sliding_window_consecutive_elements_max_left_pupil)
            sliding_window_consecutive_elements_max_right_pupil_s2 = self._inter_trial_standardization(sliding_window_consecutive_elements_max_right_pupil)
            sliding_window_consecutive_elements_sum_left_pupil_s2 = self._inter_trial_standardization(sliding_window_consecutive_elements_sum_left_pupil)
            sliding_window_consecutive_elements_sum_right_pupil_s2 = self._inter_trial_standardization(sliding_window_consecutive_elements_sum_right_pupil)
        if 'ECG' in feature_groups_to_analyze:
            sliding_window_hrv_sdnn_s2 = self._inter_trial_standardization(sliding_window_hrv_sdnn)
            sliding_window_hrv_rmssd_s2 = self._inter_trial_standardization(sliding_window_hrv_rmssd)
            # sliding_window_hrv_pnn50_s2 = self._inter_trial_standardization(sliding_window_hrv_pnn50)
        if 'cognitive' in feature_groups_to_analyze:
            sliding_window_cognitive_ies_s2 = self._inter_trial_standardization(sliding_window_cognitive_ies)

        all_features_additional_s2 = [s + '_s2' for s in all_features_additional]

        return (all_features_additional_s1, sliding_window_integral_left_pupil_s1, sliding_window_integral_right_pupil_s1,
        sliding_window_consecutive_elements_mean_left_pupil_s1, sliding_window_consecutive_elements_mean_right_pupil_s1,
        sliding_window_consecutive_elements_max_left_pupil_s1, sliding_window_consecutive_elements_max_right_pupil_s1,
        sliding_window_consecutive_elements_sum_left_pupil_s1, sliding_window_consecutive_elements_sum_right_pupil_s1,
        sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1, sliding_window_cognitive_ies_s1,
        all_features_additional_s2, sliding_window_integral_left_pupil_s2, sliding_window_integral_right_pupil_s2,
        sliding_window_consecutive_elements_mean_left_pupil_s2, sliding_window_consecutive_elements_mean_right_pupil_s2,
        sliding_window_consecutive_elements_max_left_pupil_s2, sliding_window_consecutive_elements_max_right_pupil_s2,
        sliding_window_consecutive_elements_sum_left_pupil_s2, sliding_window_consecutive_elements_sum_right_pupil_s2,
        sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2,
        sliding_window_cognitive_ies_s2)

    def _unpack_dict(self, gloc_window, sliding_window_mean_s1, number_windows, sliding_window_stddev_s1, sliding_window_max_s1,
                    sliding_window_range_s1, sliding_window_integral_left_pupil_s1, sliding_window_integral_right_pupil_s1,
                    sliding_window_consecutive_elements_mean_left_pupil_s1, sliding_window_consecutive_elements_mean_right_pupil_s1,
                    sliding_window_consecutive_elements_max_left_pupil_s1, sliding_window_consecutive_elements_max_right_pupil_s1,
                    sliding_window_consecutive_elements_sum_left_pupil_s1, sliding_window_consecutive_elements_sum_right_pupil_s1,
                    sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1, sliding_window_cognitive_ies_s1,
                    sliding_window_mean_s2, sliding_window_stddev_s2, sliding_window_max_s2, sliding_window_range_s2,
                    sliding_window_integral_left_pupil_s2, sliding_window_integral_right_pupil_s2,
                    sliding_window_consecutive_elements_mean_left_pupil_s2, sliding_window_consecutive_elements_mean_right_pupil_s2,
                    sliding_window_consecutive_elements_max_left_pupil_s2, sliding_window_consecutive_elements_max_right_pupil_s2,
                    sliding_window_consecutive_elements_sum_left_pupil_s2, sliding_window_consecutive_elements_sum_right_pupil_s2,
                    sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2, sliding_window_cognitive_ies_s2):
        """
        This function unpacks the dictionary structure to create a large features matrix (X matrix) and
        labels matrix (y matrix) for all trials being analyzed. This function will become unnecessary if
        the data remains in dataframe or arrays (rather than a dictionary).
        """

        # Find Unique Trial ID
        trial_id_in_data = list(sliding_window_mean_s1.keys())

        # Determine total length of new unpacked dictionary items
        total_rows = 0
        for i in range(np.size(trial_id_in_data)):
            total_rows += number_windows[trial_id_in_data[i]]

        # Create tuple of all dictionaries
        all_feature_dictionaries = [sliding_window_mean_s1, sliding_window_stddev_s1, sliding_window_max_s1, sliding_window_range_s1,
                                    sliding_window_integral_left_pupil_s1, sliding_window_integral_right_pupil_s1,
                                    sliding_window_consecutive_elements_mean_left_pupil_s1, sliding_window_consecutive_elements_mean_right_pupil_s1,
                                    sliding_window_consecutive_elements_max_left_pupil_s1, sliding_window_consecutive_elements_max_right_pupil_s1,
                                    sliding_window_consecutive_elements_sum_left_pupil_s1, sliding_window_consecutive_elements_sum_right_pupil_s1,
                                    sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1, sliding_window_cognitive_ies_s1,
                                    sliding_window_mean_s2, sliding_window_stddev_s2, sliding_window_max_s2,sliding_window_range_s2,
                                    sliding_window_integral_left_pupil_s2, sliding_window_integral_right_pupil_s2,
                                    sliding_window_consecutive_elements_mean_left_pupil_s2,sliding_window_consecutive_elements_mean_right_pupil_s2,
                                    sliding_window_consecutive_elements_max_left_pupil_s2, sliding_window_consecutive_elements_max_right_pupil_s2,
                                    sliding_window_consecutive_elements_sum_left_pupil_s2, sliding_window_consecutive_elements_sum_right_pupil_s2,
                                    sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2, sliding_window_cognitive_ies_s2]

        # Find all non-empty dictionaries
        non_empty_feature_dictionaries = []
        for dictionary in all_feature_dictionaries:
            if dictionary:
                non_empty_feature_dictionaries.append(dictionary)

        # Find number of columns (using non-empty dictionaries)
        num_cols = 0
        for dictionary in range(len(non_empty_feature_dictionaries)):
            current_dictionary = non_empty_feature_dictionaries[dictionary]
            num_cols = num_cols + np.shape(current_dictionary[trial_id_in_data[0]])[1]

        # Pre-allocate
        x_feature_matrix = np.zeros((total_rows, num_cols), dtype=np.float32)
        y_gloc_labels = np.zeros((total_rows, 1), dtype=np.float32)

        # Iterate through unique trial_id
        current_index = 0
        for i in range(np.size(trial_id_in_data)):

            # Find number of rows in trial
            num_rows = np.shape(sliding_window_mean_s1[trial_id_in_data[i]])[0]

            # For all non-empty dictionaries, set specific rows equal to the dictionary item corresponding to trial_id
            column_index = 0
            for dictionary in range(len(non_empty_feature_dictionaries)):

                # Find current dictionary
                current_dictionary = non_empty_feature_dictionaries[dictionary]

                # Set rows and columns in x_feature_matrix equal to current dictionary
                x_feature_matrix[current_index:num_rows + current_index,
                column_index:np.shape(current_dictionary[trial_id_in_data[i]])[1] + column_index] = current_dictionary[trial_id_in_data[i]].astype(np.float32)

                # Increment column index
                column_index += np.shape(current_dictionary[trial_id_in_data[i]])[1]

            # Set corresponding gloc labels from current trial
            y_gloc_labels[current_index:num_rows+current_index, :] = gloc_window[trial_id_in_data[i]].astype(np.float32)

            # Increment row index
            current_index += num_rows

        return y_gloc_labels, x_feature_matrix

    def _reduce_features(self, model_type, offset, stride, window_size, time_start, gloc_data_all_features_imputed_numpy, gloc_labels, features, experiment_metadata, select_features):
        if model_type[0] == "Complete" and model_type[1] == "Explicit":
            afe_indicator_column_windowed, gloc_compare, _ = self._sliding_window_max(
                experiment_metadata["AFE_indicator"], experiment_metadata["trial_id"], experiment_metadata["Time (s)"], gloc_labels,
                offset, stride, window_size, time_start
            )
            gloc_data_all_features_imputed_numpy = np.hstack([gloc_data_all_features_imputed_numpy, afe_indicator_column_windowed])
            features["All"].append("AFE_indicator_windowed")

        # Convert feature matrix to DataFrame for column selection
        gloc_data_all_features_imputed_numpy = pd.DataFrame(gloc_data_all_features_imputed_numpy, columns = features["All"])
        gloc_data_all_features_imputed_numpy = gloc_data_all_features_imputed_numpy[select_features]
        gloc_data_all_features_imputed_numpy = gloc_data_all_features_imputed_numpy.to_numpy()

        return gloc_data_all_features_imputed_numpy

    def _sliding_window_max(self, data_array, trial_column, time_column, label_array, offset, stride, window_size,time_start=0):
        """
        Compute sliding window max features and labels from a full array with a trial column.

        Input:
        - data_array: np.array of shape [num_rows, num_features] (all trials concatenated)
        - trial_column: array-like of trial IDs per row
        - time_column: array-like of timestamps per row
        - time_start: start time of first window
        - offset: offset for label window relative to feature window
        - stride: step size between windows
        - window_size: size of the sliding window

        Output:
        - all_features: np.array [total_windows, num_features]
        - all_labels: np.array [total_windows]
        - all_trials: np.array [total_windows] indicating which trial each window came from
        """

        trial_ids = pd.unique(trial_column)  # order-preserving, matching legacy script behavior

        all_features = []
        all_labels = []
        all_trials = []

        for trial_id in trial_ids:
            # Select rows for this trial
            trial_mask = (trial_column == trial_id)
            trial_times = np.array(time_column[trial_mask])
            trial_data = data_array[trial_mask, :]
            trial_gloc = np.array(label_array[trial_mask])  # replace with label column if different

            time_end = np.max(trial_times)
            number_windows = int(((time_end - offset) // stride) - (window_size // stride - 1))

            t = time_start
            for w in range(number_windows):
                # Feature window
                window_mask = (t <= trial_times) & (trial_times < t + window_size)
                window_features = np.nanmax(trial_data[window_mask, :], axis=0)

                # G-LOC window
                gloc_mask = ((t + offset) <= trial_times) & (trial_times < t + offset + window_size)
                window_label = np.any(trial_gloc[gloc_mask])

                all_features.append(window_features)
                all_labels.append(window_label)
                all_trials.append(trial_id)

                t += stride

        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        all_trials = np.array(all_trials)

        return all_features, all_labels, all_trials
    
    def _remove_constant_columns(self, x_feature_matrix, select_features):
        """
        This function removes all constant columns before feeding into the ML classifiers.
        """
        # Find all constant columns
        constant_columns = np.all(x_feature_matrix == x_feature_matrix[0,:], axis = 0)

        # Remove all constant columns from data frame
        x_feature_matrix = x_feature_matrix[:, ~constant_columns]

        select_features = [select_features[i] for i in range(len(select_features)) if ~constant_columns[i]]

        return x_feature_matrix, select_features
    
    def _process_NaN_temporal(self, y_gloc_labels, x_feature_matrix, all_features):
        """
        This is a temporary function for removing all rows with NaN values. This can be replaced by
        another method in the future, but is necessary for feeding into ML Classifiers.
        """
        # Find & remove columns if they have all NaN values
        nan_test = np.isnan(x_feature_matrix)
        index_column_all_NaN = np.all(nan_test, axis=0)
        x_feature_matrix_noNaN_cols = x_feature_matrix[:, ~index_column_all_NaN]

        # Adjust all_features to only include columns that don't have all NaN
        all_features = [all_features[i] for i in range(len(all_features)) if ~index_column_all_NaN[i]]

        # Identify rows with any NaNs
        row_nan_mask = np.isnan(x_feature_matrix_noNaN_cols).any(axis=1)

        # Save indices of removed rows
        removed_row_indices = np.where(row_nan_mask)[0]

        # Keep only rows without NaNs
        x_feature_matrix_noNaN = x_feature_matrix_noNaN_cols[~row_nan_mask]
        y_gloc_labels_noNaN = y_gloc_labels[~row_nan_mask]

        return y_gloc_labels_noNaN, x_feature_matrix_noNaN, all_features, removed_row_indices
    
    def _ready_outputs(self, x_feature_matrix, y_gloc_labels):
        x_feature_matrix = (
            x_feature_matrix.to_numpy() if hasattr(x_feature_matrix, "to_numpy") else np.asarray(x_feature_matrix)
        )
        y_gloc_labels = (
            y_gloc_labels.to_numpy().ravel() if hasattr(y_gloc_labels, "to_numpy") else np.ravel(y_gloc_labels)
        )

        return x_feature_matrix, y_gloc_labels