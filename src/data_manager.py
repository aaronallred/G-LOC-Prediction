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

class DataManager:
    # Order must match the legacy pipeline's feature accumulation order in
    # GLOC_data_processing.py → load_and_process_csv:
    #   ECG, BR, temp, fnirs, eyetracking, AFE, G, cognitive,
    #   rawEEG, processedEEG, strain, demographics
    # (fnirs and cognitive are never included but kept as positional reference)
    FEATURE_GROUPS_BY_MODEL_TYPE = {
        ("noAFE", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "AFE", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ("noAFE", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "rawEEG"),
        ("Complete", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "AFE", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ("Complete", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE"),
    }
    BASELINING_CHARACTERISTICS_BY_MODEL_TYPE = {
        "noAFE": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
        "Complete": ["v0", "v1", "v2", "v5", "v6"],
    }
    
    # Mapping of participant -> DC trial numbers for GOR EEG data files
    _EEG_PARTICIPANT_TRIALS = {
        1: [1, 2, 3],  2: [1, 2, 3],  3: [1, 2, 3],  4: [1, 2, 3],  5: [1, 2, 3],
        6: [1, 4, 6],  7: [2, 4, 6],  8: [1, 3],     9: [2, 5, 6],
        10: [2, 4, 5], 11: [1],       12: [1, 5],     13: [1, 3, 6],
    }

    _EEG_BASELINE_BANDS = ["delta", "theta", "alpha", "beta"]

    # 32 raw EEG channel names (without " - EEG" suffix)
    _RAW_EEG_CHANNELS = [
        'F1', 'Fz', 'F3', 'C3', 'C4', 'CP1', 'CP2',
        'T8', 'TP9', 'TP10', 'P7', 'P8', 'AFz', 'AF4',
        'FT9', 'FT10', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4',
        'FC6', 'C5', 'Cz', 'CP5', 'CP6', 'P5', 'P3',
        'P1', 'Pz', 'P4', 'P6',
    ]

    # Unengineered data streams used for feature selection in _generate_features.
    # EEG entries (raw + processed band power) are generated from _RAW_EEG_CHANNELS × _EEG_BASELINE_BANDS.
    _UNENGINEERED_STREAMS = frozenset(
        [
            'HR (bpm) - Equivital',
            'ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital',
            'HR_instant - Equivital', 'HR_average - Equivital', 'HR_w_average - Equivital',
            'BR (rpm) - Equivital',
            'Skin Temperature - IR Thermometer (°C) - Equivital',
            'Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii',
            'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii',
            'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii',
            'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii',
            'magnitude - Centrifuge',
            'Strain [0/1]',
            'participant_gender', 'participant_age', 'participant_height',
            'participant_weight', 'participant_BMI', 'participant_blood_volume',
            'participant_SBP_seated', 'participant_SBP_stand', 'participant_SBP_exercise',
            'participant_DBP_seated', 'participant_DBP_stand', 'participant_DBP_exercise',
            'participant_MAP_seated', 'participant_MAP_stand', 'participant_MAP_exercise',
            'participant_HR_seated', 'participant_HR_stand', 'participant_HR_exercise',
            'participant_max_leg_strength', 'participant_largest_leg_circumference',
            'participant_lower_leg_volume', 'participant_skinfolds_chest_avg',
            'participant_skinfolds_abd_avg', 'participant_skinfolds_thigh_avg',
            'participant_skinfolds_midax_avg', 'participant_skinfolds_subscap_avg',
            'participant_skinfolds_tri_avg', 'participant_skinfolds_supra_avg',
            'participant_skinfolds_sum', 'participant_percent_fat', 'participant_leg_length',
            'participant_arm_length', 'participant_midline_neck_length',
            'participant_lateral_neck_length', 'participant_torso_length_post',
            'participant_torso_length_ax', 'participant_head_to_heart', 'participant_head_girth',
            'participant_neck_girth', 'participant_chest_upper_girth', 'participant_chest_under_girth',
            'participant_waist_girth', 'participant_hip_girth', 'participant_thigh_girth',
            'participant_calf_girth', 'participant_biceps_girth_flex', 'participant_biceps_girth_relax',
            'participant_neck_flexion', 'participant_neck_extension', 'participant_neck_right_rotation',
            'participant_neck_left_rotation', 'participant_neck_left_lat_flex',
            'participant_neck_right_lat_flex', 'participant_pred_vo2',
        ]
        + [f'{ch} - EEG' for ch in _RAW_EEG_CHANNELS]
        + [f'{ch}_{band} - EEG' for ch in _RAW_EEG_CHANNELS for band in ["delta", "theta", "alpha", "beta"]]
    )

    def __init__(self, data_path: str = "../data/", testing: bool = False, random_seed: int = 42, use_reduced_dataset: bool = False) -> None:
        self.data_path = data_path
        self._data_locations = None
        self.testing = testing
        self.random_seed = random_seed
        self.use_reduced_dataset = use_reduced_dataset

    def get_data(
            self,
            model_type: Tuple[str, str],
            num_splits: int,
            kfold_ID: int,
            impute_path: str,
            subject_to_analyze: Optional[str] = None,
            trial_to_analyze: Optional[str] = None,
            impute_type: int = 1,
            n_neighbors: int = 4,
            baseline_window: float = 32.5,
            analysis_type: int = 2,
            remove_NaN_trials: bool = True,
            save_impute: bool = True,
            load_impute: bool = True
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load raw data and prepare predictor / target sets for advanced classifiers.

        Parameters:
            model_type: Tuple[str, str] — (model_kind, label_mode) e.g. ('Complete', 'Explicit')
            num_splits: Number of K-fold CV splits
            kfold_ID: Which fold to use (0 to num_splits-1)
            impute_path: Path to save/load the imputed data pickle file
            subject_to_analyze: Participant number for single-subject analysis
            trial_to_analyze: Trial number for single-trial analysis
            impute_type: Imputation method (1 = KNN on raw data)
            n_neighbors: Number of KNN imputation neighbors
            baseline_window: Baseline window duration in seconds
            analysis_type: 2=all data, 1=one participant, 0=one trial
            remove_NaN_trials: Remove trials with all-NaN sensors
            save_impute: Save imputed data to pickle
            load_impute: Load imputed data from pickle if available

        Returns:
            x_train, y_train, x_test, y_test, all_features
        """
        ################################################### FEATURES SETUP ###################################################
        feature_groups_to_analyze, baseline_methods_to_use = self._get_feature_groups_and_baseline_methods(model_type)

        ############################################# LOAD AND PROCESS DATA #############################################
        file_paths = self._get_data_locations()
        gloc_data = self._load_data(file_paths)
        gloc_data = self._filter_data_by_analysis_type(analysis_type, gloc_data, subject_to_analyze, trial_to_analyze)
        gloc_data, features = self._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, model_type, file_paths)
        gloc_labels = self._label_gloc_events(gloc_data)
        if model_type[0] != "Complete":
            gloc_data, gloc_labels = self._afe_subset(gloc_data, gloc_labels)

        ############################################# EEG Specific Imputation #############################################
        ####
        #  Note: This runs for 'complete' models, but because we are only using shared/overlapping EEG features for the
        #      'complete' case, this block doesn't do anything. Imputation occurs only for non-shared EEG features are used.
        #       This block requires 'AFE' to be an
        ####
        if model_type[0] == "Complete":
            gloc_data = self._eeg_specific_imputation(gloc_data, features)
            # Remove AFE_indicator from features (matches legacy behavior of removing 'condition')
            # AFE_indicator is stored separately in experiment_metadata during _reduce_memory
            features["All"] = [f for f in features["All"] if f != "AFE_indicator"]

        ############################################### MISSING DATA HANDLING ###############################################
        # Optional handling of raw NaN data, depending on remove_NaN_trials and impute_type
        if remove_NaN_trials:
            # This also returns a DataFrame with proportion of NaN values for each feature for each trial
            # Also modifies gloc_data and gloc_labels to remove trials with all NaNs in at least one feature
            # Note: DataFrame not used for the pipeline for memory purposes
            gloc_data, gloc_labels, _ = self._remove_all_nan_trials(gloc_data, features, gloc_labels)

        ################################################## REDUCE MEMORY ##################################################
        gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata = self._reduce_memory(
            gloc_data, gloc_labels, features, model_type
        )

        ################################################## Impute Missing ##################################################
        if impute_type == 1:
            gloc_data_all_features_imputed_numpy = self._impute_missing_data(
                gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata,
                impute_path, save_impute, load_impute, num_splits, kfold_ID, n_neighbors
            )
        else:
            gloc_data_all_features_imputed_numpy = gloc_data_all_features_numpy

        ################################################## BASELINE DATA ##################################################
        combined_baseline, combined_baseline_names = self._get_combined_baseline_data(
            gloc_data_all_features_imputed_numpy, experiment_metadata, baseline_window,
            baseline_methods_to_use, features, file_paths, model_type
        )

        ################################################ GENERATE FEATURES ################################################
        x_feature_matrix, features["All"] = self._generate_features(
            baseline_methods_to_use, combined_baseline, combined_baseline_names, experiment_metadata
        )

        ############################################# FEATURE CLEAN AND PREP ##############################################
        x_feature_matrix, y_gloc_labels, features["All"], experiment_metadata["trial_ints"] = self._feature_clean_and_prep(
            x_feature_matrix, gloc_labels_numpy, features, experiment_metadata, model_type, impute_type
        )

        ################################################ TRAIN/TEST SPLIT  ################################################
        x_train, y_train, x_test, y_test = self._get_train_test_split(
            x_feature_matrix, y_gloc_labels, experiment_metadata, num_splits, kfold_ID
        )

        return x_train, x_test, y_train, y_test, features["All"]

    def _get_feature_groups_and_baseline_methods(self, model_type: Tuple[str, str]) -> Tuple[Sequence[str], List[str]]:
        feature_groups_to_analyze = self.FEATURE_GROUPS_BY_MODEL_TYPE[model_type]
        baseline_methods_to_use = self.BASELINING_CHARACTERISTICS_BY_MODEL_TYPE[model_type[0]]

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

    @staticmethod
    def _filter_data_by_analysis_type(
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
            "EEG": {"processedEEG"}
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

    def _afe_subset(self, gloc_data: pd.DataFrame, gloc_labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
            Remove any trial that contains AFE condition (condition == 1).
        """
        trial_has_afe = gloc_data.groupby(["subject", "trial"])["AFE_indicator"].transform("max") # Mark trial as all 1 if any are 1, otherwise 0
        keep_mask = trial_has_afe != 1

        gloc_data = gloc_data.loc[keep_mask].reset_index(drop = True)
        gloc_labels = gloc_labels[keep_mask]

        return gloc_data, gloc_labels
    
    def _eeg_specific_imputation(self, gloc_data: pd.DataFrame, features: Dict[str, List[str]]) -> pd.DataFrame:
        """Mean-impute EEG channels that are exclusive to one AFE condition."""
        self._eeg_condition_impute(gloc_data, features, gloc_data["AFE_indicator"])
        return gloc_data

    def _eeg_condition_impute(self, gloc_data: pd.DataFrame, features: Dict[str, List[str]], afe_indicator_column: pd.Series, verbose: bool = True) -> None:
        """Mean-impute condition-specific EEG columns so both AFE/non-AFE have all features. Modifies gloc_data in-place."""
        # Create masks for each condition
        afe_mask = afe_indicator_column == 1
        nonafe_mask = afe_indicator_column == 0

        # Pull columns that need to be imputed for each type
        raw_eeg_feature_names = RawEEGGroup.get_separated_feature_names()
        processed_eeg_feature_names = ProcessedEEGGroup.get_separated_feature_names()
        all_afe_only_cols = raw_eeg_feature_names["AFE Only"] + processed_eeg_feature_names["AFE Only"]
        all_nonafe_only_cols = raw_eeg_feature_names["Non-AFE Only"] + processed_eeg_feature_names["Non-AFE Only"]
        eeg_feature_set = set(features["EEG"])
        afe_only_cols = [col for col in all_afe_only_cols if col in eeg_feature_set]
        nonafe_only_cols = [col for col in all_nonafe_only_cols if col in eeg_feature_set]

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

    def _remove_all_nan_trials(
            self,
            gloc_data: pd.DataFrame,
            features: Dict[str, List[str]],
            gloc_labels: np.ndarray,
            verbose: bool = True,
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
            "AFE_indicator": gloc_data["AFE_indicator"].to_numpy(dtype=np.float32).reshape(-1, 1),
        }
        # Legacy precision differs by model path:
        # - noAFE models keep features in float32 at this stage.
        # - Complete models can be promoted to float64 during the EEG condition
        #   imputation path because the condition column participates in array
        #   reconstruction. Match that behavior for strict parity tests.
        feature_dtype = np.float64 if model_type[0] == "Complete" else np.float32
        gloc_data_all_features_numpy = gloc_data[features["All"]].to_numpy(dtype=feature_dtype)
        gloc_labels_numpy = gloc_labels

        del gloc_data, gloc_labels
        return gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata

    @staticmethod
    def _convert_to_unique_ordered_integers(strings: np.ndarray) -> np.ndarray:
        """Convert strings to 1-based integers preserving first-appearance order."""
        codes, _ = pd.factorize(strings, sort=False)
        return (codes + 1).astype(np.float32)
    
    def _impute_missing_data(
            self,
            gloc_data_all_features_numpy: np.ndarray,
            gloc_labels_numpy: np.ndarray,
            experiment_metadata: Dict[str, Any],
            impute_path: str,
            save_impute: bool,
            load_impute: bool,
            num_splits: int,
            kfold_ID: int,
            n_neighbors: int,
    ) -> np.ndarray:
        # Load or compute imputed features
        # NOTE: impute_path is PROVIDED by caller; do not overwrite it.
        if load_impute and os.path.exists(impute_path):
            with open(impute_path, 'rb') as f:
                gloc_data_all_features_imputed_numpy = pickle.load(f)
            logger.info("Loaded imputed data from %s.", impute_path)
        else:
            # Only compute train/test indices when actually imputing
            _, _, _, _, train_indices, test_indices = self._groupedtrial_kfold_split(gloc_data_all_features_numpy, gloc_labels_numpy, num_splits, kfold_ID, experiment_metadata)
            gloc_data_all_features_imputed_numpy = self._faster_knn_impute_train_test(gloc_data_all_features_numpy, train_indices, test_indices, n_neighbors)

            del gloc_data_all_features_numpy # Free memory of original data after imputation

            if save_impute:
                os.makedirs(os.path.dirname(impute_path), exist_ok = True)
                with open(impute_path, 'wb') as f:
                    pickle.dump(gloc_data_all_features_imputed_numpy, f)
                logger.info("Saved imputed data to %s.", impute_path)

        return gloc_data_all_features_imputed_numpy

    def _groupedtrial_kfold_split(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            num_splits: int,
            kfold_ID: int,
            experiment_metadata: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/test using stratified group K-fold on trial groups."""
        # Grouped K-Fold setup (shuffle=False for reproducibility)
        gkf = StratifiedGroupKFold(n_splits = num_splits, shuffle = False)

        # Validate kfold_ID
        n_folds = gkf.get_n_splits()
        if kfold_ID < 0 or kfold_ID >= n_folds:
            raise ValueError(f"Fold index {kfold_ID} out of range (must be between 0 and {n_folds - 1})")

        # Get train and test indices for the specified fold
        trials = experiment_metadata["trial_ints"].reshape(-1, 1)
        train_index, test_index = next(islice(gkf.split(X, Y, trials), kfold_ID, kfold_ID + 1))

        # Extract split data
        x_train, y_train = X[train_index], Y[train_index]
        x_test, y_test = X[test_index], Y[test_index]

        return x_train, x_test, y_train, y_test, train_index, test_index

    def _faster_knn_impute_train_test(
            self,
            X: np.ndarray,
            train_ind: np.ndarray,
            test_ind: np.ndarray,
            k: int = 5,
            M: int = 32,
            efSearch: int = 64,
    ) -> np.ndarray:
        """Impute missing values via FAISS KNN, training on train set only to prevent leakage."""
        # Split into train and test
        X_train = X[train_ind]
        X_test = X[test_ind]

        # Identify missing values
        mask_train = np.isnan(X_train)
        mask_test = np.isnan(X_test)

        # Temporary mean imputation for FAISS indexing
        mean_vals = np.nanmean(X_train, axis = 0)
        X_train_temp = np.where(mask_train, mean_vals, X_train)
        X_test_temp = np.where(mask_test, mean_vals, X_test)

        if self.testing:
            faiss.omp_set_num_threads(1) # Use single thread for testing to ensure deterministic behavior (FAISS can be non-deterministic with multiple threads)

        # Build FAISS HNSW index on training data
        d = X_train.shape[1]
        index = faiss.IndexHNSWFlat(d, M)
        index.hnsw.efSearch = efSearch

        # Use fixed RNG seed for deterministic HNSW graph construction
        rng = faiss.RandomGenerator(self.random_seed)
        index.hnsw.rng = rng
        
        index.add(X_train_temp.astype(np.float32))

        # Impute training data
        distances, indices = index.search(X_train_temp.astype(np.float32), k + 1)
        X_train_imputed = X_train.copy()
        for i in range(X_train.shape[0]):
            neighbors = indices[i, 1:]  # skip self
            for j in range(X_train.shape[1]):
                if mask_train[i, j]:
                    neighbor_values = X_train_temp[neighbors, j]
                    X_train_imputed[i, j] = np.nanmean(neighbor_values)

        # Impute test data
        distances_test, indices_test = index.search(X_test_temp.astype(np.float32), k)
        X_test_imputed = X_test.copy()
        for i in range(X_test.shape[0]):
            neighbors = indices_test[i]
            for j in range(X_test.shape[1]):
                if mask_test[i, j]:
                    neighbor_values = X_train_temp[neighbors, j]
                    X_test_imputed[i, j] = np.nanmean(neighbor_values)

        # Rebuild into single array
        X_imputed = X.copy()
        X_imputed[train_ind] = X_train_imputed
        X_imputed[test_ind] = X_test_imputed

        return X_imputed

    def _get_combined_baseline_data(
            self,
            gloc_data_all_features_imputed_numpy: np.ndarray,
            experiment_metadata: Dict[str, Any],
            baseline_window: float,
            baseline_methods_to_use: List[str],
            features: Dict[str, List[str]],
            file_paths: Dict[str, Any],
            model_type: Tuple[str, str],
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Compute baselines for all specified methods and combine into a single feature set."""
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

        # Build feature-group index arrays using set lookups for O(1) membership
        phys_set, ecg_set, eeg_set = set(features["Phys"]), set(features["ECG"]), set(features["EEG"])
        phys_indices = [i for i, f in enumerate(features["All"]) if f in phys_set]
        ecg_indices = [i for i, f in enumerate(features["All"]) if f in ecg_set]
        eeg_indices = [i for i, f in enumerate(features["All"]) if f in eeg_set]

        context = BaselineContext(
            trial_column=experiment_metadata["trial_id"],
            time_column=experiment_metadata["Time (s)"],
            event_validated_column=experiment_metadata["event_validated"],
            subject_column=experiment_metadata["subject"],
            data_by_features={
                "All": gloc_data_all_features_imputed_numpy,
                "Phys": gloc_data_all_features_imputed_numpy[:, phys_indices],
                "ECG": gloc_data_all_features_imputed_numpy[:, ecg_indices],
                "EEG": gloc_data_all_features_imputed_numpy[:, eeg_indices],
            },
            features=features,
            baseline_window=baseline_window,
            model_type=model_type,
            participant_baseline_data=participant_baseline_rhr,
            eeg_baseline_data=eeg_baseline_data,
        )

        combined_baseline, combined_names, _, _, trial_order = baseline_data(baseline_methods_to_use, context)
        # Store trial_order for use in _generate_features to compute trial_ints in correct order
        experiment_metadata["trial_order"] = trial_order
        return combined_baseline, combined_names
        
    def _generate_features(
            self,
            baseline_methods_to_use: List[str],
            combined_baseline: Dict[str, np.ndarray],
            combined_baseline_names: List[str],
            experiment_metadata: Dict[str, Any],
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate feature matrices from baseline data using only unengineered data streams."""
        # Get trial_order from metadata (passed from baseline_data)
        trial_order = experiment_metadata.get("trial_order")
        trial_column = experiment_metadata["trial_id"]
        
        if trial_order is None:
            # Fallback: Use pandas unique to preserve first-appearance order like legacy code
            trial_order = pd.unique(trial_column)
        
        # Concatenate trial arrays along first axis
        x_feature_matrix = np.concatenate(
            [combined_baseline[tid] for tid in trial_order], 
            axis = 0
        ).astype(np.float32)
        
        # Build baseline suffixes as frozenset for faster membership testing
        baseline_suffixes = frozenset(baseline_methods_to_use)
        
        # Use boolean indexing instead of nested loops
        ue_indices = np.array([
            i for i, feature in enumerate(combined_baseline_names)
            if feature in self._UNENGINEERED_STREAMS or self._is_baselined_stream(
                feature, baseline_suffixes
            )
        ])
        
        # Compute trial_ints in the same order as features were concatenated (trial_order).
        trial_ids_for_rows = np.concatenate([
            np.full(combined_baseline[tid].shape[0], tid, dtype=object)
            for tid in trial_order
        ])
        trial_ints = self._convert_to_unique_ordered_integers(trial_ids_for_rows)
        
        x_feature_matrix = x_feature_matrix[:, ue_indices]
        x_feature_matrix = np.hstack([
            x_feature_matrix,
            trial_ints.reshape(-1, 1)
        ])
        
        all_features = [combined_baseline_names[i] for i in ue_indices]
        
        return x_feature_matrix, all_features
    
    def _is_baselined_stream(self, feature_name: str, baseline_suffixes: frozenset) -> bool:
        """Check if feature_name matches '{unengineered_stream}_{baseline_method}' pattern."""
        # Early exit if feature doesn't contain underscore (optimization)
        if '_' not in feature_name:
            return False
        
        # Extract potential stream and suffix
        parts = feature_name.rsplit('_', 1)
        if len(parts) != 2:
            return False
        
        stream_candidate, suffix = parts
        
        # Check if suffix is a baseline method and stream is unengineered
        return suffix in baseline_suffixes and stream_candidate in self._UNENGINEERED_STREAMS

    def _feature_clean_and_prep(
            self,
            x_feature_matrix: np.ndarray,
            gloc_labels_numpy: np.ndarray,
            features: Dict[str, List[str]],
            experiment_metadata: Dict[str, Any],
            model_type: Tuple[str, str],
            impute_type: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Remove constant columns, optionally add AFE indicator, and remove NaN rows."""
        # CRITICAL: Extract trial_ints BEFORE removing constant columns,
        # otherwise the last column might be removed and we'll extract the wrong column
        trial_ints = x_feature_matrix[:, -1].copy()
        
        # Separate trial_ints from feature matrix for constant column removal
        # The last column is trial_ints which was added by _generate_features
        x_features_without_trials = x_feature_matrix[:, :-1]
        feature_names_without_trials = features["All"]  # features["All"] should NOT include trial_ints
        
        # Remove constant columns (typically no constant columns)
        x_features_without_trials, feature_names_without_trials = self._remove_constant_columns(
            x_features_without_trials, feature_names_without_trials
        )

        # Add AFE_indicator as 2nd-to-last column (before trial_ints) for explicit only
        # Since trial_ints is already separated, just append AFE_indicator at the end of features
        model_kind, label_mode = model_type
        if model_kind == "Complete" and label_mode == "Explicit":
            x_features_without_trials = np.hstack([
                x_features_without_trials,
                experiment_metadata["AFE_indicator"].reshape(-1, 1),
            ])

        # Restore trial_ints as the last column
        x_feature_matrix = np.hstack([x_features_without_trials, trial_ints.reshape(-1, 1)])
        all_features = feature_names_without_trials  # Don't include trial_ints in feature names

        # List-wise deletion or clean any residual NaNs
        if impute_type in (1, 2):
            # Remove rows with NaN
            x_feature_matrix_noNaN, y_gloc_labels_noNaN, all_features, trials_noNaN = self._process_NaN(
                x_feature_matrix,
                gloc_labels_numpy,
                all_features,
                trial_ints
            )
        else:
            x_feature_matrix_noNaN = x_feature_matrix
            y_gloc_labels_noNaN = gloc_labels_numpy
            trials_noNaN = trial_ints

        return x_feature_matrix_noNaN, y_gloc_labels_noNaN, all_features, trials_noNaN

    @staticmethod
    def _remove_constant_columns(x_feature_matrix_noNaN: np.ndarray, all_features: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Remove columns with zero variance (constant across all rows)."""
        # Find all constant columns
        constant_columns = np.all(x_feature_matrix_noNaN == x_feature_matrix_noNaN[0,:], axis = 0)

        # Remove all constant columns from data frame
        x_feature_matrix_noNaN = x_feature_matrix_noNaN[:, ~constant_columns]

        all_features = [all_features[i] for i in range(len(all_features)) if ~constant_columns[i]]

        return x_feature_matrix_noNaN, all_features

    @staticmethod
    def _process_NaN(
            x_feature_matrix: np.ndarray,
            y_gloc_labels: np.ndarray,
            all_features: List[str],
            trials: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Remove all-NaN columns and any rows containing NaN values."""
        nan_mask = np.isnan(x_feature_matrix)

        # Find & remove columns if they have all NaN values
        index_column_all_NaN = nan_mask.all(axis=0)
        if index_column_all_NaN.any():
            x_feature_matrix = x_feature_matrix[:, ~index_column_all_NaN]
            all_features = [f for f, keep in zip(all_features, ~index_column_all_NaN) if keep]
            nan_mask = nan_mask[:, ~index_column_all_NaN]

        # Find & Remove rows in label/trial arrays if they have NaN values
        row_has_nan = nan_mask.any(axis=1)
        if row_has_nan.any():
            keep_rows = ~row_has_nan
            x_feature_matrix = x_feature_matrix[keep_rows]
            y_gloc_labels = y_gloc_labels[keep_rows]
            trials = trials[keep_rows]

        return x_feature_matrix, y_gloc_labels, all_features, trials

    def _get_train_test_split(
            self,
            x_feature_matrix: np.ndarray,
            y_gloc_labels: np.ndarray,
            experiment_metadata: Dict[str, Any],
            num_splits: int,
            kfold_ID: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split into train/test, standardize features, and preserve trial indices in the last column."""
        # Perform stratified group k-fold split
        x_train, x_test, y_train, y_test, _, _ = self._groupedtrial_kfold_split(
            x_feature_matrix, y_gloc_labels, num_splits, kfold_ID, experiment_metadata)

        # Extract trial indices from last column (use direct slicing for memory efficiency)
        train_trials = x_train[:, -1:]
        test_trials = x_test[:, -1:]
        
        # Remove trial column from features
        x_train = x_train[:, :-1]
        x_test = x_test[:, :-1]

        # Standardize features based on training data distribution
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Reattach trial indices as final column
        x_train = np.hstack([x_train, train_trials])
        x_test = np.hstack([x_test, test_trials])

        return x_train, y_train, x_test, y_test