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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .baseline import BaselineContext, baseline_data
from .features import FEATURE_REGISTRY, RawEEGGroup, ProcessedEEGGroup
from src.models.base import BaseModel

logger = logging.getLogger(__name__)

@dataclass(frozen = True)
class ModelType:
    afe_filter: str  # "noAFE" or "Complete"
    feature_set: str # "Explicit" or "Implicit"

@dataclass(frozen = True)
class ExperimentMetadata:
    trial_id: np.ndarray
    trial_ints: np.ndarray
    time_s: np.ndarray
    event_validated: np.ndarray
    subject: np.ndarray
    afe_indicator: np.ndarray
    
class DataPipeline:
    """ Unified data pipeline for Traditional and Advanced Classifiers """

    FEATURE_GROUPS_BY_MODEL_TYPE = {
        ModelType("noAFE", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ModelType("noAFE", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "rawEEG"),
        ModelType("Complete", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ModelType("Complete", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "rawEEG"),
    }
    BASELINING_CHARACTERISTICS_BY_MODEL_TYPE = {
        "noAFE": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
        "Complete": ["v0", "v1", "v2", "v5", "v6"],
    }

    # Predefined hyperparameters for traditional classifiers
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

    # Mapping of participant -> DC trial numbers for GOR EEG data files
    _EEG_PARTICIPANT_TRIALS = {
        1: [1, 2, 3],  2: [1, 2, 3],  3: [1, 2, 3],   4: [1, 2, 3],  5: [1, 2, 3],
        6: [1, 4, 6],  7: [2, 4, 6],  8: [1, 3],      9: [2, 5, 6],
        10: [2, 4, 5], 11: [1],       12: [1, 5],     13: [1, 3, 6],
    }

    _EEG_BASELINE_BANDS = ["delta", "theta", "alpha", "beta"]

    def __init__(
            self, 
            data_folder: str, 
            model: BaseModel, 
            model_type: ModelType,
            kfold_ID: int = 0,
            baseline_window: float = 32.5,
            window_size: Optional[float] = None,
            stride: Optional[float] = None,
            feature_reduction_type: Optional[str] = None,
            imbalance_type: str = "none",
            n_neighbors: int = 4,
            num_splits: int = 10,
            random_state: int = 42,
            subject_to_analyze: Optional[str] = None,
            trial_to_analyze: Optional[str] = None,
            analysis_type: int = 2, # TODO: Change to enum
            remove_NaN_trials: bool = True,
            should_impute: bool = True,
            impute_file_name: str = "gloc_data_imputed.pkl",
            impute_k: int = 5,
            impute_M: int = 32,
            impute_efSearch: int = 64,
            backstep: int = 0, # Seconds
            data_rate: int = 25, # Hz
            verbose: bool = False
        ):
        self.data_folder = data_folder
        self.model = model
        self.model_type = model_type
        self.random_state = random_state
        self.verbose = verbose

        # Data Processing Hyperparameters
        self.baseline_window = baseline_window
        self.window_size = window_size
        self.stride = stride
        self.feature_reduction_type = feature_reduction_type
        self.feature_groups_to_analyze = None
        self.baseline_methods_to_use = None
        self.imbalance_type = imbalance_type
        self.impute_type = None
        self.n_neighbors = n_neighbors

        # Dataset Splitting Parameters
        self.num_splits = num_splits

        if kfold_ID < 0 or kfold_ID >= num_splits:
            raise ValueError(f"kfold_ID must be between 0 and num_splits-1 (inclusive). Got kfold_ID = {kfold_ID} with num_splits = {num_splits}.")
        self.kfold_ID = kfold_ID

        self.subject_to_analyze = subject_to_analyze
        self.trial_to_analyze = trial_to_analyze
        self.analysis_type = analysis_type
        self.remove_NaN_trials = remove_NaN_trials

        if self.model.is_traditional:
            self.data_split_method = StratifiedKFold(n_splits = self.num_splits, shuffle = True, random_state = self.random_state)
        else:
            self.data_split_method = StratifiedGroupKFold(n_splits = self.num_splits, shuffle = True, random_state = self.random_state)
        
        # Imputation Parameters
        self.should_impute = should_impute
        self.impute_file_name = impute_file_name
        self.impute_k = impute_k
        self.impute_M = impute_M
        self.impute_efSearch = impute_efSearch

        # Temporal Offset Parameters
        self.backstep = backstep
        self.data_rate = data_rate

        # Internal State
        self.data_locations = None
        self.gloc_data = None
        self.gloc_labels = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.nan_proportion_df = None
        self.experiment_metadata = None
    
    def get_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.gloc_data, self.gloc_labels

    def process_and_get_data(self):
        ################################################### FEATURES SETUP ###################################################
        logger.info("Assigning hyperparameters and feature groups based on model type '%s' and model '%s'.", self.model_type, self.model.get_name())
        self._assign_hyperparameters_by_classifier()
        self._assign_feature_groups_and_baseline_methods()

        ############################################# LOAD AND PROCESS DATA ##################################################
        logger.info("Assigning data locations and loading data.")
        self._assign_data_locations()
        self._load_data()
        self._filter_data_by_analysis_type()
        self._process_and_get_feature_names()
        self._label_gloc_events()

        ############################################# EEG Specific Imputation ################################################
        # Note: This is currently not needed since the only EEG features we are using for the 'Complete' model are the shared/overlapping features that are present in both conditions. 
        # If we were to use non-shared EEG features, then this imputation would be necessary to fill in the NaNs for the missing condition.
        # logger.info("Performing EEG condition imputation if necessary based on model type '%s'.", self.model_type)
        # self._eeg_condition_impute() 

        ################################################# AFE Subsetting #####################################################
        logger.info("Performing AFE subsetting if necessary based on model type '%s'.", self.model_type)
        self._afe_subset()

        ############################################### MISSING DATA HANDLING ################################################
        logger.info("Performing NaN trial removal if remove_NaN_trials is set to True.")
        self._remove_all_nan_trials()

        ################################################## REDUCE MEMORY #####################################################
        logger.info("Extracting experiment metadata and reducing memory footprint.")
        self._extract_experiment_metadata()
        self._reduce_memory()

        ################################################### Impute Missing ###################################################
        logger.info("Performing train/test split and imputation if should_impute is set to True.")
        self._kfold_split()
        self._impute_missing_data()

        ################################################# Prediction Offset ##################################################
        self._apply_prediction_offset()

        return self.gloc_data, self.gloc_labels
    
    def get_nan_proportion_df(self) -> Optional[pd.DataFrame]:
        """Returns dataframe with proportion of NaNs for each feature by trial, if calculated during _remove_all_nan_trials."""
        return self.nan_proportion_df
    
    def _assign_hyperparameters_by_classifier(self) -> None:
        """Set hyperparameters for a given classifier type."""

        params = self._CLASSIFIER_HYPERPARAMETERS.get(self.model.get_name())
        if params is None:
            # Using advanced classifier which should have provided parameters
            logger.warning(f"Classifier type '{self.model.get_name()}' not found in predefined hyperparameters. Using initialization values.")
            return

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

        if not self.model.is_traditional:
            # Only update baseline methods to use for advanced classifier as they were predefined for traditional classifiers
            self.baseline_methods_to_use = self.BASELINING_CHARACTERISTICS_BY_MODEL_TYPE[self.model_type.afe_filter]

        # NOTE:
        # AFE indicator is required for EEG imputation in Complete models,
        # but is only included as a predictive feature for Explicit models.

    def _assign_data_locations(self) -> None:
        """Get file locations for all relevant data files and aassign to self._data_locations."""

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

        self._data_locations = {
            "main": os.path.join(self.data_path, "all_trials_25_hz_stacked_null_str_filled.csv"),
            "baseline": os.path.join(self.data_path, "ParticipantBaseline.csv"),
            "demographics": os.path.join(self.data_path, "GLOC_Effectiveness_Final.csv"),
            "eeg_list": list_of_eeg_data_file_paths,
            "baseline_eeg_processed_list": list_of_baseline_eeg_processed_file_paths,
        }

    def _load_data(self) -> None:
        """Load and cache data. Uses a fully processed cache to avoid repeated EEG Excel reads."""

        main_csv_file = self.data_locations["main"]
        main_data_pickle_file = main_csv_file.replace(".csv", ".pkl")
        enriched_pickle_file = main_csv_file.replace(".csv", "_with_EEG_GOR.pkl")

        if os.path.isfile(enriched_pickle_file):
            logger.info("Loading preprocessed data from pickle at %s.", enriched_pickle_file)
            self.gloc_data = pd.read_pickle(enriched_pickle_file)
            return
        
        logger.info("Base pickle not found at %s. Loading from CSV and caching.", main_data_pickle_file)
        self.gloc_data = pd.read_csv(main_csv_file)

        # Add GOR and EEG data once, then persist the enriched version.
        self._add_EEG_GOR()

        condition = self.gloc_data.pop("condition")
        afe_indicator = condition.map({"N": False, "AFE": True})
        if afe_indicator.isna().any():
            logger.warning("Found condition values outside {'N', 'AFE'}. Defaulting unknown values to False.")
        self.gloc_data["AFE_indicator"] = afe_indicator.fillna(False).astype(np.bool_)

        # Convert float64 to float32 to reduce memory footprint.
        float64_cols = self.gloc_data.select_dtypes(include = "float64").columns
        if len(float64_cols) > 0:
            self.gloc_data = self.gloc_data.astype({col: "float32" for col in float64_cols})

        # Vectorized extraction of subject/trial IDs is significantly faster than Python loops.
        trial_parts = self.gloc_data["trial_id"].str.split("-", n = 1, expand = True)
        subject_vals = pd.to_numeric(trial_parts[0], errors = "coerce").to_numpy()
        trial_vals = pd.to_numeric(trial_parts[1], errors = "coerce").to_numpy()
        if np.isnan(subject_vals).any() or np.isnan(trial_vals).any():
            logger.warning("Found malformed trial IDs while extracting subject/trial. Defaulting malformed rows to 0.")
        self.gloc_data["subject"] = np.nan_to_num(subject_vals, nan = 0).astype(np.uint8)
        self.gloc_data["trial"] = np.nan_to_num(trial_vals, nan = 0).astype(np.uint8)

        self.gloc_data.to_pickle(enriched_pickle_file)
    
    def _add_EEG_GOR(self) -> None:
        """Slot in GOR EEG band power data from xlsx files, replacing NaNs in the main CSV."""

        list_of_eeg_data_files = self.data_locations["eeg_list"]
        trial_indices_map = self.gloc_data.groupby("trial_id", sort=False).indices
        event_validated = self.gloc_data["event_validated"].to_numpy()
        trial_ids = self.gloc_data["trial_id"].to_numpy()
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
            corresponding_trial = f"{int(match.group(1)):02d}-{int(match.group(2)):02d}"

            if not os.path.isfile(current_file):
                logger.warning("EEG source file not found: %s", current_file)
                continue

            # Read all sheets in one call to reduce repeated workbook parsing overhead.
            band_sheets = pd.read_excel(current_file, sheet_name = band_names)
            column_names = band_sheets[band_names[0]].columns[:-1]
            band_arrays = {
                band: band_sheets[band].iloc[:, :-1].to_numpy(dtype = np.float32, copy = False)
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
            n_rows = min(band_arrays["delta"].shape[0], len(trial_indices) - start_pos)
            if n_rows <= 0:
                logger.warning("No overlapping rows while inserting EEG for trial %s.", corresponding_trial)
                continue
            trial_indexer = trial_indices[start_pos : start_pos + n_rows]

            # Build column names and assign values for each band
            for band in band_names:
                cols = [f"{c}_{band} - EEG" for c in column_names]
                self.gloc_data.loc[trial_indexer, cols] = band_arrays[band][:n_rows]

    def _filter_data_by_analysis_type(self) -> None:
        """Filter out sections of gloc_data specified using analysis_type."""
        if self.analysis_type == 0: # One Trial / One Subject
            mask = (self.gloc_data["subject"] == self.subject_to_analyze) & (self.gloc_data["trial"] == self.trial_to_analyze)
        elif self.analysis_type == 1: # All Trials for One Subject
            mask = (self.gloc_data["subject"] == self.subject_to_analyze)
        # All Trials for All Subjects (analysis_type == 2) does not require filtering
            
        self.gloc_data = self.gloc_data[mask].reset_index(drop = True)
    
    def _process_and_get_feature_names(self) -> None:
        """Process data and extract feature names based on specified feature groups."""
        # Defining which features go into which group of feature groups
        GROUPS_OF_FEATURE_GROUPS = {
            # "Phys": {"ECG", "BR", "temp", "fnirs", "eyetracking", "rawEEG", "processedEEG"}, # fNIRS not used due to warning
            "Phys": {"ECG", "BR", "temp", "eyetracking", "rawEEG", "processedEEG"},
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

        for group_name in self.feature_groups_to_analyze:
            if group_name not in FEATURE_REGISTRY:
                logger.warning("Feature group '%s' not recognized. Skipping.", group_name)
                continue

            processor = FEATURE_REGISTRY[group_name]

            # Process data for the feature group
            self.gloc_data = processor.process(self.gloc_data, self.data_locations)
            feature_names = processor.get_feature_names(self.model_type)

            # Adding features to relevant groups
            if group_name in GROUPS_OF_FEATURE_GROUPS["Phys"]:
                features_phys.extend(feature_names)

            if group_name in GROUPS_OF_FEATURE_GROUPS["ECG"]:
                features_ecg.extend(feature_names)

            if group_name in GROUPS_OF_FEATURE_GROUPS["EEG"]:
                features_eeg.extend(feature_names)

            features_all.extend(feature_names)

        self.feature_names = features

    def _label_gloc_events(self) -> None:
        """
        This function creates a g-loc label for the data based on the event_validated column. The event
        is labeled as 1 between GLOC and Return to Consciousness.
        """
        event_validated = self.gloc_data["event_validated"]

        # Find all GLOC and RTC indices, pair them in order, and label between each pair
        gloc_indices = np.where(event_validated.to_numpy() == "GLOC")[0]
        rtc_indices = np.where(event_validated.to_numpy() == "return to consciousness")[0]

        trial_ids = self.gloc_data["trial_id"].to_numpy()
        gloc_labels = np.zeros(len(self.gloc_data))

        for i in range(len(gloc_indices)):
            start = gloc_indices[i]
            end = rtc_indices[i]
            if trial_ids[start] == trial_ids[end]:
                gloc_labels[start:end] = 1

        self.gloc_labels = gloc_labels

    def _eeg_condition_impute(self) -> None:
        """
            Mean-impute condition-specific EEG columns so both AFE/non-AFE have all features. Modifies gloc_data in-place.
        
            Note: 
                * This runs for 'complete' models, but because we are only using shared/overlapping EEG features for the 'complete' case, this block doesn't do anything. 
                * Imputation occurs only for non-shared EEG features are used.
        """
        if self.model_type.afe_filter != "Complete":
            logger.info("Skipping EEG imputation since model type '%s' does not include all AFE features.", self.model_type)
            return

        # Create masks for each condition
        afe_mask = self.gloc_data["AFE_indicator"] == 1
        nonafe_mask = self.gloc_data["AFE_indicator"] == 0

        # Pull columns that need to be imputed for each type
        raw_eeg_feature_names = RawEEGGroup.get_separated_feature_names()
        processed_eeg_feature_names = ProcessedEEGGroup.get_separated_feature_names()
        all_afe_only_cols = raw_eeg_feature_names["AFE Only"] + processed_eeg_feature_names["AFE Only"]
        all_nonafe_only_cols = raw_eeg_feature_names["Non-AFE Only"] + processed_eeg_feature_names["Non-AFE Only"]
        eeg_feature_set = set(self.feature_names["EEG"])
        afe_only_cols = [col for col in all_afe_only_cols if col in eeg_feature_set]
        nonafe_only_cols = [col for col in all_nonafe_only_cols if col in eeg_feature_set]

        # Mean imputation processing
        if afe_only_cols:
            means = self.gloc_data.loc[afe_mask, afe_only_cols].mean(skipna = True)
            if self.verbose:
                missing_counts = self.gloc_data.loc[nonafe_mask, afe_only_cols].isna().sum()
            self.gloc_data.loc[nonafe_mask, afe_only_cols] = self.gloc_data.loc[nonafe_mask, afe_only_cols].fillna(means)

            # Show columns imputed and how many rows were imputed for each column
            if self.verbose:
                for col, n in missing_counts.items():
                    logger.debug("Imputed %d values in '%s' for non-AFE rows.", n, col)

        if nonafe_only_cols:
            means = self.gloc_data.loc[nonafe_mask, nonafe_only_cols].mean(skipna = True)
            if self.verbose:
                missing_counts = self.gloc_data.loc[afe_mask, nonafe_only_cols].isna().sum()
            self.gloc_data.loc[afe_mask, nonafe_only_cols] = self.gloc_data.loc[afe_mask, nonafe_only_cols].fillna(means)

            # Show columns imputed and how many rows were imputed for each column
            if self.verbose:
                for col, n in missing_counts.items():
                    logger.debug("Imputed %d values in '%s' for AFE rows.", n, col)

    def _afe_subset(self) -> None:
        """
            Remove any trial that contains AFE condition (condition == 1).
        """
        if self.model_type.afe_filter != "noAFE":
            logger.info("Skipping AFE subsetting since model type '%s' is not 'noAFE'.", self.model_type)
            return

        trial_has_afe = self.gloc_data.groupby(["subject", "trial"])["AFE_indicator"].transform("max") # Mark trial as all 1 if any are 1, otherwise 0
        keep_mask = trial_has_afe != 1

        self.gloc_data = self.gloc_data.loc[keep_mask].reset_index(drop = True)
        self.gloc_labels = self.gloc_labels[keep_mask]

    def _remove_all_nan_trials(self) -> None:
        """
            Optional handling of raw NaN data, depending on remove_NaN_trials and impute_type
            Remove trials where at least one feature is entirely NaN. 
            Modifies gloc_data and gloc_labels in-place.
            Calculates and stores a dataframe with the proportion of NaNs for each feature by trial, which can be accessed using get_nan_proportion_df().
        """
        if self.remove_NaN_trials == False:
            logger.info("Skipping NaN trial removal since remove_NaN_trials is set to False.")
            return

        # All features and subject trial info to be put into a reduced dataframe from gloc_data
        all_features = self.feature_names["All"]
        all_features_with_ids = all_features + ["subject", "trial"]
        reduced_data_frame = self.gloc_data[all_features_with_ids]

        nan_flags = reduced_data_frame[all_features].isna()
        group_keys = [reduced_data_frame["subject"], reduced_data_frame["trial"]]
        grouped = nan_flags.groupby(group_keys, sort=False)

        nan_proportion_df = grouped.mean()
        all_nan_cols_df = grouped.all()
        bad_trials = all_nan_cols_df.any(axis=1)

        if self.verbose and bad_trials.any():
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

        rows_to_remove = self.gloc_data.index[~keep_mask]
        self.gloc_data.drop(rows_to_remove, inplace=True)
        self.gloc_data.reset_index(drop=True, inplace=True)

        kept_labels = self.gloc_labels[keep_mask]
        self.gloc_labels.resize(kept_labels.shape, refcheck=False)
        self.gloc_labels[:] = kept_labels

        N = int(bad_trials.shape[0])
        M = int(bad_trials.sum())

        if self.verbose:
            # Print NaN findings
            logger.info("%d trials with all NaNs for at least one feature out of %d trials. %d remaining.", M, N, N - M)

        self.nan_proportion_df = nan_proportion_df

    def _extract_experiment_metadata(self) -> None:
        """Extract out experiment metadata from gloc_data and store in self.experiment_metadata. This is done before memory reduction since it relies on the DataFrame format."""
        trial_id_arr = self.gloc_data["trial_id"].to_numpy(dtype = str)
        self.experiment_metadata = ExperimentMetadata(
            trial_id = trial_id_arr,
            trial_ints = self._convert_to_unique_ordered_integers(trial_id_arr),
            time_s = self.gloc_data["Time (s)"].to_numpy(dtype = np.float32),
            event_validated = self.gloc_data["event_validated"].to_numpy(dtype = str),
            subject = self.gloc_data["subject"].to_numpy(dtype = np.uint8),
            afe_indicator = self.gloc_data["AFE_indicator"].to_numpy(dtype = np.bool_)
        )

    def _reduce_memory(self) -> None:
        """Extract numpy arrays from DataFrame and free the DataFrame to reduce memory usage."""

        # Convert gloc_data into a numpy array with only feature columns
        # Feature columns only consist of float32 datatypes
        self.gloc_data = self.gloc_data[self.feature_names["All"]].to_numpy(dtype = np.float32)

        # Downsize gloc_labels to boolean datatype
        self.gloc_labels = self.gloc_labels.to_numpy(dtype = np.bool_)

    def _convert_to_unique_ordered_integers(self, strings: np.ndarray) -> np.ndarray:
        """Convert strings to 1-based integers preserving first-appearance order."""
        codes, _ = pd.factorize(strings, sort=False)
        return (codes + 1).astype(np.uint8)
    
    def _kfold_split(self) -> None:
        """
            Split data into train/test using stratified K-fold for traditional classifiers and stratified group K-fold for traditional classifiers.

            Creates new class attributes:
                self.X_train
                self.X_test
                self.y_train
                self.y_test
            
            Deletes previous self.gloc_data and self.gloc_labels to free memory, as they are now split into train/test sets.
        """

        # Get train and test indices for the specified fold
        # If StratifiedKFold, split on gloc_data and gloc_labels directly. If StratifiedGroupKFold, split on gloc_data, gloc_labels, and trial groups from experiment_metadata.
        train_indices, test_indices = next(islice(self.data_split_method.split(self.gloc_data, self.gloc_labels, self.experiment_metadata.trial_ints), self.kfold_ID, self.  kfold_ID + 1))

        # Extract split data
        X_train, y_train = self.gloc_data[train_indices], self.gloc_labels[train_indices]
        X_test, y_test = self.gloc_data[test_indices], self.gloc_labels[test_indices]

        # Free memory of original data after split
        del self.gloc_data
        del self.gloc_labels

        # Update new data attributes
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_indices = train_indices
        self.test_indices = test_indices

    def _impute_missing_data(self) -> None:
        """Impute missing data using KNN imputation with FAISS for efficient neighbor search. Only runs if should_impute is True."""
        if not self.should_impute:
            logger.info("Skipping imputation since should_impute is set to False.")
            return
        
        if self.X_train is None or self.y_train is None or self.X_test is None or self.y_test is None:
            logger.info("Train/test split not found. Performing K-fold split before imputation.")
            self._kfold_split()

        # Load cached imputed data if available
        impute_path = os.path.join(self.data_path, "Imputed_Data", self.model_type.afe_filter + "_" + self.model_type.feature_set, f"Fold_{self.kfold_I}", self.model.get_name() + "_" + self.impute_file_name)
        if os.path.exists(impute_path):
            with open(impute_path, 'rb') as f:
                self.X_train, self.y_train, self.X_test, self.y_test = pickle.load(f)
            logger.info("Loaded imputed train/test split from %s.", impute_path)
            return
        
        self._knn_train_test_imputation()

        # Save imputed train/test split for future use
        os.makedirs(os.path.dirname(impute_path), exist_ok = True)
        with open(impute_path, 'wb') as f:
            pickle.dump((self.X_train, self.y_train, self.X_test, self.y_test), f)
        logger.info("Saved imputed train/test split to %s.", impute_path)
    
    def _knn_train_test_imputation(self) -> None:
        """Perform KNN imputation using FAISS for efficient neighbor search. Modifies self.X_train and self.X_test in-place with imputed values."""
        num_training_rows = self.X_train.shape[0]
        num_training_cols = self.X_train.shape[1]
        num_testing_rows = self.X_test.shape[0]
        num_testing_cols = self.X_test.shape[1]

        # Identify missing values
        mask_train = np.isnan(self.X_train)
        mask_test = np.isnan(self.X_test)

        # Temporary mean imputation for FAISS indexing
        mean_vals = np.nanmean(self.X_train, axis = 0)
        X_train_temp = np.where(mask_train, mean_vals, self.X_train).astype(np.float32)
        X_test_temp = np.where(mask_test, mean_vals, self.X_test).astype(np.float32)

        # Build FAISS HNSW index on training data
        index = faiss.IndexHNSWFlat(num_training_cols, self.impute_M)
        index.hnsw.efSearch = self.impute_efSearch
        index.add(X_train_temp)

        # Impute training data
        _, train_indices = index.search(X_train_temp, self.impute_k + 1)
        X_train_imputed = self.X_train.copy()
        for i in range(num_training_rows):
            neighbors = train_indices[i, 1:]  # skip self
            for j in range(num_training_cols):
                if mask_train[i, j]:
                    neighbor_values = X_train_temp[neighbors, j]
                    X_train_imputed[i, j] = np.nanmean(neighbor_values)

        # Impute test data
        _, indices_test = index.search(X_test_temp, self.impute_k)
        X_test_imputed = self.X_test.copy()
        for i in range(num_testing_rows):
            neighbors = indices_test[i]
            for j in range(num_testing_cols):
                if mask_test[i, j]:
                    neighbor_values = X_train_temp[neighbors, j]
                    X_test_imputed[i, j] = np.nanmean(neighbor_values)

        # Update class attributes with imputed data
        self.X_train = X_train_imputed
        self.X_test = X_test_imputed

    def _apply_prediction_offset(self) -> None:
        """
        Shift both train and test GLOC labels to the left by `backstep * data_rate`
        frames within each trial. The beginning is truncated and the end padded with zeros.
        """

        offset = int(self.backstep * self.data_rate)
        if offset <= 0:
            logger.info("Skipping y prediction offset since backstep is set to 0.")
            return

        if self.y_train is None or self.y_test is None:
            logger.info("Skipping y prediction offset since train/test labels are not initialized.")
            return

        if not hasattr(self, "train_indices") or not hasattr(self, "test_indices"):
            logger.info("Train/test indices not found. Performing K-fold split before y prediction offset.")
            self._kfold_split()

        train_trials = self.experiment_metadata.trial_ints[self.train_indices]
        test_trials = self.experiment_metadata.trial_ints[self.test_indices]

        def _shift_by_trial(labels: np.ndarray, trial_groups: np.ndarray, shift: int) -> np.ndarray:
            shifted = np.asarray(labels).copy()
            unique_trials = np.unique(trial_groups)

            for trial in unique_trials:
                trial_idx = np.nonzero(trial_groups == trial)[0]
                current_y = shifted[trial_idx]

                if not np.any(current_y):
                    continue

                if shift >= current_y.shape[0]:
                    shifted[trial_idx] = np.zeros_like(current_y)
                    continue

                y_shifted = current_y[shift:]
                shifted[trial_idx] = np.concatenate([
                    y_shifted,
                    np.zeros(shift, dtype = current_y.dtype),
                ])[: current_y.shape[0]]

            return shifted

        self.y_train = _shift_by_trial(self.y_train, train_trials, offset)
        self.y_test = _shift_by_trial(self.y_test, test_trials, offset)

class AdvancedDataPipeline(DataPipeline):
    """
    Advanced data pipeline for GLOC event prediction, refactored from load_and_prepare_data_advanced.
    
    Performance vs. Legacy Pipeline (full GLOC dataset, 5-fold CV, KNN k=4, baseline 32.5s):
    ──────────────────────────────────────────────────────────────────────────────────
    Model Type           │ Speedup │ Time Reduction │ Memory Impact
    ──────────────────────────────────────────────────────────────────────────────────
    Complete/Explicit    │  1.46x  │     31.3%      │  7.4% less
    Complete/Implicit    │  2.24x  │     55.4%      │  16.6% more*
    noAFE/Explicit       │  1.21x  │     17.5%      │  16.4% less
    noAFE/Implicit       │  2.00x  │     49.9%      │  47.0% more*
    ──────────────────────────────────────────────────────────────────────────────────
    Aggregate (all)      │  1.48x  │     32.6%      │  1.6% net change
    
    * Implicit modes show higher memory usage, likely due to intermediate array allocation
      during feature engineering. This is acceptable given 50%+ speed improvements.
    """
    # Order must match the legacy pipeline's feature accumulation order in
    # GLOC_data_processing.py → load_and_process_csv:
    #   ECG, BR, temp, fnirs, eyetracking, AFE, G, cognitive,
    #   rawEEG, processedEEG, strain, demographics
    # (fnirs and cognitive are never included but kept as positional reference)
    FEATURE_GROUPS_BY_MODEL_TYPE = {
        ("noAFE", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "AFE", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ("noAFE", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "rawEEG"),
        ("Complete", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "AFE", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ("Complete", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "AFE", "rawEEG"),
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

    def __init__(self, data_path: str = "../data/", testing: bool = False, random_seed: int = 42) -> None:
        self.data_folder = data_path
        self._data_locations = None
        self.random_seed = random_seed

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

        self._data_locations = {
            "main": os.path.join(self.data_path, "all_trials_25_hz_stacked_null_str_filled.csv"),
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
        feature_dtype = np.float64 if model_type[0] == "Complete" else np.float32 # TODO: Make this a user input parameter
        gloc_data_all_features_numpy = gloc_data[features["All"]].to_numpy(dtype=feature_dtype)
        gloc_labels_numpy = gloc_labels

        del gloc_data, gloc_labels
        return gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata

    def _convert_to_unique_ordered_integers(self, strings: np.ndarray) -> np.ndarray:
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

    def _remove_constant_columns(self, x_feature_matrix_noNaN: np.ndarray, all_features: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Remove columns with zero variance (constant across all rows)."""
        # Find all constant columns
        constant_columns = np.all(x_feature_matrix_noNaN == x_feature_matrix_noNaN[0,:], axis = 0)

        # Remove all constant columns from data frame
        x_feature_matrix_noNaN = x_feature_matrix_noNaN[:, ~constant_columns]

        all_features = [all_features[i] for i in range(len(all_features)) if ~constant_columns[i]]

        return x_feature_matrix_noNaN, all_features

    def _process_NaN(
            self,
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

class TraditionalDataPipeline(DataPipeline):
    """Legacy-compatible data pipeline for temporal/traditional GLOC modeling."""

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

    _CLASSIFIER_HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
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

    def __init__(self, data_path: str = "../data/", testing: bool = False, random_seed: int = 42, use_reduced_dataset: bool = False) -> None:
        self.data_path = data_path
        self._data_locations = None
        self.testing = testing
        self.random_seed = random_seed
        self.use_reduced_dataset = use_reduced_dataset

    def get_data(
            self,
            backstep,
            data_rate,
            classifier_type,
            model_type,
            select_features,
            remove_NaN_trials,
            offset,
            time_start,
            subject_to_analyze,
            trial_to_analyze,
            analysis_type,
    ):
        """Return data for a given set of parameters."""
        (
            baseline_window,
            window_size,
            stride,
            _feature_reduction_type,
            baseline_methods_to_use,
            _imbalance_type,
            impute_type,
            n_neighbors,
        ) = self._get_hyperparameters_by_classifier(classifier_type)
        feature_groups_to_analyze, baseline_methods_to_use = self._get_feature_groups_and_baseline_methods(model_type, baseline_methods_to_use)
        model_kind, label_mode = model_type
        is_complete_explicit = model_kind == "Complete" and label_mode == "Explicit"

        ############################################# LOAD AND PROCESS DATA #############################################
        # "Grabs GLOC event and predictor data, depending on 'analysis_type' and 'feature_groups_to_analyze"
        file_paths = self._get_data_locations()

        # Load data and slot in GOR EEG features from xlsx files, then filter to specified analysis type and process features based on specified feature groups
        gloc_data = self._load_data(file_paths)
        gloc_data = self._filter_data_by_analysis_type(analysis_type, gloc_data, subject_to_analyze, trial_to_analyze)
        gloc_data, features = self._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, model_type, file_paths)
        
        # Create GLOC categorical vector
        gloc_labels = self._label_gloc_events(gloc_data)

        if is_complete_explicit:
            # Impute raw (using mean) value of the missing channels for each AFE condition
            gloc_data = self._eeg_specific_imputation(gloc_data, features)

        pipeline_features = {name: feature_names.copy() for name, feature_names in features.items()}
        if is_complete_explicit:
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
            # Legacy parity is model-path dependent:
            # - Complete Explicit follows a float64/pandas-default memory path.
            # - Other paths match float32 C-order input.
            if is_complete_explicit:
                impute_input = gloc_data[pipeline_features["All"]].to_numpy(dtype=np.float64)
            else:
                impute_input = np.asarray(
                    gloc_data[pipeline_features["All"]].to_numpy(dtype=np.float32),
                    dtype=np.float32,
                    order="C",
                )
            imputed_features = self._faster_knn_impute(impute_input, k=n_neighbors)
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
        if is_complete_explicit:
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

        # Backward compatibility: legacy feature lists may still reference "condition".
        translated_select_features = [
            feature_name.replace("condition", "AFE_indicator") for feature_name in select_features
        ]

        # Select columns by index to avoid an expensive full DataFrame materialization.
        feature_index = {feature_name: i for i, feature_name in enumerate(pipeline_features["All"])}
        selected_indices = [feature_index[feature_name] for feature_name in translated_select_features]
        gloc_data_all_features_numpy = gloc_data_all_features_numpy[:, selected_indices]

        gloc_data_all_features_numpy, select_features = self._remove_constant_columns(gloc_data_all_features_numpy, translated_select_features)

        ################################################ NaN Processing ################################################
        gloc_labels_numpy, gloc_data_all_features_numpy, pipeline_features["All"], _removed_ind = self._process_NaN_temporal(
            gloc_labels_numpy,
            gloc_data_all_features_numpy,
            select_features,
        )

        ################################################ Get Outputs Ready ############################################
        gloc_data_all_features_numpy, gloc_labels_numpy = self._ready_outputs(gloc_data_all_features_numpy, gloc_labels_numpy)

        return gloc_data_all_features_numpy, gloc_labels_numpy
    
    def _get_hyperparameters_by_classifier(self, classifier_type):
        """Return hyperparameters for a given classifier type."""
        params = self._CLASSIFIER_HYPERPARAMETERS.get(classifier_type)
        if params is None:
            available = ", ".join(sorted(self._CLASSIFIER_HYPERPARAMETERS.keys()))
            raise ValueError(f"Unknown classifier_type '{classifier_type}'. Available: {available}")

        return (
            params["baseline_window"],
            params["window_size"],
            params["stride"],
            params["feature_reduction_type"],
            params["baseline_methods_to_use"],
            params["imbalance_type"],
            params["impute_type"],
            params["n_neighbors"],
        )
    
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
        # Legacy parity depends on model path:
        # - Complete Explicit retains float64 through imputation/baseline feature generation.
        # - Other paths use float32.
        model_kind, label_mode = model_type
        feature_dtype = np.float64 if (model_kind == "Complete" and label_mode == "Explicit") else np.float32 # TODO: Make this a user input parameter
        gloc_data_all_features_numpy = np.asarray(
            gloc_data[features["All"]].to_numpy(dtype=feature_dtype),
            dtype=feature_dtype,
            order="C",
        )
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
        
        rng = faiss.RandomGenerator(self.random_seed)
        index.hnsw.rng = rng
        
        index.add(X_temp.astype(np.float32))

        # Find k nearest neighbors
        distances, indices = index.search(X_temp.astype(np.float32), k + 1)

        # Impute missing values (skip self, which is always the first neighbor)
        for i in range(X.shape[0]):
            missing_cols = np.flatnonzero(mask[i])
            if missing_cols.size == 0:
                continue
            neighbors = indices[i, 1:] # skip self
            for j in missing_cols:
                neighbor_values = X_temp[neighbors, j]
                X_imputed[i, j] = np.nanmean(neighbor_values)

        return X_imputed
    
    def _y_prediction_offset(self, y, backstep, data_rate, trial_set):
        """
        Shifts GLOC flags to the left by 'backstep' frames.
        Truncates the beginning and pads the end with zeros.
        """
        y = np.asarray(y).copy()
        offset = int(backstep * data_rate) # the actual number of indices to offset.
        # if backstep is given as seconds and data rate as hz
        # the result would be something like 5 seconds back * 25hz so 125 indices shift

        # y is passed as every single subject and trial in one so we have to break out the indices.

        unique_trials = np.unique(trial_set) # finds the unique trials within the set. Gives an array of name of each unique

        if offset <= 0:
            return y

        for trial in unique_trials:
            trial_indices = np.nonzero(trial_set == trial)[0]
            current_y = y[trial_indices]

            if not np.any(current_y):
                continue

            if offset >= current_y.shape[0]:
                y[trial_indices] = np.zeros_like(current_y)
                continue

            y_shifted = current_y[offset:]
            y[trial_indices] = np.concatenate([y_shifted, np.zeros(offset, dtype=current_y.dtype)])[: current_y.shape[0]]

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
        constant_columns = np.all(x_feature_matrix == x_feature_matrix[0, :], axis=0)
        keep_columns = ~constant_columns

        # Remove all constant columns from feature matrix and names
        x_feature_matrix = x_feature_matrix[:, keep_columns]
        select_features = [feature for feature, keep in zip(select_features, keep_columns) if keep]

        return x_feature_matrix, select_features
    
    def _process_NaN_temporal(self, y_gloc_labels, x_feature_matrix, all_features):
        """
        This is a temporary function for removing all rows with NaN values. This can be replaced by
        another method in the future, but is necessary for feeding into ML Classifiers.
        """
        # Find & remove columns if they have all NaN values
        nan_test = np.isnan(x_feature_matrix)
        index_column_all_NaN = np.all(nan_test, axis=0)
        keep_columns = ~index_column_all_NaN
        x_feature_matrix_noNaN_cols = x_feature_matrix[:, keep_columns]

        # Adjust all_features to only include columns that don't have all NaN
        all_features = [feature_name for feature_name, keep in zip(all_features, keep_columns) if keep]

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