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
            analysis_type: int = 2, # TODO: Change to enum
            remove_NaN_trials: bool = True,
            impute_file_name: str = "gloc_data_imputed.pkl",
            save_impute: bool = True,
            load_impute: bool = True,
            verbose: bool = False
        ):
        self.data_folder = data_folder
        self.model = model
        self.model_type = model_type
        self.verbose = verbose

        # Data Processing Hyperparameters
        self.baseline_window = baseline_window
        self.window_size = window_size
        self.stride = stride
        self.feature_reduction_type = feature_reduction_type
        self.feature_groups_to_analyze = None
        self.baseline_methods_to_use = None
        self.imbalance_type = imbalance_type
        self.should_impute = should_impute
        self.impute_type = None
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

        # Internal State
        self.data_locations = None
        self.gloc_data = None
        self.gloc_labels = None
        self.feature_names = None
        self.nan_proportion_df = None
    
    def get_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.gloc_data, self.gloc_labels

    def process_and_get_data(self):
        ################################################### FEATURES SETUP ###################################################
        self._assign_hyperparameters_by_classifier()
        self._assign_feature_groups_and_baseline_methods()

        ############################################# LOAD AND PROCESS DATA ##################################################
        self._assign_data_locations()
        self._load_data()
        self._filter_data_by_analysis_type()
        self._process_and_get_feature_names()
        self._label_gloc_events()

        ############################################# EEG Specific Imputation ################################################
        self._eeg_condition_impute()

        ################################################# AFE Subsetting #####################################################
        self._afe_subset()

        ############################################### MISSING DATA HANDLING ###############################################
        self._remove_all_nan_trials()

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