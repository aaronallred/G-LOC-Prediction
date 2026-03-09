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

    def faster_knn_impute(self, X, k = 5, M = 32, efSearch = 64):
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
    
    def y_prediction_offset(self, y, backstep, data_rate, trial_set):
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