import pandas as pd
import numpy as np
import os

from features import *

class DataManager:
    FEATURE_GROUPS_BY_MODEL_TYPE = {
        ("noAFE", "Explicit"): {"ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE", "G", "processedEEG", "demographics", "strain"},
        ("noAFE", "Implicit"): {"ECG", "BR", "temp", "eyetracking", "rawEEG"},
        ("Complete", "Explicit"): {"ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE", "G", "processedEEG", "demographics", "strain"},
        ("Complete", "Implicit"): {"ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE"},
    }
    BASELINING_CHARACTERISTICS_BY_MODEL_TYPE = {
        "noAFE": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
        "Complete": ["v0", "v1", "v2", "v5", "v6"],
    }

    def __init__(self, data_path = "../data/"):
        self.data_path = data_path
        self._data_locations = None

    def get_data(
            self,
            model_type,
            num_splits,
            kfold_ID,
            impute_path,
            subject_to_analyze = None,
            trial_to_analyze = None,
            impute_type = 1,
            n_neighbors = 4,
            baseline_window = 32.5,
            datafolder = "../data/",
            analysis_type = 2,
            remove_NaN_trials = True,
            save_impute = True,
            load_impute = True
        ):
        """
            Function Loads Raw data and Prepares the Predictor / Target Sets for Advanced Classifiers

            Parameters:
                model_type (tuple[str])  --> A list of model type characteristics i.e. 'Complete/nonAFE' and 'explicit/implicit'
                num_splits (int)         --> The number of splits of the training and test data set for K-fold CV. Nominally set to 10
                kfold_ID (int)           --> The id of the train / test split. If num split is 10, kfold is [0, 9]
                impute_path (str)        --> The path to save/load the imputed data pickle file
                subject_to_analyze (str) --> If analysis type is 1, the participant number to analyze
                trial_to_analyze (str)   --> If analysis type is 0, the trial number to analyze
                impute_type (int)        --> Sets the imputation method. Since Sequential, use impute type equal to 1 (input raw data)
                n_neighbors (int)        --> Sets the KNN imputation # of neighbors. Since Sequential, use 4 neighbors
                baseline_window (float)  --> Sets the baseline window duration. Since Sequential, use 32.5 s
                datafolder (str)         --> Location of AFRL provided data from the experiment: raw data that is processed
                analysis_type (int)      --> Determines what data to use. 2: all data. 1: one participant (set in function), 0: one trial
                remove_NaN_trials (bool) --> Removes trials that have an all NaN sensor instead of imputing an all NaN array
                save_impute (bool)       --> Dumps data post-impute into a pickle file (this is convenient as imputation has large compute)
                load_impute (bool)       --> Checks if there is a saved impute pickle and loads it if available

            Returns:
                x_train, x_test --> Array of predictors. rows are over time, columns are over predictors
                    * Additional Note: The last column is the trial ID. Needed for the advanced classifier slicing
                y_train, y_test --> An array of binary labels corresponding to GLOC or no GLOC (1 or 0, respectively)
                    * Additional Note: Not shifted by horizon for advanced classifiers (happens in the data loader inside)
                all_features    --> List of all feature names in x_train and x_test
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
        if model_type[0] != "Complete":
            self._eeg_specific_imputation(gloc_data, features)

        ############################################### MISSING DATA HANDLING ###############################################
        """
        Optional handling of raw NaN data, depending on 'remove_NaN_trials' 'impute_type' <= 1
        """
        if remove_NaN_trials:
            nan_proportion_df = self._remove_all_nan_trials(gloc_data, features, gloc_labels)

    def _get_feature_groups_and_baseline_methods(self, model_type):
        feature_groups_to_analyze = self.FEATURE_GROUPS_BY_MODEL_TYPE[model_type]
        baseline_methods_to_use = self.BASELINING_CHARACTERISTICS_BY_MODEL_TYPE[model_type[0]]

        # NOTE:
        # AFE indicator is required for EEG imputation in complete models,
        # but is only included as a predictive feature for explicit models.

        return feature_groups_to_analyze, baseline_methods_to_use

    def _get_data_locations(self):
        """
            Get the file locations for all relevant data files
            
            Parameters:
                datafolder (str) --> location of AFRL provided data from the experiment: raw data that is processed

            Returns:
                dict: A dictionary containing file paths for the data with the following structure:
                - "main": Path to the main data CSV file.
                - "baseline": Path to the baseline data CSV file.
                - "demographic": Path to the demographic data CSV file.
                - "eeg_list": List of paths to individual EEG data files.
                - "baseline_eeg_processed_list": List of paths to baseline EEG processed data files.   
        """
        if self._data_locations is not None:
            return self._data_locations

        # Data CSV
        print("USING REDUCED DATASET!!!!!!!!!!!!!!!")
        main_data_file_path = os.path.join(self.data_path, "all_trials_25_hz_stacked_null_str_filled_reduced.csv")

        # Baseline Data (HR)
        baseline_data_file_path = os.path.join(self.data_path, "ParticipantBaseline.csv")

        # Modified Demographic Data (put in order of participant 1-13, removed excess calculations, and converted from .xlsx to .csv)
        demographic_data_file_path = os.path.join(self.data_path, "GLOC_Effectiveness_Final.csv")

        # Input GOR EEG data from separate files
        list_of_eeg_data_file_paths = [
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_01_DC1_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_01_DC2_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_01_DC3_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_02_DC1_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_02_DC2_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_02_DC3_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_03_DC1_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_03_DC2_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_03_DC3_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_04_DC1_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_04_DC2_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_04_DC3_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_05_DC1_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_05_DC2_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_05_DC3_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_06_DC1_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_06_DC4_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_06_DC6_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_07_DC2_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_07_DC4_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_07_DC6_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_08_DC1_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_08_DC3_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_09_DC2_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_09_DC5_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_09_DC6_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_10_DC2_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_10_DC4_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_10_DC5_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_11_DC1_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_12_DC1_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_12_DC5_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_13_DC1_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_13_DC3_25Hz_EEG_power_wMAR.xlsx"),
            os.path.join(self.data_path, "GLOC_GOR_EEG_data_participants_1-13/GLOC_13_DC6_25Hz_EEG_power_wMAR.xlsx")
        ]

        # Input baseline EEG data from separate files
        list_of_baseline_eeg_processed_file_paths = [
            os.path.join(self.data_path, "GLOC_EEG_baseline_delta_noAFE1.csv"),
            os.path.join(self.data_path, "GLOC_EEG_baseline_theta_noAFE1.csv"),
            os.path.join(self.data_path, "GLOC_EEG_baseline_alpha_noAFE1.csv"),
            os.path.join(self.data_path, "GLOC_EEG_baseline_beta_noAFE1.csv")
        ]

        file_paths = {
            "main": main_data_file_path,
            "baseline": baseline_data_file_path,
            "demographic": demographic_data_file_path,
            "eeg_list": list_of_eeg_data_file_paths,
            "baseline_eeg_processed_list": list_of_baseline_eeg_processed_file_paths
        }

        self._data_locations = file_paths
        return self._data_locations
    
    def _load_data(self, file_paths):
        """Load data from CSV or pickle files. If pickle does not exist, create it from CSV."""

        main_data_pickle_file = file_paths["main"].replace(".csv", ".pkl")

        # Check if pickle exists, if not create it then save it
        if not os.path.isfile(main_data_pickle_file):
            print(f"Pickle file not found at {main_data_pickle_file}. Loading from CSV and creating pickle file.")
            gloc_data = pd.read_csv(file_paths["main"])
            gloc_data.to_pickle(main_data_pickle_file)
        else:
            print(f"Loading data from pickle file at {main_data_pickle_file}.")
            gloc_data = pd.read_pickle(main_data_pickle_file)
        
        # Add GOR and EEG data from other files
        gloc_data = self._process_EEG_GOR(file_paths["eeg_list"], gloc_data)

        # Adjust AFE condition column always
        gloc_data["condition"] = gloc_data["condition"].map({"N": 0, "AFE": 1})

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

    def _filter_data_by_analysis_type(self, analysis_type, gloc_data, subject_to_analyze = None, trial_to_analyze = None):
        """Analyze only section of gloc_data specified using analysis_type"""
        
        if analysis_type == 0: # One Trial / One Subject
            mask = (gloc_data["subject"] == subject_to_analyze) & (gloc_data["trial"] == trial_to_analyze)
        elif analysis_type == 1: # All Trials for One Subject
            mask = (gloc_data["subject"] == subject_to_analyze)
        else: # All Trials for All Subjects
            return gloc_data
        
        return gloc_data[mask].copy()

    def _process_and_get_feature_names(self, gloc_data, feature_groups_to_analyze, model_type, file_names):
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
                print(f"Warning: Feature group '{group_name}' not recognized. Skipping.")
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

    def _label_gloc_events(self, gloc_data):
        """
        This function creates a g-loc label for the data based on the event_validated column. The event
        is labeled as 1 between GLOC and Return to Consciousness.
        """
        event_validated = gloc_data["event_validated"]
        trial_ids = gloc_data["trial_id"]

        # Find first GLOC and first RTC per trial
        gloc_idx = event_validated.eq("GLOC").groupby(trial_ids).idxmax()
        rtc_idx = event_validated.eq("return to consciousness").groupby(trial_ids).idxmax()

        gloc_labels = np.zeros(len(gloc_data), dtype = np.int8)

        # Loop over trials only (much smaller than rows)
        for t in gloc_idx.index:
            start = gloc_idx[t]
            end = rtc_idx[t]
            if start < end:
                gloc_labels[start:end] = 1

        return gloc_labels

    def _process_EEG_GOR(self, list_of_eeg_data_files, gloc_data):
        """
        This function slots in the GOR EEG data for the nonAFE condition based on the list of xlsx files.
        The NaNs in the initial csv are replaced.
        """
        trial_indices_map = gloc_data.groupby("trial_id", sort=False).indices
        event_validated = gloc_data["event_validated"].to_numpy()
        trial_ids = gloc_data["trial_id"].to_numpy()
        begin_mask = event_validated == "begin GOR"
        begin_idx = np.flatnonzero(begin_mask)
        begin_trial_ids = trial_ids[begin_mask]
        begin_idx_map = (
            pd.Series(begin_idx, index=begin_trial_ids)
            .groupby(level=0, sort=False)
            .first()
            .to_dict()
        )

        # Iterate through all EEG files and write directly to gloc_data
        for current_file in list_of_eeg_data_files:

            # Grab corresponding trial based on file name
            # corresponding_trial = current_file[47] + current_file[48] + '-0' + current_file[52]
            corresponding_trial = current_file[-31] + current_file[-30] + '-0' + current_file[-26]

            # Define data frame for delta, theta, alpha, and beta bands
            df_delta = pd.read_excel(current_file, sheet_name="delta")
            df_theta = pd.read_excel(current_file, sheet_name="theta")
            df_alpha = pd.read_excel(current_file, sheet_name="alpha")
            df_beta = pd.read_excel(current_file, sheet_name="beta")

            # Remove time column from all spreadsheets that were read in
            df_delta = df_delta.iloc[:, :-1]
            df_theta = df_theta.iloc[:, :-1]
            df_alpha = df_alpha.iloc[:, :-1]
            df_beta = df_beta.iloc[:, :-1]

            trial_indices = trial_indices_map.get(corresponding_trial)
            if trial_indices is None:
                print(f"Could not find 'begin GOR' for trial {corresponding_trial}")
                continue

            index_begin_GOR = begin_idx_map.get(corresponding_trial)
            if index_begin_GOR is None:
                print(f"Could not find 'begin GOR' for trial {corresponding_trial}")
                continue

            # Find end index of GOR EEG data
            start_pos = np.searchsorted(trial_indices, index_begin_GOR)
            end_pos = start_pos + len(df_delta)
            trial_indexer = trial_indices[start_pos:end_pos]

            # Build column names once
            column_names = df_delta.columns
            cols_delta = [f"{c}_delta - EEG" for c in column_names]
            cols_theta = [f"{c}_theta - EEG" for c in column_names]
            cols_alpha = [f"{c}_alpha - EEG" for c in column_names]
            cols_beta = [f"{c}_beta - EEG" for c in column_names]

            # Convert once
            delta_vals = df_delta.to_numpy(dtype=np.float32)
            theta_vals = df_theta.to_numpy(dtype=np.float32)
            alpha_vals = df_alpha.to_numpy(dtype=np.float32)
            beta_vals = df_beta.to_numpy(dtype=np.float32)

            # Assign in blocks
            gloc_data.loc[trial_indexer, cols_delta] = delta_vals
            gloc_data.loc[trial_indexer, cols_theta] = theta_vals
            gloc_data.loc[trial_indexer, cols_alpha] = alpha_vals
            gloc_data.loc[trial_indexer, cols_beta] = beta_vals

        return gloc_data

    def _afe_subset(self, gloc_data, gloc_labels):
        """
            Remove any trial that contains AFE condition (condition == 1).
        """
        trial_has_afe = gloc_data.groupby("trial")["condition"].transform("max") # Mark trial as all 1 if any are 1, otherwise 0
        keep_mask = trial_has_afe != 1

        gloc_data = gloc_data.loc[keep_mask].reset_index(drop = True)
        gloc_labels = gloc_labels[keep_mask]

        return gloc_data, gloc_labels
    
    def _eeg_specific_imputation(self, gloc_data, features):
        # Compute AFE / NonAFE condition indicator column
        afe_indicator_column = gloc_data["condition"]

        # Impute (using mean) the value of the missing channels for each AFE condition
        self._eeg_condition_impute(gloc_data, features, afe_indicator_column)

        # Rename column for indicating AFE status
        gloc_data.rename(columns = {"condition": "AFE_indicator"}, inplace = True)

    def _eeg_condition_impute(self, gloc_data, features, afe_indicator_column, verbose = True):
        """
            Ensures both AFE (1) and non-AFE (0) conditions have the same feature columns.
            Missing columns are imputed with mean values in gloc_data and reflected in feature arrays.

            Modified the gloc_data DataFrame inplace
        """
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
                    print(f"Imputed {n} values in '{col}' for non-AFE rows")

        if nonafe_only_cols:
            means = gloc_data.loc[nonafe_mask, nonafe_only_cols].mean(skipna = True)
            if verbose:
                missing_counts = gloc_data.loc[afe_mask, nonafe_only_cols].isna().sum()
            gloc_data.loc[afe_mask, nonafe_only_cols] = gloc_data.loc[afe_mask, nonafe_only_cols].fillna(means)

            # Show columns imputed and how many rows were imputed for each column
            if verbose:
                for col, n in missing_counts.items():
                    print(f"Imputed {n} values in '{col}' for AFE rows")

    def _remove_all_nan_trials(self, gloc_data, features, gloc_labels, verbose = True):
        """
            Remove trials where there is at least one data stream that is all NaN
            Also returns a NaN proportionality table that says for each trial, what prop are NaN for each data stream
        """
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
                    print(f"Subject {subject}, Trial {trial}: features entirely NaN → {nan_features}")

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
        print(f"There are {M} trials with all NaNs for at least one feature out of {N} trials. {N - M} trials remaining.")

        return nan_proportion_df