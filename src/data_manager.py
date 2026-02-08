from tokenize import group
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, KFold
import warnings

from features import *

class DataManager:
    def __init__(self, data_path = "../data/"):
        self.data_path = data_path

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
        if model_type[0] != "Complete":
            gloc_data = gloc_data

    def _get_feature_groups_and_baseline_methods(self, model_type):
        FEATURE_GROUPS_BY_MODEL_TYPE = {
            ("noAFE", "Explicit"): {"ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE", "G", "processedEEG", "demographics", "strain"},
            ("noAFE", "Implicit"): {"ECG", "BR", "temp", "eyetracking", "rawEEG"},
            ("Complete", "Explicit"): {"ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE", "G", "processedEEG", "demographics", "strain"},
            ("Complete", "Implicit"): {"ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE"} # AFE is removed downstream
        }
        BASELINING_CHARACTERISTICS_BY_MODEL_TYPE = {
            "noAFE": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
            "Complete": ["v0", "v1", "v2", "v5", "v6"]
        }

        feature_groups_to_analyze = FEATURE_GROUPS_BY_MODEL_TYPE[model_type]
        baseline_methods_to_use = BASELINING_CHARACTERISTICS_BY_MODEL_TYPE[model_type[0]]

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

        return file_paths
    
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
        # gloc_data = self._process_EEG_GOR(file_paths["eeg_list"], gloc_data)

        # Adjust AFE condition column always
        gloc_data["condition"] = gloc_data["condition"].map({"N": 0, "AFE": 1})

        # Convert float64 to float32 to save memory, and copy to defragment the DataFrame
        gloc_data = gloc_data.astype({col: "float32" for col in gloc_data.select_dtypes(include = "float64").columns}).copy()
        
        # Extracting subject and trial into separate columns
        trial_ids = gloc_data["trial_id"].to_numpy().astype("str")
        trial_ids = np.array(np.char.split(trial_ids, "-").tolist())
        gloc_data["subject"] = trial_ids[:, 0]
        gloc_data["trial"] = trial_ids[:, 1]

        # Decouple from original dataframe to prevent unwanted modifications later on
        return gloc_data.copy()

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
                features["Phys"].extend(feature_names)

            if group_name in GROUPS_OF_FEATURE_GROUPS["ECG"]:
                features["ECG"].extend(feature_names)

            if group_name in GROUPS_OF_FEATURE_GROUPS["EEG"]:
                features["EEG"].extend(feature_names)

            features["All"].extend(feature_names)

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
        # Initialize EEG dictionaries
        eeg_dict_delta = dict()
        eeg_dict_theta = dict()
        eeg_dict_alpha = dict()
        eeg_dict_beta = dict()

        # Iterate through all EEG files
        for file in range(len(list_of_eeg_data_files)):

            # Define current file
            current_file = list_of_eeg_data_files[file]

            # Grab corresponding trial based on file name
            # corresponding_trial = current_file[47] + current_file[48] + '-0' + current_file[52]
            corresponding_trial = current_file[-31] + current_file[-30] + '-0' + current_file[-26]

            # Define data frame for delta, theta, alpha, and beta bands
            df_delta = pd.read_excel(current_file, sheet_name='delta')
            df_theta = pd.read_excel(current_file, sheet_name='theta')
            df_alpha = pd.read_excel(current_file, sheet_name='alpha')
            df_beta = pd.read_excel(current_file, sheet_name='beta')

            # Remove time column from all spreadsheets that were read in
            df_delta = df_delta.iloc[:, :-1]
            df_theta = df_theta.iloc[:, :-1]
            df_alpha = df_alpha.iloc[:, :-1]
            df_beta = df_beta.iloc[:, :-1]

            # Add each data frame to dictionary corresponding to the trial
            eeg_dict_delta[corresponding_trial] = df_delta
            eeg_dict_theta[corresponding_trial] = df_theta
            eeg_dict_alpha[corresponding_trial] = df_alpha
            eeg_dict_beta[corresponding_trial] = df_beta

        # For each key in the dictionary, look at gloc_data_reduced for that trial
        all_trial_dictionary = list(eeg_dict_delta.keys())
        for key in range(len(all_trial_dictionary)):

            # Find current trial's data in gloc_data
            current_key = all_trial_dictionary[key]
            current_trial_data = gloc_data[gloc_data['trial_id'] == current_key]

            # Find first instance of 'begin GOR' in event_validated column for current trial
            event_validated_current_trial = np.array(current_trial_data['event_validated'])
            index_begin_GOR = np.argwhere(event_validated_current_trial == 'begin GOR')[0]

            # Find end index of GOR EEG data
            index_end_GOR_eeg = index_begin_GOR + len(eeg_dict_delta[current_key])

            # Iterate through all columns & insert data from Excel file
            column_names = eeg_dict_delta[current_key].columns
            for col in range(len(column_names)):

                # Get current column name
                column_name = column_names[col]

                # Modify column name
                modified_name_delta = column_name + '_delta' + ' - EEG'
                modified_name_theta = column_name + '_theta' + ' - EEG'
                modified_name_alpha = column_name + '_alpha' + ' - EEG'
                modified_name_beta = column_name + '_beta' + ' - EEG'

                # For each dictionary column, insert GOR EEG data in current_trial_data
                current_trial_data.iloc[index_begin_GOR[0]:index_end_GOR_eeg[0], current_trial_data.columns.get_loc(modified_name_delta)] = eeg_dict_delta[current_key][column_name].astype(np.float32)
                current_trial_data.iloc[index_begin_GOR[0]:index_end_GOR_eeg[0], current_trial_data.columns.get_loc(modified_name_theta)] = eeg_dict_theta[current_key][column_name].astype(np.float32)
                current_trial_data.iloc[index_begin_GOR[0]:index_end_GOR_eeg[0], current_trial_data.columns.get_loc(modified_name_alpha)] = eeg_dict_alpha[current_key][column_name].astype(np.float32)
                current_trial_data.iloc[index_begin_GOR[0]:index_end_GOR_eeg[0], current_trial_data.columns.get_loc(modified_name_beta)] = eeg_dict_beta[current_key][column_name].astype(np.float32)

            # Replace previously empty processed EEG data with current_trial_data
            gloc_data[gloc_data['trial_id'] == current_key] = current_trial_data

        return gloc_data

    def _afe_subset(self, gloc_data, gloc_labels):
        """
            Remove any trial that contains AFE condition (condition == 1).
        """
        trial_has_afe = gloc_data.groupby("trial")["condition"].transform("max") # Mark trial as all 1 if any are 1, otherwise 0
        keep_mask = trial_has_afe != 1

        gloc_data = gloc_data.loc[keep_mask].reset_index(drop = True)
        gloc_labels = gloc_labels[keep_mask]

        return gloc_data.copy(), gloc_labels.copy()