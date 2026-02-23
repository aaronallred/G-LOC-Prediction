import pandas as pd
import numpy as np
import os

from features import *
from sklearn.model_selection import StratifiedGroupKFold
from itertools import islice
import faiss
import pickle
from baseline import BaselineContext, baseline_data, BaselineProcessor

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
    
    # Cache unengineered streams as a frozen set for O(1) lookups
    _UNENGINEERED_STREAMS = frozenset([
        'HR (bpm) - Equivital',
        'ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital',
        'HR_instant - Equivital', 'HR_average - Equivital', 'HR_w_average - Equivital',
        'BR (rpm) - Equivital',
        'Skin Temperature - IR Thermometer (°C) - Equivital',

        'Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii',
        'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii',
        'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii',
        'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii',

        'F1 - EEG', 'Fz - EEG', 'F3 - EEG', 'C3 - EEG', 'C4 - EEG', 'CP1 - EEG', 'CP2 - EEG',
        'T8 - EEG', 'TP9 - EEG', 'TP10 - EEG', 'P7 - EEG', 'P8 - EEG', 'AFz - EEG', 'AF4 - EEG',
        'FT9 - EEG', 'FT10 - EEG', 'FC5 - EEG', 'FC3 - EEG', 'FC1 - EEG', 'FC2 - EEG', 'FC4 - EEG',
        'FC6 - EEG', 'C5 - EEG', 'Cz - EEG', 'CP5 - EEG', 'CP6 - EEG', 'P5 - EEG', 'P3 - EEG',
        'P1 - EEG', 'Pz - EEG', 'P4 - EEG', 'P6 - EEG',

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

        'F1_delta - EEG', 'F1_theta - EEG', 'F1_alpha - EEG', 'F1_beta - EEG',
        'Fz_delta - EEG', 'Fz_theta - EEG', 'Fz_alpha - EEG', 'Fz_beta - EEG',
        'F3_delta - EEG', 'F3_theta - EEG', 'F3_alpha - EEG', 'F3_beta - EEG',
        'C3_delta - EEG', 'C3_theta - EEG', 'C3_alpha - EEG', 'C3_beta - EEG',
        'C4_delta - EEG', 'C4_theta - EEG', 'C4_alpha - EEG', 'C4_beta - EEG',
        'CP1_delta - EEG', 'CP1_theta - EEG', 'CP1_alpha - EEG', 'CP1_beta - EEG',
        'CP2_delta - EEG', 'CP2_theta - EEG', 'CP2_alpha - EEG', 'CP2_beta - EEG',
        'T8_delta - EEG', 'T8_theta - EEG', 'T8_alpha - EEG', 'T8_beta - EEG',
        'TP9_delta - EEG', 'TP9_theta - EEG', 'TP9_alpha - EEG', 'TP9_beta - EEG',
        'TP10_delta - EEG', 'TP10_theta - EEG', 'TP10_alpha - EEG', 'TP10_beta - EEG',
        'P7_delta - EEG', 'P7_theta - EEG', 'P7_alpha - EEG', 'P7_beta - EEG',
        'P8_delta - EEG', 'P8_theta - EEG', 'P8_alpha - EEG', 'P8_beta - EEG',
        'AFz_delta - EEG', 'AFz_theta - EEG', 'AFz_alpha - EEG', 'AFz_beta - EEG',
        'AF4_delta - EEG', 'AF4_theta - EEG', 'AF4_alpha - EEG', 'AF4_beta - EEG',
        'FT9_delta - EEG', 'FT9_theta - EEG', 'FT9_alpha - EEG', 'FT9_beta - EEG',
        'FT10_delta - EEG', 'FT10_theta - EEG', 'FT10_alpha - EEG', 'FT10_beta - EEG',
        'FC5_delta - EEG', 'FC5_theta - EEG', 'FC5_alpha - EEG', 'FC5_beta - EEG',
        'FC3_delta - EEG', 'FC3_theta - EEG', 'FC3_alpha - EEG', 'FC3_beta - EEG',
        'FC1_delta - EEG', 'FC1_theta - EEG', 'FC1_alpha - EEG', 'FC1_beta - EEG',
        'FC2_delta - EEG', 'FC2_theta - EEG', 'FC2_alpha - EEG', 'FC2_beta - EEG',
        'FC4_delta - EEG', 'FC4_theta - EEG', 'FC4_alpha - EEG', 'FC4_beta - EEG',
        'FC6_delta - EEG', 'FC6_theta - EEG', 'FC6_alpha - EEG', 'FC6_beta - EEG',
        'C5_delta - EEG', 'C5_theta - EEG', 'C5_alpha - EEG', 'C5_beta - EEG',
        'Cz_delta - EEG', 'Cz_theta - EEG', 'Cz_alpha - EEG', 'Cz_beta - EEG',
        'CP5_delta - EEG', 'CP5_theta - EEG', 'CP5_alpha - EEG', 'CP5_beta - EEG',
        'CP6_delta - EEG', 'CP6_theta - EEG', 'CP6_alpha - EEG', 'CP6_beta - EEG',
        'P5_delta - EEG', 'P5_theta - EEG', 'P5_alpha - EEG', 'P5_beta - EEG',
        'P3_delta - EEG', 'P3_theta - EEG', 'P3_alpha - EEG', 'P3_beta - EEG',
        'P1_delta - EEG', 'P1_theta - EEG', 'P1_alpha - EEG', 'P1_beta - EEG',
        'Pz_delta - EEG', 'Pz_theta - EEG', 'Pz_alpha - EEG', 'Pz_beta - EEG',
        'P4_delta - EEG', 'P4_theta - EEG', 'P4_alpha - EEG', 'P4_beta - EEG',
        'P6_delta - EEG', 'P6_theta - EEG', 'P6_alpha - EEG', 'P6_beta - EEG'
    ])

    def __init__(self, data_path = "../data/", testing = False, random_seed = 42):
        self.data_path = data_path
        self._data_locations = None
        self.testing = testing
        self.random_seed = random_seed

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
            gloc_data = self._eeg_specific_imputation(gloc_data, features)

        ############################################### MISSING DATA HANDLING ###############################################
        """
        Optional handling of raw NaN data, depending on 'remove_NaN_trials' 'impute_type' <= 1
        """
        if remove_NaN_trials:
            # This also returns a DataFrame with proportion of NaN values for each feature for each trial
            # Also modifies gloc_data and gloc_labels to remove trials with all NaNs in at least one feature
            # Note: DataFrame not used for the pipeline for memory purposes
            gloc_data, gloc_labels, _ = self._remove_all_nan_trials(gloc_data, features, gloc_labels)

        ################################################## REDUCE MEMORY ##################################################
        gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata = self._reduce_memory(gloc_data, model_type, features)

        ################################################## Impute Missing ##################################################
        """
            Imputes data using train / test split within imputation to prevent data leakage
        """
        ### Impute missing row data
        if impute_type == 1:
            gloc_data_all_features_imputed_numpy = self._imput_missing_data(gloc_data_all_features_numpy, gloc_labels_numpy, features, impute_path, num_splits, kfold_ID, n_neighbors, save_impute, load_impute)
        else:
            gloc_data_all_features_imputed_numpy = gloc_data_all_features_numpy

        ################################################## BASELINE DATA ##################################################
        """
            Baselines pre-feature data based on 'baseline_methods_to_use'
        """
        combined_baseline, combined_baseline_names = self._get_combined_baseline(gloc_data_all_features_imputed_numpy, experiment_metadata, baseline_window, baseline_methods_to_use, features, file_paths, model_type)

        ################################################ GENERATE FEATURES ################################################
        """
            Generates unengineered features from baseline data using same naming convention as traditional models
        """
        x_feature_matrix, features["All"] = self._generate_features(baseline_methods_to_use, combined_baseline, combined_baseline_names, experiment_metadata)

        ############################################# FEATURE CLEAN AND PREP ##############################################
        """
            Optional handling of raw NaN data
        """

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

    def _filter_data_by_analysis_type(self, analysis_type, gloc_data, subject_to_analyze = None, trial_to_analyze = None):
        """Analyze only section of gloc_data specified using analysis_type"""
        
        if analysis_type == 0: # One Trial / One Subject
            mask = (gloc_data["subject"] == subject_to_analyze) & (gloc_data["trial"] == trial_to_analyze)
        elif analysis_type == 1: # All Trials for One Subject
            mask = (gloc_data["subject"] == subject_to_analyze)
        else: # All Trials for All Subjects
            return gloc_data
        
        return gloc_data[mask]

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

    def _afe_subset(self, gloc_data, gloc_labels):
        """
            Remove any trial that contains AFE condition (condition == 1).
        """
        trial_has_afe = gloc_data.groupby("trial")["AFE_indicator"].transform("max") # Mark trial as all 1 if any are 1, otherwise 0
        keep_mask = trial_has_afe != 1

        gloc_data = gloc_data.loc[keep_mask].reset_index(drop = True)
        gloc_labels = gloc_labels[keep_mask]

        return gloc_data, gloc_labels
    
    def _eeg_specific_imputation(self, gloc_data, features):
        # Compute AFE / NonAFE condition indicator column
        afe_indicator_column = gloc_data["AFE_indicator"]

        # Impute (using mean) the value of the missing channels for each AFE condition
        self._eeg_condition_impute(gloc_data, features, afe_indicator_column)

        return gloc_data

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

        return gloc_data, gloc_labels, nan_proportion_df
    
    def _reduce_memory(self, gloc_data, gloc_labels, features):
        """"""
        # Grab columns from gloc_data and remove gloc_data_reduced variable from memory
        experiment_metadata = {
            "trial_id": gloc_data["trial_id"].to_numpy(),
            "trial_ints": self._convert_to_unique_ordered_integers(gloc_data["trial_id"].to_numpy()),
            "Time (s)": gloc_data["Time (s)"].to_numpy(dtype = np.float32),
            "event_validated": gloc_data["event_validated"].to_numpy(),
            "subject": gloc_data["subject"].to_numpy(),
            "AFE_indicator": gloc_data["AFE_indicator"].to_numpy(dtype = np.bool_).reshape(-1, 1)
        }
        gloc_data_all_features_numpy = gloc_data[features["All"]].to_numpy(dtype = np.float32)
        gloc_labels_numpy = gloc_labels.astype(np.bool_)

        del gloc_data, gloc_labels

        return gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata

    def _convert_to_unique_ordered_integers(self, strings):
        """
            Convert a list of strings to unique integers in the order they appear.
            For example, ['trial1', 'trial2', 'trial1'] would be converted to [1, 2, 1].
        """
        mapping = {}
        result = []
        current_id = 1
        for s in strings:
            if s not in mapping:
                mapping[s] = current_id
                current_id += 1

            result.append(mapping[s])

        return np.array(result, dtype = np.uint32)
    
    def _impute_missing_data(self, gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata, impute_path, save_impute, load_impute, num_splits, kfold_ID, n_neighbors):
        # Load or compute imputed features
        # NOTE: impute_path is PROVIDED by caller; do not overwrite it.
        if load_impute and os.path.exists(impute_path):
            with open(impute_path, 'rb') as f:
                gloc_data_all_features_imputed_numpy = pickle.load(f)
            print(f"Loaded imputed data from {impute_path}")
        else:
            # Only compute train/test indices when actually imputing
            _, _, _, _, train_indices, test_indices = self._groupedtrial_kfold_split(gloc_data_all_features_numpy, gloc_labels_numpy, num_splits, kfold_ID, experiment_metadata)
            gloc_data_all_features_imputed_numpy = self._faster_knn_impute_train_test(gloc_data_all_features_numpy, train_indices, test_indices, n_neighbors)

            del gloc_data_all_features_numpy # Free memory of original data after imputation

            if save_impute:
                os.makedirs(os.path.dirname(impute_path), exist_ok = True)
                with open(impute_path, 'wb') as f:
                    pickle.dump(gloc_data_all_features_imputed_numpy, f)
                print(f"Saved imputed data to {impute_path}")

        return gloc_data_all_features_imputed_numpy

    def _groupedtrial_kfold_split(self, X, Y, num_splits, kfold_ID, experiment_metadata):
        """
        Split data into training and test sets using stratified group K-fold.
        
        Parameters:
            Y: Labels (array)
            X: Feature data (DataFrame)
            trials: Trial identifiers for grouping
            num_splits: Number of K-fold splits
            kfold_ID: Which fold to use (0 to num_splits-1)
            
        Returns:
            x_train, x_test: Split feature data
            y_train, y_test: Split labels
            train_index, test_index: Indices for the splits
        """
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

    def _faster_knn_impute_train_test(self, X, train_ind, test_ind, k = 5, M = 32, efSearch = 64):
        """
        Impute missing values using FAISS KNN, training on train set only to prevent data leakage.

        Parameters:
            X: Input DataFrame
            train_ind: Training set indices
            test_ind: Test set indices
            k: Number of neighbors for KNN
            M, efSearch: FAISS HNSW graph parameters

        Returns:
            X_imputed: DataFrame with imputed values
        """
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

        # Set a fixed random seed for more reproducibility in HNSW graph construction
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

    def _get_combined_baseline_data(self, gloc_data_all_features_imputed_numpy, experiment_metadata, baseline_window, baseline_methods_to_use, features, file_paths, model_type):        
        # Load baseline data (if needed)
        participant_baseline = pd.read_csv(file_paths["baseline"])
        participant_baseline_rhr = participant_baseline["resting HR [seated]"][:-1]
        participant_baseline_rhr.index = [f"{i:02d}" for i in range(1, 14)]
        
        # Load EEG baseline data (if needed)
        eeg_baseline_data = {}
        for filepath in file_paths["baseline_eeg_processed_list"]:
            df = pd.read_csv(filepath)
            df.index = [f"{i:02d}" for i in range(1, 14)]
            band = filepath.split("_")[-1].split(".")[0]  # Extract band name from filename
            eeg_baseline_data[band] = df
        
        # Organize features
        phys_indices = [i for i, feature in enumerate(features["All"]) if feature in features["Phys"]]
        ecg_indices = [i for i, feature in enumerate(features["All"]) if feature in features["ECG"]]
        eeg_indices = [i for i, feature in enumerate(features["All"]) if feature in features["EEG"]]

        data_by_features = {
            "All": gloc_data_all_features_imputed_numpy,
            "Phys": gloc_data_all_features_imputed_numpy[:, phys_indices],
            "ECG": gloc_data_all_features_imputed_numpy[:, ecg_indices],
            "EEG": gloc_data_all_features_imputed_numpy[:, eeg_indices],
        }
        
        # Create context object
        context = BaselineContext(
            trial_column = experiment_metadata["trial_id"],
            time_column = experiment_metadata["Time (s)"],
            event_validated_column = experiment_metadata["event_validated"],
            subject_column = experiment_metadata["subject"],
            data_by_features = data_by_features,
            features = features,
            baseline_window = baseline_window,
            model_type = model_type,
            participant_baseline_data = participant_baseline_rhr,
            eeg_baseline_data = eeg_baseline_data
        )
        
        combined_baseline, combined_names, baseline_v0, baseline_v0_names = baseline_data(baseline_methods_to_use, context)
        
        # Result:
        # - combined_baseline: Dict[trial_id -> baseline_array (n_samples x n_features*3)]
        # - combined_names: List of feature names (base + derivative + 2nd derivative)
        # - baseline_v0: Reference v0 baseline
        # - baseline_v0_names: Feature names for v0
        
        return combined_baseline, combined_names
        
    def _generate_features(self, baseline_methods_to_use, combined_baseline, combined_baseline_names, experiment_metadata):
        """
        Generate feature matrices from baseline data using only unengineered data streams.
        
        Parameters:
            baseline_methods_to_use (list): Baseline methods applied (e.g., ["v0", "v1", "v2"])
            combined_baseline (dict): Dictionary mapping trial_id -> feature array
            combined_names (list): Feature names from baseline processing
            experiment_metadata (dict): Metadata including trial_id information
            
        Returns:
            x_feature_matrix (np.ndarray): Feature matrix with trial indices appended
            all_features (list): Names of selected features
        """
        # Concatenate trial arrays along first axis (handles variable sample counts per trial)
        trial_ids = list(combined_baseline.keys())
        x_feature_matrix = np.concatenate(
            [combined_baseline[tid] for tid in trial_ids], 
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
        ], dtype = np.uint32)
        
        # Compute trial integers before using them
        trial_ints = self._convert_to_unique_ordered_integers(experiment_metadata["trial_id"])
        
        x_feature_matrix = x_feature_matrix[:, ue_indices]
        x_feature_matrix = np.hstack([
            x_feature_matrix,
            trial_ints.reshape(-1, 1).astype(np.uint32)
        ])
        
        all_features = [combined_baseline_names[i] for i in ue_indices]
        
        return x_feature_matrix, all_features
    
    def _is_baselined_stream(self, feature_name, baseline_suffixes):
        """
        Check if feature name matches pattern stream_suffix for any unengineered stream.
        
        Parameters:
            feature_name (str): Feature name to check
            baseline_suffixes (frozenset): Baseline method names
            
        Returns:
            bool: True if feature matches baselined stream pattern
        """
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

    def _feature_clean_and_prep(self, x_feature_matrix, gloc_labels_numpy, features, experiment_metadata, model_type, impute_type):
        """
        Perform final cleaning and preparation of feature matrix before modeling.
        
        Steps:
        - Remove features with zero variance
        - Standardize features (zero mean, unit variance)
        
        Parameters:
            x_feature_matrix (np.ndarray): Raw feature matrix
            all_features (list): Corresponding feature names
        """
        # Remove constant columns (typically no constant columns)
        x_feature_matrix, features["All"] = self._remove_constant_columns(x_feature_matrix, features["All"])

        # Add back in as 2nd to last column for explicit only
        # (needs to be 2nd to last for advanced pipeline - could be last for traditional)
        model_kind, label_mode = model_type
        if model_kind == "Complete" and label_mode == "Explicit":
            x_feature_matrix = np.hstack([
                x_feature_matrix[:, :-1],
                experiment_metadata["AFE_indicator"].reshape(-1, 1),
                x_feature_matrix[:, -1:]
            ])

        # List-wise deletion or clean any residual NaNs
        if impute_type in (1, 2):
            # Remove rows with NaN
            x_feature_matrix_noNaN, y_gloc_labels_noNaN, all_features, trials_noNaN = self._process_NaN(
                x_feature_matrix,
                gloc_labels_numpy,
                features["All"],
                experiment_metadata["trial_ints"]
            )
        else:
            x_feature_matrix_noNaN, y_gloc_labels_noNaN, trials_noNaN = x_feature_matrix, gloc_labels_numpy, experiment_metadata["trial_ints"]

        return x_feature_matrix_noNaN, y_gloc_labels_noNaN, all_features, trials_noNaN

    def _remove_constant_columns(self, x_feature_matrix_noNaN, all_features):
        """
        This function removes all constant columns before feeding into the ML classifiers.
        """
        # Find all constant columns
        constant_columns = np.all(x_feature_matrix_noNaN == x_feature_matrix_noNaN[0,:], axis = 0)

        # Remove all constant columns from data frame
        x_feature_matrix_noNaN = x_feature_matrix_noNaN[:, ~constant_columns]

        all_features = [all_features[i] for i in range(len(all_features)) if ~constant_columns[i]]

        return x_feature_matrix_noNaN, all_features

    def _process_NaN(self, x_feature_matrix, y_gloc_labels, all_features, trials):
        """
        This is a temporary function for removing all rows with NaN values. This can be replaced by
        another method in the future, but is necessary for feeding into ML Classifiers.
        """
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