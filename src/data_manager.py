import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, KFold

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
                model_type (List[str])   --> A list of model type characteristics i.e. 'Complete/nonAFE' and 'explicit/implicit'
                num_splits (int)         --> The number of splits of the training and test data set for K-fold CV. Nominally set to 10
                kfold_ID (int)           --> The id of the train / test split. If num split is 10, kfold is [0, 9]
                impute_path (str)        --> The path to save/load the imputed data pickle file
                subject_to_analyze (str) --> If analysis type is 1, the participant number to analyze
                trial_to_analyze (str)   --> If analysis type is 0, the trial number to analyze
                impute_type (int)        --> Sets the imputation method. Since Sequential, use impute type equal to 1 (input raw data)
                n_neighbors (int)        --> Sets the KNN imputation # of neighbors. Since Sequential, use 4 neighbors
                baseline_window (float)  --> Sets the baseline window duration. Since Sequential, use 32.5 s
                datafolder (str)         --> location of AFRL provided data from the experiment: raw data that is processed
                analysis_type (int)      --> Determines what data to use. 2: all data. 1: one participant (set in function), 0: one trial
                remove_NaN_trials (bool) --> removes trials that have an all NaN sensor instead of imputing an all NaN array
                save_impute (bool)       --> dumps data post-impute into a pickle file (this is convenient as imputation has large compute)
                load_impute (bool)       --> checks if there is a saved impute pickle and loads it if available

            Returns:
                x_train, x_test --> Array of predictors. rows are over time, columns are over predictors
                    *Additional Note: The last column is the trial ID. Needed for the advanced classifier slicing
                y_train, y_test --> An array of binary labels corresponding to GLOC or no GLOC (1 or 0, respectively)
                    *Additional Note: Not shifted by horizon for advanced classifiers (happens in the data loader inside)
                all_features    --> List of all feature names in x_train and x_test
        """
        ################################################### FEATURES SETUP ###################################################
        # Feature Groups to Analyze
        IMPLICIT_FEATURE_GROUPS = {"ECG", "BR", "temp", "eyetracking", "rawEEG"} # All physiological signals
        EXPLICIT_FEATURE_GROUPS = IMPLICIT_FEATURE_GROUPS.union({"AFE", "G", "processedEEG", "demographics", "strain"}) # All physiological and participant info
        COMPLETE_FEATURE_GROUPS = {"AFE"} # Include nonAFE and AFE trials
        
        # TODO: Change model type to an ENUM or similar structure for clarity
        feature_groups_to_analyze = set()
        if model_type[0] == "Complete":
            feature_groups_to_analyze = COMPLETE_FEATURE_GROUPS
        # noAFE restriction is imposed later when filtering out rows

        if model_type[1] == "Implicit":
            feature_groups_to_analyze = feature_groups_to_analyze.union(IMPLICIT_FEATURE_GROUPS)
        elif model_type[1] == "Explicit":
            feature_groups_to_analyze = feature_groups_to_analyze.union(EXPLICIT_FEATURE_GROUPS)

        # NOTE:
        # AFE indicator is required for EEG imputation in complete models,
        # but is only included as a predictive feature for explicit models.

        # Baselining Characteristics
        if model_type[0] == "noAFE":
            baseline_methods_to_use = ["v0", "v1", "v2", "v5", "v6", "v7", "v8"]
        else:
            baseline_methods_to_use = ["v0", "v1", "v2", "v5", "v6"]



        ############################################# LOAD AND PROCESS DATA #############################################
        file_paths = self.get_data_locations()

        

    def get_data_locations(self):
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
        main_data_file_path = os.path.join(self.data_path, "all_trials_25_hz_stacked_null_str_filled.csv")

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
    
# TODO: Change analysis type to an ENUM
def process_csv_by_analysis_type(analysis_type, filename, feature_groups_to_analyze, demographic_data_filename,
                                   model_type, list_of_eeg_data_files, trial_to_analyze, subject_to_analyze):
    if analysis_type == 0:  # One Trial / One Subject
        (gloc_data_reduced, features, features_phys, features_ecg, features_eeg, all_features, all_features_phys,
         all_features_ecg, all_features_eeg) = (load_and_process_csv(filename, analysis_type, feature_groups_to_analyze,
                                                                     demographic_data_filename, model_type,
                                                                     list_of_eeg_data_files,
                                                                     trial_to_analyze=trial_to_analyze,
                                                                     subject_to_analyze=subject_to_analyze))

    elif analysis_type == 1:  # All Trials for One Subject
        (gloc_data_reduced, features, features_phys, features_ecg, features_eeg, all_features, all_features_phys,
         all_features_ecg, all_features_eeg) = (load_and_process_csv(filename, analysis_type, feature_groups_to_analyze,
                                                                     demographic_data_filename, model_type,
                                                                     list_of_eeg_data_files,
                                                                     subject_to_analyze=subject_to_analyze))

    else:  # All Trials for All Subjects
        (gloc_data_reduced, features, features_phys, features_ecg, features_eeg, all_features, all_features_phys,
         all_features_ecg, all_features_eeg) = (load_and_process_csv(filename, analysis_type, feature_groups_to_analyze,
                                                                     demographic_data_filename, model_type,
                                                                     list_of_eeg_data_files))

    return (gloc_data_reduced, features, features_phys, features_ecg, features_eeg, all_features, all_features_phys,
            all_features_ecg, all_features_eeg)