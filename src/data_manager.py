import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, KFold
import warnings

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
        # Feature Groups to Analyze and Baselining Characteristics
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

        ############################################# LOAD AND PROCESS DATA #############################################
        file_paths = self._get_data_locations()

        

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
    def _process_csv_by_analysis_type(self, analysis_type, file_paths, subject_to_analyze, trial_to_analyze):
        """
        Processes the CSV data based on the specified analysis type.

        Parameters:
            analysis_type (int): The type of analysis to perform (0: One Trial / One Subject, 1: All Trials for One Subject, 2: All Trials for All Subjects).
            file_paths (dict): A dictionary containing file paths for the data.
            subject_to_analyze (str): The subject identifier to analyze (used for analysis types 0 and 1).
            trial_to_analyze (str): The trial identifier to analyze (used for analysis type 0).

        Returns:
            tuple: A tuple containing processed data and features.
        """
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

    

    def _load_and_process_csv(self, analysis_type, feature_groups_to_analyze, file_paths, model_type, **kwargs):
        """
        This function first checks for a pickle file to import (much quicker than loading csv). If the
        .pkl does not exist, it will create that and open it the next time. Additionally, it creates
        an array of relevant feature columns from the data file.
        """

        ############################################# Data Loading #############################################
        # pickle file name
        main_data_pickle_file = file_paths["main"].replace(".csv", ".pkl")

        # Check if pickle exists, if not create it
        if not os.path.isfile(main_data_pickle_file):
            # Load CSV
            gloc_data = pd.read_csv(file_paths["main"])

            # Save data to pickle file
            gloc_data.to_pickle(main_data_pickle_file)
        else:
            # Load data from pickle file
            gloc_data = pd.read_pickle(main_data_pickle_file)

        # Add GOR and EEG data from other files
        gloc_data = self.process_EEG_GOR(file_paths["eeg"], gloc_data)

        # Convert float64 to float32 to save memory
        gloc_data = gloc_data.astype({col: "float32" for col in gloc_data.select_dtypes(include = "float64").columns})
        
        # Decouple from original dataframe to prevent unwanted modifications later on
        gloc_data = gloc_data.copy()

        ############################################# Data Processing #############################################
        # Extracting subject and trial into separate columns
        trial_ids = gloc_data["trial_id"].to_numpy().astype("str")
        trial_ids = np.array(np.char.split(trial_ids, "-").tolist())
        gloc_data["subject"] = trial_ids[:, 0]
        gloc_data["trial"] = trial_ids[:, 1]

        # Free up memory
        del trial_ids

        # TODO: Remove the kwargs and pass subject and trial directly
        # TODO: Change this to return indices so we don't create another DataFrame yet
        # Analyze only section of gloc_data specified using analysis_type
        if analysis_type == 0: # One Trial / One Subject
            # Find data from subject & trial of interest
            gloc_data_reduced = gloc_data[(gloc_data["subject"] == kwargs["subject_to_analyze"]) & (gloc_data["trial"] == kwargs["trial_to_analyze"])]

        elif analysis_type == 1: # All Trials for One Subject
            # Find data from subject of interest
            gloc_data_reduced = gloc_data[(gloc_data["subject"] == kwargs["subject_to_analyze"])]

        elif analysis_type == 2: # All Trials for All Subjects
            gloc_data_reduced = gloc_data

        #############################################   Features   #############################################
        # Initialize feature names lists
        ecg_features = []
        br_features = []
        temp_features = []
        fnirs_features = []
        eyetracking_features = []
        afe_features = []
        g_features = []
        cognitive_features = []
        raw_eeg_shared_features = []
        raw_eeg_condition_specific = []
        processed_eeg_shared_features = []
        processed_eeg_condition_specific = []
        strain_features = []
        demographics_features = []

        # Include features from feature_groups_to_analyze

        
        
        # Get feature columns
        if 'ECG' in feature_groups_to_analyze:
            ecg_features = ['HR (bpm) - Equivital', 'ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital', 'HR_instant - Equivital','HR_average - Equivital', 'HR_w_average - Equivital']
        else:
            ecg_features = []

        if 'BR' in feature_groups_to_analyze:
            br_features = ['BR (rpm) - Equivital']
        else:
            br_features = []

        if 'temp' in feature_groups_to_analyze:
            temp_features = ['Skin Temperature - IR Thermometer (°C) - Equivital']
        else:
            temp_features = []

        if 'fnirs' in feature_groups_to_analyze:
            fnirs_features = ['HbO2 - fNIRS', 'Hbd - fNIRS']

            ######### Generate Additional fnirs specific features #########
            # HbO2/Hbd
            ox_deox_ratio = gloc_data_reduced['HbO2 - fNIRS'] / gloc_data_reduced['Hbd - fNIRS']
            gloc_data_reduced['HbO2 / Hbd'] = ox_deox_ratio

            # append fnirs_features
            fnirs_features.append('HbO2 / Hbd')

            # output warning message for fnirs
            warnings.warn("Per information from Chris on 01/15/25, FNIRS data was impacted by eye tracking glasses and should not be used.")
        else:
            fnirs_features = []

        if 'eyetracking' in feature_groups_to_analyze:
            eyetracking_features = ['Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii', 'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii',
                'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii', 'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii']

            ######### Generate additional pupil specific features #########
            # Pupil Difference
            pupil_difference = gloc_data_reduced['Pupil diameter left [mm] - Tobii'] - gloc_data_reduced['Pupil diameter right [mm] - Tobii']
            gloc_data_reduced['Pupil Difference [mm]'] = pupil_difference

            # append eyetracking_features
            eyetracking_features.append('Pupil Difference [mm]')

        else:
            eyetracking_features = []

        # Adjust columns of data frame for feature always
        # gloc_data_reduced['condition'].replace('N', 0, inplace=True)
        gloc_data_reduced.replace({'condition': 'N',}, 0, inplace=True)
        # gloc_data_reduced['condition'].replace('AFE', 1, inplace=True)
        gloc_data_reduced.replace({'condition': 'AFE'}, 1, inplace=True)
        if 'AFE' in feature_groups_to_analyze:
            afe_features = ['condition']

        else:
            afe_features = []

        if 'G' in feature_groups_to_analyze and 'explicit' in model_type:
            # Process magnitude Centrifuge column to include 1.2g instead of NaN
            # gloc_data_reduced['magnitude - Centrifuge'].fillna(1.2, inplace=True)
            gloc_data_reduced.fillna({'magnitude - Centrifuge': 1.2}, inplace=True)

            # Grab g feature column
            g_features = ['magnitude - Centrifuge']
        elif 'G' in feature_groups_to_analyze and 'implicit' in model_type:
            # output warning message for implicit vs. explicit models
            warnings.warn("G cannot be used as a feature in implicit models. Feature removed.")

            g_features = []
        else:
            g_features = []

        if 'cognitive' in feature_groups_to_analyze:
            cognitive_features = ['deviation - Cog', 'RespTime - Cog', 'Correct - Cog', 'joystickPosMag - Cog', 'joystickVelMag - Cog'] #, 'tgtposX - Cog', 'tgtposY - Cog']

            # Adjust columns of data frame for feature
            gloc_data_reduced['Correct - Cog'].replace('correct', 1, inplace=True)
            gloc_data_reduced['Correct - Cog'].replace('no response', 0, inplace=True)
            gloc_data_reduced['Correct - Cog'].replace('incorrect', -1, inplace=True)
            gloc_data_reduced['Correct - Cog'].replace('NO VALUE', np.nan, inplace=True)

            # output warning message for fnirs
            warnings.warn(
                "Per information from Chris on 03/12/25, Cognitive data only collected right before and right after ROR.")
            # Note post 3/12 meeting: the target is stationary, while the participant moves the tracker
            # so this metric no longer makes sense
            # ######### Generate additional cognitive task specific features #########
            # # Deviation/Screen Pos
            # deviation_wrt_target_position =  gloc_data_reduced['deviation - Cog'] / np.sqrt(gloc_data_reduced['tgtposX - Cog']**2 +  gloc_data_reduced['tgtposY - Cog']**2)
            # gloc_data_reduced['Deviation wrt Target Position'] = deviation_wrt_target_position
            #
            # # append cognitive features
            # cognitive_features.append('Deviation wrt Target Position')
        else:
            cognitive_features = []

        if 'rawEEG' in feature_groups_to_analyze:
            _, _, _, raw_eeg_shared_features, raw_eeg_afe_only, raw_eeg_nonafe_only = pull_eeg_sets()

            # Pull condition specific EEG streams
            if 'AFE' in model_type:
                raw_eeg_condition_specific = raw_eeg_afe_only
            elif 'noAFE' in model_type:
                raw_eeg_condition_specific = raw_eeg_nonafe_only
            else:
                # Use full dataset
                raw_eeg_condition_specific = raw_eeg_afe_only + raw_eeg_nonafe_only
                raw_eeg_condition_specific = []
        else:
            raw_eeg_shared_features = []
            raw_eeg_condition_specific = []

        if 'processedEEG' in feature_groups_to_analyze:
            processed_eeg_shared_features, processed_eeg_afe_only, processed_eeg_nonafe_only,_,_,_ = pull_eeg_sets()

            # Pull condition specific EEG streams
            if 'AFE' in model_type:
                processed_eeg_condition_specific = processed_eeg_afe_only
            elif 'noAFE' in model_type:
                processed_eeg_condition_specific = processed_eeg_nonafe_only
            else:
                # Use full dataset
                processed_eeg_condition_specific = processed_eeg_afe_only + processed_eeg_nonafe_only
                processed_eeg_condition_specific = []
        else:
            processed_eeg_shared_features = []
            processed_eeg_condition_specific = []

        if 'strain' in feature_groups_to_analyze and 'explicit' in model_type:
            strain_features = []

            # Add missing strain during GOR labels based on GLOC Effectiveness Spreadsheet
            gloc_data_reduced, gloc_trial = process_strain_data(gloc_data_reduced)

            ######### Generate Strain specific features #########
            # Create Strain Vector
            event = gloc_data_reduced['event'].to_numpy()
            event_validated = gloc_data_reduced['event_validated'].to_numpy()
            strain_event = np.zeros(event.shape)

            # Find labeled 'strain' and 'end GOR' markings in the event column
            strain_indices = np.argwhere(event == 'strain during GOR')
            end_GOR_indices = np.argwhere(event_validated == 'end GOR')

            # Determine which trial strain label and end GOR label occur
            trial_strain = gloc_trial[strain_indices[:,0]]
            trial_end_GOR = gloc_trial[end_GOR_indices[:,0]]

            # when strain and eng GOR label occur on the same trial, set chunk from
            # start of strain to end of GOR to 1, otherwise 0. This was implemented because
            # some labels were missed.
            for i in range(trial_strain.shape[0]):
                if trial_end_GOR.str.contains(trial_strain.iloc[i]).any():
                    trial_end_GOR_contains_strain = trial_end_GOR.str.contains(trial_strain.iloc[i])
                    end_gor_trial_index = trial_end_GOR_contains_strain[trial_end_GOR_contains_strain].index.tolist()
                    strain_event[strain_indices[i, 0]:end_gor_trial_index[0]] = 1

            gloc_data_reduced['Strain [0/1]'] = strain_event

            # append strain features
            strain_features.append('Strain [0/1]')
        elif 'strain' in feature_groups_to_analyze and 'implicit' in model_type:
            # output warning message for implicit vs. explicit models
            warnings.warn("Strain cannot be used as a feature in implicit models. Feature removed.")

            strain_features = []
        else:
            strain_features = []

        if 'demographics' in feature_groups_to_analyze and 'explicit' in model_type:
            # Read Demographics Spreadsheet and Append to gloc_data_reduced
            [gloc_data_reduced, demographics_names] = read_and_process_demographics(demographic_data_filename, gloc_data_reduced)
            demographics_features = demographics_names
        elif 'demographics' in feature_groups_to_analyze and 'implicit' in model_type:
            # output warning message for implicit vs. explicit models
            warnings.warn("Demographics cannot be used as a feature in implicit models. Feature removed.")

            demographics_features = []
        else:
            demographics_features = []

        # Combine names of different feature categories for baseline methods
        all_features = (ecg_features + br_features + temp_features + fnirs_features + eyetracking_features + afe_features
                        + g_features + cognitive_features + raw_eeg_shared_features + raw_eeg_condition_specific + processed_eeg_shared_features +
                        processed_eeg_condition_specific + strain_features + demographics_features)
        all_features_phys = (ecg_features + br_features + temp_features + fnirs_features + eyetracking_features +
                            raw_eeg_shared_features + raw_eeg_condition_specific + processed_eeg_shared_features + processed_eeg_condition_specific)
        all_features_ecg = ecg_features
        all_features_eeg = processed_eeg_shared_features + processed_eeg_condition_specific

        # Create matrix of all features for data being analyzed
        features = gloc_data_reduced[all_features].to_numpy(dtype=np.float32)
        features_phys = gloc_data_reduced[all_features_phys].to_numpy(dtype=np.float32)
        features_ecg = gloc_data_reduced[all_features_ecg].to_numpy(dtype=np.float32)
        features_eeg = gloc_data_reduced[all_features_eeg].to_numpy(dtype=np.float32)

        return gloc_data_reduced, features, features_phys, features_ecg, features_eeg, all_features, all_features_phys, all_features_ecg, all_features_eeg

    def process_EEG_GOR(self, list_of_eeg_data_files, gloc_data):
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
                # current_trial_data[modified_name_delta][index_begin_GOR[0]:index_end_GOR_eeg[0]] = eeg_dict_delta[current_key][column_name]
                # current_trial_data.loc[index_begin_GOR[0]:index_end_GOR_eeg[0], modified_name_delta] = eeg_dict_delta[current_key][column_name].astype(np.float32)
                current_trial_data.iloc[index_begin_GOR[0]:index_end_GOR_eeg[0], current_trial_data.columns.get_loc(modified_name_delta)] = eeg_dict_delta[current_key][column_name].astype(np.float32)

                # current_trial_data[modified_name_theta][index_begin_GOR[0]:index_end_GOR_eeg[0]] = eeg_dict_theta[current_key][column_name]
                # current_trial_data.loc[index_begin_GOR[0]:index_end_GOR_eeg[0], modified_name_theta] = eeg_dict_theta[current_key][column_name].astype(np.float32)
                current_trial_data.iloc[index_begin_GOR[0]:index_end_GOR_eeg[0], current_trial_data.columns.get_loc(modified_name_theta)] = eeg_dict_theta[current_key][column_name].astype(np.float32)

                # current_trial_data[modified_name_alpha][index_begin_GOR[0]:index_end_GOR_eeg[0]] = eeg_dict_alpha[current_key][column_name]
                # current_trial_data.loc[index_begin_GOR[0]:index_end_GOR_eeg[0], modified_name_alpha] = eeg_dict_alpha[current_key][column_name].astype(np.float32)
                current_trial_data.iloc[index_begin_GOR[0]:index_end_GOR_eeg[0], current_trial_data.columns.get_loc(modified_name_alpha)] = eeg_dict_alpha[current_key][column_name].astype(np.float32)

                # current_trial_data[modified_name_beta][index_begin_GOR[0]:index_end_GOR_eeg[0]] = eeg_dict_beta[current_key][column_name]
                # current_trial_data.loc[index_begin_GOR[0]:index_end_GOR_eeg[0], modified_name_beta] = eeg_dict_beta[current_key][column_name].astype(np.float32)
                current_trial_data.iloc[index_begin_GOR[0]:index_end_GOR_eeg[0], current_trial_data.columns.get_loc(modified_name_beta)] = eeg_dict_beta[current_key][column_name].astype(np.float32)

            # Replace previously empty processed EEG data with current_trial_data
            gloc_data[gloc_data['trial_id'] == current_key] = current_trial_data

        return gloc_data