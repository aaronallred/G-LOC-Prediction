import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import warnings

from data_manager import DataManager
from features import RawEEGGroup, ProcessedEEGGroup

class TestDataManager:
    def test_get_feature_groups(self):
        IMPLICIT_FEATURE_GROUPS = {"ECG", "BR", "temp", "eyetracking", "rawEEG"}
        EXPLICIT_FEATURE_GROUPS = IMPLICIT_FEATURE_GROUPS.union({"AFE", "G", "processedEEG", "demographics", "strain"})
        COMPLETE_FEATURE_GROUPS = {"AFE"}

        # Testing Complete and Explicit Feature Groups
        model_type = ("Complete", "Explicit")

        feature_groups_to_analyze = set()
        if model_type[0] == "Complete":
            feature_groups_to_analyze = COMPLETE_FEATURE_GROUPS

        if model_type[1] == "Implicit":
            feature_groups_to_analyze = feature_groups_to_analyze.union(IMPLICIT_FEATURE_GROUPS)
        elif model_type[1] == "Explicit":
            feature_groups_to_analyze = feature_groups_to_analyze.union(EXPLICIT_FEATURE_GROUPS)

        true_feature_groups = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G', 'rawEEG', 'processedEEG', 'demographics', 'strain']
        assert feature_groups_to_analyze == set(true_feature_groups)

        # Testing Complete and Implicit Feature Groups
        model_type = ("Complete", "Implicit")

        feature_groups_to_analyze = set()
        if model_type[0] == "Complete":
            feature_groups_to_analyze = COMPLETE_FEATURE_GROUPS

        if model_type[1] == "Implicit":
            feature_groups_to_analyze = feature_groups_to_analyze.union(IMPLICIT_FEATURE_GROUPS)
        elif model_type[1] == "Explicit":
            feature_groups_to_analyze = feature_groups_to_analyze.union(EXPLICIT_FEATURE_GROUPS)

        true_feature_groups = ['ECG', 'BR', 'temp', 'eyetracking', 'rawEEG', 'AFE']
        assert feature_groups_to_analyze == set(true_feature_groups)

        # Testing Non-AFE and Explicit Feature Groups
        model_type = ("Non-AFE", "Explicit")

        feature_groups_to_analyze = set()
        if model_type[0] == "Complete":
            feature_groups_to_analyze = COMPLETE_FEATURE_GROUPS

        if model_type[1] == "Implicit":
            feature_groups_to_analyze = feature_groups_to_analyze.union(IMPLICIT_FEATURE_GROUPS)
        elif model_type[1] == "Explicit":
            feature_groups_to_analyze = feature_groups_to_analyze.union(EXPLICIT_FEATURE_GROUPS)

        true_feature_groups = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G', 'rawEEG', 'processedEEG', 'demographics', 'strain']
        assert feature_groups_to_analyze == set(true_feature_groups)

        # Testing Non-AFE and Implicit Feature Groups
        model_type = ("Non-AFE", "Implicit")

        feature_groups_to_analyze = set()
        if model_type[0] == "Complete":
            feature_groups_to_analyze = COMPLETE_FEATURE_GROUPS

        if model_type[1] == "Implicit":
            feature_groups_to_analyze = feature_groups_to_analyze.union(IMPLICIT_FEATURE_GROUPS)
        elif model_type[1] == "Explicit":
            feature_groups_to_analyze = feature_groups_to_analyze.union(EXPLICIT_FEATURE_GROUPS)

        true_feature_groups = ['ECG', 'BR', 'temp', 'eyetracking', 'rawEEG']
        assert feature_groups_to_analyze == set(true_feature_groups)

    def test_getting_feature_names_complete_explicit(self):
        manager = DataManager()
        file_paths = manager._get_data_locations()
        gloc_data = manager._load_data(file_paths)
        actual_gloc_data = manager._load_data(file_paths)

        feature_groups_to_analyze = {"ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE", "G", "processedEEG", "demographics", "strain"}

        gloc_data, features = manager._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, ("Complete", "Explicit"), file_paths)

        def get_actual_features(gloc_data_reduced, feature_groups_to_analyze, demographic_data_filename, model_type):
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
                    # raw_eeg_condition_specific = []
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
                    # processed_eeg_condition_specific = []
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

        def pull_eeg_sets():
            # list of shared eeg channels
            raw_eeg_shared_features = ['Fz - EEG', 'F3 - EEG', 'C3 - EEG', 'C4 - EEG',
                                        'CP1 - EEG', 'CP2 - EEG', 'T8 - EEG', 'TP9 - EEG', 'TP10 - EEG',
                                        'P7 - EEG', 'P8 - EEG']

            processed_eeg_shared_features = ['Fz_delta - EEG', 'Fz_theta - EEG', 'Fz_alpha - EEG', 'Fz_beta - EEG',
                                            'F3_delta - EEG', 'F3_theta - EEG', 'F3_alpha - EEG', 'F3_beta - EEG',
                                            'C3_delta - EEG', 'C3_theta - EEG', 'C3_alpha - EEG', 'C3_beta - EEG',
                                            'C4_delta - EEG', 'C4_theta - EEG', 'C4_alpha - EEG', 'C4_beta - EEG',
                                            'CP1_delta - EEG', 'CP1_theta - EEG', 'CP1_alpha - EEG', 'CP1_beta - EEG',
                                            'CP2_delta - EEG', 'CP2_theta - EEG', 'CP2_alpha - EEG', 'CP2_beta - EEG',
                                            'T8_delta - EEG', 'T8_theta - EEG', 'T8_alpha - EEG', 'T8_beta - EEG',
                                            'TP9_delta - EEG', 'TP9_theta - EEG', 'TP9_alpha - EEG', 'TP9_beta - EEG',
                                            'TP10_delta - EEG', 'TP10_theta - EEG', 'TP10_alpha - EEG', 'TP10_beta - EEG',
                                            'P7_delta - EEG', 'P7_theta - EEG', 'P7_alpha - EEG', 'P7_beta - EEG',
                                            'P8_delta - EEG', 'P8_theta - EEG', 'P8_alpha - EEG', 'P8_beta - EEG']

            # list of AFE only eeg channels
            raw_eeg_afe_only = ['F4 - EEG', 'T7 - EEG', 'O1 - EEG', 'O2 - EEG']

            processed_eeg_afe_only =['F4_delta - EEG', 'F4_theta - EEG', 'F4_alpha - EEG', 'F4_beta - EEG',
                                                        'T7_delta - EEG', 'T7_theta - EEG', 'T7_alpha - EEG', 'T7_beta - EEG',
                                                        'O1_delta - EEG', 'O1_theta - EEG', 'O1_alpha - EEG', 'O1_beta - EEG',
                                                        'O2_delta - EEG', 'O2_theta - EEG', 'O2_alpha - EEG', 'O2_beta - EEG']
            # list of Non-AFE only eeg channels
            raw_eeg_nonafe_only = ['F1 - EEG', 'AFz - EEG', 'AF4 - EEG', 'FT9 - EEG', 'FT10 - EEG', 'FC5 - EEG',
                                            'FC3 - EEG', 'FC1 - EEG', 'FC2 - EEG', 'FC4 - EEG', 'FC6 - EEG', 'C5 - EEG',
                                            'Cz - EEG', 'CP5 - EEG', 'CP6 - EEG', 'P5 - EEG', 'P3 - EEG', 'P1 - EEG',
                                            'Pz - EEG', 'P4 - EEG', 'P6 - EEG']

            processed_eeg_nonafe_only =['F1_delta - EEG', 'F1_theta - EEG', 'F1_alpha - EEG', 'F1_beta - EEG',
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
                                                        'CP6_delta - EEG', 'CP6_theta - EEG', 'CP6_alpha - EEG','CP6_beta - EEG',
                                                        'P5_delta - EEG', 'P5_theta - EEG', 'P5_alpha - EEG', 'P5_beta - EEG',
                                                        'P3_delta - EEG', 'P3_theta - EEG', 'P3_alpha - EEG', 'P3_beta - EEG',
                                                        'P1_delta - EEG', 'P1_theta - EEG', 'P1_alpha - EEG', 'P1_beta - EEG',
                                                        'Pz_delta - EEG', 'Pz_theta - EEG', 'Pz_alpha - EEG', 'Pz_beta - EEG',
                                                        'P4_delta - EEG', 'P4_theta - EEG', 'P4_alpha - EEG', 'P4_beta - EEG',
                                                        'P6_delta - EEG', 'P6_theta - EEG', 'P6_alpha - EEG', 'P6_beta - EEG']

            return (processed_eeg_shared_features, processed_eeg_afe_only, processed_eeg_nonafe_only,
                    raw_eeg_shared_features, raw_eeg_afe_only, raw_eeg_nonafe_only)
        
        def read_and_process_demographics(demographic_data_filename, gloc_data_reduced):
            """
            This function imports the demographics spreadsheet and appends the demographics
            to the gloc_data_reduced variable. Each column represents one of the demographics
            variables below. The column is filled by determining the participant for a specific
            trial and filling the entire trial with that participants demographic data.
            """

            # Import demographics spreadsheet
            demographics = pd.read_csv(demographic_data_filename)
            demographics = demographics.astype({col: 'float32' for col in demographics.select_dtypes(include='float64').columns})
            demographics = demographics.copy()

            # Grab variables of interest
            participant_index = demographics['GLOC ID']                                                         # Corresponds to subject 1-13
            participant_gender = pd.Series(demographics['Gender code [1/0]'])        # 0 = Female, 1 = Male
            participant_age = pd.Series(demographics['Age [yr]'])
            participant_height = pd.Series(demographics['height [m]'])
            participant_weight = pd.Series(demographics['weight (kg)'])
            participant_BMI = pd.Series(demographics['BMI [kg/m^2]'])
            participant_blood_volume = pd.Series(demographics['Blood Volume [L]'])   # Based on Nadler's approximation
            # participant_PWV = pd.Series(demographics['Avg PWV'])                   # Pulse Wave Velocity
                # (no val for participant 4)
            # participant_PTT = demographics['Avg PTT [ms]'])                        # Pulse Transit Time
                # (no val for participant 4)
            participant_SBP_seated = pd.Series(demographics['Resting SBP (seat)'])   # Systolic Blood Pressure
            participant_SBP_stand = pd.Series(demographics['resting SBP (stand)'])
            participant_SBP_exercise = pd.Series(demographics['SBP after squat'])
            participant_DBP_seated = pd.Series(demographics['Resting DBP (seat)'] )  # Diastolic Blood Pressure
            participant_DBP_stand = pd.Series(demographics['resting DBP (stand)'])
            participant_DBP_exercise = pd.Series(demographics['DBP after squat'])
            participant_MAP_seated = pd.Series(demographics['Resting MAP'])          # Mean Arterial Pressure
            participant_MAP_stand = pd.Series(demographics['Resting MAP (stand'])
            participant_MAP_exercise = pd.Series(demographics['Post-Squat MAP'])
            participant_HR_seated = pd.Series(demographics['resting HR [seated]'])
            participant_HR_stand = pd.Series(demographics['resting HR (stand)'])
            participant_HR_exercise = pd.Series(demographics['HR after squat'])
            participant_max_leg_strength = pd.Series(demographics['Max (N)'])        # Max Leg Strength
            participant_largest_leg_circumference = pd.Series(demographics['largest leg circ. [cm]'])
            participant_lower_leg_volume = pd.Series(demographics['lower leg volume [mL]'])
            participant_skinfolds_chest_avg = pd.Series(demographics['chest avg'])   # Skin Folds to Approximate Body Fat %
            participant_skinfolds_abd_avg = pd.Series(demographics['abd avg'])
            participant_skinfolds_thigh_avg = pd.Series(demographics['thigh avg'])
            participant_skinfolds_midax_avg = pd.Series(demographics['midax avg'])
            participant_skinfolds_subscap_avg = pd.Series(demographics['subscap avg'])
            participant_skinfolds_tri_avg = pd.Series(demographics['tri avg'])
            participant_skinfolds_supra_avg = pd.Series(demographics['supra avg'])
            participant_skinfolds_sum = pd.Series(demographics['sum'])
            participant_percent_fat = pd.Series(demographics['% fat'])
            participant_leg_length = pd.Series(demographics['leg avg'])
            participant_arm_length = pd.Series(demographics['arm avg'])
            participant_midline_neck_length = pd.Series(demographics['neck (MNL) avg'])
            participant_lateral_neck_length = pd.Series(demographics['neck (LNL) avg'])
            participant_torso_length_post = pd.Series(demographics['torso (post) avg '])
            participant_torso_length_ax = pd.Series(demographics['torso (ax) avg '])
            participant_head_to_heart = pd.Series(demographics['head to heart avg'])
            participant_head_girth = pd.Series(demographics['head avg'])
            participant_neck_girth = pd.Series(demographics['neck avg'])
            participant_chest_upper_girth = pd.Series(demographics['chest upper avg'])
            participant_chest_under_girth = pd.Series(demographics['chest under avg'])
            participant_waist_girth = pd.Series(demographics['waist avg'])
            participant_hip_girth = pd.Series(demographics['hip avg'])
            participant_thigh_girth = pd.Series(demographics['thigh avg'])
            participant_calf_girth = pd.Series(demographics['calf avg'])
            participant_biceps_girth_flex = pd.Series(demographics['bicep flex avg'])
            participant_biceps_girth_relax = pd.Series(demographics['bicep relax avg'])
            participant_neck_flexion = pd.Series(demographics['avg (N) flexion'])
            participant_neck_extension = pd.Series(demographics['avg (N) extens'])
            participant_neck_right_rotation = pd.Series(demographics['avg (N) Rt. Rot'])
            participant_neck_left_rotation = pd.Series(demographics['avg (N) left rot'])
            participant_neck_left_lat_flex = pd.Series(demographics['avg (N) left lat flex'])
            participant_neck_right_lat_flex = pd.Series(demographics['avg (N) rt lat flex'])
            participant_pred_vo2 = pd.Series(demographics['pred. Vo2'])              # Predicted VO2

            # Concatenate all demographics of interest
            all_demographics = pd.concat([participant_gender, participant_age, participant_height, participant_weight,
                                        participant_BMI, participant_blood_volume, participant_SBP_seated,
                                        participant_SBP_stand, participant_SBP_exercise, participant_DBP_seated,
                                        participant_DBP_stand, participant_DBP_exercise, participant_MAP_seated,
                                        participant_MAP_stand, participant_MAP_exercise, participant_HR_seated,
                                        participant_HR_stand, participant_HR_exercise, participant_max_leg_strength,
                                        participant_largest_leg_circumference, participant_lower_leg_volume,
                                        participant_skinfolds_chest_avg, participant_skinfolds_abd_avg,
                                        participant_skinfolds_thigh_avg, participant_skinfolds_midax_avg,
                                        participant_skinfolds_subscap_avg, participant_skinfolds_tri_avg,
                                        participant_skinfolds_supra_avg, participant_skinfolds_sum,
                                        participant_percent_fat, participant_leg_length, participant_arm_length,
                                        participant_midline_neck_length, participant_lateral_neck_length,
                                        participant_torso_length_post, participant_torso_length_ax, participant_head_to_heart,
                                        participant_head_girth, participant_neck_girth, participant_chest_upper_girth,
                                        participant_chest_under_girth, participant_waist_girth, participant_hip_girth,
                                        participant_thigh_girth, participant_calf_girth, participant_biceps_girth_flex,
                                        participant_biceps_girth_relax, participant_neck_flexion,
                                        participant_neck_extension, participant_neck_right_rotation, participant_neck_left_rotation,
                                        participant_neck_left_lat_flex, participant_neck_right_lat_flex,
                                        participant_pred_vo2], axis=1)

            participant_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

            # Initialize variables
            participant_demographics = np.zeros((len(gloc_data_reduced), all_demographics.shape[1]))
            length_previous_participant_data = -1

            # Organize all participant data in array same length as gloc_data_reduced
            for i in range(len(participant_list)):
                length_current_participant_data = len(np.argwhere((gloc_data_reduced['subject'] == participant_list[i])))
                participant_demographics[length_previous_participant_data+1:length_previous_participant_data+1+length_current_participant_data, :] = all_demographics.iloc[i]
                length_previous_participant_data = length_current_participant_data + length_previous_participant_data

            # Create demographics data frame
            demographics_names = ['participant_gender', 'participant_age', 'participant_height', 'participant_weight',
                                        'participant_BMI', 'participant_blood_volume', 'participant_SBP_seated',
                                        'participant_SBP_stand', 'participant_SBP_exercise', 'participant_DBP_seated',
                                        'participant_DBP_stand', 'participant_DBP_exercise', 'participant_MAP_seated',
                                        'participant_MAP_stand', 'participant_MAP_exercise', 'participant_HR_seated',
                                        'participant_HR_stand', 'participant_HR_exercise', 'participant_max_leg_strength',
                                        'participant_largest_leg_circumference', 'participant_lower_leg_volume',
                                        'participant_skinfolds_chest_avg', 'participant_skinfolds_abd_avg',
                                        'participant_skinfolds_thigh_avg', 'participant_skinfolds_midax_avg',
                                        'participant_skinfolds_subscap_avg', 'participant_skinfolds_tri_avg',
                                        'participant_skinfolds_supra_avg', 'participant_skinfolds_sum',
                                        'participant_percent_fat', 'participant_leg_length', 'participant_arm_length',
                                        'participant_midline_neck_length', 'participant_lateral_neck_length',
                                        'participant_torso_length_post', 'participant_torso_length_ax', 'participant_head_to_heart',
                                        'participant_head_girth', 'participant_neck_girth', 'participant_chest_upper_girth',
                                        'participant_chest_under_girth', 'participant_waist_girth', 'participant_hip_girth',
                                        'participant_thigh_girth', 'participant_calf_girth', 'participant_biceps_girth_flex',
                                        'participant_biceps_girth_relax', 'participant_neck_flexion',
                                        'participant_neck_extension', 'participant_neck_right_rotation', 'participant_neck_left_rotation',
                                        'participant_neck_left_lat_flex', 'participant_neck_right_lat_flex',
                                        'participant_pred_vo2']
            demographics_concat = pd.DataFrame(participant_demographics, columns = demographics_names)

            # Append all demographic data to gloc data reduced
            gloc_data_reduced = pd.concat([gloc_data_reduced, demographics_concat], axis=1)
            return gloc_data_reduced, demographics_names

        def process_strain_data(gloc_data_reduced):
            ######### Add missing strain during GOR labels based on GLOC Effectiveness Spreadsheet #########
            # Add individual strain labels for trials where label was 'missed' (from column E of GLOC_Effectiveness spreadsheet)
            # Define trial & g magnitude variable
            gloc_trial = gloc_data_reduced['trial_id']
            magnitude_g = gloc_data_reduced['magnitude - Centrifuge'].to_numpy()
            event = gloc_data_reduced['event']

            return gloc_data_reduced, gloc_trial

            ######## Trial 04-06 (GLOC_Effectiveness stain value of 6.1g) ########
            trial_individual_coding = '04-06'
            g_level_strain = 6.1

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            #
            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 06-02 (GLOC_Effectiveness stain value of 8.1g) ########
            trial_individual_coding = '06-02'
            g_level_strain = 8.1

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 06-06 End GOR Label in Event Validated ########
            trial_individual_coding = '06-06'

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])

            ## Add missing 'end GOR' label
            # Find first nan in g magnitude post GOR peak
            return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]

            # Find the index of new end GOR label in full length csv
            end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index

            # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 07-06 (GLOC_Effectiveness stain value of 4.6g) ########
            trial_individual_coding = '07-06'
            g_level_strain = 4.6

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 08-02 (GLOC_Effectiveness stain value of 8.3g) ########
            trial_individual_coding = '08-02'
            g_level_strain = 8.3

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 08-05 (GLOC_Effectiveness stain value of 4.3g) ########
            trial_individual_coding = '08-05'
            g_level_strain = 4.3

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ## Add missing 'end GOR' label
            # Find first nan in g magnitude post GOR peak
            return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]

            # Find the index of new end GOR label in full length csv
            end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index

            # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 08-06 (GLOC_Effectiveness stain value of 8.2g) ########
            trial_individual_coding = '08-06'
            g_level_strain = 8.2

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 09-03 (GLOC_Effectiveness stain value of 8.7g) ########
            trial_individual_coding = '09-03'
            g_level_strain = 8.7

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 09-05 (GLOC_Effectiveness stain value of 4.8g) ########
            trial_individual_coding = '09-05'
            g_level_strain = 4.8

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 10-05 (GLOC_Effectiveness stain value of 3.8g) ########
            trial_individual_coding = '10-05'
            g_level_strain = 3.8

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 10-06 (GLOC_Effectiveness stain value of 5.5g) ########
            trial_individual_coding = '10-06'
            g_level_strain = 5.5

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]
            current_event = event[trial_index]

            #### Remove original label in event column ####
            # Find row containing the string
            mislabel_mask_strain = current_event.str.contains('strain during GOR')
            # mislabel_mask_end_GOR = current_event.str.contains('end GOR')

            # Get the index of that row
            mislabel_index_strain = current_event[mislabel_mask_strain].index
            # mislabel_index_end_GOR = current_event[mislabel_mask_end_GOR].index

            # Find the index of new strain label in full length csv
            strain_relabel_index = mislabel_index_strain
            # end_GOR_relabel_index = mislabel_index_end_GOR

            # gloc_data_reduced['event'][strain_relabel_index] = None
            gloc_data_reduced.loc[strain_relabel_index, 'event'] = None
            # # gloc_data_reduced['event'][end_GOR_relabel_index] = None
            # gloc_data_reduced.loc[end_GOR_relabel_index, 'event'] = None

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 11-02 (GLOC_Effectiveness stain value of 5.5g) ########
            trial_individual_coding = '11-02'
            g_level_strain = 5.5

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 12-03 (GLOC_Effectiveness stain value of 3.7g) ########
            trial_individual_coding = '12-03'
            g_level_strain = 3.7

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 13-02 (GLOC_Effectiveness stain value of 7.3g) ########
            trial_individual_coding = '13-02'
            g_level_strain = 7.3

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 13-04 (GLOC_Effectiveness stain value of 7.4g) ########
            trial_individual_coding = '13-04'
            g_level_strain = 7.4

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            return gloc_data_reduced, gloc_trial

        _, _, _, _, _, actual_all_features, actual_all_features_phys, actual_all_features_ecg, actual_all_features_eeg = get_actual_features(actual_gloc_data, feature_groups_to_analyze, file_paths["demographic"] , ("complete", "explicit"))

        # Testing all features similarity
        all_features_set = set(features["All"])
        actual_all_features_set = set(actual_all_features)
        assert all_features_set == actual_all_features_set, f"Missing features in actual_all_features: {all_features_set - actual_all_features_set}"

        # Testing physiological features similarity
        features_phys_set = set(features["Phys"])
        actual_all_features_phys_set = set(actual_all_features_phys)
        assert features_phys_set == actual_all_features_phys_set, f"Missing features in actual_all_features_phys: {features_phys_set - actual_all_features_phys_set}"

        # Testing ECG features similarity
        features_ecg_set = set(features["ECG"])
        actual_all_features_ecg_set = set(actual_all_features_ecg)
        assert features_ecg_set == actual_all_features_ecg_set, f"Missing features in actual_all_features_ecg: {features_ecg_set - actual_all_features_ecg_set}"

        # Testing EEG features similarity
        features_eeg_set = set(features["EEG"])
        actual_all_features_eeg_set = set(actual_all_features_eeg)
        assert features_eeg_set == actual_all_features_eeg_set, f"Missing features in actual_all_features_eeg: {features_eeg_set - actual_all_features_eeg_set}"

    def test_getting_feature_names_complete_implicit(self):
        manager = DataManager()
        file_paths = manager._get_data_locations()
        gloc_data = manager._load_data(file_paths)
        actual_gloc_data = manager._load_data(file_paths)

        feature_groups_to_analyze, _ = manager._get_feature_groups_and_baseline_methods(("Complete", "Implicit"))

        gloc_data, features = manager._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, ("Complete", "Implicit"), file_paths)

        def get_actual_features(gloc_data_reduced, feature_groups_to_analyze, demographic_data_filename, model_type):
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
                    # raw_eeg_condition_specific = []
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
                    # processed_eeg_condition_specific = []
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

        def pull_eeg_sets():
            # list of shared eeg channels
            raw_eeg_shared_features = ['Fz - EEG', 'F3 - EEG', 'C3 - EEG', 'C4 - EEG',
                                        'CP1 - EEG', 'CP2 - EEG', 'T8 - EEG', 'TP9 - EEG', 'TP10 - EEG',
                                        'P7 - EEG', 'P8 - EEG']

            processed_eeg_shared_features = ['Fz_delta - EEG', 'Fz_theta - EEG', 'Fz_alpha - EEG', 'Fz_beta - EEG',
                                            'F3_delta - EEG', 'F3_theta - EEG', 'F3_alpha - EEG', 'F3_beta - EEG',
                                            'C3_delta - EEG', 'C3_theta - EEG', 'C3_alpha - EEG', 'C3_beta - EEG',
                                            'C4_delta - EEG', 'C4_theta - EEG', 'C4_alpha - EEG', 'C4_beta - EEG',
                                            'CP1_delta - EEG', 'CP1_theta - EEG', 'CP1_alpha - EEG', 'CP1_beta - EEG',
                                            'CP2_delta - EEG', 'CP2_theta - EEG', 'CP2_alpha - EEG', 'CP2_beta - EEG',
                                            'T8_delta - EEG', 'T8_theta - EEG', 'T8_alpha - EEG', 'T8_beta - EEG',
                                            'TP9_delta - EEG', 'TP9_theta - EEG', 'TP9_alpha - EEG', 'TP9_beta - EEG',
                                            'TP10_delta - EEG', 'TP10_theta - EEG', 'TP10_alpha - EEG', 'TP10_beta - EEG',
                                            'P7_delta - EEG', 'P7_theta - EEG', 'P7_alpha - EEG', 'P7_beta - EEG',
                                            'P8_delta - EEG', 'P8_theta - EEG', 'P8_alpha - EEG', 'P8_beta - EEG']

            # list of AFE only eeg channels
            raw_eeg_afe_only = ['F4 - EEG', 'T7 - EEG', 'O1 - EEG', 'O2 - EEG']

            processed_eeg_afe_only =['F4_delta - EEG', 'F4_theta - EEG', 'F4_alpha - EEG', 'F4_beta - EEG',
                                                        'T7_delta - EEG', 'T7_theta - EEG', 'T7_alpha - EEG', 'T7_beta - EEG',
                                                        'O1_delta - EEG', 'O1_theta - EEG', 'O1_alpha - EEG', 'O1_beta - EEG',
                                                        'O2_delta - EEG', 'O2_theta - EEG', 'O2_alpha - EEG', 'O2_beta - EEG']
            # list of Non-AFE only eeg channels
            raw_eeg_nonafe_only = ['F1 - EEG', 'AFz - EEG', 'AF4 - EEG', 'FT9 - EEG', 'FT10 - EEG', 'FC5 - EEG',
                                            'FC3 - EEG', 'FC1 - EEG', 'FC2 - EEG', 'FC4 - EEG', 'FC6 - EEG', 'C5 - EEG',
                                            'Cz - EEG', 'CP5 - EEG', 'CP6 - EEG', 'P5 - EEG', 'P3 - EEG', 'P1 - EEG',
                                            'Pz - EEG', 'P4 - EEG', 'P6 - EEG']

            processed_eeg_nonafe_only =['F1_delta - EEG', 'F1_theta - EEG', 'F1_alpha - EEG', 'F1_beta - EEG',
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
                                                        'CP6_delta - EEG', 'CP6_theta - EEG', 'CP6_alpha - EEG','CP6_beta - EEG',
                                                        'P5_delta - EEG', 'P5_theta - EEG', 'P5_alpha - EEG', 'P5_beta - EEG',
                                                        'P3_delta - EEG', 'P3_theta - EEG', 'P3_alpha - EEG', 'P3_beta - EEG',
                                                        'P1_delta - EEG', 'P1_theta - EEG', 'P1_alpha - EEG', 'P1_beta - EEG',
                                                        'Pz_delta - EEG', 'Pz_theta - EEG', 'Pz_alpha - EEG', 'Pz_beta - EEG',
                                                        'P4_delta - EEG', 'P4_theta - EEG', 'P4_alpha - EEG', 'P4_beta - EEG',
                                                        'P6_delta - EEG', 'P6_theta - EEG', 'P6_alpha - EEG', 'P6_beta - EEG']

            return (processed_eeg_shared_features, processed_eeg_afe_only, processed_eeg_nonafe_only,
                    raw_eeg_shared_features, raw_eeg_afe_only, raw_eeg_nonafe_only)
        
        def read_and_process_demographics(demographic_data_filename, gloc_data_reduced):
            """
            This function imports the demographics spreadsheet and appends the demographics
            to the gloc_data_reduced variable. Each column represents one of the demographics
            variables below. The column is filled by determining the participant for a specific
            trial and filling the entire trial with that participants demographic data.
            """

            # Import demographics spreadsheet
            demographics = pd.read_csv(demographic_data_filename)
            demographics = demographics.astype({col: 'float32' for col in demographics.select_dtypes(include='float64').columns})
            demographics = demographics.copy()

            # Grab variables of interest
            participant_index = demographics['GLOC ID']                                                         # Corresponds to subject 1-13
            participant_gender = pd.Series(demographics['Gender code [1/0]'])        # 0 = Female, 1 = Male
            participant_age = pd.Series(demographics['Age [yr]'])
            participant_height = pd.Series(demographics['height [m]'])
            participant_weight = pd.Series(demographics['weight (kg)'])
            participant_BMI = pd.Series(demographics['BMI [kg/m^2]'])
            participant_blood_volume = pd.Series(demographics['Blood Volume [L]'])   # Based on Nadler's approximation
            # participant_PWV = pd.Series(demographics['Avg PWV'])                   # Pulse Wave Velocity
                # (no val for participant 4)
            # participant_PTT = demographics['Avg PTT [ms]'])                        # Pulse Transit Time
                # (no val for participant 4)
            participant_SBP_seated = pd.Series(demographics['Resting SBP (seat)'])   # Systolic Blood Pressure
            participant_SBP_stand = pd.Series(demographics['resting SBP (stand)'])
            participant_SBP_exercise = pd.Series(demographics['SBP after squat'])
            participant_DBP_seated = pd.Series(demographics['Resting DBP (seat)'] )  # Diastolic Blood Pressure
            participant_DBP_stand = pd.Series(demographics['resting DBP (stand)'])
            participant_DBP_exercise = pd.Series(demographics['DBP after squat'])
            participant_MAP_seated = pd.Series(demographics['Resting MAP'])          # Mean Arterial Pressure
            participant_MAP_stand = pd.Series(demographics['Resting MAP (stand'])
            participant_MAP_exercise = pd.Series(demographics['Post-Squat MAP'])
            participant_HR_seated = pd.Series(demographics['resting HR [seated]'])
            participant_HR_stand = pd.Series(demographics['resting HR (stand)'])
            participant_HR_exercise = pd.Series(demographics['HR after squat'])
            participant_max_leg_strength = pd.Series(demographics['Max (N)'])        # Max Leg Strength
            participant_largest_leg_circumference = pd.Series(demographics['largest leg circ. [cm]'])
            participant_lower_leg_volume = pd.Series(demographics['lower leg volume [mL]'])
            participant_skinfolds_chest_avg = pd.Series(demographics['chest avg'])   # Skin Folds to Approximate Body Fat %
            participant_skinfolds_abd_avg = pd.Series(demographics['abd avg'])
            participant_skinfolds_thigh_avg = pd.Series(demographics['thigh avg'])
            participant_skinfolds_midax_avg = pd.Series(demographics['midax avg'])
            participant_skinfolds_subscap_avg = pd.Series(demographics['subscap avg'])
            participant_skinfolds_tri_avg = pd.Series(demographics['tri avg'])
            participant_skinfolds_supra_avg = pd.Series(demographics['supra avg'])
            participant_skinfolds_sum = pd.Series(demographics['sum'])
            participant_percent_fat = pd.Series(demographics['% fat'])
            participant_leg_length = pd.Series(demographics['leg avg'])
            participant_arm_length = pd.Series(demographics['arm avg'])
            participant_midline_neck_length = pd.Series(demographics['neck (MNL) avg'])
            participant_lateral_neck_length = pd.Series(demographics['neck (LNL) avg'])
            participant_torso_length_post = pd.Series(demographics['torso (post) avg '])
            participant_torso_length_ax = pd.Series(demographics['torso (ax) avg '])
            participant_head_to_heart = pd.Series(demographics['head to heart avg'])
            participant_head_girth = pd.Series(demographics['head avg'])
            participant_neck_girth = pd.Series(demographics['neck avg'])
            participant_chest_upper_girth = pd.Series(demographics['chest upper avg'])
            participant_chest_under_girth = pd.Series(demographics['chest under avg'])
            participant_waist_girth = pd.Series(demographics['waist avg'])
            participant_hip_girth = pd.Series(demographics['hip avg'])
            participant_thigh_girth = pd.Series(demographics['thigh avg'])
            participant_calf_girth = pd.Series(demographics['calf avg'])
            participant_biceps_girth_flex = pd.Series(demographics['bicep flex avg'])
            participant_biceps_girth_relax = pd.Series(demographics['bicep relax avg'])
            participant_neck_flexion = pd.Series(demographics['avg (N) flexion'])
            participant_neck_extension = pd.Series(demographics['avg (N) extens'])
            participant_neck_right_rotation = pd.Series(demographics['avg (N) Rt. Rot'])
            participant_neck_left_rotation = pd.Series(demographics['avg (N) left rot'])
            participant_neck_left_lat_flex = pd.Series(demographics['avg (N) left lat flex'])
            participant_neck_right_lat_flex = pd.Series(demographics['avg (N) rt lat flex'])
            participant_pred_vo2 = pd.Series(demographics['pred. Vo2'])              # Predicted VO2

            # Concatenate all demographics of interest
            all_demographics = pd.concat([participant_gender, participant_age, participant_height, participant_weight,
                                        participant_BMI, participant_blood_volume, participant_SBP_seated,
                                        participant_SBP_stand, participant_SBP_exercise, participant_DBP_seated,
                                        participant_DBP_stand, participant_DBP_exercise, participant_MAP_seated,
                                        participant_MAP_stand, participant_MAP_exercise, participant_HR_seated,
                                        participant_HR_stand, participant_HR_exercise, participant_max_leg_strength,
                                        participant_largest_leg_circumference, participant_lower_leg_volume,
                                        participant_skinfolds_chest_avg, participant_skinfolds_abd_avg,
                                        participant_skinfolds_thigh_avg, participant_skinfolds_midax_avg,
                                        participant_skinfolds_subscap_avg, participant_skinfolds_tri_avg,
                                        participant_skinfolds_supra_avg, participant_skinfolds_sum,
                                        participant_percent_fat, participant_leg_length, participant_arm_length,
                                        participant_midline_neck_length, participant_lateral_neck_length,
                                        participant_torso_length_post, participant_torso_length_ax, participant_head_to_heart,
                                        participant_head_girth, participant_neck_girth, participant_chest_upper_girth,
                                        participant_chest_under_girth, participant_waist_girth, participant_hip_girth,
                                        participant_thigh_girth, participant_calf_girth, participant_biceps_girth_flex,
                                        participant_biceps_girth_relax, participant_neck_flexion,
                                        participant_neck_extension, participant_neck_right_rotation, participant_neck_left_rotation,
                                        participant_neck_left_lat_flex, participant_neck_right_lat_flex,
                                        participant_pred_vo2], axis=1)

            participant_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

            # Initialize variables
            participant_demographics = np.zeros((len(gloc_data_reduced), all_demographics.shape[1]))
            length_previous_participant_data = -1

            # Organize all participant data in array same length as gloc_data_reduced
            for i in range(len(participant_list)):
                length_current_participant_data = len(np.argwhere((gloc_data_reduced['subject'] == participant_list[i])))
                participant_demographics[length_previous_participant_data+1:length_previous_participant_data+1+length_current_participant_data, :] = all_demographics.iloc[i]
                length_previous_participant_data = length_current_participant_data + length_previous_participant_data

            # Create demographics data frame
            demographics_names = ['participant_gender', 'participant_age', 'participant_height', 'participant_weight',
                                        'participant_BMI', 'participant_blood_volume', 'participant_SBP_seated',
                                        'participant_SBP_stand', 'participant_SBP_exercise', 'participant_DBP_seated',
                                        'participant_DBP_stand', 'participant_DBP_exercise', 'participant_MAP_seated',
                                        'participant_MAP_stand', 'participant_MAP_exercise', 'participant_HR_seated',
                                        'participant_HR_stand', 'participant_HR_exercise', 'participant_max_leg_strength',
                                        'participant_largest_leg_circumference', 'participant_lower_leg_volume',
                                        'participant_skinfolds_chest_avg', 'participant_skinfolds_abd_avg',
                                        'participant_skinfolds_thigh_avg', 'participant_skinfolds_midax_avg',
                                        'participant_skinfolds_subscap_avg', 'participant_skinfolds_tri_avg',
                                        'participant_skinfolds_supra_avg', 'participant_skinfolds_sum',
                                        'participant_percent_fat', 'participant_leg_length', 'participant_arm_length',
                                        'participant_midline_neck_length', 'participant_lateral_neck_length',
                                        'participant_torso_length_post', 'participant_torso_length_ax', 'participant_head_to_heart',
                                        'participant_head_girth', 'participant_neck_girth', 'participant_chest_upper_girth',
                                        'participant_chest_under_girth', 'participant_waist_girth', 'participant_hip_girth',
                                        'participant_thigh_girth', 'participant_calf_girth', 'participant_biceps_girth_flex',
                                        'participant_biceps_girth_relax', 'participant_neck_flexion',
                                        'participant_neck_extension', 'participant_neck_right_rotation', 'participant_neck_left_rotation',
                                        'participant_neck_left_lat_flex', 'participant_neck_right_lat_flex',
                                        'participant_pred_vo2']
            demographics_concat = pd.DataFrame(participant_demographics, columns = demographics_names)

            # Append all demographic data to gloc data reduced
            gloc_data_reduced = pd.concat([gloc_data_reduced, demographics_concat], axis=1)
            return gloc_data_reduced, demographics_names

        def process_strain_data(gloc_data_reduced):
            ######### Add missing strain during GOR labels based on GLOC Effectiveness Spreadsheet #########
            # Add individual strain labels for trials where label was 'missed' (from column E of GLOC_Effectiveness spreadsheet)
            # Define trial & g magnitude variable
            gloc_trial = gloc_data_reduced['trial_id']
            magnitude_g = gloc_data_reduced['magnitude - Centrifuge'].to_numpy()
            event = gloc_data_reduced['event']

            return gloc_data_reduced, gloc_trial

            ######## Trial 04-06 (GLOC_Effectiveness stain value of 6.1g) ########
            trial_individual_coding = '04-06'
            g_level_strain = 6.1

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            #
            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 06-02 (GLOC_Effectiveness stain value of 8.1g) ########
            trial_individual_coding = '06-02'
            g_level_strain = 8.1

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 06-06 End GOR Label in Event Validated ########
            trial_individual_coding = '06-06'

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])

            ## Add missing 'end GOR' label
            # Find first nan in g magnitude post GOR peak
            return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]

            # Find the index of new end GOR label in full length csv
            end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index

            # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 07-06 (GLOC_Effectiveness stain value of 4.6g) ########
            trial_individual_coding = '07-06'
            g_level_strain = 4.6

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 08-02 (GLOC_Effectiveness stain value of 8.3g) ########
            trial_individual_coding = '08-02'
            g_level_strain = 8.3

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 08-05 (GLOC_Effectiveness stain value of 4.3g) ########
            trial_individual_coding = '08-05'
            g_level_strain = 4.3

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ## Add missing 'end GOR' label
            # Find first nan in g magnitude post GOR peak
            return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]

            # Find the index of new end GOR label in full length csv
            end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index

            # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 08-06 (GLOC_Effectiveness stain value of 8.2g) ########
            trial_individual_coding = '08-06'
            g_level_strain = 8.2

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 09-03 (GLOC_Effectiveness stain value of 8.7g) ########
            trial_individual_coding = '09-03'
            g_level_strain = 8.7

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 09-05 (GLOC_Effectiveness stain value of 4.8g) ########
            trial_individual_coding = '09-05'
            g_level_strain = 4.8

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 10-05 (GLOC_Effectiveness stain value of 3.8g) ########
            trial_individual_coding = '10-05'
            g_level_strain = 3.8

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 10-06 (GLOC_Effectiveness stain value of 5.5g) ########
            trial_individual_coding = '10-06'
            g_level_strain = 5.5

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]
            current_event = event[trial_index]

            #### Remove original label in event column ####
            # Find row containing the string
            mislabel_mask_strain = current_event.str.contains('strain during GOR')
            # mislabel_mask_end_GOR = current_event.str.contains('end GOR')

            # Get the index of that row
            mislabel_index_strain = current_event[mislabel_mask_strain].index
            # mislabel_index_end_GOR = current_event[mislabel_mask_end_GOR].index

            # Find the index of new strain label in full length csv
            strain_relabel_index = mislabel_index_strain
            # end_GOR_relabel_index = mislabel_index_end_GOR

            # gloc_data_reduced['event'][strain_relabel_index] = None
            gloc_data_reduced.loc[strain_relabel_index, 'event'] = None
            # # gloc_data_reduced['event'][end_GOR_relabel_index] = None
            # gloc_data_reduced.loc[end_GOR_relabel_index, 'event'] = None

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 11-02 (GLOC_Effectiveness stain value of 5.5g) ########
            trial_individual_coding = '11-02'
            g_level_strain = 5.5

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 12-03 (GLOC_Effectiveness stain value of 3.7g) ########
            trial_individual_coding = '12-03'
            g_level_strain = 3.7

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 13-02 (GLOC_Effectiveness stain value of 7.3g) ########
            trial_individual_coding = '13-02'
            g_level_strain = 7.3

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 13-04 (GLOC_Effectiveness stain value of 7.4g) ########
            trial_individual_coding = '13-04'
            g_level_strain = 7.4

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            return gloc_data_reduced, gloc_trial

        _, _, _, _, _, actual_all_features, actual_all_features_phys, actual_all_features_ecg, actual_all_features_eeg = get_actual_features(actual_gloc_data, feature_groups_to_analyze, file_paths["demographic"] , ("complete", "implicit"))

        # Testing all features similarity
        all_features_set = set(features["All"])
        actual_all_features_set = set(actual_all_features)
        assert all_features_set == actual_all_features_set, f"Missing features in actual_all_features: {all_features_set - actual_all_features_set}"

        # Testing physiological features similarity
        features_phys_set = set(features["Phys"])
        actual_all_features_phys_set = set(actual_all_features_phys)
        assert features_phys_set == actual_all_features_phys_set, f"Missing features in actual_all_features_phys: {features_phys_set - actual_all_features_phys_set}"

        # Testing ECG features similarity
        features_ecg_set = set(features["ECG"])
        actual_all_features_ecg_set = set(actual_all_features_ecg)
        assert features_ecg_set == actual_all_features_ecg_set, f"Missing features in actual_all_features_ecg: {features_ecg_set - actual_all_features_ecg_set}"

        # Testing EEG features similarity
        features_eeg_set = set(features["EEG"])
        actual_all_features_eeg_set = set(actual_all_features_eeg)
        assert features_eeg_set == actual_all_features_eeg_set, f"Missing features in actual_all_features_eeg: {features_eeg_set - actual_all_features_eeg_set}"

    def test_getting_feature_names_noAFE_explicit(self):
        manager = DataManager()
        file_paths = manager._get_data_locations()
        gloc_data = manager._load_data(file_paths)
        actual_gloc_data = manager._load_data(file_paths)

        feature_groups_to_analyze = {"ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE", "G", "processedEEG", "demographics", "strain"}

        gloc_data, features = manager._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, ("noAFE", "Explicit"), file_paths)

        def get_actual_features(gloc_data_reduced, feature_groups_to_analyze, demographic_data_filename, model_type):
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
                    # raw_eeg_condition_specific = []
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
                    # processed_eeg_condition_specific = []
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

        def pull_eeg_sets():
            # list of shared eeg channels
            raw_eeg_shared_features = ['Fz - EEG', 'F3 - EEG', 'C3 - EEG', 'C4 - EEG',
                                        'CP1 - EEG', 'CP2 - EEG', 'T8 - EEG', 'TP9 - EEG', 'TP10 - EEG',
                                        'P7 - EEG', 'P8 - EEG']

            processed_eeg_shared_features = ['Fz_delta - EEG', 'Fz_theta - EEG', 'Fz_alpha - EEG', 'Fz_beta - EEG',
                                            'F3_delta - EEG', 'F3_theta - EEG', 'F3_alpha - EEG', 'F3_beta - EEG',
                                            'C3_delta - EEG', 'C3_theta - EEG', 'C3_alpha - EEG', 'C3_beta - EEG',
                                            'C4_delta - EEG', 'C4_theta - EEG', 'C4_alpha - EEG', 'C4_beta - EEG',
                                            'CP1_delta - EEG', 'CP1_theta - EEG', 'CP1_alpha - EEG', 'CP1_beta - EEG',
                                            'CP2_delta - EEG', 'CP2_theta - EEG', 'CP2_alpha - EEG', 'CP2_beta - EEG',
                                            'T8_delta - EEG', 'T8_theta - EEG', 'T8_alpha - EEG', 'T8_beta - EEG',
                                            'TP9_delta - EEG', 'TP9_theta - EEG', 'TP9_alpha - EEG', 'TP9_beta - EEG',
                                            'TP10_delta - EEG', 'TP10_theta - EEG', 'TP10_alpha - EEG', 'TP10_beta - EEG',
                                            'P7_delta - EEG', 'P7_theta - EEG', 'P7_alpha - EEG', 'P7_beta - EEG',
                                            'P8_delta - EEG', 'P8_theta - EEG', 'P8_alpha - EEG', 'P8_beta - EEG']

            # list of AFE only eeg channels
            raw_eeg_afe_only = ['F4 - EEG', 'T7 - EEG', 'O1 - EEG', 'O2 - EEG']

            processed_eeg_afe_only =['F4_delta - EEG', 'F4_theta - EEG', 'F4_alpha - EEG', 'F4_beta - EEG',
                                                        'T7_delta - EEG', 'T7_theta - EEG', 'T7_alpha - EEG', 'T7_beta - EEG',
                                                        'O1_delta - EEG', 'O1_theta - EEG', 'O1_alpha - EEG', 'O1_beta - EEG',
                                                        'O2_delta - EEG', 'O2_theta - EEG', 'O2_alpha - EEG', 'O2_beta - EEG']
            # list of Non-AFE only eeg channels
            raw_eeg_nonafe_only = ['F1 - EEG', 'AFz - EEG', 'AF4 - EEG', 'FT9 - EEG', 'FT10 - EEG', 'FC5 - EEG',
                                            'FC3 - EEG', 'FC1 - EEG', 'FC2 - EEG', 'FC4 - EEG', 'FC6 - EEG', 'C5 - EEG',
                                            'Cz - EEG', 'CP5 - EEG', 'CP6 - EEG', 'P5 - EEG', 'P3 - EEG', 'P1 - EEG',
                                            'Pz - EEG', 'P4 - EEG', 'P6 - EEG']

            processed_eeg_nonafe_only =['F1_delta - EEG', 'F1_theta - EEG', 'F1_alpha - EEG', 'F1_beta - EEG',
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
                                                        'CP6_delta - EEG', 'CP6_theta - EEG', 'CP6_alpha - EEG','CP6_beta - EEG',
                                                        'P5_delta - EEG', 'P5_theta - EEG', 'P5_alpha - EEG', 'P5_beta - EEG',
                                                        'P3_delta - EEG', 'P3_theta - EEG', 'P3_alpha - EEG', 'P3_beta - EEG',
                                                        'P1_delta - EEG', 'P1_theta - EEG', 'P1_alpha - EEG', 'P1_beta - EEG',
                                                        'Pz_delta - EEG', 'Pz_theta - EEG', 'Pz_alpha - EEG', 'Pz_beta - EEG',
                                                        'P4_delta - EEG', 'P4_theta - EEG', 'P4_alpha - EEG', 'P4_beta - EEG',
                                                        'P6_delta - EEG', 'P6_theta - EEG', 'P6_alpha - EEG', 'P6_beta - EEG']

            return (processed_eeg_shared_features, processed_eeg_afe_only, processed_eeg_nonafe_only,
                    raw_eeg_shared_features, raw_eeg_afe_only, raw_eeg_nonafe_only)
        
        def read_and_process_demographics(demographic_data_filename, gloc_data_reduced):
            """
            This function imports the demographics spreadsheet and appends the demographics
            to the gloc_data_reduced variable. Each column represents one of the demographics
            variables below. The column is filled by determining the participant for a specific
            trial and filling the entire trial with that participants demographic data.
            """

            # Import demographics spreadsheet
            demographics = pd.read_csv(demographic_data_filename)
            demographics = demographics.astype({col: 'float32' for col in demographics.select_dtypes(include='float64').columns})
            demographics = demographics.copy()

            # Grab variables of interest
            participant_index = demographics['GLOC ID']                                                         # Corresponds to subject 1-13
            participant_gender = pd.Series(demographics['Gender code [1/0]'])        # 0 = Female, 1 = Male
            participant_age = pd.Series(demographics['Age [yr]'])
            participant_height = pd.Series(demographics['height [m]'])
            participant_weight = pd.Series(demographics['weight (kg)'])
            participant_BMI = pd.Series(demographics['BMI [kg/m^2]'])
            participant_blood_volume = pd.Series(demographics['Blood Volume [L]'])   # Based on Nadler's approximation
            # participant_PWV = pd.Series(demographics['Avg PWV'])                   # Pulse Wave Velocity
                # (no val for participant 4)
            # participant_PTT = demographics['Avg PTT [ms]'])                        # Pulse Transit Time
                # (no val for participant 4)
            participant_SBP_seated = pd.Series(demographics['Resting SBP (seat)'])   # Systolic Blood Pressure
            participant_SBP_stand = pd.Series(demographics['resting SBP (stand)'])
            participant_SBP_exercise = pd.Series(demographics['SBP after squat'])
            participant_DBP_seated = pd.Series(demographics['Resting DBP (seat)'] )  # Diastolic Blood Pressure
            participant_DBP_stand = pd.Series(demographics['resting DBP (stand)'])
            participant_DBP_exercise = pd.Series(demographics['DBP after squat'])
            participant_MAP_seated = pd.Series(demographics['Resting MAP'])          # Mean Arterial Pressure
            participant_MAP_stand = pd.Series(demographics['Resting MAP (stand'])
            participant_MAP_exercise = pd.Series(demographics['Post-Squat MAP'])
            participant_HR_seated = pd.Series(demographics['resting HR [seated]'])
            participant_HR_stand = pd.Series(demographics['resting HR (stand)'])
            participant_HR_exercise = pd.Series(demographics['HR after squat'])
            participant_max_leg_strength = pd.Series(demographics['Max (N)'])        # Max Leg Strength
            participant_largest_leg_circumference = pd.Series(demographics['largest leg circ. [cm]'])
            participant_lower_leg_volume = pd.Series(demographics['lower leg volume [mL]'])
            participant_skinfolds_chest_avg = pd.Series(demographics['chest avg'])   # Skin Folds to Approximate Body Fat %
            participant_skinfolds_abd_avg = pd.Series(demographics['abd avg'])
            participant_skinfolds_thigh_avg = pd.Series(demographics['thigh avg'])
            participant_skinfolds_midax_avg = pd.Series(demographics['midax avg'])
            participant_skinfolds_subscap_avg = pd.Series(demographics['subscap avg'])
            participant_skinfolds_tri_avg = pd.Series(demographics['tri avg'])
            participant_skinfolds_supra_avg = pd.Series(demographics['supra avg'])
            participant_skinfolds_sum = pd.Series(demographics['sum'])
            participant_percent_fat = pd.Series(demographics['% fat'])
            participant_leg_length = pd.Series(demographics['leg avg'])
            participant_arm_length = pd.Series(demographics['arm avg'])
            participant_midline_neck_length = pd.Series(demographics['neck (MNL) avg'])
            participant_lateral_neck_length = pd.Series(demographics['neck (LNL) avg'])
            participant_torso_length_post = pd.Series(demographics['torso (post) avg '])
            participant_torso_length_ax = pd.Series(demographics['torso (ax) avg '])
            participant_head_to_heart = pd.Series(demographics['head to heart avg'])
            participant_head_girth = pd.Series(demographics['head avg'])
            participant_neck_girth = pd.Series(demographics['neck avg'])
            participant_chest_upper_girth = pd.Series(demographics['chest upper avg'])
            participant_chest_under_girth = pd.Series(demographics['chest under avg'])
            participant_waist_girth = pd.Series(demographics['waist avg'])
            participant_hip_girth = pd.Series(demographics['hip avg'])
            participant_thigh_girth = pd.Series(demographics['thigh avg'])
            participant_calf_girth = pd.Series(demographics['calf avg'])
            participant_biceps_girth_flex = pd.Series(demographics['bicep flex avg'])
            participant_biceps_girth_relax = pd.Series(demographics['bicep relax avg'])
            participant_neck_flexion = pd.Series(demographics['avg (N) flexion'])
            participant_neck_extension = pd.Series(demographics['avg (N) extens'])
            participant_neck_right_rotation = pd.Series(demographics['avg (N) Rt. Rot'])
            participant_neck_left_rotation = pd.Series(demographics['avg (N) left rot'])
            participant_neck_left_lat_flex = pd.Series(demographics['avg (N) left lat flex'])
            participant_neck_right_lat_flex = pd.Series(demographics['avg (N) rt lat flex'])
            participant_pred_vo2 = pd.Series(demographics['pred. Vo2'])              # Predicted VO2

            # Concatenate all demographics of interest
            all_demographics = pd.concat([participant_gender, participant_age, participant_height, participant_weight,
                                        participant_BMI, participant_blood_volume, participant_SBP_seated,
                                        participant_SBP_stand, participant_SBP_exercise, participant_DBP_seated,
                                        participant_DBP_stand, participant_DBP_exercise, participant_MAP_seated,
                                        participant_MAP_stand, participant_MAP_exercise, participant_HR_seated,
                                        participant_HR_stand, participant_HR_exercise, participant_max_leg_strength,
                                        participant_largest_leg_circumference, participant_lower_leg_volume,
                                        participant_skinfolds_chest_avg, participant_skinfolds_abd_avg,
                                        participant_skinfolds_thigh_avg, participant_skinfolds_midax_avg,
                                        participant_skinfolds_subscap_avg, participant_skinfolds_tri_avg,
                                        participant_skinfolds_supra_avg, participant_skinfolds_sum,
                                        participant_percent_fat, participant_leg_length, participant_arm_length,
                                        participant_midline_neck_length, participant_lateral_neck_length,
                                        participant_torso_length_post, participant_torso_length_ax, participant_head_to_heart,
                                        participant_head_girth, participant_neck_girth, participant_chest_upper_girth,
                                        participant_chest_under_girth, participant_waist_girth, participant_hip_girth,
                                        participant_thigh_girth, participant_calf_girth, participant_biceps_girth_flex,
                                        participant_biceps_girth_relax, participant_neck_flexion,
                                        participant_neck_extension, participant_neck_right_rotation, participant_neck_left_rotation,
                                        participant_neck_left_lat_flex, participant_neck_right_lat_flex,
                                        participant_pred_vo2], axis=1)

            participant_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

            # Initialize variables
            participant_demographics = np.zeros((len(gloc_data_reduced), all_demographics.shape[1]))
            length_previous_participant_data = -1

            # Organize all participant data in array same length as gloc_data_reduced
            for i in range(len(participant_list)):
                length_current_participant_data = len(np.argwhere((gloc_data_reduced['subject'] == participant_list[i])))
                participant_demographics[length_previous_participant_data+1:length_previous_participant_data+1+length_current_participant_data, :] = all_demographics.iloc[i]
                length_previous_participant_data = length_current_participant_data + length_previous_participant_data

            # Create demographics data frame
            demographics_names = ['participant_gender', 'participant_age', 'participant_height', 'participant_weight',
                                        'participant_BMI', 'participant_blood_volume', 'participant_SBP_seated',
                                        'participant_SBP_stand', 'participant_SBP_exercise', 'participant_DBP_seated',
                                        'participant_DBP_stand', 'participant_DBP_exercise', 'participant_MAP_seated',
                                        'participant_MAP_stand', 'participant_MAP_exercise', 'participant_HR_seated',
                                        'participant_HR_stand', 'participant_HR_exercise', 'participant_max_leg_strength',
                                        'participant_largest_leg_circumference', 'participant_lower_leg_volume',
                                        'participant_skinfolds_chest_avg', 'participant_skinfolds_abd_avg',
                                        'participant_skinfolds_thigh_avg', 'participant_skinfolds_midax_avg',
                                        'participant_skinfolds_subscap_avg', 'participant_skinfolds_tri_avg',
                                        'participant_skinfolds_supra_avg', 'participant_skinfolds_sum',
                                        'participant_percent_fat', 'participant_leg_length', 'participant_arm_length',
                                        'participant_midline_neck_length', 'participant_lateral_neck_length',
                                        'participant_torso_length_post', 'participant_torso_length_ax', 'participant_head_to_heart',
                                        'participant_head_girth', 'participant_neck_girth', 'participant_chest_upper_girth',
                                        'participant_chest_under_girth', 'participant_waist_girth', 'participant_hip_girth',
                                        'participant_thigh_girth', 'participant_calf_girth', 'participant_biceps_girth_flex',
                                        'participant_biceps_girth_relax', 'participant_neck_flexion',
                                        'participant_neck_extension', 'participant_neck_right_rotation', 'participant_neck_left_rotation',
                                        'participant_neck_left_lat_flex', 'participant_neck_right_lat_flex',
                                        'participant_pred_vo2']
            demographics_concat = pd.DataFrame(participant_demographics, columns = demographics_names)

            # Append all demographic data to gloc data reduced
            gloc_data_reduced = pd.concat([gloc_data_reduced, demographics_concat], axis=1)
            return gloc_data_reduced, demographics_names

        def process_strain_data(gloc_data_reduced):
            ######### Add missing strain during GOR labels based on GLOC Effectiveness Spreadsheet #########
            # Add individual strain labels for trials where label was 'missed' (from column E of GLOC_Effectiveness spreadsheet)
            # Define trial & g magnitude variable
            gloc_trial = gloc_data_reduced['trial_id']
            magnitude_g = gloc_data_reduced['magnitude - Centrifuge'].to_numpy()
            event = gloc_data_reduced['event']

            return gloc_data_reduced, gloc_trial

            ######## Trial 04-06 (GLOC_Effectiveness stain value of 6.1g) ########
            trial_individual_coding = '04-06'
            g_level_strain = 6.1

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            #
            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 06-02 (GLOC_Effectiveness stain value of 8.1g) ########
            trial_individual_coding = '06-02'
            g_level_strain = 8.1

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 06-06 End GOR Label in Event Validated ########
            trial_individual_coding = '06-06'

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])

            ## Add missing 'end GOR' label
            # Find first nan in g magnitude post GOR peak
            return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]

            # Find the index of new end GOR label in full length csv
            end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index

            # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 07-06 (GLOC_Effectiveness stain value of 4.6g) ########
            trial_individual_coding = '07-06'
            g_level_strain = 4.6

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 08-02 (GLOC_Effectiveness stain value of 8.3g) ########
            trial_individual_coding = '08-02'
            g_level_strain = 8.3

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 08-05 (GLOC_Effectiveness stain value of 4.3g) ########
            trial_individual_coding = '08-05'
            g_level_strain = 4.3

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ## Add missing 'end GOR' label
            # Find first nan in g magnitude post GOR peak
            return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]

            # Find the index of new end GOR label in full length csv
            end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index

            # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 08-06 (GLOC_Effectiveness stain value of 8.2g) ########
            trial_individual_coding = '08-06'
            g_level_strain = 8.2

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 09-03 (GLOC_Effectiveness stain value of 8.7g) ########
            trial_individual_coding = '09-03'
            g_level_strain = 8.7

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 09-05 (GLOC_Effectiveness stain value of 4.8g) ########
            trial_individual_coding = '09-05'
            g_level_strain = 4.8

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 10-05 (GLOC_Effectiveness stain value of 3.8g) ########
            trial_individual_coding = '10-05'
            g_level_strain = 3.8

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 10-06 (GLOC_Effectiveness stain value of 5.5g) ########
            trial_individual_coding = '10-06'
            g_level_strain = 5.5

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]
            current_event = event[trial_index]

            #### Remove original label in event column ####
            # Find row containing the string
            mislabel_mask_strain = current_event.str.contains('strain during GOR')
            # mislabel_mask_end_GOR = current_event.str.contains('end GOR')

            # Get the index of that row
            mislabel_index_strain = current_event[mislabel_mask_strain].index
            # mislabel_index_end_GOR = current_event[mislabel_mask_end_GOR].index

            # Find the index of new strain label in full length csv
            strain_relabel_index = mislabel_index_strain
            # end_GOR_relabel_index = mislabel_index_end_GOR

            # gloc_data_reduced['event'][strain_relabel_index] = None
            gloc_data_reduced.loc[strain_relabel_index, 'event'] = None
            # # gloc_data_reduced['event'][end_GOR_relabel_index] = None
            # gloc_data_reduced.loc[end_GOR_relabel_index, 'event'] = None

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 11-02 (GLOC_Effectiveness stain value of 5.5g) ########
            trial_individual_coding = '11-02'
            g_level_strain = 5.5

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 12-03 (GLOC_Effectiveness stain value of 3.7g) ########
            trial_individual_coding = '12-03'
            g_level_strain = 3.7

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 13-02 (GLOC_Effectiveness stain value of 7.3g) ########
            trial_individual_coding = '13-02'
            g_level_strain = 7.3

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 13-04 (GLOC_Effectiveness stain value of 7.4g) ########
            trial_individual_coding = '13-04'
            g_level_strain = 7.4

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            return gloc_data_reduced, gloc_trial

        _, _, _, _, _, actual_all_features, actual_all_features_phys, actual_all_features_ecg, actual_all_features_eeg = get_actual_features(actual_gloc_data, feature_groups_to_analyze, file_paths["demographic"] , ("noAFE", "explicit"))

        # Testing all features similarity
        all_features_set = set(features["All"])
        actual_all_features_set = set(actual_all_features)
        assert all_features_set == actual_all_features_set, f"Missing features in actual_all_features: {all_features_set - actual_all_features_set}"

        # Testing physiological features similarity
        features_phys_set = set(features["Phys"])
        actual_all_features_phys_set = set(actual_all_features_phys)
        assert features_phys_set == actual_all_features_phys_set, f"Missing features in actual_all_features_phys: {features_phys_set - actual_all_features_phys_set}"

        # Testing ECG features similarity
        features_ecg_set = set(features["ECG"])
        actual_all_features_ecg_set = set(actual_all_features_ecg)
        assert features_ecg_set == actual_all_features_ecg_set, f"Missing features in actual_all_features_ecg: {features_ecg_set - actual_all_features_ecg_set}"

        # Testing EEG features similarity
        features_eeg_set = set(features["EEG"])
        actual_all_features_eeg_set = set(actual_all_features_eeg)
        assert features_eeg_set == actual_all_features_eeg_set, f"Missing features in actual_all_features_eeg: {features_eeg_set - actual_all_features_eeg_set}"

    def test_getting_feature_names_noAFE_implicit(self):
        manager = DataManager()
        file_paths = manager._get_data_locations()
        gloc_data = manager._load_data(file_paths)
        actual_gloc_data = manager._load_data(file_paths)

        feature_groups_to_analyze, _ = manager._get_feature_groups_and_baseline_methods(("noAFE", "Implicit"))

        gloc_data, features = manager._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, ("noAFE", "Implicit"), file_paths)

        def get_actual_features(gloc_data_reduced, feature_groups_to_analyze, demographic_data_filename, model_type):
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
                    # raw_eeg_condition_specific = []
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
                    # processed_eeg_condition_specific = []
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

        def pull_eeg_sets():
            # list of shared eeg channels
            raw_eeg_shared_features = ['Fz - EEG', 'F3 - EEG', 'C3 - EEG', 'C4 - EEG',
                                        'CP1 - EEG', 'CP2 - EEG', 'T8 - EEG', 'TP9 - EEG', 'TP10 - EEG',
                                        'P7 - EEG', 'P8 - EEG']

            processed_eeg_shared_features = ['Fz_delta - EEG', 'Fz_theta - EEG', 'Fz_alpha - EEG', 'Fz_beta - EEG',
                                            'F3_delta - EEG', 'F3_theta - EEG', 'F3_alpha - EEG', 'F3_beta - EEG',
                                            'C3_delta - EEG', 'C3_theta - EEG', 'C3_alpha - EEG', 'C3_beta - EEG',
                                            'C4_delta - EEG', 'C4_theta - EEG', 'C4_alpha - EEG', 'C4_beta - EEG',
                                            'CP1_delta - EEG', 'CP1_theta - EEG', 'CP1_alpha - EEG', 'CP1_beta - EEG',
                                            'CP2_delta - EEG', 'CP2_theta - EEG', 'CP2_alpha - EEG', 'CP2_beta - EEG',
                                            'T8_delta - EEG', 'T8_theta - EEG', 'T8_alpha - EEG', 'T8_beta - EEG',
                                            'TP9_delta - EEG', 'TP9_theta - EEG', 'TP9_alpha - EEG', 'TP9_beta - EEG',
                                            'TP10_delta - EEG', 'TP10_theta - EEG', 'TP10_alpha - EEG', 'TP10_beta - EEG',
                                            'P7_delta - EEG', 'P7_theta - EEG', 'P7_alpha - EEG', 'P7_beta - EEG',
                                            'P8_delta - EEG', 'P8_theta - EEG', 'P8_alpha - EEG', 'P8_beta - EEG']

            # list of AFE only eeg channels
            raw_eeg_afe_only = ['F4 - EEG', 'T7 - EEG', 'O1 - EEG', 'O2 - EEG']

            processed_eeg_afe_only =['F4_delta - EEG', 'F4_theta - EEG', 'F4_alpha - EEG', 'F4_beta - EEG',
                                                        'T7_delta - EEG', 'T7_theta - EEG', 'T7_alpha - EEG', 'T7_beta - EEG',
                                                        'O1_delta - EEG', 'O1_theta - EEG', 'O1_alpha - EEG', 'O1_beta - EEG',
                                                        'O2_delta - EEG', 'O2_theta - EEG', 'O2_alpha - EEG', 'O2_beta - EEG']
            # list of Non-AFE only eeg channels
            raw_eeg_nonafe_only = ['F1 - EEG', 'AFz - EEG', 'AF4 - EEG', 'FT9 - EEG', 'FT10 - EEG', 'FC5 - EEG',
                                            'FC3 - EEG', 'FC1 - EEG', 'FC2 - EEG', 'FC4 - EEG', 'FC6 - EEG', 'C5 - EEG',
                                            'Cz - EEG', 'CP5 - EEG', 'CP6 - EEG', 'P5 - EEG', 'P3 - EEG', 'P1 - EEG',
                                            'Pz - EEG', 'P4 - EEG', 'P6 - EEG']

            processed_eeg_nonafe_only =['F1_delta - EEG', 'F1_theta - EEG', 'F1_alpha - EEG', 'F1_beta - EEG',
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
                                                        'CP6_delta - EEG', 'CP6_theta - EEG', 'CP6_alpha - EEG','CP6_beta - EEG',
                                                        'P5_delta - EEG', 'P5_theta - EEG', 'P5_alpha - EEG', 'P5_beta - EEG',
                                                        'P3_delta - EEG', 'P3_theta - EEG', 'P3_alpha - EEG', 'P3_beta - EEG',
                                                        'P1_delta - EEG', 'P1_theta - EEG', 'P1_alpha - EEG', 'P1_beta - EEG',
                                                        'Pz_delta - EEG', 'Pz_theta - EEG', 'Pz_alpha - EEG', 'Pz_beta - EEG',
                                                        'P4_delta - EEG', 'P4_theta - EEG', 'P4_alpha - EEG', 'P4_beta - EEG',
                                                        'P6_delta - EEG', 'P6_theta - EEG', 'P6_alpha - EEG', 'P6_beta - EEG']

            return (processed_eeg_shared_features, processed_eeg_afe_only, processed_eeg_nonafe_only,
                    raw_eeg_shared_features, raw_eeg_afe_only, raw_eeg_nonafe_only)
        
        def read_and_process_demographics(demographic_data_filename, gloc_data_reduced):
            """
            This function imports the demographics spreadsheet and appends the demographics
            to the gloc_data_reduced variable. Each column represents one of the demographics
            variables below. The column is filled by determining the participant for a specific
            trial and filling the entire trial with that participants demographic data.
            """

            # Import demographics spreadsheet
            demographics = pd.read_csv(demographic_data_filename)
            demographics = demographics.astype({col: 'float32' for col in demographics.select_dtypes(include='float64').columns})
            demographics = demographics.copy()

            # Grab variables of interest
            participant_index = demographics['GLOC ID']                                                         # Corresponds to subject 1-13
            participant_gender = pd.Series(demographics['Gender code [1/0]'])        # 0 = Female, 1 = Male
            participant_age = pd.Series(demographics['Age [yr]'])
            participant_height = pd.Series(demographics['height [m]'])
            participant_weight = pd.Series(demographics['weight (kg)'])
            participant_BMI = pd.Series(demographics['BMI [kg/m^2]'])
            participant_blood_volume = pd.Series(demographics['Blood Volume [L]'])   # Based on Nadler's approximation
            # participant_PWV = pd.Series(demographics['Avg PWV'])                   # Pulse Wave Velocity
                # (no val for participant 4)
            # participant_PTT = demographics['Avg PTT [ms]'])                        # Pulse Transit Time
                # (no val for participant 4)
            participant_SBP_seated = pd.Series(demographics['Resting SBP (seat)'])   # Systolic Blood Pressure
            participant_SBP_stand = pd.Series(demographics['resting SBP (stand)'])
            participant_SBP_exercise = pd.Series(demographics['SBP after squat'])
            participant_DBP_seated = pd.Series(demographics['Resting DBP (seat)'] )  # Diastolic Blood Pressure
            participant_DBP_stand = pd.Series(demographics['resting DBP (stand)'])
            participant_DBP_exercise = pd.Series(demographics['DBP after squat'])
            participant_MAP_seated = pd.Series(demographics['Resting MAP'])          # Mean Arterial Pressure
            participant_MAP_stand = pd.Series(demographics['Resting MAP (stand'])
            participant_MAP_exercise = pd.Series(demographics['Post-Squat MAP'])
            participant_HR_seated = pd.Series(demographics['resting HR [seated]'])
            participant_HR_stand = pd.Series(demographics['resting HR (stand)'])
            participant_HR_exercise = pd.Series(demographics['HR after squat'])
            participant_max_leg_strength = pd.Series(demographics['Max (N)'])        # Max Leg Strength
            participant_largest_leg_circumference = pd.Series(demographics['largest leg circ. [cm]'])
            participant_lower_leg_volume = pd.Series(demographics['lower leg volume [mL]'])
            participant_skinfolds_chest_avg = pd.Series(demographics['chest avg'])   # Skin Folds to Approximate Body Fat %
            participant_skinfolds_abd_avg = pd.Series(demographics['abd avg'])
            participant_skinfolds_thigh_avg = pd.Series(demographics['thigh avg'])
            participant_skinfolds_midax_avg = pd.Series(demographics['midax avg'])
            participant_skinfolds_subscap_avg = pd.Series(demographics['subscap avg'])
            participant_skinfolds_tri_avg = pd.Series(demographics['tri avg'])
            participant_skinfolds_supra_avg = pd.Series(demographics['supra avg'])
            participant_skinfolds_sum = pd.Series(demographics['sum'])
            participant_percent_fat = pd.Series(demographics['% fat'])
            participant_leg_length = pd.Series(demographics['leg avg'])
            participant_arm_length = pd.Series(demographics['arm avg'])
            participant_midline_neck_length = pd.Series(demographics['neck (MNL) avg'])
            participant_lateral_neck_length = pd.Series(demographics['neck (LNL) avg'])
            participant_torso_length_post = pd.Series(demographics['torso (post) avg '])
            participant_torso_length_ax = pd.Series(demographics['torso (ax) avg '])
            participant_head_to_heart = pd.Series(demographics['head to heart avg'])
            participant_head_girth = pd.Series(demographics['head avg'])
            participant_neck_girth = pd.Series(demographics['neck avg'])
            participant_chest_upper_girth = pd.Series(demographics['chest upper avg'])
            participant_chest_under_girth = pd.Series(demographics['chest under avg'])
            participant_waist_girth = pd.Series(demographics['waist avg'])
            participant_hip_girth = pd.Series(demographics['hip avg'])
            participant_thigh_girth = pd.Series(demographics['thigh avg'])
            participant_calf_girth = pd.Series(demographics['calf avg'])
            participant_biceps_girth_flex = pd.Series(demographics['bicep flex avg'])
            participant_biceps_girth_relax = pd.Series(demographics['bicep relax avg'])
            participant_neck_flexion = pd.Series(demographics['avg (N) flexion'])
            participant_neck_extension = pd.Series(demographics['avg (N) extens'])
            participant_neck_right_rotation = pd.Series(demographics['avg (N) Rt. Rot'])
            participant_neck_left_rotation = pd.Series(demographics['avg (N) left rot'])
            participant_neck_left_lat_flex = pd.Series(demographics['avg (N) left lat flex'])
            participant_neck_right_lat_flex = pd.Series(demographics['avg (N) rt lat flex'])
            participant_pred_vo2 = pd.Series(demographics['pred. Vo2'])              # Predicted VO2

            # Concatenate all demographics of interest
            all_demographics = pd.concat([participant_gender, participant_age, participant_height, participant_weight,
                                        participant_BMI, participant_blood_volume, participant_SBP_seated,
                                        participant_SBP_stand, participant_SBP_exercise, participant_DBP_seated,
                                        participant_DBP_stand, participant_DBP_exercise, participant_MAP_seated,
                                        participant_MAP_stand, participant_MAP_exercise, participant_HR_seated,
                                        participant_HR_stand, participant_HR_exercise, participant_max_leg_strength,
                                        participant_largest_leg_circumference, participant_lower_leg_volume,
                                        participant_skinfolds_chest_avg, participant_skinfolds_abd_avg,
                                        participant_skinfolds_thigh_avg, participant_skinfolds_midax_avg,
                                        participant_skinfolds_subscap_avg, participant_skinfolds_tri_avg,
                                        participant_skinfolds_supra_avg, participant_skinfolds_sum,
                                        participant_percent_fat, participant_leg_length, participant_arm_length,
                                        participant_midline_neck_length, participant_lateral_neck_length,
                                        participant_torso_length_post, participant_torso_length_ax, participant_head_to_heart,
                                        participant_head_girth, participant_neck_girth, participant_chest_upper_girth,
                                        participant_chest_under_girth, participant_waist_girth, participant_hip_girth,
                                        participant_thigh_girth, participant_calf_girth, participant_biceps_girth_flex,
                                        participant_biceps_girth_relax, participant_neck_flexion,
                                        participant_neck_extension, participant_neck_right_rotation, participant_neck_left_rotation,
                                        participant_neck_left_lat_flex, participant_neck_right_lat_flex,
                                        participant_pred_vo2], axis=1)

            participant_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

            # Initialize variables
            participant_demographics = np.zeros((len(gloc_data_reduced), all_demographics.shape[1]))
            length_previous_participant_data = -1

            # Organize all participant data in array same length as gloc_data_reduced
            for i in range(len(participant_list)):
                length_current_participant_data = len(np.argwhere((gloc_data_reduced['subject'] == participant_list[i])))
                participant_demographics[length_previous_participant_data+1:length_previous_participant_data+1+length_current_participant_data, :] = all_demographics.iloc[i]
                length_previous_participant_data = length_current_participant_data + length_previous_participant_data

            # Create demographics data frame
            demographics_names = ['participant_gender', 'participant_age', 'participant_height', 'participant_weight',
                                        'participant_BMI', 'participant_blood_volume', 'participant_SBP_seated',
                                        'participant_SBP_stand', 'participant_SBP_exercise', 'participant_DBP_seated',
                                        'participant_DBP_stand', 'participant_DBP_exercise', 'participant_MAP_seated',
                                        'participant_MAP_stand', 'participant_MAP_exercise', 'participant_HR_seated',
                                        'participant_HR_stand', 'participant_HR_exercise', 'participant_max_leg_strength',
                                        'participant_largest_leg_circumference', 'participant_lower_leg_volume',
                                        'participant_skinfolds_chest_avg', 'participant_skinfolds_abd_avg',
                                        'participant_skinfolds_thigh_avg', 'participant_skinfolds_midax_avg',
                                        'participant_skinfolds_subscap_avg', 'participant_skinfolds_tri_avg',
                                        'participant_skinfolds_supra_avg', 'participant_skinfolds_sum',
                                        'participant_percent_fat', 'participant_leg_length', 'participant_arm_length',
                                        'participant_midline_neck_length', 'participant_lateral_neck_length',
                                        'participant_torso_length_post', 'participant_torso_length_ax', 'participant_head_to_heart',
                                        'participant_head_girth', 'participant_neck_girth', 'participant_chest_upper_girth',
                                        'participant_chest_under_girth', 'participant_waist_girth', 'participant_hip_girth',
                                        'participant_thigh_girth', 'participant_calf_girth', 'participant_biceps_girth_flex',
                                        'participant_biceps_girth_relax', 'participant_neck_flexion',
                                        'participant_neck_extension', 'participant_neck_right_rotation', 'participant_neck_left_rotation',
                                        'participant_neck_left_lat_flex', 'participant_neck_right_lat_flex',
                                        'participant_pred_vo2']
            demographics_concat = pd.DataFrame(participant_demographics, columns = demographics_names)

            # Append all demographic data to gloc data reduced
            gloc_data_reduced = pd.concat([gloc_data_reduced, demographics_concat], axis=1)
            return gloc_data_reduced, demographics_names

        def process_strain_data(gloc_data_reduced):
            ######### Add missing strain during GOR labels based on GLOC Effectiveness Spreadsheet #########
            # Add individual strain labels for trials where label was 'missed' (from column E of GLOC_Effectiveness spreadsheet)
            # Define trial & g magnitude variable
            gloc_trial = gloc_data_reduced['trial_id']
            magnitude_g = gloc_data_reduced['magnitude - Centrifuge'].to_numpy()
            event = gloc_data_reduced['event']

            return gloc_data_reduced, gloc_trial

            ######## Trial 04-06 (GLOC_Effectiveness stain value of 6.1g) ########
            trial_individual_coding = '04-06'
            g_level_strain = 6.1

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            #
            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 06-02 (GLOC_Effectiveness stain value of 8.1g) ########
            trial_individual_coding = '06-02'
            g_level_strain = 8.1

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 06-06 End GOR Label in Event Validated ########
            trial_individual_coding = '06-06'

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])

            ## Add missing 'end GOR' label
            # Find first nan in g magnitude post GOR peak
            return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]

            # Find the index of new end GOR label in full length csv
            end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index

            # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 07-06 (GLOC_Effectiveness stain value of 4.6g) ########
            trial_individual_coding = '07-06'
            g_level_strain = 4.6

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 08-02 (GLOC_Effectiveness stain value of 8.3g) ########
            trial_individual_coding = '08-02'
            g_level_strain = 8.3

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 08-05 (GLOC_Effectiveness stain value of 4.3g) ########
            trial_individual_coding = '08-05'
            g_level_strain = 4.3

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ## Add missing 'end GOR' label
            # Find first nan in g magnitude post GOR peak
            return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]

            # Find the index of new end GOR label in full length csv
            end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index

            # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 08-06 (GLOC_Effectiveness stain value of 8.2g) ########
            trial_individual_coding = '08-06'
            g_level_strain = 8.2

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 09-03 (GLOC_Effectiveness stain value of 8.7g) ########
            trial_individual_coding = '09-03'
            g_level_strain = 8.7

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 09-05 (GLOC_Effectiveness stain value of 4.8g) ########
            trial_individual_coding = '09-05'
            g_level_strain = 4.8

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 10-05 (GLOC_Effectiveness stain value of 3.8g) ########
            trial_individual_coding = '10-05'
            g_level_strain = 3.8

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 10-06 (GLOC_Effectiveness stain value of 5.5g) ########
            trial_individual_coding = '10-06'
            g_level_strain = 5.5

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]
            current_event = event[trial_index]

            #### Remove original label in event column ####
            # Find row containing the string
            mislabel_mask_strain = current_event.str.contains('strain during GOR')
            # mislabel_mask_end_GOR = current_event.str.contains('end GOR')

            # Get the index of that row
            mislabel_index_strain = current_event[mislabel_mask_strain].index
            # mislabel_index_end_GOR = current_event[mislabel_mask_end_GOR].index

            # Find the index of new strain label in full length csv
            strain_relabel_index = mislabel_index_strain
            # end_GOR_relabel_index = mislabel_index_end_GOR

            # gloc_data_reduced['event'][strain_relabel_index] = None
            gloc_data_reduced.loc[strain_relabel_index, 'event'] = None
            # # gloc_data_reduced['event'][end_GOR_relabel_index] = None
            # gloc_data_reduced.loc[end_GOR_relabel_index, 'event'] = None

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 11-02 (GLOC_Effectiveness stain value of 5.5g) ########
            trial_individual_coding = '11-02'
            g_level_strain = 5.5

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 12-03 (GLOC_Effectiveness stain value of 3.7g) ########
            trial_individual_coding = '12-03'
            g_level_strain = 3.7

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            ######## Trial 13-02 (GLOC_Effectiveness stain value of 7.3g) ########
            trial_individual_coding = '13-02'
            g_level_strain = 7.3

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            ######## Trial 13-04 (GLOC_Effectiveness stain value of 7.4g) ########
            trial_individual_coding = '13-04'
            g_level_strain = 7.4

            # Find indices associated with current trial
            trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
            magnitude_g_trial = magnitude_g[trial_index]

            # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
            gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
            magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
            closest_value_strain_index = np.nanargmin(magnitude_difference)

            # Find the index of new strain label in full length csv
            strain_label_index = trial_index.idxmax() + closest_value_strain_index

            # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
            gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

            # ## Add missing 'end GOR' label
            # # Find first nan in g magnitude post GOR peak
            # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
            #
            # # Find the index of new end GOR label in full length csv
            # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
            #
            # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

            return gloc_data_reduced, gloc_trial

        _, _, _, _, _, actual_all_features, actual_all_features_phys, actual_all_features_ecg, actual_all_features_eeg = get_actual_features(actual_gloc_data, feature_groups_to_analyze, file_paths["demographic"] , ("noAFE", "implicit"))

        # Testing all features similarity
        all_features_set = set(features["All"])
        actual_all_features_set = set(actual_all_features)
        assert all_features_set == actual_all_features_set, f"Missing features in actual_all_features: {all_features_set - actual_all_features_set}"

        # Testing physiological features similarity
        features_phys_set = set(features["Phys"])
        actual_all_features_phys_set = set(actual_all_features_phys)
        assert features_phys_set == actual_all_features_phys_set, f"Missing features in actual_all_features_phys: {features_phys_set - actual_all_features_phys_set}"

        # Testing ECG features similarity
        features_ecg_set = set(features["ECG"])
        actual_all_features_ecg_set = set(actual_all_features_ecg)
        assert features_ecg_set == actual_all_features_ecg_set, f"Missing features in actual_all_features_ecg: {features_ecg_set - actual_all_features_ecg_set}"

        # Testing EEG features similarity
        features_eeg_set = set(features["EEG"])
        actual_all_features_eeg_set = set(actual_all_features_eeg)
        assert features_eeg_set == actual_all_features_eeg_set, f"Missing features in actual_all_features_eeg: {features_eeg_set - actual_all_features_eeg_set}"

    def test_gloc_labeling(self):
        manager = DataManager()
        file_paths = manager._get_data_locations()
        gloc_data = manager._load_data(file_paths)

        feature_groups_to_analyze = {"ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE", "G", "processedEEG", "demographics", "strain"}
        model_type = ("Complete", "Explicit")
        gloc_data, _ = manager._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, model_type, file_paths)

        event_validated = gloc_data["event_validated"].to_numpy()
        trial_id = gloc_data["trial_id"].to_numpy()

        # Find indices where 'GLOC' and 'return to consciousness' occur
        gloc_indices = np.argwhere(event_validated == "GLOC")
        rtc_indices = np.argwhere(event_validated == "return to consciousness")

        expected_gloc_labels = np.zeros(event_validated.shape)
        for i in range(gloc_indices.shape[0]):
            # Check the index for gloc and return to consciousness occurs on the same trial
            if trial_id[gloc_indices[i]] == trial_id[rtc_indices[i]]:
                expected_gloc_labels[gloc_indices[i, 0]:rtc_indices[i, 0]] = 1

        gloc_labels = manager._label_gloc_events(gloc_data)

        assert np.array_equal(gloc_labels, expected_gloc_labels), "The GLOC labels do not match the expected labels based on event_validated and trial_id."

    def test_afe_subset(self):
        def afe_subset(model_type, gloc_data, all_features, gloc_labels):
            """
                Remove trials where there is atl east one data stream that is all NaN
                Also returns a NaN proportionality table that says for each trial, what prop are NaN for each data stream
            """

            if model_type[0] == 'AFE':
                cond = 1
            else:
                cond = 0

            # All features and subject trial info to be put into a reduced dataframe from gloc_data_reduced
            # add on 'condition' to always check | requires second '.any()' statement below in the condition
            all_features_with_ids = all_features + ['condition','subject','trial']
            reduced_data_frame = gloc_data[all_features_with_ids]

            rows_to_remove = []

            N = 0 # number of trials total
            M = 0 # number of trials with missing data streams
            for (subject, trial), group in reduced_data_frame.groupby(['subject', 'trial']):
                trial_data = reduced_data_frame[(reduced_data_frame['subject'] == subject) &
                                                (reduced_data_frame['trial'] == trial)]


                # Check if the chosen AFE condition is violated at all during the trial
                if trial_data['condition'].any().any() != cond:
                    # If so, add these indices to the list of rows to remove
                    rows_to_remove.append(trial_data.index)
                    M = M+1 # to be removed

                N = N+1 # count trials

            # Flatten list of indices and remove them from the DataFrame
            rows_to_remove = [item for sublist in rows_to_remove for item in sublist]

            # Get rid of rows in the DF and array
            gloc_data = gloc_data.drop(rows_to_remove)
            gloc_data = gloc_data.reset_index(drop = True)

            gloc_labels = np.delete(gloc_labels, rows_to_remove, axis=0)

            # Print NaN findings
            print("There are ", N - M, " trials that match the chosen AFE condition out of ", N,
                "trials. ")

            return gloc_data, gloc_labels

        manager = DataManager()
        file_paths = manager._get_data_locations()
        gloc_data = manager._load_data(file_paths)

        feature_groups_to_analyze = {"ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE", "G", "processedEEG", "demographics", "strain"}
        model_type = ("Complete", "Explicit")
        gloc_data, features = manager._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, model_type, file_paths)
        gloc_labels = manager._label_gloc_events(gloc_data)
        gloc_data, gloc_labels = manager._afe_subset(gloc_data, gloc_labels)

        actual_gloc_data, actual_gloc_labels = gloc_data.copy(), gloc_labels.copy()
        actual_gloc_data, actual_gloc_labels = afe_subset(model_type, gloc_data, features["All"], gloc_labels)

        assert gloc_data.equals(actual_gloc_data), "The gloc_data after afe_subset does not match the expected gloc_data."
        assert np.array_equal(gloc_labels, actual_gloc_labels), "The gloc_labels after afe_subset does not match the expected gloc_labels."

    def test_eeg_specific_imputation(self):
        def pull_eeg_sets():
            # list of shared eeg channels
            raw_eeg_shared_features = ['Fz - EEG', 'F3 - EEG', 'C3 - EEG', 'C4 - EEG',
                                        'CP1 - EEG', 'CP2 - EEG', 'T8 - EEG', 'TP9 - EEG', 'TP10 - EEG',
                                        'P7 - EEG', 'P8 - EEG']

            processed_eeg_shared_features = ['Fz_delta - EEG', 'Fz_theta - EEG', 'Fz_alpha - EEG', 'Fz_beta - EEG',
                                            'F3_delta - EEG', 'F3_theta - EEG', 'F3_alpha - EEG', 'F3_beta - EEG',
                                            'C3_delta - EEG', 'C3_theta - EEG', 'C3_alpha - EEG', 'C3_beta - EEG',
                                            'C4_delta - EEG', 'C4_theta - EEG', 'C4_alpha - EEG', 'C4_beta - EEG',
                                            'CP1_delta - EEG', 'CP1_theta - EEG', 'CP1_alpha - EEG', 'CP1_beta - EEG',
                                            'CP2_delta - EEG', 'CP2_theta - EEG', 'CP2_alpha - EEG', 'CP2_beta - EEG',
                                            'T8_delta - EEG', 'T8_theta - EEG', 'T8_alpha - EEG', 'T8_beta - EEG',
                                            'TP9_delta - EEG', 'TP9_theta - EEG', 'TP9_alpha - EEG', 'TP9_beta - EEG',
                                            'TP10_delta - EEG', 'TP10_theta - EEG', 'TP10_alpha - EEG', 'TP10_beta - EEG',
                                            'P7_delta - EEG', 'P7_theta - EEG', 'P7_alpha - EEG', 'P7_beta - EEG',
                                            'P8_delta - EEG', 'P8_theta - EEG', 'P8_alpha - EEG', 'P8_beta - EEG']

            # list of AFE only eeg channels
            raw_eeg_afe_only = ['F4 - EEG', 'T7 - EEG', 'O1 - EEG', 'O2 - EEG']

            processed_eeg_afe_only =['F4_delta - EEG', 'F4_theta - EEG', 'F4_alpha - EEG', 'F4_beta - EEG',
                                                        'T7_delta - EEG', 'T7_theta - EEG', 'T7_alpha - EEG', 'T7_beta - EEG',
                                                        'O1_delta - EEG', 'O1_theta - EEG', 'O1_alpha - EEG', 'O1_beta - EEG',
                                                        'O2_delta - EEG', 'O2_theta - EEG', 'O2_alpha - EEG', 'O2_beta - EEG']
            # list of Non-AFE only eeg channels
            raw_eeg_nonafe_only = ['F1 - EEG', 'AFz - EEG', 'AF4 - EEG', 'FT9 - EEG', 'FT10 - EEG', 'FC5 - EEG',
                                            'FC3 - EEG', 'FC1 - EEG', 'FC2 - EEG', 'FC4 - EEG', 'FC6 - EEG', 'C5 - EEG',
                                            'Cz - EEG', 'CP5 - EEG', 'CP6 - EEG', 'P5 - EEG', 'P3 - EEG', 'P1 - EEG',
                                            'Pz - EEG', 'P4 - EEG', 'P6 - EEG']

            processed_eeg_nonafe_only =['F1_delta - EEG', 'F1_theta - EEG', 'F1_alpha - EEG', 'F1_beta - EEG',
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
                                                        'CP6_delta - EEG', 'CP6_theta - EEG', 'CP6_alpha - EEG','CP6_beta - EEG',
                                                        'P5_delta - EEG', 'P5_theta - EEG', 'P5_alpha - EEG', 'P5_beta - EEG',
                                                        'P3_delta - EEG', 'P3_theta - EEG', 'P3_alpha - EEG', 'P3_beta - EEG',
                                                        'P1_delta - EEG', 'P1_theta - EEG', 'P1_alpha - EEG', 'P1_beta - EEG',
                                                        'Pz_delta - EEG', 'Pz_theta - EEG', 'Pz_alpha - EEG', 'Pz_beta - EEG',
                                                        'P4_delta - EEG', 'P4_theta - EEG', 'P4_alpha - EEG', 'P4_beta - EEG',
                                                        'P6_delta - EEG', 'P6_theta - EEG', 'P6_alpha - EEG', 'P6_beta - EEG']

            return (processed_eeg_shared_features, processed_eeg_afe_only, processed_eeg_nonafe_only,
                    raw_eeg_shared_features, raw_eeg_afe_only, raw_eeg_nonafe_only)

        def eeg_condition_impute(gloc_data_reduced, all_features_eeg, afe_indicator_column, verbose = True):
            """
                Ensures both AFE (1) and non-AFE (0) conditions have the same feature columns.
                Missing columns are imputed with mean values in gloc_data_reduced and reflected in feature arrays.

                Returns df, the imputed dataframe corresponding to gloc_data_reduced
                Returns updated features, features_phys, and features_eeg that are affected by these imputations
            """

            # Create masks for each condition
            df = gloc_data_reduced.copy()
            afe_mask = afe_indicator_column == 1
            nonafe_mask = afe_indicator_column == 0

            # Pull columns that need to be imputed for each type
            _, processed_eeg_afe_only, processed_eeg_nonafe_only, _, raw_eeg_afe_only, raw_eeg_nonafe_only = pull_eeg_sets()
            afe_only_cols = processed_eeg_afe_only + raw_eeg_afe_only
            nonafe_only_cols = processed_eeg_nonafe_only + raw_eeg_nonafe_only

            # Impute AFE-only columns for non-AFE rows
            for col in afe_only_cols:
                if col in all_features_eeg:
                    # Check if all values in this column for non-AFE rows are NaN
                    #if df.loc[nonafe_mask, col].isna().all():
                    mean_val = df.loc[afe_mask, col].mean(skipna=True)
                    n_missing = df.loc[nonafe_mask, col].isna().sum()
                    df.loc[nonafe_mask, col] = df.loc[nonafe_mask, col].fillna(mean_val)
                    if verbose:
                        print(f"Imputed {n_missing} values in '{col}' for non-AFE rows")

            #  Impute non-AFE-only columns for AFE rows
            for col in nonafe_only_cols:
                if col in  all_features_eeg:
                    # Check if all values in this column for AFE rows are NaN
                    #if df.loc[afe_mask, col].isna().all():
                    mean_val = df.loc[nonafe_mask, col].mean(skipna=True)
                    n_missing = df.loc[afe_mask, col].isna().sum()
                    df.loc[afe_mask, col] = df.loc[afe_mask, col].fillna(mean_val)
                    if verbose:
                        print(f"Imputed {n_missing} values in '{col}' for AFE rows")

            return df

        # Setup data
        manager = DataManager()
        file_paths = manager._get_data_locations()
        gloc_data = manager._load_data(file_paths)
        
        feature_groups_to_analyze = {"ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE", "G", "processedEEG", "demographics", "strain"}
        model_type = ("Complete", "Explicit")
        gloc_data, features = manager._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, model_type, file_paths)
        gloc_labels = manager._label_gloc_events(gloc_data)
        gloc_data, gloc_labels = manager._afe_subset(gloc_data, gloc_labels)



        # Setup the actual result
        actual_gloc_data = gloc_data.copy()

        # Compute AFE / NonAFE condition indicator column
        condition_idx = features["All"].index('condition')
        afe_indicator_column = actual_gloc_data.iloc[:, condition_idx]

        # Impute (using mean) the value of the missing channels for each AFE condition
        actual_gloc_data = eeg_condition_impute(actual_gloc_data, features["EEG"], afe_indicator_column, verbose = False)
        actual_gloc_data.rename(columns = {"condition": "AFE_indicator"}, inplace = True) # Rename condition column to AFE_indicator to maintain ordering of columns



        # Setup data to compare to the actual
        manager._eeg_specific_imputation(gloc_data, features)

        print(gloc_data.columns[264], actual_gloc_data.columns[264])
        pd.testing.assert_frame_equal(gloc_data, actual_gloc_data)
        assert gloc_data.equals(actual_gloc_data), "The gloc_data after eeg_specific_imputation does not match the expected gloc_data."