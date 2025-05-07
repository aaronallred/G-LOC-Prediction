import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def data_locations(datafolder):
    ## File Name & Path
    # Data CSV
    filename = os.path.join(datafolder,'all_trials_25_hz_stacked_null_str_filled.csv')

    # Baseline Data (HR)
    baseline_data_filename = os.path.join(datafolder,'ParticipantBaseline.csv')

    # Modified Demographic Data (put in order of participant 1-13, removed excess calculations, and converted from .xlsx to .csv)
    demographic_data_filename = os.path.join(datafolder,'GLOC_Effectiveness_Final.csv')

    # Input GOR EEG data from separate files
    list_of_eeg_data_files = [os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_01_DC1_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_01_DC2_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_01_DC3_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_02_DC1_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_02_DC2_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_02_DC3_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_03_DC1_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_03_DC2_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_03_DC3_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_04_DC1_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_04_DC2_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_04_DC3_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_05_DC1_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_05_DC2_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_05_DC3_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_06_DC1_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_06_DC4_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_06_DC6_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_07_DC2_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_07_DC4_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_07_DC6_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_08_DC1_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_08_DC3_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_09_DC2_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_09_DC5_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_09_DC6_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_10_DC2_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_10_DC4_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_10_DC5_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_11_DC1_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_12_DC1_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_12_DC5_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_13_DC1_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_13_DC3_25Hz_EEG_power_wMAR.xlsx'),
                              os.path.join(datafolder,'GLOC_GOR_EEG_data_participants_1-13/GLOC_13_DC6_25Hz_EEG_power_wMAR.xlsx')]

    # Input baseline EEG data from separate files
    list_of_baseline_eeg_processed_files = [os.path.join(datafolder,'GLOC_EEG_baseline_delta_noAFE1.csv'),
                                            os.path.join(datafolder,'GLOC_EEG_baseline_theta_noAFE1.csv'),
                                            os.path.join(datafolder,'GLOC_EEG_baseline_alpha_noAFE1.csv'),
                                            os.path.join(datafolder,'GLOC_EEG_baseline_beta_noAFE1.csv')]

    return filename, baseline_data_filename, demographic_data_filename, list_of_eeg_data_files, list_of_baseline_eeg_processed_files

def analysis_driven_csv_processing(analysis_type, filename, feature_groups_to_analyze, demographic_data_filename,
                                   model_type,list_of_eeg_data_files,trial_to_analyze,subject_to_analyze):

    # Process CSV
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

def load_and_process_csv(filename, analysis_type, feature_groups_to_analyze, demographic_data_filename, model_type, list_of_eeg_data_files, **kwargs):
    """
    This function first checks for a pickle file to import (much quicker than loading csv). If the
    .pkl does not exist, it will create that and open it the next time. Additionally, it creates
    an array of relevant feature columns from the data file.
    """

    ############################################# File Import #############################################
    # pickle file name
    pickle_filename = filename[0:-1-3] + '.pkl'

    # Check if pickle exists, if not create it
    if not os.path.isfile(pickle_filename):
        # Load CSV
        gloc_data = pd.read_csv(filename)
        gloc_data = gloc_data.astype({col: 'float32' for col in gloc_data.select_dtypes(include='float64').columns})
        gloc_data = gloc_data.copy()

        # Save pickle file
        gloc_data.to_pickle(pickle_filename)
    else:
        # Load Pickle file
        gloc_data = pd.read_pickle(pickle_filename)


    # Slot in GOR EEG data from other files
    gloc_data = process_EEG_GOR(list_of_eeg_data_files, gloc_data)
    gloc_data = gloc_data.astype({col: 'float32' for col in gloc_data.select_dtypes(include='float64').columns})
    gloc_data = gloc_data.copy()

    ############################################# Data Processing #############################################
    # Separate Subject/Trial Column
    trial_id = gloc_data['trial_id'].to_numpy().astype('str')
    trial_id = np.array(np.char.split(trial_id, '-').tolist())
    subject = trial_id[:, 0]
    trial = trial_id[:, 1]

    # Add new subject & trial columns to gloc_data data frame
    gloc_data['subject'] = pd.Series(subject, index=gloc_data.index)
    gloc_data['trial'] = pd.Series(trial, index=gloc_data.index)

    # Analyze only section of gloc_data specified using analysis_type
    if analysis_type == 0: # One Trial / One Subject
        subject_to_analyze = kwargs['subject_to_analyze']
        trial_to_analyze = kwargs['trial_to_analyze']

        # Find data from subject & trial of interest
        gloc_data_reduced = gloc_data[(gloc_data['subject'] == subject_to_analyze) & (gloc_data['trial'] == trial_to_analyze)]

    elif analysis_type == 1: # All Trials for One Subject
        subject_to_analyze = kwargs['subject_to_analyze']

        # Find data from subject of interest
        gloc_data_reduced = gloc_data[(gloc_data['subject'] == subject_to_analyze)]

    elif analysis_type == 2: # All Trials for All Subjects
        gloc_data_reduced = gloc_data

    #############################################   Features   #############################################
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
        raw_eeg_shared_features = ['F1 - EEG', 'Fz - EEG', 'F3 - EEG', 'C3 - EEG', 'C4 - EEG',
                                   'CP1 - EEG', 'CP2 - EEG', 'T8 - EEG', 'TP9 - EEG', 'TP10 - EEG',
                                   'P7 - EEG', 'P8 - EEG']
        if 'AFE' in model_type:
            raw_eeg_condition_specific = ['F4 - EEG', 'T7 - EEG', 'O1 - EEG', 'O2 - EEG']
        elif 'noAFE' in model_type:
            raw_eeg_condition_specific = ['AFz - EEG', 'AF4 - EEG', 'FT9 - EEG', 'FT10 - EEG', 'FC5 - EEG',
                                      'FC3 - EEG', 'FC1 - EEG', 'FC2 - EEG', 'FC4 - EEG', 'FC6 - EEG', 'C5 - EEG',
                                      'Cz - EEG', 'CP5 - EEG', 'CP6 - EEG', 'P5 - EEG', 'P3 - EEG', 'P1 - EEG',
                                      'Pz - EEG', 'P4 - EEG', 'P6 - EEG']
    else:
        raw_eeg_shared_features = []
        raw_eeg_condition_specific = []

    if 'processedEEG' in feature_groups_to_analyze:
        processed_eeg_shared_features = ['F1_delta - EEG', 'F1_theta - EEG', 'F1_alpha - EEG', 'F1_beta - EEG',
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
                                   'P8_delta - EEG', 'P8_theta - EEG', 'P8_alpha - EEG', 'P8_beta - EEG']
        if 'AFE' in model_type:
            processed_eeg_condition_specific = ['F4_delta - EEG', 'F4_theta - EEG', 'F4_alpha - EEG', 'F4_beta - EEG',
                                                'T7_delta - EEG', 'T7_theta - EEG', 'T7_alpha - EEG', 'T7_beta - EEG',
                                                'O1_delta - EEG', 'O1_theta - EEG', 'O1_alpha - EEG', 'O1_beta - EEG',
                                                'O2_delta - EEG', 'O2_theta - EEG', 'O2_alpha - EEG', 'O2_beta - EEG']
        elif 'noAFE' in model_type:
            processed_eeg_condition_specific = ['AFz_delta - EEG', 'AFz_theta - EEG', 'AFz_alpha - EEG', 'AFz_beta - EEG',
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

def label_gloc_events(gloc_data_reduced):
    """
    This function creates a g-loc label for the data based on the event_validated column. The event
    is labeled as 1 between GLOC and Return to Consciousness.
    """

    # Grab event validated column & convert to numpy array
    event_validated = gloc_data_reduced['event_validated'].to_numpy()

    # Grab trial_id column & convert to numpy array
    trial_id = gloc_data_reduced['trial_id'].to_numpy()

    # Find indices where 'GLOC' and 'return to consciousness' occur
    gloc_indices = np.argwhere(event_validated == 'GLOC')
    rtc_indices = np.argwhere(event_validated == 'return to consciousness')

    # Create GLOC Classifier Vector
    gloc_classifier = np.zeros(event_validated.shape)
    for i in range(gloc_indices.shape[0]):
        # Check the index for gloc and return to consciousness occurs on the same trial
        if trial_id[gloc_indices[i]] == trial_id[rtc_indices[i]]:
            gloc_classifier[gloc_indices[i, 0]:rtc_indices[i, 0]] = 1

    return gloc_classifier

def process_EEG_GOR(list_of_eeg_data_files, gloc_data):
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
            current_trial_data.loc[index_begin_GOR[0]:index_end_GOR_eeg[0], modified_name_delta] =  eeg_dict_delta[current_key][column_name].astype(np.float32)

            # current_trial_data[modified_name_theta][index_begin_GOR[0]:index_end_GOR_eeg[0]] = eeg_dict_theta[current_key][column_name]
            current_trial_data.loc[index_begin_GOR[0]:index_end_GOR_eeg[0], modified_name_theta] = eeg_dict_theta[current_key][column_name].astype(np.float32)

            # current_trial_data[modified_name_alpha][index_begin_GOR[0]:index_end_GOR_eeg[0]] = eeg_dict_alpha[current_key][column_name]
            current_trial_data.loc[index_begin_GOR[0]:index_end_GOR_eeg[0], modified_name_alpha] = eeg_dict_alpha[current_key][column_name].astype(np.float32)

            # current_trial_data[modified_name_beta][index_begin_GOR[0]:index_end_GOR_eeg[0]] = eeg_dict_beta[current_key][column_name]
            current_trial_data.loc[index_begin_GOR[0]:index_end_GOR_eeg[0], modified_name_beta] = eeg_dict_beta[current_key][column_name].astype(np.float32)

        # Replace previously empty processed EEG data with current_trial_data
        gloc_data[gloc_data['trial_id'] == current_key] = current_trial_data

    return gloc_data

def process_strain_data(gloc_data_reduced):
    ######### Add missing strain during GOR labels based on GLOC Effectiveness Spreadsheet #########
    # Add individual strain labels for trials where label was 'missed' (from column E of GLOC_Effectiveness spreadsheet)
    # Define trial & g magnitude variable
    gloc_trial = gloc_data_reduced['trial_id']
    magnitude_g = gloc_data_reduced['magnitude - Centrifuge'].to_numpy()
    event = gloc_data_reduced['event']

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

def tabulateNaN(feature_baseline, all_features, gloc, gloc_data_reduced):
    """
    This function tabulates NaN values for each feature for each trial.
    """

    # Find Unique Trial ID
    trial_id_in_data = list(feature_baseline.keys())

    # Initialize NaN variables
    NaN_count = np.zeros((len(trial_id_in_data), len(all_features)))
    NaN_prop = np.zeros((len(trial_id_in_data), len(all_features)))
    NaN_gloc = np.zeros((len(trial_id_in_data), 1))

    # Loop through dictionary values and count NaNs per trial/feature
    sum_gloc_trials = 0
    for i in range(len(trial_id_in_data)):

        # Count number of NaNs
        NaN_count[i,:] = np.count_nonzero(pd.isna(feature_baseline[trial_id_in_data[i]]), axis=0, keepdims=True)

        # Calculate proportion of trial that are NaNs
        NaN_prop[i,:] = NaN_count[i,:] / np.shape(feature_baseline[trial_id_in_data[i]])[0]

        # Create trimmed gloc data to count number of GLOC trials corresponding to NaN
        gloc_trimmed = gloc[(gloc_data_reduced.trial_id == trial_id_in_data[i])]
        no_gloc = np.count_nonzero(gloc_trimmed == 1) == 0

        # Parse all columns to find if any values are NaN for that row, if they are
        # set NaN_index for that column equal to True
        NaN_index = np.any(pd.isna(feature_baseline[trial_id_in_data[i]]), axis = 1)

        # Set NaN_gloc variable based on if NaNs occur during a gloc trial
        if no_gloc:
            NaN_gloc[i, :] = np.nan
        else:
            # Find proportion of gloc time steps that have a Nan in at least one column
            NaN_gloc[i,:] = np.count_nonzero(((gloc_trimmed == 1) & (NaN_index == True))) / np.count_nonzero(gloc_trimmed == 1)
            sum_gloc_trials += 1

    # Output in Data Frame
    NaN_table = pd.DataFrame(NaN_count, columns = all_features, index = trial_id_in_data)
    NaN_proportion = pd.DataFrame(NaN_prop, columns = all_features, index = trial_id_in_data)
    NaN_gloc_proportion = pd.DataFrame(NaN_gloc, index = trial_id_in_data)

    # Sum all NaN rows
    NaN_rows = (NaN_proportion == 1).any(axis = 1)
    number_NaN_rows = NaN_rows.values.sum()

    # Sum all NaN GLOC rows
    NaN_gloc_rows = NaN_gloc_proportion == 1
    number_NaN_gloc_rows = NaN_gloc_rows.values.sum()

    # Total number of trials
    total_rows = NaN_proportion.shape[0]

    # Print NaN findings
    print("There are ", number_NaN_rows, " trials with all NaNs for at least one feature out of ", total_rows, "trials. ", total_rows - number_NaN_rows, " trials remaining.")
    print("There are ", number_NaN_gloc_rows, " trials with all NaNs during GLOC out of ", sum_gloc_trials, "trials with GLOC. ")
    return NaN_table, NaN_proportion, NaN_gloc_proportion

def unpack_dict(gloc_window, sliding_window_mean_s1, number_windows, sliding_window_stddev_s1, sliding_window_max_s1,
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

def process_NaN(y_gloc_labels, x_feature_matrix, all_features, trials):
    """
    This is a temporary function for removing all rows with NaN values. This can be replaced by
    another method in the future, but is necessary for feeding into ML Classifiers.
    """
    # Find & remove columns if they have all NaN values
    nan_test = np.isnan(x_feature_matrix)
    index_column_all_NaN = np.all(nan_test, axis=0)
    x_feature_matrix_noNaN_cols = x_feature_matrix[:, ~index_column_all_NaN]

    # Adjust all_features to only include columns that don't have all NaN
    all_features = [all_features[i] for i in range(len(all_features)) if ~index_column_all_NaN[i]]

    # Find & Remove rows in label array if they have NaN values
    y_gloc_labels_noNaN = y_gloc_labels[~np.isnan(x_feature_matrix_noNaN_cols).any(axis=1)]

    # Find & Remove rows in trial array if they have NaN values
    trials_noNaN = trials[~np.isnan(x_feature_matrix_noNaN_cols).any(axis=1)]

    # Find & Remove rows in X matrix if the features have any NaN values in that row
    x_feature_matrix_noNaN = x_feature_matrix_noNaN_cols[~np.isnan(x_feature_matrix_noNaN_cols).any(axis=1)]

    return y_gloc_labels_noNaN, x_feature_matrix_noNaN, all_features, trials_noNaN

def remove_constant_columns(x_feature_matrix_noNaN, all_features):
    """
    This function removes all constant columns before feeding into the ML classifiers.
    """
    # Find all constant columns
    constant_columns = np.all(x_feature_matrix_noNaN == x_feature_matrix_noNaN[0,:], axis = 0)

    # Remove all constant columns from data frame
    x_feature_matrix_noNaN = x_feature_matrix_noNaN[:, ~constant_columns]

    all_features = [all_features[i] for i in range(len(all_features)) if ~constant_columns[i]]

    return x_feature_matrix_noNaN, all_features

def summarize_performance_metrics(accuracy_logreg, accuracy_rf, accuracy_lda, accuracy_knn, accuracy_svm, accuracy_gb,
                                  precision_logreg, precision_rf, precision_lda, precision_knn, precision_svm, precision_gb,
                                  recall_logreg, recall_rf, recall_lda, recall_knn, recall_svm, recall_gb,
                                  f1_logreg, f1_rf, f1_lda, f1_knn, f1_svm, f1_gb,
                                  specificity_logreg, specificity_rf, specificity_lda, specificity_knn, specificity_svm, specificity_gb,
                                  g_mean_logreg, g_mean_rf, g_mean_lda, g_mean_knn, g_mean_svm, g_mean_gb):
    """
    This function takes the performance metrics from each classifier and outputs a data frame
    with a performance summary.
    """

    # Define classifiers being used and summary performance meetrics to use
    classifiers = ['Log Reg', 'RF', 'LDA', 'KNN', 'SVM' , 'Ensemble w/ GB']
    performance_metrics = ['accuracy', 'precision', 'recall', 'f1-score', 'specificity', 'g mean']

    # For each performance metric, combine each machine learning method into np array
    accuracy = np.array([accuracy_logreg, accuracy_rf, accuracy_lda, accuracy_knn, accuracy_svm, accuracy_gb])
    precision = np.array([precision_logreg, precision_rf, precision_lda, precision_knn, precision_svm, precision_gb])
    recall = np.array([recall_logreg, recall_rf, recall_lda, recall_knn, recall_svm, recall_gb])
    f1 = np.array([f1_logreg, f1_rf, f1_lda, f1_knn, f1_svm, f1_gb])
    specificity = np.array([specificity_logreg, specificity_rf, specificity_lda, specificity_knn, specificity_svm, specificity_gb])
    g_mean = np.array([g_mean_logreg, g_mean_rf, g_mean_lda, g_mean_knn, g_mean_svm, g_mean_gb])

    # create combined stack of all performance metrics
    combined_metrics = np.column_stack((accuracy, precision, recall, f1, specificity, g_mean))

    # label combined metrics by classifier name and performance metric name
    performance_metric_summary = pd.DataFrame(combined_metrics, index = classifiers, columns = performance_metrics)

    return performance_metric_summary

def single_classifier_performance_summary(accuracy, precision, recall,  f1, specificity, g_mean, classifier=['']):
    metrics = ['accuracy', 'precision', 'recall', 'f1-score', 'specificity', 'g mean']

    # create combined stack of all performance metrics
    performance_metrics = np.column_stack((accuracy, precision, recall, f1, specificity, g_mean))

    # label combined metrics by classifier name and performance metric name
    performance_metric_summary = pd.DataFrame(performance_metrics, index = classifier, columns=metrics)

    return performance_metric_summary

def find_prediction_window(gloc_data_reduced, gloc, time_variable):
    """
    This function that finds the Loss of Consciousness Induction Time (LOCINDTI). This function
    also generates a histogram that summarizes the LOCINDTI findings along with plots of
    post ROR onset acceleration and GLOC.
    """

    # Find unique trial ids in data
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Initialize max prediction offset
    max_prediction_offset = np.zeros(len(trial_id_in_data))

    # Iterate through trials to find time after ROR onset to LOC
    for i in range(len(trial_id_in_data)):

        # Indices of current trial
        current_index = gloc_data_reduced['trial_id'] == trial_id_in_data[i]

        # Pull time and acceleration from gloc_data_reduced
        time = gloc_data_reduced[time_variable]
        accel = gloc_data_reduced['magnitude - Centrifuge']

        # Index time, acceleration, and gloc to match current trial
        current_time = np.array(time[current_index])
        current_accel = np.array(accel[current_index])
        current_gloc = gloc[current_index]

        # Individual coding for trial 7, 8, and 19, where ROR is later than the other trials
        if i == 7 or i == 8 or i == 19:
            ror = [m for m in range(len(current_time)) if current_time[m] >= 600]
        else:
            ror = [t for t in range(len(current_time)) if current_time[t] >= 400]

        # Shorten time, acceleration, and gloc variables to be post-ROR onset
        reduced_time = current_time[ror]
        reduced_accel = current_accel[ror]
        reduced_gloc = current_gloc[ror]

        # Plot acceleration and gloc post ROR onset
        fig, ax = plt.subplots()
        ax.plot(reduced_time, reduced_accel)
        ax.plot(reduced_time, reduced_gloc)
        plt.show()

        # Find first instance of GLOC in trial
        gloc_vals = [j for j in range(len(reduced_gloc)) if reduced_gloc[j] == 1]
        if len(gloc_vals) == 0:
            gloc_index = np.nan
        else:
            gloc_index = gloc_vals[0]

        # Find when ROR onset happens (currently have it hard-coded to be when acceleration exceeds 1.3g)
        accel_increase = [k for k in range(len(reduced_accel)) if reduced_accel[k] >= 1.3]
        if len(accel_increase) == 0:
            accel_index = np.nan
        else:
            accel_index = accel_increase[0]

        # When GLOC occurs, find time post ROR onset to GLOC (similar to LOCINDTI)
        if (np.isnan(accel_index)) | (np.isnan(gloc_index)):
            max_prediction_offset[i] = np.nan

        else:
            max_prediction_offset[i] = reduced_time[gloc_index] - reduced_time[accel_index]

    # Creating a customized histogram with a density plot
    sns.histplot(max_prediction_offset)

    # Adding labels and title
    plt.xlabel('LOCINDTI')
    plt.ylabel('Number of Trials')
    plt.title('Time Prior to LOC post Acceleration')

    # Display the plot
    plt.show()

    # Find Mean, Median, Max, Min, Range, Std. Dev
    mean_locindti = np.nanmean(max_prediction_offset)
    median_locindti = np.nanmedian(max_prediction_offset)
    max_locindti = np.nanmax(max_prediction_offset)
    min_locindti = np.nanmin(max_prediction_offset)
    range_locindti = max_locindti - min_locindti
    stddev_locindti = np.nanstd(max_prediction_offset)

def process_NaN_raw(gloc, features, gloc_data_reduced):
    """
    This is a temporary function for removing all rows with NaN values. This can be replaced by
    another method in the future, but is necessary for feeding into ML Classifiers.
    """

    # Find & Remove rows in X matrix if they have NaN values
    features_noNaN = features[~np.isnan(features).any(axis=1)]
    gloc_data_reduced_noNaN = gloc_data_reduced[~np.isnan(features).any(axis=1)]

    # Find & Remove rows in label array if the features have any NaN values in that row
    gloc_noNaN = gloc[~np.isnan(features).any(axis=1)]

    return gloc_noNaN, features_noNaN, gloc_data_reduced_noNaN

def remove_all_nan_trials(gloc_data_reduced,all_features,
                          features,features_phys, features_ecg, features_eeg, gloc):
    """
        Remove trials where there is atl east one data stream that is all NaN
        Also returns a NaN proportionality table that says for each trial, what prop are NaN for each data stream
    """

    # All features and subject trial info to be put into a reduced dataframe from gloc_data_reduced
    all_features_with_ids = all_features + ['subject','trial']
    reduced_data_frame = gloc_data_reduced[all_features_with_ids]

    rows_to_remove = []
    nan_proportion_table = []

    N = 0 # number of trials total
    M = 0 # number of trials with missing data streams
    for (subject, trial), group in reduced_data_frame.groupby(['subject', 'trial']):
        trial_data = reduced_data_frame[(reduced_data_frame['subject'] == subject) &
                                        (reduced_data_frame['trial'] == trial)]

        # Compute proportion of NaN values for each feature
        nan_proportions = trial_data[all_features].isna().mean().to_dict()

        # Store subject-trial and NaN proportions
        nan_proportion_table.append({'subject-trial': f"{subject}-{trial}", **nan_proportions})

        # Check if any of the columns in the trial data are entirely NaN
        if trial_data[all_features].isna().all().any():
            # If so, add these indices to the list of rows to remove
            rows_to_remove.append(trial_data.index)
            M = M+1 # count missing trials
        N = N+1 # count trials

    # Flatten list of indices and remove them from the DataFrame
    rows_to_remove = [item for sublist in rows_to_remove for item in sublist]

    # Convert from a dict to a DF
    nan_proportion_df = pd.DataFrame(nan_proportion_table)

    # Get rid of rows in the DF and array
    gloc_data_reduced = gloc_data_reduced.drop(rows_to_remove)
    gloc_data_reduced = gloc_data_reduced.reset_index(drop=True)

    features = np.delete(features, rows_to_remove, axis=0)
    features_phys = np.delete(features_phys, rows_to_remove, axis=0)
    features_ecg = np.delete(features_ecg, rows_to_remove, axis=0)
    features_eeg = np.delete(features_eeg, rows_to_remove, axis=0)
    gloc = np.delete(gloc, rows_to_remove, axis=0)

    # Print NaN findings
    print("There are ", M, " trials with all NaNs for at least one feature out of ", N,
          "trials. ", N - M, " trials remaining.")

    return gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc, nan_proportion_df

def afe_subset(model_type, gloc_data_reduced,all_features,
               features,features_phys, features_ecg, features_eeg,gloc):
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
    reduced_data_frame = gloc_data_reduced[all_features_with_ids]

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
    gloc_data_reduced = gloc_data_reduced.drop(rows_to_remove)
    gloc_data_reduced = gloc_data_reduced.reset_index(drop=True)

    features = np.delete(features, rows_to_remove, axis=0)
    features_phys = np.delete(features_phys, rows_to_remove, axis=0)
    features_ecg = np.delete(features_ecg, rows_to_remove, axis=0)
    features_eeg = np.delete(features_eeg, rows_to_remove, axis=0)
    gloc = np.delete(gloc, rows_to_remove, axis=0)

    # Print NaN findings
    print("There are ", N - M, " trials that match the chosen AFE condition out of ", N,
          "trials. ")

    return gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc

def read_csv_float32(filepath, preview_rows=100, **kwargs):
    """
    Previews the first few rows of a CSV and reads it with float32 dtype.

    Parameters:
    - filepath (str): Path to the CSV file.
    - preview_rows (int): Number of rows to preview. Default is 5.
    - **kwargs: Additional keyword arguments passed to pd.read_csv()

    Returns:
    - df (DataFrame): The loaded DataFrame with float32 for float columns.
    """

    preview = pd.read_csv(filepath, nrows=preview_rows, **kwargs)

    print_preview = 0
    if print_preview:
        # Preview the file
        print("Preview of the CSV file:")
        print(preview)

    # Identify float columns in the preview
    float_cols = preview.select_dtypes(include=['float64', 'float']).columns
    dtype_mapping = {col: 'float32' for col in float_cols}

    # Merge user-provided dtypes with float32 overrides (float32 takes precedence)
    user_dtypes = kwargs.get("dtype", {})
    if isinstance(user_dtypes, dict):
        dtype_mapping.update(user_dtypes)

    # Read full CSV with combined dtype mapping
    df = pd.read_csv(filepath, dtype=dtype_mapping, **kwargs)

    return df

def process_swp_pkl():
    with open("C:\\Users\\nicol\\Downloads\\Sequential_Optimization_Sliding_Window_dictionary_model_type_explicit_KNN.pkl", 'rb') as file:
        knn_pkl = pickle.load(file)

    column_names = list(knn_pkl[next(iter(data))].columns)

    # Flatten and convert each value to a 1D NumPy array
    flattened_rows = [np.array(v).reshape(-1) for v in knn_pkl.values()]  # Converts from DataFrame -> array -> (6,)

    # Stack into a 2D NumPy array
    data_array = np.vstack(flattened_rows)

    # Create the DataFrame with custom row labels
    knn_results = pd.DataFrame(data_array, index=knn_pkl.keys(), columns=column_names)

    with open("C:\\Users\\nicol\\Downloads\\Sequential_Optimization_Sliding_Window_dictionary_model_type_explicit_LDA.pkl", 'rb') as file:
        lda_pkl = pickle.load(file)

    column_names = list(lda_pkl[next(iter(data))].columns)

    # Flatten and convert each value to a 1D NumPy array
    flattened_rows = [np.array(v).reshape(-1) for v in lda_pkl.values()]  # Converts from DataFrame -> array -> (6,)

    # Stack into a 2D NumPy array
    data_array = np.vstack(flattened_rows)

    # Create the DataFrame with custom row labels
    lda_results = pd.DataFrame(data_array, index=lda_pkl.keys(), columns=column_names)

    with open("C:\\Users\\nicol\\Downloads\\Sequential_Optimization_Sliding_Window_dictionary_model_type_explicit_logreg.pkl", 'rb') as file:
        logreg_pkl = pickle.load(file)

    column_names = list(logreg_pkl[next(iter(data))].columns)

    # Flatten and convert each value to a 1D NumPy array
    flattened_rows = [np.array(v).reshape(-1) for v in logreg_pkl.values()]  # Converts from DataFrame -> array -> (6,)

    # Stack into a 2D NumPy array
    data_array = np.vstack(flattened_rows)

    # Create the DataFrame with custom row labels
    logreg_results = pd.DataFrame(data_array, index=logreg_pkl.keys(), columns=column_names)

    with open("C:\\Users\\nicol\\Downloads\\Sequential_Optimization_Sliding_Window_dictionary_model_type_explicit_rf.pkl", 'rb') as file:
        rf_pkl = pickle.load(file)

    column_names = list(rf_pkl[next(iter(data))].columns)

    # Flatten and convert each value to a 1D NumPy array
    flattened_rows = [np.array(v).reshape(-1) for v in rf_pkl.values()]  # Converts from DataFrame -> array -> (6,)

    # Stack into a 2D NumPy array
    data_array = np.vstack(flattened_rows)

    # Create the DataFrame with custom row labels
    rf_results = pd.DataFrame(data_array, index=rf_pkl.keys(), columns=column_names)

    with open("C:\\Users\\nicol\\Downloads\\Sequential_Optimization_Sliding_Window_dictionary_model_type_explicit_egb_fast.pkl", 'rb') as file:
        egb_fast_pkl = pickle.load(file)

    column_names = list(egb_fast_pkl[next(iter(data))].columns)

    # Flatten and convert each value to a 1D NumPy array
    flattened_rows = [np.array(v).reshape(-1) for v in egb_fast_pkl.values()]  # Converts from DataFrame -> array -> (6,)

    # Stack into a 2D NumPy array
    data_array = np.vstack(flattened_rows)

    # Create the DataFrame with custom row labels
    egb_fast_results = pd.DataFrame(data_array, index=egb_fast_pkl.keys(), columns=column_names)

def convert_to_unique_ordered_integers(strings):
    mapping = {}
    result = []
    current_id = 1
    for s in strings:
        if s not in mapping:
            mapping[s] = current_id
            current_id += 1
        result.append(mapping[s])

    return np.array(result,dtype=np.float32).reshape(-1, 1)