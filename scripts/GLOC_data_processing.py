import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_process_csv(filename, analysis_type, feature_groups_to_analyze, demographic_data_filename, **kwargs):
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

        # Save pickle file
        gloc_data.to_pickle(pickle_filename)
    else:
        # Load Pickle file
        gloc_data = pd.read_pickle(pickle_filename)

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

    if 'AFE' in feature_groups_to_analyze:
        afe_features = ['condition']

        # Adjust columns of data frame for feature
        gloc_data_reduced['condition'].replace('N', 0, inplace=True)
        gloc_data_reduced['condition'].replace('AFE', 1, inplace=True)
    else:
        afe_features = []

    if 'G' in feature_groups_to_analyze:
        g_features = ['magnitude - Centrifuge']
    else:
        g_features = []

    if 'cognitive' in feature_groups_to_analyze:
        cognitive_features = ['deviation - Cog', 'RespTime - Cog', 'Correct - Cog', 'joystickPosMag - Cog', 'joystickVelMag - Cog'] #, 'tgtposX - Cog', 'tgtposY - Cog']

        # Adjust columns of data frame for feature
        gloc_data_reduced['Correct - Cog'].replace('correct', 1, inplace=True)
        gloc_data_reduced['Correct - Cog'].replace('no response', 0, inplace=True)
        gloc_data_reduced['Correct - Cog'].replace('incorrect', -1, inplace=True)
        gloc_data_reduced['Correct - Cog'].replace('NO VALUE', np.nan, inplace=True)

        ######### Generate additional cognitive task specific features #########
        # Deviation/Screen Pos
        deviation_wrt_target_position =  gloc_data_reduced['deviation - Cog'] / np.sqrt(gloc_data_reduced['tgtposX - Cog']**2 +  gloc_data_reduced['tgtposY - Cog']**2)
        gloc_data_reduced['Deviation wrt Target Position'] = deviation_wrt_target_position

        # append cognitive features
        cognitive_features.append('Deviation wrt Target Position')
    else:
        cognitive_features = []

    if 'rawEEG_shared' in feature_groups_to_analyze:
        raw_eeg_shared_features = ['F1 - EEG', 'Fz - EEG', 'F3 - EEG', 'C3 - EEG', 'C4 - EEG',
                        'CP1 - EEG', 'CP2 - EEG', 'T8 - EEG', 'TP9 - EEG', 'TP10 - EEG',
                        'P7 - EEG', 'P8 - EEG']
    else:
        raw_eeg_shared_features = []

    if 'processedEEG_shared' in feature_groups_to_analyze:
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
    else:
        processed_eeg_shared_features = []

    if 'strain' in feature_groups_to_analyze:
        strain_features = []

        ######### Add missing strain during GOR labels based on GLOC Effectiveness Spreadsheet #########
        # Add individual strain labels for trials where label was 'missed' (from column E of GLOC_Effectiveness spreadsheet)
        # Define trial & g magnitude variable
        gloc_trial = gloc_data_reduced['trial_id']
        magnitude_g = gloc_data_reduced['magnitude - Centrifuge'].to_numpy()

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

        ## Add missing 'end GOR' label
        # Find first nan in g magnitude post GOR peak
        first_nan_post_GOR_peak = np.argmax(magnitude_g_trial[gor_peak:-1])

        # Find the index of new end GOR label in full length csv
        end_GOR_label_index = trial_index.idxmax() + gor_peak + first_nan_post_GOR_peak

        gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

        ## Add missing 'end GOR' label
        # Find first nan in g magnitude post GOR peak
        first_nan_post_GOR_peak = np.argmax(magnitude_g_trial[gor_peak:-1])

        # Find the index of new end GOR label in full length csv
        end_GOR_label_index = trial_index.idxmax() + gor_peak + first_nan_post_GOR_peak

        gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

        ## Add missing 'end GOR' label
        # Find first nan in g magnitude post GOR peak
        first_nan_post_GOR_peak = np.argmax(magnitude_g_trial[gor_peak:-1])

        # Find the index of new end GOR label in full length csv
        end_GOR_label_index = trial_index.idxmax() + gor_peak + first_nan_post_GOR_peak

        gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

        ## Add missing 'end GOR' label
        # Find first nan in g magnitude post GOR peak
        first_nan_post_GOR_peak = np.argmax(magnitude_g_trial[gor_peak:-1])

        # Find the index of new end GOR label in full length csv
        end_GOR_label_index = trial_index.idxmax() + gor_peak + first_nan_post_GOR_peak

        gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

        ## Add missing 'end GOR' label
        # Find first nan in g magnitude post GOR peak
        first_nan_post_GOR_peak = np.argmax(magnitude_g_trial[gor_peak:-1])

        # Find the index of new end GOR label in full length csv
        end_GOR_label_index = trial_index.idxmax() + gor_peak + first_nan_post_GOR_peak

        gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

        ## Add missing 'end GOR' label
        # Find first nan in g magnitude post GOR peak
        first_nan_post_GOR_peak = np.argmax(magnitude_g_trial[gor_peak:-1])

        # Find the index of new end GOR label in full length csv
        end_GOR_label_index = trial_index.idxmax() + gor_peak + first_nan_post_GOR_peak

        gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'

        # fig, ax = plt.subplots()
        # current_time = np.array(gloc_data_reduced['Time (s)'])
        # time = current_time[trial_index]
        #
        # ax.plot(time, magnitude_g_trial)
        # plt.show()

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

        ## Add missing 'end GOR' label
        # Find first nan in g magnitude post GOR peak
        first_nan_post_GOR_peak = np.argmax(magnitude_g_trial[gor_peak:-1])

        # Find the index of new end GOR label in full length csv
        end_GOR_label_index = trial_index.idxmax() + gor_peak + first_nan_post_GOR_peak

        gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

        ## Add missing 'end GOR' label
        # Find first nan in g magnitude post GOR peak
        first_nan_post_GOR_peak = np.argmax(magnitude_g_trial[gor_peak:-1])

        # Find the index of new end GOR label in full length csv
        end_GOR_label_index = trial_index.idxmax() + gor_peak + first_nan_post_GOR_peak

        gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

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

        gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'

        ## Add missing 'end GOR' label
        # Find first nan in g magnitude post GOR peak
        first_nan_post_GOR_peak = np.argmax(magnitude_g_trial[gor_peak:-1])

        # Find the index of new end GOR label in full length csv
        end_GOR_label_index = trial_index.idxmax() + gor_peak + first_nan_post_GOR_peak

        gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'

        ######### Generate Strain specific features #########
        # Create Strain Vector
        event = gloc_data_reduced['event'].to_numpy()
        strain_event = np.zeros(event.shape)

        # Find labeled 'strain' and 'end GOR' markings in the event column
        strain_indices = np.argwhere(event == 'strain during GOR')
        end_GOR_indices = np.argwhere(event == 'end GOR')

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

    else:
        strain_features = []

    if 'demographics' in feature_groups_to_analyze:
        # Read Demographics Spreadsheet and Append to gloc_data_reduced
        [gloc_data_reduced, demographics_names] = read_and_process_demographics(demographic_data_filename, gloc_data_reduced)
        demographics_features = demographics_names
    else:
        demographics_features = []

    # Combine names of different feature categories for baseline methods
    all_features = (ecg_features + br_features + temp_features + fnirs_features + eyetracking_features + afe_features
                    + g_features + cognitive_features + raw_eeg_shared_features + processed_eeg_shared_features +
                    strain_features + demographics_features)
    all_features_phys = (ecg_features + br_features + temp_features + fnirs_features + eyetracking_features +
                         raw_eeg_shared_features + processed_eeg_shared_features)
    all_features_ecg = ecg_features

    # Create matrix of all features for data being analyzed
    features = gloc_data_reduced[all_features].to_numpy()
    features_phys = gloc_data_reduced[all_features_phys].to_numpy()
    features_ecg = gloc_data_reduced[all_features_ecg].to_numpy()

    return gloc_data_reduced, features, features_phys, features_ecg, all_features, all_features_phys, all_features_ecg

def label_gloc_events(gloc_data_reduced):
    """
    This function creates a g-loc label for the data based on the event_validated column. The event
    is labeled as 1 between GLOC and Return to Consciousness.
    """

    # Grab event validated column & convert to numpy array
    event_validated = gloc_data_reduced['event_validated'].to_numpy()

    # Find indices where 'GLOC' and 'return to consciousness' occur
    gloc_indices = np.argwhere(event_validated == 'GLOC')
    rtc_indices = np.argwhere(event_validated == 'return to consciousness')

    # Create GLOC Classifier Vector
    gloc_classifier = np.zeros(event_validated.shape)
    for i in range(gloc_indices.shape[0]):
        gloc_classifier[gloc_indices[i, 0]:rtc_indices[i, 0]] = 1

    return gloc_classifier

def read_and_process_demographics(demographic_data_filename, gloc_data_reduced):
    """
    This function imports the demographics spreadsheet and appends the demographics
    to the gloc_data_reduced variable. Each column represents one of the demographics
    variables below. The column is filled by determining the participant for a specific
    trial and filling the entire trial with that participants demographic data.
    """

    # Import demographics spreadsheet
    demographics = pd.read_csv(demographic_data_filename)

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

def combine_all_baseline(gloc_data_reduced, baseline, baseline_derivative, baseline_second_derivative, baseline_names):
    """
    This function combines the features, derivative of features, and second derivative of features into one np array.
    """
    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Iterate through all unique trial_id & combine the baseline, baseline derivative, and baseline second derivative
    combined_baseline = dict()
    for id in trial_id_in_data:
        all_baseline_data = []
        for method in baseline.keys():
            all_baseline_data.append(baseline[method][id])
            all_baseline_data.append(baseline_derivative[method][id])
            all_baseline_data.append(baseline_second_derivative[method][id])

        combined_baseline[id] = np.column_stack(tuple(all_baseline_data))

    combined_baseline_names = sum([baseline_names[method] + [s + '_derivative' for s in baseline_names[method]] +
                                       [s + '_2derivative' for s in baseline_names[method]] for method in baseline_names.keys()], [])

    return combined_baseline, combined_baseline_names

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

def unpack_dict(gloc_window, sliding_window_mean_s1, number_windows, sliding_window_stddev_s1,sliding_window_max_s1,
                sliding_window_range_s1,sliding_window_integral_left_pupil_s1,sliding_window_integral_right_pupil_s1,
                sliding_window_consecutive_elements_mean_left_pupil_s1, sliding_window_consecutive_elements_mean_right_pupil_s1,
                sliding_window_consecutive_elements_max_left_pupil_s1, sliding_window_consecutive_elements_max_right_pupil_s1,
                sliding_window_consecutive_elements_sum_left_pupil_s1, sliding_window_consecutive_elements_sum_right_pupil_s1,
                sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1, sliding_window_hrv_pnn50_s1, sliding_window_cognitive_IES_s1,
                sliding_window_mean_s2, sliding_window_stddev_s2, sliding_window_max_s2, sliding_window_range_s2,
                sliding_window_integral_left_pupil_s2, sliding_window_integral_right_pupil_s2,
                sliding_window_consecutive_elements_mean_left_pupil_s2, sliding_window_consecutive_elements_mean_right_pupil_s2,
                sliding_window_consecutive_elements_max_left_pupil_s2, sliding_window_consecutive_elements_max_right_pupil_s2,
                sliding_window_consecutive_elements_sum_left_pupil_s2, sliding_window_consecutive_elements_sum_right_pupil_s2,
                sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2, sliding_window_hrv_pnn50_s2, sliding_window_cognitive_IES_s2):
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
                                sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1, sliding_window_hrv_pnn50_s1, sliding_window_cognitive_IES_s1,
                                sliding_window_mean_s2, sliding_window_stddev_s2, sliding_window_max_s2,sliding_window_range_s2,
                                sliding_window_integral_left_pupil_s2, sliding_window_integral_right_pupil_s2,
                                sliding_window_consecutive_elements_mean_left_pupil_s2,sliding_window_consecutive_elements_mean_right_pupil_s2,
                                sliding_window_consecutive_elements_max_left_pupil_s2, sliding_window_consecutive_elements_max_right_pupil_s2,
                                sliding_window_consecutive_elements_sum_left_pupil_s2, sliding_window_consecutive_elements_sum_right_pupil_s2,
                                sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2, sliding_window_hrv_pnn50_s2, sliding_window_cognitive_IES_s2]

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
    x_feature_matrix = np.zeros((total_rows, num_cols))
    y_gloc_labels = np.zeros((total_rows, 1))

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
            column_index:np.shape(current_dictionary[trial_id_in_data[i]])[1] + column_index] = current_dictionary[trial_id_in_data[i]]

            # Increment column index
            column_index += np.shape(current_dictionary[trial_id_in_data[i]])[1]

        # Set corresponding gloc labels from current trial
        y_gloc_labels[current_index:num_rows+current_index, :] = gloc_window[trial_id_in_data[i]]

        # Increment row index
        current_index += num_rows

    return y_gloc_labels, x_feature_matrix

def process_NaN(y_gloc_labels, x_feature_matrix):
    """
    This is a temporary function for removing all rows with NaN values. This can be replaced by
    another method in the future, but is necessary for feeding into ML Classifiers.
    """

    # Find & Remove rows in X matrix if they have NaN values
    y_gloc_labels_noNaN = y_gloc_labels[~np.isnan(x_feature_matrix).any(axis=1)]

    # Find & Remove rows in label array if the features have any NaN values in that row
    x_feature_matrix_noNaN = x_feature_matrix[~np.isnan(x_feature_matrix).any(axis=1)]

    return y_gloc_labels_noNaN, x_feature_matrix_noNaN

def summarize_performance_metrics(accuracy_logreg, accuracy_rf, accuracy_lda, accuracy_knn, accuracy_svm, accuracy_gb,
                                  precision_logreg, precision_rf, precision_lda, precision_knn, precision_svm, precision_gb,
                                  recall_logreg, recall_rf, recall_lda, recall_knn, recall_svm, recall_gb,
                                  f1_logreg, f1_rf, f1_lda, f1_knn, f1_svm, f1_gb,
                                  specificity_logreg, specificity_rf, specificity_lda, specificity_knn, specificity_svm, specificity_gb):
    """
    This function takes the performance metrics from each classifier and outputs a data frame
    with a performance summary.
    """

    # Define classifiers being used and summary performance meetrics to use
    classifiers = ['Log Reg', 'RF', 'LDA', 'KNN', 'SVM' , 'Ensemble w/ GB']
    performance_metrics = ['accuracy', 'precision', 'recall', 'f1-score', 'specificity']

    # For each performance metric, combine each machine learning method into np array
    accuracy = np.array([accuracy_logreg, accuracy_rf, accuracy_lda, accuracy_knn, accuracy_svm, accuracy_gb])
    precision = np.array([precision_logreg, precision_rf, precision_lda, precision_knn, precision_svm, precision_gb])
    recall = np.array([recall_logreg, recall_rf, recall_lda, recall_knn, recall_svm, recall_gb])
    f1 = np.array([f1_logreg, f1_rf, f1_lda, f1_knn, f1_svm, f1_gb])
    specificity = np.array([specificity_logreg, specificity_rf, specificity_lda, specificity_knn, specificity_svm, specificity_gb])

    # create combined stack of all performance metrics
    combined_metrics = np.column_stack((accuracy, precision, recall, f1, specificity))

    # label combined metrics by classifier name and performance metric name
    performance_metric_summary = pd.DataFrame(combined_metrics, index = classifiers, columns = performance_metrics)

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

def tabulateNaN_raw(features, all_features, gloc, gloc_data_reduced):
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