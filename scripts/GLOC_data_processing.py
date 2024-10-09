import numpy as np
import pandas as pd
import os

def load_and_process_csv(filename, analysis_type, feature_to_analyze, time_variable, **kwargs):

    # pickle file name
    global ecg_features, br_features, temp_features, fnirs_features, eeg_features, eyetracking_features
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

        # Find data from subject of interest
        #subject_indices = gloc_data.index[gloc_data['subject'] == subject_to_analyze].tolist()

        # Find data from trials of interest
        #trial_indices = gloc_data.index[gloc_data['trial'] == trial_to_analyze].tolist()

        # Reduce gloc_data data frame to represent rows of interest
        #gloc_data = gloc_data[subject_indices & trial_indices]

    elif analysis_type == 1: # All Trials for One Subject
        subject_to_analyze = kwargs['subject_to_analyze']

        # Find data from subject of interest
        gloc_data_reduced = gloc_data[(gloc_data['subject'] == subject_to_analyze)]

    elif analysis_type == 2: # All Trials for All Subjects
        gloc_data_reduced = gloc_data

    # time = gloc_data_reduced[time_variable].to_numpy()

    # ECG ('HR (bpm) - Equivital','ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital', 'HR_instant - Equivital', 'HR_average - Equivital', 'HR_w_average - Equivital')
    # BR ('BR (rpm) - Equivital')
    # temp ('Skin Temperature - IR Thermometer (°C) - Equivital')
    # fnirs ('HbO2 - fNIRS', 'Hbd - fNIRS')
    # eyetracking ('Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii', 'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii'
            # 'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii', 'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii')

    # Get feature columns
    if 'ECG' in feature_to_analyze:
        ecg_features = ['HR (bpm) - Equivital', 'ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital', 'HR_instant - Equivital','HR_average - Equivital', 'HR_w_average - Equivital']
    else:
        ecg_features = []

    if 'BR' in feature_to_analyze:
       br_features = ['BR (rpm) - Equivital']
    else:
        br_features = []

    if 'temp' in feature_to_analyze:
        temp_features = ['Skin Temperature - IR Thermometer (°C) - Equivital']
    else:
        temp_features = []

    if 'fnirs' in feature_to_analyze:
        fnirs_features = ['HbO2 - fNIRS', 'Hbd - fNIRS']
    else:
        fnirs_features = []

    if 'eyetracking' in feature_to_analyze:
        eyetracking_features = ['Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii', 'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii',
            'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii', 'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii']
    else:
        eyetracking_features = []

    if 'EEG' in feature_to_analyze:
        eeg_features = []
    else:
        eeg_features = []
    all_features = ecg_features + br_features + temp_features + fnirs_features + eyetracking_features + eeg_features

    features = gloc_data_reduced[all_features].to_numpy()
    # g = gloc_data_reduced['magnitude - Centrifuge']

    return gloc_data_reduced, features, all_features

def baseline_features(baseline_window, gloc_data_reduced, features, time_variable):
    # Baseline Feature

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    feature_baseline = dict()
    for i in range(np.size(trial_id_in_data)):

            # Find baseline average based on specified baseline window
            baseline_feature = np.mean(features[(gloc_data_reduced[time_variable]<baseline_window) &
                                                (gloc_data_reduced.trial_id == trial_id_in_data[i])], axis = 0)

            # Divide features for that trial by baselined data
            feature_baseline[trial_id_in_data[i]] = np.array(features[(gloc_data_reduced.trial_id == trial_id_in_data[i])]/baseline_feature)

    return feature_baseline

def sliding_window_mean_calc(time_start, offset, stride, window_size, feature_baseline, gloc, gloc_data_reduced, time_variable):
    # Find sliding window mean for every feature

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    sliding_window_mean = dict()
    gloc_window = dict()

    for i in range(np.size(trial_id_in_data)):
        current_index = (gloc_data_reduced['trial_id'] == trial_id_in_data[i])
        current_time = np.array(gloc_data_reduced[time_variable])
        time_trimmed = current_time[current_index]

        time_end = np.max(time_trimmed)
        number_windows = np.int32(((time_end - offset) // stride) - (window_size // stride - 1))

        # Pre-allocate
        sliding_window_mean_current = np.zeros((number_windows, np.shape(feature_baseline[trial_id_in_data[i]])[1]))
        gloc_window_current = np.zeros((number_windows, 1))

        # Create trimmed gloc data for the specific
        gloc_trimmed = gloc[(gloc_data_reduced.trial_id == trial_id_in_data[i])]

        time_iteration = time_start

        for j in range(number_windows):

            time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))

            # temp fix: replace with zeros before mean, then re-fill with nans
            temp = feature_baseline[trial_id_in_data[i]][time_period_feature]

            sliding_window_mean_current[j,:] = np.nanmean(temp, axis = 0, keepdims=True)

            time_period_gloc = (((time_iteration + offset) <= time_trimmed) &
                                (time_trimmed < (time_iteration + offset + window_size)))

            gloc_window_current[j] = np.any(gloc_trimmed[time_period_gloc])

            time_iteration = stride + time_iteration

        sliding_window_mean[trial_id_in_data[i]] = sliding_window_mean_current
        gloc_window[trial_id_in_data[i]] = gloc_window_current

    return gloc_window, sliding_window_mean, number_windows