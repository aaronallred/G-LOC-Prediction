import pickle
import numpy as np
import pandas as pd
import os

def load_and_process_csv(filename, analysis_type, feature_to_analyze, time_variable, **kwargs):
    """
    This function first checks for a pickle file to import (much quicker than loading csv). If the
    .pkl does not exist, it will create that and open it the next time. Additionally, it creates
    arrays that are useful in data processing, and creates an array of relevant feature columns
    from the data file.
    """

    ############################################# File Import #############################################
    # pickle file name
    global ecg_features, br_features, temp_features, fnirs_features, eeg_features, eyetracking_features
    pickle_filename = filename[0:-1-3] + '.pkl'

    # Check if pickle exists, if not create it
    if not os.path.isfile(pickle_filename):
        # Load CSV
        gloc_data = pd.read_csv(filename)

        # Save pickle file
        with open(pickle_filename, "wb") as f:
            pickle.dump(gloc_data, f)
    else:
        # Load Pickle file
        with open(pickle_filename, 'rb') as f:
            gloc_data = pickle.load(f)

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

    # Create matrix of all features for data being analyzed
    features = gloc_data_reduced[all_features].to_numpy()

    return gloc_data_reduced, features, all_features

def baseline_features(baseline_window, gloc_data_reduced, features, time_variable):
    """
    This function baselines features for each trial being analyzed. The baseline window is specified
    in main. Each feature is defined by the average of the baseline feature for that trial.
    """

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
    """
    This function creates the engineered features and gloc labels for the data. This includes a
    sliding window mean for each of the features for each trial_id. The number of windows is
    determined from the specified stride, window size, and offset. The gloc label is determined
    by finding if there are any 1 GLOC labels within the window (at some offset from the engineered
    feature window). A dictionary for the engineered feature, engineered label, and number of windows
    is returned. These dictionaries are sorted by trial_id.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    sliding_window_mean = dict()
    gloc_window = dict()
    number_windows = dict()

    # Iterate through all unique trial_id
    for i in range(np.size(trial_id_in_data)):

        # Determine index from current trial_id
        current_index = (gloc_data_reduced['trial_id'] == trial_id_in_data[i])

        # Create time array based on current_index
        current_time = np.array(gloc_data_reduced[time_variable])
        time_trimmed = current_time[current_index]

        # Find end time for specific trial
        time_end = np.max(time_trimmed)

        # Determine number of windows
        number_windows_current = np.int32(((time_end - offset) // stride) - (window_size // stride - 1))

        # Pre-allocate arrays
        sliding_window_mean_current = np.zeros((number_windows_current, np.shape(feature_baseline[trial_id_in_data[i]])[1]))
        gloc_window_current = np.zeros((number_windows_current, 1))

        # Create trimmed gloc data for the specific
        gloc_trimmed = gloc[(gloc_data_reduced.trial_id == trial_id_in_data[i])]

        # Define iteration time
        time_iteration = time_start

        # Iterate through all windows to compute relevant parameters
        for j in range(number_windows_current):

            # Find index for current window
            time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))

            # Find feature for current window
            feature_window = feature_baseline[trial_id_in_data[i]][time_period_feature]

            # Take nanmean for the window (one value per column (feature))
            sliding_window_mean_current[j,:] = np.nanmean(feature_window, axis = 0, keepdims=True)

            # Find the offset time for G-LOC label
            time_period_gloc = (((time_iteration + offset) <= time_trimmed) &
                                (time_trimmed < (time_iteration + offset + window_size)))

            # Create engineered label set to 1 if any values in window are 1
            gloc_window_current[j] = np.any(gloc_trimmed[time_period_gloc])

            # Adjust iteration_time
            time_iteration = stride + time_iteration

        # Define dictionary item for trial_id
        sliding_window_mean[trial_id_in_data[i]] = sliding_window_mean_current
        gloc_window[trial_id_in_data[i]] = gloc_window_current
        number_windows[trial_id_in_data[i]] = number_windows_current

    return gloc_window, sliding_window_mean, number_windows

def unpack_dict(gloc_window, sliding_window_mean, number_windows):
    """
    This function unpacks the dictionary structure to create a large features matrix (X matrix) and
    labels matrix (y matrix) for all trials being analyzed. This function will become unnecessary if
    the data remains in dataframe or arrays (rather than a dictionary).
    """

    # Find Unique Trial ID
    trial_id_in_data = list(sliding_window_mean.keys())

    # Determine total length of new unpacked dictionary items
    total_rows = 0
    for i in range(np.size(trial_id_in_data)):
        total_rows += number_windows[trial_id_in_data[i]]

    # Pre-allocate
    x_feature_matrix = np.zeros((total_rows, np.shape(sliding_window_mean[trial_id_in_data[0]])[1]))
    y_gloc_labels = np.zeros((total_rows, 1))

    current_index = 0

    # Iterate through unique trial_id
    for i in range(np.size(trial_id_in_data)):
        num_rows = np.shape(sliding_window_mean[trial_id_in_data[i]])[0]

        # Set specific rows equal to the dictionary item corresponding to trial_id
        x_feature_matrix[current_index:num_rows+current_index, :] = sliding_window_mean[trial_id_in_data[i]]
        y_gloc_labels[current_index:num_rows+current_index, :] = gloc_window[trial_id_in_data[i]]
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

def load_and_process_pkl(filename, feature_to_analyze, time_variable):
    # Load CSV (can read just a chunk of the file)
    # chunksize = 10**3
    with open(filename, 'rb') as file:
        # Call load method to deserialze
        gloc_data = pickle.load(file)

    # Separate Subject/Trial Column
    trial_id = gloc_data['trial_id'].to_numpy().astype('str')
    trial_id = np.array(np.char.split(trial_id, '-').tolist())
    subject = trial_id[:,0]
    trial = trial_id[:,1]
    time = gloc_data[time_variable].to_numpy()
    feature = gloc_data[feature_to_analyze].to_numpy()

    return gloc_data, subject, trial, time, feature

def find_missing_values(gloc_data,feature_to_analyze):
    nan_indices = gloc_data.isna().stack()
    missing_heart_rate = gloc_data[gloc_data[feature_to_analyze].isna()]
    subject_ids_missing = missing_heart_rate['trial_id']