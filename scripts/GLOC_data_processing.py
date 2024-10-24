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
    if analysis_type == 0:  # One Trial / One Subject
        subject_to_analyze = kwargs['subject_to_analyze']
        trial_to_analyze = kwargs['trial_to_analyze']

        # Find data from subject & trial of interest
        gloc_data_reduced = gloc_data[
            (gloc_data['subject'] == subject_to_analyze) & (gloc_data['trial'] == trial_to_analyze)]

    elif analysis_type == 1:  # All Trials for One Subject
        subject_to_analyze = kwargs['subject_to_analyze']

        # Find data from subject of interest
        gloc_data_reduced = gloc_data[(gloc_data['subject'] == subject_to_analyze)]

    elif analysis_type == 2:  # All Trials for All Subjects
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
        ecg_features = ['HR (bpm) - Equivital', 'ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital',
                        'HR_instant - Equivital', 'HR_average - Equivital', 'HR_w_average - Equivital']
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
        eyetracking_features = ['Pupil position left X [HUCS mm] - Tobii',
                                'Pupil position left Y [HUCS mm] - Tobii',
                                'Pupil position left Z [HUCS mm] - Tobii',
                                'Pupil position right X [HUCS mm] - Tobii',
                                'Pupil position right Y [HUCS mm] - Tobii',
                                'Pupil position right Z [HUCS mm] - Tobii', 'Pupil diameter left [mm] - Tobii',
                                'Pupil diameter right [mm] - Tobii']
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
    feature_baseline_derivative = dict()

    for i in range(np.size(trial_id_in_data)):
        # Find time window
        index_window = ((gloc_data_reduced[time_variable] < baseline_window) & (
                    gloc_data_reduced.trial_id == trial_id_in_data[i]))
        time_index = (gloc_data_reduced.trial_id == trial_id_in_data[i])
        time_array = gloc_data_reduced[time_variable]

        # Find baseline average based on specified baseline window
        baseline_feature = np.mean(features[index_window], axis=0)

        # Divide features for that trial by baselined data
        feature_baseline[trial_id_in_data[i]] = np.array(
            features[(gloc_data_reduced.trial_id == trial_id_in_data[i])] / baseline_feature)

        # Compute derivative
        diff_feature_baseline = np.diff(feature_baseline[trial_id_in_data[i]])
        diff_time = np.diff(time_array[time_index])
        diff_time = np.append(np.nan, diff_time)
        diff_time = np.reshape(diff_time, (len(diff_time), 1))

        feature_baseline_derivative[trial_id_in_data[i]] = diff_feature_baseline / diff_time

    return feature_baseline

def tabulateNaN(feature_baseline, all_features):
    """
    This function tabulates NaN values for each feature for each trial.
    """

    # Find Unique Trial ID
    trial_id_in_data = list(feature_baseline.keys())

    # Initialize table
    NaN_count = np.zeros((len(trial_id_in_data), len(all_features)))
    NaN_prop = np.zeros((len(trial_id_in_data), len(all_features)))

    # Loop through dictionary values and count NaNs per trial/feature
    for i in range(len(trial_id_in_data)):
        NaN_count[i, :] = np.count_nonzero(np.isnan(feature_baseline[trial_id_in_data[i]]), axis=0, keepdims=True)
        NaN_prop[i, :] = NaN_count[i, :] / np.shape(feature_baseline[trial_id_in_data[i]])[0]

    # Output in Data Frame
    NaN_table = pd.DataFrame(NaN_count, columns=all_features, index=trial_id_in_data)
    NaN_proportion = pd.DataFrame(NaN_prop, columns=all_features, index=trial_id_in_data)

    NaN_rows = (NaN_proportion == 1).any(axis=1)
    number_NaN_rows = NaN_rows.values.sum()
    total_rows = NaN_proportion.shape[0]

    print("There are ", number_NaN_rows, " trials with all NaNs for at least one feature out of ", total_rows,
          "trials. ", total_rows - number_NaN_rows, " trials remaining.")

    return NaN_table, NaN_proportion

def unpack_dict(gloc_window, sliding_window_mean, number_windows, sliding_window_stddev, sliding_window_max,
                sliding_window_range):
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

    # Find number of columns
    num_cols = (np.shape(sliding_window_mean[trial_id_in_data[0]])[1] +
                np.shape(sliding_window_stddev[trial_id_in_data[0]])[1]
                + np.shape(sliding_window_max[trial_id_in_data[0]])[1] +
                np.shape(sliding_window_range[trial_id_in_data[0]])[1])

    # Pre-allocate
    x_feature_matrix = np.zeros((total_rows, num_cols))
    y_gloc_labels = np.zeros((total_rows, 1))

    current_index = 0

    # Iterate through unique trial_id
    for i in range(np.size(trial_id_in_data)):
        num_rows = np.shape(sliding_window_mean[trial_id_in_data[i]])[0]

        # Set specific rows equal to the dictionary item corresponding to trial_id
        x_feature_matrix[current_index:num_rows + current_index, :] = np.column_stack(
            (sliding_window_mean[trial_id_in_data[i]],
             sliding_window_stddev[trial_id_in_data[i]],
             sliding_window_max[trial_id_in_data[i]],
             sliding_window_range[trial_id_in_data[i]]))
        y_gloc_labels[current_index:num_rows + current_index, :] = gloc_window[trial_id_in_data[i]]
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

def summarize_performance_metrics(accuracy_logreg, accuracy_rf, accuracy_lda, accuracy_knn, accuracy_svm,
                                  accuracy_gb,
                                  precision_logreg, precision_rf, precision_lda, precision_knn, precision_svm,
                                  precision_gb,
                                  recall_logreg, recall_rf, recall_lda, recall_knn, recall_svm, recall_gb,
                                  f1_logreg, f1_rf, f1_lda, f1_knn, f1_svm, f1_gb):
    classifiers = ['Log Reg', 'RF', 'LDA', 'KNN', 'SVM', 'Ensemble w/ GB']
    performance_metrics = ['accuracy', 'precision', 'recall', 'f1-score']

    accuracy = np.array([accuracy_logreg, accuracy_rf, accuracy_lda, accuracy_knn, accuracy_svm, accuracy_gb])
    precision = np.array(
        [precision_logreg, precision_rf, precision_lda, precision_knn, precision_svm, precision_gb])
    recall = np.array([recall_logreg, recall_rf, recall_lda, recall_knn, recall_svm, recall_gb])
    f1 = np.array([f1_logreg, f1_rf, f1_lda, f1_knn, f1_svm, f1_gb])
    combined_metrics = np.column_stack((accuracy, precision, recall, f1))

    performance_metric_summary = pd.DataFrame(combined_metrics, index=classifiers, columns=performance_metrics)

    return performance_metric_summary

def unpack_dict_id(gloc_window, sliding_window_mean, number_windows, sliding_window_stddev, sliding_window_max,
                sliding_window_range):
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

    # Find number of columns
    num_cols = (np.shape(sliding_window_mean[trial_id_in_data[0]])[1] +
                np.shape(sliding_window_stddev[trial_id_in_data[0]])[1]
                + np.shape(sliding_window_max[trial_id_in_data[0]])[1] +
                np.shape(sliding_window_range[trial_id_in_data[0]])[1])

    # Pre-allocate
    x_feature_matrix = np.zeros((total_rows, num_cols+1))
    y_gloc_labels = np.zeros((total_rows, 1))

    current_index = 0

    # Iterate through unique trial_id
    for i in range(np.size(trial_id_in_data)):
        num_rows = np.shape(sliding_window_mean[trial_id_in_data[i]])[0]

        # Set specific rows equal to the dictionary item corresponding to trial_id
        x_feature_matrix[current_index:num_rows + current_index, 0:-1] = np.column_stack(
            (sliding_window_mean[trial_id_in_data[i]],
             sliding_window_stddev[trial_id_in_data[i]],
             sliding_window_max[trial_id_in_data[i]],
             sliding_window_range[trial_id_in_data[i]]))
        x_feature_matrix[current_index:num_rows + current_index, -1] = np.ones((num_rows,))*i
        y_gloc_labels[current_index:num_rows + current_index, :] = gloc_window[trial_id_in_data[i]]
        current_index += num_rows

    return y_gloc_labels, x_feature_matrix