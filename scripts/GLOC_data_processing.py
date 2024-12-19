import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_process_csv(filename, analysis_type, feature_to_analyze, time_variable, **kwargs):
    """
    This function first checks for a pickle file to import (much quicker than loading csv). If the
    .pkl does not exist, it will create that and open it the next time. Additionally, it creates
    arrays that are useful in data processing, and creates an array of relevant feature columns
    from the data file.
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

        ######### Generate Additional fnirs specific features #########
        # HbO2/Hbd
        ox_deox_ratio = gloc_data_reduced['HbO2 - fNIRS'] / gloc_data_reduced['Hbd - fNIRS']
        gloc_data_reduced['HbO2 / Hbd'] = ox_deox_ratio

        # augment fnirs_features
        fnirs_features.append('HbO2 / Hbd')
    else:
        fnirs_features = []

    if 'eyetracking' in feature_to_analyze:
        eyetracking_features = ['Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii', 'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii',
            'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii', 'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii']

        ######### Generate Additional pupil specific features #########
        # Pupil Difference
        pupil_difference = gloc_data_reduced['Pupil diameter left [mm] - Tobii'] - gloc_data_reduced['Pupil diameter right [mm] - Tobii']
        gloc_data_reduced['Pupil Difference [mm]'] = pupil_difference

        # augment eyetracking_features
        eyetracking_features.append('Pupil Difference [mm]')

    else:
        eyetracking_features = []

    if 'AFE' in feature_to_analyze:
        afe_features = ['condition']

        # Adjust columns of data frame for feature
        gloc_data_reduced['condition'].replace('N', 0, inplace=True)
        gloc_data_reduced['condition'].replace('AFE', 1, inplace=True)
    else:
        afe_features = []

    if 'G' in feature_to_analyze:
        g_features = ['magnitude - Centrifuge']
    else:
        g_features = []

    if 'cognitive' in feature_to_analyze:
        cognitive_features = ['deviation - Cog', 'RespTime - Cog', 'Correct - Cog', 'joystickVelMag - Cog'] #, 'tgtposX - Cog', 'tgtposY - Cog']

        # Adjust columns of data frame for feature
        gloc_data_reduced['Correct - Cog'].replace('correct', 1, inplace=True)
        gloc_data_reduced['Correct - Cog'].replace('no response', 0, inplace=True)
        gloc_data_reduced['Correct - Cog'].replace('incorrect', -1, inplace=True)
        gloc_data_reduced['Correct - Cog'].replace('NO VALUE', np.nan, inplace=True)
    else:
        cognitive_features = []

    if 'EEG' in feature_to_analyze:
        eeg_features = []
    else:
        eeg_features = []

    if 'strain' in feature_to_analyze:
        strain_features = []
    else:
        strain_features = []

    # Combine names of different feature categories for baseline methods
    all_features = ecg_features + br_features + temp_features + fnirs_features + eyetracking_features + afe_features + g_features + cognitive_features + eeg_features + strain_features
    all_features_phys = ecg_features + br_features + temp_features + fnirs_features + eyetracking_features + eeg_features
    all_features_ecg = ecg_features

    # Create matrix of all features for data being analyzed
    features = gloc_data_reduced[all_features].to_numpy()
    features_phys = gloc_data_reduced[all_features_phys].to_numpy()
    features_ecg = gloc_data_reduced[all_features_ecg].to_numpy()

    return gloc_data_reduced, features, features_phys, features_ecg, all_features, all_features_phys, all_features_ecg

def combine_all_baseline(gloc_data_reduced, features_v0, features_v0_derivative, features_v0_second_derivative,
                                             features_v1, features_v1_derivative, features_v1_second_derivative,
                                             features_v2, features_v2_derivative, features_v2_second_derivative,
                                             features_v3, features_v3_derivative, features_v3_second_derivative,
                                             features_v4, features_v4_derivative, features_v4_second_derivative,
                                             features_v5, features_v5_derivative, features_v5_second_derivative,
                                             features_v6, features_v6_derivative, features_v6_second_derivative,
                                             baseline_names_v0, baseline_names_v1, baseline_names_v2, baseline_names_v3,
                                             baseline_names_v4, baseline_names_v5, baseline_names_v6):
    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    combined_baseline = dict()

    # Iterate through all unique trial_id
    for i in range(np.size(trial_id_in_data)):
        combined_baseline[trial_id_in_data[i]] = np.column_stack((features_v0[trial_id_in_data[i]], features_v0_derivative[trial_id_in_data[i]], features_v0_second_derivative[trial_id_in_data[i]],
                                                                  features_v1[trial_id_in_data[i]], features_v1_derivative[trial_id_in_data[i]], features_v1_second_derivative[trial_id_in_data[i]],
                                                                  features_v2[trial_id_in_data[i]], features_v2_derivative[trial_id_in_data[i]], features_v2_second_derivative[trial_id_in_data[i]],
                                                                  features_v3[trial_id_in_data[i]], features_v3_derivative[trial_id_in_data[i]], features_v3_second_derivative[trial_id_in_data[i]],
                                                                  features_v4[trial_id_in_data[i]], features_v4_derivative[trial_id_in_data[i]], features_v4_second_derivative[trial_id_in_data[i]],
                                                                  features_v5[trial_id_in_data[i]], features_v5_derivative[trial_id_in_data[i]], features_v5_second_derivative[trial_id_in_data[i]],
                                                                  features_v6[trial_id_in_data[i]], features_v6_derivative[trial_id_in_data[i]], features_v6_second_derivative[trial_id_in_data[i]]))

    combined_baseline_names = (baseline_names_v0 + [s + '_derivative' for s in baseline_names_v0] + [s + '_2derivative' for s in baseline_names_v0] +
                               baseline_names_v1 + [s + '_derivative' for s in baseline_names_v1] + [s + '_2derivative' for s in baseline_names_v1] +
                               baseline_names_v2 + [s + '_derivative' for s in baseline_names_v2] + [s + '_2derivative' for s in baseline_names_v2] +
                               baseline_names_v3 + [s + '_derivative' for s in baseline_names_v3] + [s + '_2derivative' for s in baseline_names_v3] +
                               baseline_names_v4 + [s + '_derivative' for s in baseline_names_v4] + [s + '_2derivative' for s in baseline_names_v4] +
                               baseline_names_v5 + [s + '_derivative' for s in baseline_names_v5] + [s + '_2derivative' for s in baseline_names_v5] +
                               baseline_names_v6 + [s + '_derivative' for s in baseline_names_v6] + [s + '_2derivative' for s in baseline_names_v6])

    return combined_baseline, combined_baseline_names

def tabulateNaN(feature_baseline, all_features, gloc, gloc_data_reduced):
    """
    This function tabulates NaN values for each feature for each trial.
    """

    # Find Unique Trial ID
    trial_id_in_data = list(feature_baseline.keys())

    # Initialize table
    NaN_count = np.zeros((len(trial_id_in_data), len(all_features)))
    NaN_prop = np.zeros((len(trial_id_in_data), len(all_features)))
    NaN_gloc = np.zeros((len(trial_id_in_data), 1))

    # Loop through dictionary values and count NaNs per trial/feature
    sum_gloc_trials = 0
    for i in range(len(trial_id_in_data)):

        NaN_index = np.zeros((len(feature_baseline[trial_id_in_data[i]]), 1))

        NaN_count[i,:] = np.count_nonzero(pd.isna(feature_baseline[trial_id_in_data[i]]), axis=0, keepdims=True)
        NaN_prop[i,:] = NaN_count[i,:] / np.shape(feature_baseline[trial_id_in_data[i]])[0]

        # Create trimmed gloc data to count number of GLOC trials corresponding to NaN
        gloc_trimmed = gloc[(gloc_data_reduced.trial_id == trial_id_in_data[i])]
        NaN_index = np.any(pd.isna(feature_baseline[trial_id_in_data[i]]), axis = 1)
        if (np.count_nonzero(gloc_trimmed == 1) == 0):
            NaN_gloc[i, :] = np.nan
        else:
            NaN_gloc[i,:] = np.count_nonzero(((gloc_trimmed == 1) & (NaN_index == True))) / np.count_nonzero(gloc_trimmed == 1)
            sum_gloc_trials = sum_gloc_trials + 1

    # Output in Data Frame
    NaN_table = pd.DataFrame(NaN_count, columns = all_features, index = trial_id_in_data)
    NaN_proportion = pd.DataFrame(NaN_prop, columns = all_features, index = trial_id_in_data)
    NaN_gloc_proportion = pd.DataFrame(NaN_gloc, index = trial_id_in_data)

    NaN_rows = (NaN_proportion == 1).any(axis = 1)
    number_NaN_rows = NaN_rows.values.sum()

    NaN_gloc_rows = NaN_gloc_proportion == 1
    number_NaN_gloc_rows = NaN_gloc_rows.values.sum()

    total_rows = NaN_proportion.shape[0]

    print("There are ", number_NaN_rows, " trials with all NaNs for at least one feature out of ", total_rows, "trials. ", total_rows - number_NaN_rows, " trials remaining.")
    print("There are ", number_NaN_gloc_rows, " trials with all NaNs during GLOC out of ", sum_gloc_trials, "trials with GLOC. ")
    return NaN_table, NaN_proportion, NaN_gloc_proportion

def unpack_dict(gloc_window, sliding_window_mean, number_windows, sliding_window_stddev,
                                                  sliding_window_max, sliding_window_range,
                                                  sliding_window_integral_left_pupil,sliding_window_integral_right_pupil,
                                                  sliding_window_consecutive_elements_mean_left_pupil, sliding_window_consecutive_elements_mean_right_pupil,
                                                  sliding_window_consecutive_elements_max_left_pupil, sliding_window_consecutive_elements_max_right_pupil,
                                                  sliding_window_consecutive_elements_sum_left_pupil, sliding_window_consecutive_elements_sum_right_pupil,
                                                  sliding_window_hrv_sdnn, sliding_window_hrv_rmssd):
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
    num_cols = ((np.shape(sliding_window_mean[trial_id_in_data[0]])[1] + np.shape(sliding_window_stddev[trial_id_in_data[0]])[1]
                + np.shape(sliding_window_max[trial_id_in_data[0]])[1] + np.shape(sliding_window_range[trial_id_in_data[0]])[1]
                + np.shape(sliding_window_pupil_difference[trial_id_in_data[0]])[1]
                + np.shape(sliding_window_ox_deox_ratio[trial_id_in_data[0]])[1] + np.shape(sliding_window_integral_left_pupil[trial_id_in_data[0]])[1]
                + np.shape(sliding_window_integral_right_pupil[trial_id_in_data[0]])[1] + np.shape(sliding_window_consecutive_elements_mean_left_pupil[trial_id_in_data[0]])[1]
                + np.shape(sliding_window_consecutive_elements_mean_right_pupil[trial_id_in_data[0]])[1] + np.shape(sliding_window_consecutive_elements_max_left_pupil[trial_id_in_data[0]])[1]
                + np.shape(sliding_window_consecutive_elements_max_right_pupil[trial_id_in_data[0]])[1] + np.shape(sliding_window_consecutive_elements_sum_left_pupil[trial_id_in_data[0]])[1]
                + np.shape(sliding_window_consecutive_elements_sum_right_pupil[trial_id_in_data[0]])[1] + np.shape(sliding_window_hrv_sdnn[trial_id_in_data[0]])[1]
                + np.shape(sliding_window_hrv_rmssd[trial_id_in_data[0]])[1]))

    # Pre-allocate
    x_feature_matrix = np.zeros((total_rows, num_cols))
    y_gloc_labels = np.zeros((total_rows, 1))

    current_index = 0

    # Iterate through unique trial_id
    for i in range(np.size(trial_id_in_data)):
        num_rows = np.shape(sliding_window_mean[trial_id_in_data[i]])[0]

        # Set specific rows equal to the dictionary item corresponding to trial_id
        x_feature_matrix[current_index:num_rows+current_index, :] = np.column_stack((sliding_window_mean[trial_id_in_data[i]],
                                                                                     sliding_window_stddev[trial_id_in_data[i]],
                                                                                     sliding_window_max[trial_id_in_data[i]],
                                                                                     sliding_window_range[trial_id_in_data[i]],
                                                                                     sliding_window_pupil_difference[trial_id_in_data[i]],
                                                                                     sliding_window_ox_deox_ratio[trial_id_in_data[i]],
                                                                                     sliding_window_integral_left_pupil[trial_id_in_data[i]],
                                                                                     sliding_window_integral_right_pupil[trial_id_in_data[i]],
                                                                                     sliding_window_consecutive_elements_mean_left_pupil[trial_id_in_data[i]],
                                                                                     sliding_window_consecutive_elements_mean_right_pupil[trial_id_in_data[i]],
                                                                                     sliding_window_consecutive_elements_max_left_pupil[trial_id_in_data[i]],
                                                                                     sliding_window_consecutive_elements_max_right_pupil[trial_id_in_data[i]],
                                                                                     sliding_window_consecutive_elements_sum_left_pupil[trial_id_in_data[i]],
                                                                                     sliding_window_consecutive_elements_sum_right_pupil[trial_id_in_data[i]],
                                                                                     sliding_window_hrv_sdnn[trial_id_in_data[i]],
                                                                                     sliding_window_hrv_rmssd[trial_id_in_data[i]]))

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

    return y_gloc_labels_noNaN, x_feature_matrix_noNaN

def summarize_performance_metrics(accuracy_logreg, accuracy_rf, accuracy_lda, accuracy_knn, accuracy_svm, accuracy_gb,
                                  precision_logreg, precision_rf, precision_lda, precision_knn, precision_svm, precision_gb,
                                  recall_logreg, recall_rf, recall_lda, recall_knn, recall_svm, recall_gb,
                                  f1_logreg, f1_rf, f1_lda, f1_knn, f1_svm, f1_gb,
                                  specificity_logreg, specificity_rf, specificity_lda, specificity_knn, specificity_svm, specificity_gb):
    classifiers = ['Log Reg', 'RF', 'LDA', 'KNN', 'SVM' , 'Ensemble w/ GB']
    performance_metrics = ['accuracy', 'precision', 'recall', 'f1-score', 'specificity']

    accuracy = np.array([accuracy_logreg, accuracy_rf, accuracy_lda, accuracy_knn, accuracy_svm, accuracy_gb])
    precision = np.array([precision_logreg, precision_rf, precision_lda, precision_knn, precision_svm, precision_gb])
    recall = np.array([recall_logreg, recall_rf, recall_lda, recall_knn, recall_svm, recall_gb])
    f1 = np.array([f1_logreg, f1_rf, f1_lda, f1_knn, f1_svm, f1_gb])
    specificity = np.array([specificity_logreg, specificity_rf, specificity_lda, specificity_knn, specificity_svm, specificity_gb])
    combined_metrics = np.column_stack((accuracy, precision, recall, f1, specificity))

    performance_metric_summary = pd.DataFrame(combined_metrics, index = classifiers, columns = performance_metrics)

    return performance_metric_summary

def find_prediction_window(gloc_data_reduced, gloc, time_variable):
    trial_id_in_data = gloc_data_reduced.trial_id.unique()
    max_prediction_offset = np.zeros(len(trial_id_in_data))
    for i in range(len(trial_id_in_data)):
        current_index = gloc_data_reduced['trial_id'] == trial_id_in_data[i]

        time = gloc_data_reduced[time_variable]
        accel = gloc_data_reduced['magnitude - Centrifuge']

        current_time = np.array(time[current_index])
        current_accel = np.array(accel[current_index])
        current_gloc = gloc[current_index]

        if i == 7 or i == 8 or i == 19:
            ror = [m for m in range(len(current_time)) if current_time[m] >= 600]
        else:
            ror = [t for t in range(len(current_time)) if current_time[t] >= 400]

        reduced_time = current_time[ror]
        reduced_accel = current_accel[ror]
        reduced_gloc = current_gloc[ror]

        fig, ax = plt.subplots()
        ax.plot(reduced_time, reduced_accel)
        ax.plot(reduced_time, reduced_gloc)
        plt.show()

        gloc_vals = [j for j in range(len(reduced_gloc)) if reduced_gloc[j] == 1]
        if len(gloc_vals) == 0:
            gloc_index = np.nan
        else:
            gloc_index = gloc_vals[0]

        accel_increase = [k for k in range(len(reduced_accel)) if reduced_accel[k] >= 1.3]
        if len(accel_increase) == 0:
            accel_index = np.nan
        else:
            accel_index = accel_increase[0]

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

    y = 1
