import numpy as np
import pandas as pd
import os
from sklearn import linear_model

def sliding_window_mean_calc(time_start, offset, stride, window_size, combined_baseline, gloc, gloc_data_reduced, time_variable, combined_baseline_names):
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
        sliding_window_mean_current = np.zeros((number_windows_current, np.shape(combined_baseline[trial_id_in_data[i]])[1]))
        gloc_window_current = np.zeros((number_windows_current, 1))

        # Create trimmed gloc data for the specific
        gloc_trimmed = gloc[(gloc_data_reduced.trial_id == trial_id_in_data[i])]

        # Define iteration time
        time_iteration = time_start

        # Iterate through all windows to compute relevant parameters
        for j in range(number_windows_current):

            # Find index for current window
            time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))
            current_combined_baseline = combined_baseline[trial_id_in_data[i]][time_period_feature]

            # Take nanmean for the window (one value per column (feature))
            sliding_window_mean_current[j,:] = np.nanmean(current_combined_baseline, axis = 0, keepdims=True)

            # Find the offset time for G-LOC label
            time_period_gloc = (((time_iteration + offset) <= time_trimmed) &
                                (time_trimmed < (time_iteration + offset + window_size)))

            # Create engineered label set to 1 if any values in window are 1
            gloc_window_current[j] = np.any(gloc_trimmed[time_period_gloc])

            # Adjust iteration_time
            time_iteration = stride + time_iteration

        ## Compute z-score to standardize (intra-trial standardization)
        # if np.nanstd(sliding_window_mean_current == 0:
        #
        # else:
        sliding_window_mean_current_z_score = ((sliding_window_mean_current - np.nanmean(sliding_window_mean_current, axis = 0, keepdims=True))
                                               / np.nanstd(sliding_window_mean_current, axis = 0, keepdims=True))

        # Define dictionary item for trial_id
        sliding_window_mean[trial_id_in_data[i]] = sliding_window_mean_current_z_score
        gloc_window[trial_id_in_data[i]] = gloc_window_current
        number_windows[trial_id_in_data[i]] = number_windows_current

        # Name all features
        all_features_mean = [s + '_mean' for s in combined_baseline_names]

    ## Compute z-score to standardize (inter-trial standardization)
    #inter_trial_z_score =

    return gloc_window, sliding_window_mean, number_windows, all_features_mean

def sliding_window_calc(time_start, stride, window_size, combined_baseline, gloc_data_reduced, time_variable, number_windows, combined_baseline_names):
    """
    This function creates the engineered features and gloc labels for the data. This includes a
    sliding window standard deviation for each of the features for each trial_id. A dictionary
    sorted by trial_id for the engineered feature is returned.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    sliding_window_stddev = dict()
    sliding_window_max = dict()
    sliding_window_range = dict()

    # Iterate through all unique trial_id
    for i in range(np.size(trial_id_in_data)):

        # Determine index from current trial_id
        current_index = (gloc_data_reduced['trial_id'] == trial_id_in_data[i])

        # Create time array based on current_index
        current_time = np.array(gloc_data_reduced[time_variable])
        time_trimmed = current_time[current_index]

        # Determine number of windows
        number_windows_current = number_windows[trial_id_in_data[i]]

        # Pre-allocate arrays
        sliding_window_stddev_current = np.zeros((number_windows_current, np.shape(combined_baseline[trial_id_in_data[i]])[1]))
        sliding_window_max_current = np.zeros((number_windows_current, np.shape(combined_baseline[trial_id_in_data[i]])[1]))
        sliding_window_range_current = np.zeros((number_windows_current, np.shape(combined_baseline[trial_id_in_data[i]])[1]))

        # Define iteration time
        time_iteration = time_start

        # Iterate through all windows to compute relevant parameters
        for j in range(number_windows_current):

            # Find index for current window
            time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))

            # Find feature for current window
            current_combined_baseline = combined_baseline[trial_id_in_data[i]][time_period_feature]

            # Take nan stddev for the window (one value per column (feature))
            sliding_window_stddev_current[j,:] = np.nanstd(current_combined_baseline, axis = 0, keepdims=True)

            # Take nan max for the window (one value per column (feature))
            sliding_window_max_current[j, :] = np.nanmax(current_combined_baseline, axis=0, keepdims=True)

            # Take nan range for the window (one value per column (feature))
            sliding_window_range_current[j, :] = np.nanmax(current_combined_baseline, axis=0, keepdims=True) - np.nanmin(current_combined_baseline, axis=0, keepdims=True)

            # Adjust iteration_time
            time_iteration = stride + time_iteration

        # Compute z-score to standardize
        sliding_window_stddev_current_z_score = ((sliding_window_stddev_current - np.nanmean(sliding_window_stddev_current, axis = 0, keepdims=True))
                                               / np.nanstd(sliding_window_stddev_current, axis = 0, keepdims=True))

        sliding_window_max_current_z_score = ((sliding_window_max_current - np.nanmean(sliding_window_max_current, axis = 0, keepdims=True))
                                               / np.nanstd(sliding_window_max_current, axis = 0, keepdims=True))

        sliding_window_range_current_z_score = ((sliding_window_range_current - np.nanmean(sliding_window_range_current, axis = 0, keepdims=True))
                                               / np.nanstd(sliding_window_range_current, axis = 0, keepdims=True))


        # Define dictionary item for trial_id
        sliding_window_stddev[trial_id_in_data[i]] = sliding_window_stddev_current_z_score
        sliding_window_max[trial_id_in_data[i]] = sliding_window_max_current_z_score
        sliding_window_range[trial_id_in_data[i]] = sliding_window_range_current_z_score

        # Name features
        all_features_stddev = [s + '_stddev' for s in combined_baseline_names]
        all_features_max = [s + '_max' for s in combined_baseline_names]
        all_features_range = [s + '_range' for s in combined_baseline_names]

    return sliding_window_stddev, sliding_window_max, sliding_window_range, all_features_stddev, all_features_max, all_features_range

def sliding_window_other_features(time_start, stride, window_size, gloc_data_reduced, time_variable, number_windows,
                                  baseline_names_v0, baseline_v0, feature_groups_to_analyze):
    """
    This function creates the engineered features and gloc labels for the data. This includes a
    sliding window mean of the difference between left and right pupil and HbO/Hbd ratio.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    if 'eyetracking' in feature_groups_to_analyze:
        # Find indices of left and right pupil
        index_left_pupil = baseline_names_v0.index('Pupil diameter left [mm] - Tobii_v0')
        index_right_pupil = baseline_names_v0.index('Pupil diameter right [mm] - Tobii_v0')

        # Define eyetracking feature names
        eye_tracking_features = ['Left Pupil Integral (Non-Baseline)', 'Right Pupil Integral (Non-Baseline)',
                                   'Left Pupil Mean of Consecutive Difference (Non-Baseline)',
                                   'Right Pupil Mean of Consecutive Difference (Non-Baseline)',
                                   'Left Pupil Max of Consecutive Difference (Non-Baseline)',
                                   'Right Pupil Max of Consecutive Difference (Non-Baseline)',
                                   'Left Pupil Sum of Consecutive Difference (Non-Baseline)',
                                   'Right Pupil Sum of Consecutive Difference (Non-Baseline)']
    else:
        eye_tracking_features = []

    if 'ECG' in feature_groups_to_analyze:
        # Find indices of HR
        index_HR = baseline_names_v0.index('HR (bpm) - Equivital_v0')

        # Define ECG feature names
        ecg_features = ['HRV (SDNN)', 'HRV (RMSSD)']#, 'HRV (PNN50)']
    else:
        ecg_features = []

    if 'cognitive' in feature_groups_to_analyze:
        # Find indices of Cognitive Response Time and Correct
        index_response_time = baseline_names_v0.index('RespTime - Cog_v0')
        index_correct = baseline_names_v0.index('Correct - Cog_v0')

        # Define ECG feature names
        cognitive_features = ['Cognitive IES']
    else:
        cognitive_features = []

    # Build Dictionary for each trial_id
    sliding_window_integral_left_pupil = dict()
    sliding_window_integral_right_pupil = dict()

    sliding_window_consecutive_elements_mean_left_pupil = dict()
    sliding_window_consecutive_elements_mean_right_pupil = dict()

    sliding_window_consecutive_elements_max_left_pupil = dict()
    sliding_window_consecutive_elements_max_right_pupil = dict()

    sliding_window_consecutive_elements_sum_left_pupil = dict()
    sliding_window_consecutive_elements_sum_right_pupil = dict()

    sliding_window_hrv_sdnn = dict()
    sliding_window_hrv_rmssd = dict()
    sliding_window_hrv_pnn50 = dict()

    sliding_window_cognitive_IES = dict()

    # Iterate through all unique trial_id
    for i in range(np.size(trial_id_in_data)):

        # Determine index from current trial_id
        current_index = (gloc_data_reduced['trial_id'] == trial_id_in_data[i])

        # Create time array based on current_index
        current_time = np.array(gloc_data_reduced[time_variable])
        time_trimmed = current_time[current_index]

        # Determine number of windows
        number_windows_current = number_windows[trial_id_in_data[i]]

        # Pre-allocate arrays
        if 'eyetracking' in feature_groups_to_analyze:
            sliding_window_integral_left_pupil_current = np.zeros((number_windows_current, 1))
            sliding_window_integral_right_pupil_current = np.zeros((number_windows_current, 1))
            sliding_window_consecutive_elements_mean_left_pupil_current = np.zeros((number_windows_current, 1))
            sliding_window_consecutive_elements_mean_right_pupil_current = np.zeros((number_windows_current, 1))
            sliding_window_consecutive_elements_max_left_pupil_current = np.zeros((number_windows_current, 1))
            sliding_window_consecutive_elements_max_right_pupil_current = np.zeros((number_windows_current, 1))
            sliding_window_consecutive_elements_sum_left_pupil_current = np.zeros((number_windows_current, 1))
            sliding_window_consecutive_elements_sum_right_pupil_current = np.zeros((number_windows_current, 1))
        if 'ECG' in feature_groups_to_analyze:
            sliding_window_hrv_sdnn_current = np.zeros((number_windows_current, 1))
            sliding_window_hrv_rmssd_current = np.zeros((number_windows_current, 1))
            # sliding_window_hrv_pnn50_current = np.zeros((number_windows_current, 1))
        if 'cognitive' in feature_groups_to_analyze:
            sliding_window_cognitive_IES_current = np.zeros((number_windows_current, 1))

        # Define iteration time
        time_iteration = time_start

        # Iterate through all windows to compute relevant parameters
        for j in range(number_windows_current):

            # Find index for current window
            time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))

            # Find non-baseline feature for current window
            feature_window_no_baseline = baseline_v0[trial_id_in_data[i]][time_period_feature]

            if 'ECG' in feature_groups_to_analyze:
                # Compute HRV
                RR_interval = 60000 / feature_window_no_baseline[:,index_HR]
                sliding_window_hrv_sdnn_current[j] = np.nanstd(RR_interval)

                successive_difference = np.diff(RR_interval)
                sliding_window_hrv_rmssd_current[j] = np.sqrt(np.nanmean(successive_difference**2))

                # # Compute PNN50
                # count_50ms_diff_current = np.sum(np.abs(successive_difference) > 50)
                # sliding_window_hrv_pnn50_current[j] = (count_50ms_diff_current / len(successive_difference)) * 100

            if 'cognitive' in feature_groups_to_analyze:
                # Compute IES (Inverse Efficiency Score)
                sliding_window_cognitive_IES_current[j] = np.mean(feature_window_no_baseline[:,index_response_time]) / (np.mean(feature_window_no_baseline[:,index_correct]))

            if 'eyetracking' in feature_groups_to_analyze:
                # Compute non-baseline pupil features
                left_pupil_no_baseline = feature_window_no_baseline[:, index_left_pupil]
                right_pupil_no_baseline = feature_window_no_baseline[:, index_right_pupil]

                # Integral (using Trapezoid rule)
                sliding_window_integral_left_pupil_current[j] = (window_size / 2) * (left_pupil_no_baseline[-1] + left_pupil_no_baseline[0])
                sliding_window_integral_right_pupil_current[j] = (window_size / 2) * (right_pupil_no_baseline[-1] + right_pupil_no_baseline[0])

                # Compute average difference between consecutive elements
                left_pupil_consecutive_difference = np.diff(left_pupil_no_baseline)
                left_pupil_consecutive_difference_full = np.append(left_pupil_consecutive_difference, np.nan)

                right_pupil_consecutive_difference = np.diff(right_pupil_no_baseline)
                right_pupil_consecutive_difference_full = np.append(right_pupil_consecutive_difference, np.nan)

                sliding_window_consecutive_elements_mean_left_pupil_current[j] = np.nanmean(left_pupil_consecutive_difference_full)
                sliding_window_consecutive_elements_mean_right_pupil_current[j] = np.nanmean(right_pupil_consecutive_difference_full)

                # Compute max difference between consecutive elements
                sliding_window_consecutive_elements_max_left_pupil_current[j] = np.nanmax(left_pupil_consecutive_difference_full)
                sliding_window_consecutive_elements_max_right_pupil_current[j] = np.nanmax(right_pupil_consecutive_difference_full)

                # Compute sum of difference between consecutive elements
                sliding_window_consecutive_elements_sum_left_pupil_current[j] = np.nansum(left_pupil_consecutive_difference_full)
                sliding_window_consecutive_elements_sum_right_pupil_current[j] = np.nansum(right_pupil_consecutive_difference_full)

            # Adjust iteration_time
            time_iteration = stride + time_iteration

        # Compute Z-score
        if 'eyetracking' in feature_groups_to_analyze:
            # Compute z-score to standardize integral left pupil
            sliding_window_integral_left_pupil_current_z_score = ((sliding_window_integral_left_pupil_current - np.nanmean(sliding_window_integral_left_pupil_current, axis = 0, keepdims=True))
                                                   / np.nanstd(sliding_window_integral_left_pupil_current, axis = 0, keepdims=True))

            # Compute z-score to standardize integral right pupil
            sliding_window_integral_right_pupil_current_z_score = ((sliding_window_integral_right_pupil_current - np.nanmean(sliding_window_integral_right_pupil_current, axis = 0, keepdims=True))
                                                   / np.nanstd(sliding_window_integral_right_pupil_current, axis = 0, keepdims=True))

            # Compute z-score to standardize mean of difference of consecutive elements-left pupil
            sliding_window_consecutive_elements_mean_left_pupil_current_z_score = ((sliding_window_consecutive_elements_mean_left_pupil_current - np.nanmean(sliding_window_consecutive_elements_mean_left_pupil_current, axis = 0, keepdims=True))
                                                   / np.nanstd(sliding_window_consecutive_elements_mean_left_pupil_current, axis = 0, keepdims=True))

            # Compute z-score to standardize mean of difference of consecutive elements-right pupil
            sliding_window_consecutive_elements_mean_right_pupil_current_z_score = ((sliding_window_consecutive_elements_mean_right_pupil_current - np.nanmean(sliding_window_consecutive_elements_mean_right_pupil_current, axis = 0, keepdims=True))
                                                   / np.nanstd(sliding_window_consecutive_elements_mean_right_pupil_current, axis = 0, keepdims=True))

            # Compute z-score to standardize max of difference of consecutive elements-left pupil
            sliding_window_consecutive_elements_max_left_pupil_current_z_score = ((sliding_window_consecutive_elements_max_left_pupil_current - np.nanmean(sliding_window_consecutive_elements_max_left_pupil_current, axis = 0, keepdims=True))
                                                   / np.nanstd(sliding_window_consecutive_elements_max_left_pupil_current, axis = 0, keepdims=True))

            # Compute z-score to standardize max of difference of consecutive elements-right pupil
            sliding_window_consecutive_elements_max_right_pupil_current_z_score = ((sliding_window_consecutive_elements_max_right_pupil_current - np.nanmean(sliding_window_consecutive_elements_max_right_pupil_current, axis = 0, keepdims=True))
                                                   / np.nanstd(sliding_window_consecutive_elements_max_right_pupil_current, axis = 0, keepdims=True))

            # Compute z-score to standardize sum of difference of consecutive elements-left pupil
            sliding_window_consecutive_elements_sum_left_pupil_current_z_score = ((sliding_window_consecutive_elements_sum_left_pupil_current - np.nanmean(sliding_window_consecutive_elements_sum_left_pupil_current, axis = 0, keepdims=True))
                                                   / np.nanstd(sliding_window_consecutive_elements_sum_left_pupil_current, axis = 0, keepdims=True))

            # Compute z-score to standardize sum of difference of consecutive elements-right pupil
            sliding_window_consecutive_elements_sum_right_pupil_current_z_score = ((sliding_window_consecutive_elements_sum_right_pupil_current - np.nanmean(sliding_window_consecutive_elements_sum_right_pupil_current, axis = 0, keepdims=True))
                                                   / np.nanstd(sliding_window_consecutive_elements_sum_right_pupil_current, axis = 0, keepdims=True))
        if 'ECG' in feature_groups_to_analyze:
            # Compute z-score to standardize hrv sdnn
            sliding_window_hrv_sdnn_current_z_score = ((sliding_window_hrv_sdnn_current - np.nanmean(sliding_window_hrv_sdnn_current, axis = 0, keepdims=True))
                                                   / np.nanstd(sliding_window_hrv_sdnn_current, axis = 0, keepdims=True))

            # Compute z-score to standardize hrv rmssd
            sliding_window_hrv_rmssd_current_z_score = ((sliding_window_hrv_rmssd_current - np.nanmean(sliding_window_hrv_rmssd_current, axis = 0, keepdims=True))
                                                   / np.nanstd(sliding_window_hrv_rmssd_current, axis = 0, keepdims=True))

            # # Compute z-score to standardize hrv pnn50
            # sliding_window_hrv_pnn50_current_z_score = ((sliding_window_hrv_pnn50_current - np.nanmean(sliding_window_hrv_pnn50_current, axis=0, keepdims=True))
            #                                        / np.nanstd(sliding_window_hrv_pnn50_current, axis=0, keepdims=True))

        if 'cognitive' in feature_groups_to_analyze:
            # Compute z-score to standardize cognitive IES
            sliding_window_cognitive_IES_current_z_score = ((sliding_window_cognitive_IES_current - np.nanmean(sliding_window_cognitive_IES_current, axis=0, keepdims=True))
                                                   / np.nanstd(sliding_window_cognitive_IES_current, axis=0, keepdims=True))

        # Define dictionary item for trial_id
        if 'eyetracking' in feature_groups_to_analyze:
            sliding_window_integral_left_pupil[trial_id_in_data[i]] = sliding_window_integral_left_pupil_current_z_score
            sliding_window_integral_right_pupil[trial_id_in_data[i]] = sliding_window_integral_right_pupil_current_z_score
            sliding_window_consecutive_elements_mean_left_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_mean_left_pupil_current_z_score
            sliding_window_consecutive_elements_mean_right_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_mean_right_pupil_current_z_score
            sliding_window_consecutive_elements_max_left_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_max_left_pupil_current_z_score
            sliding_window_consecutive_elements_max_right_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_max_right_pupil_current_z_score
            sliding_window_consecutive_elements_sum_left_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_sum_left_pupil_current_z_score
            sliding_window_consecutive_elements_sum_right_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_sum_right_pupil_current_z_score
        if 'ECG' in feature_groups_to_analyze:
            sliding_window_hrv_sdnn[trial_id_in_data[i]] = sliding_window_hrv_sdnn_current_z_score
            sliding_window_hrv_rmssd[trial_id_in_data[i]] = sliding_window_hrv_rmssd_current_z_score
            # sliding_window_hrv_pnn50[trial_id_in_data[i]] = sliding_window_hrv_pnn50_current_z_score
        if 'cognitive' in feature_groups_to_analyze:
            sliding_window_cognitive_IES[trial_id_in_data[i]] = sliding_window_cognitive_IES_current_z_score

        # Name all features
        all_features_additional = eye_tracking_features + ecg_features + cognitive_features

    return (all_features_additional, sliding_window_integral_left_pupil, sliding_window_integral_right_pupil,
            sliding_window_consecutive_elements_mean_left_pupil, sliding_window_consecutive_elements_mean_right_pupil,
            sliding_window_consecutive_elements_max_left_pupil, sliding_window_consecutive_elements_max_right_pupil,
            sliding_window_consecutive_elements_sum_left_pupil, sliding_window_consecutive_elements_sum_right_pupil,
            sliding_window_hrv_sdnn, sliding_window_hrv_rmssd, sliding_window_hrv_pnn50, sliding_window_cognitive_IES)