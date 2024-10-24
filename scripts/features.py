import numpy as np
import pandas as pd
import os

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

        # Compute z-score to standardize
        sliding_window_mean_current_z_score = ((sliding_window_mean_current - np.nanmean(sliding_window_mean_current, axis = 0, keepdims=True))
                                               / np.nanstd(sliding_window_mean_current, axis = 0, keepdims=True))

        # Define dictionary item for trial_id
        sliding_window_mean[trial_id_in_data[i]] = sliding_window_mean_current_z_score
        gloc_window[trial_id_in_data[i]] = gloc_window_current
        number_windows[trial_id_in_data[i]] = number_windows_current

    return gloc_window, sliding_window_mean, number_windows

def sliding_window_calc(time_start, stride, window_size, feature_baseline, gloc_data_reduced, time_variable, number_windows):
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
        sliding_window_stddev_current = np.zeros((number_windows_current, np.shape(feature_baseline[trial_id_in_data[i]])[1]))
        sliding_window_max_current = np.zeros((number_windows_current, np.shape(feature_baseline[trial_id_in_data[i]])[1]))
        sliding_window_range_current = np.zeros((number_windows_current, np.shape(feature_baseline[trial_id_in_data[i]])[1]))

        # Define iteration time
        time_iteration = time_start

        # Iterate through all windows to compute relevant parameters
        for j in range(number_windows_current):

            # Find index for current window
            time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))

            # Find feature for current window
            feature_window = feature_baseline[trial_id_in_data[i]][time_period_feature]

            # Take nan stddev for the window (one value per column (feature))
            sliding_window_stddev_current[j,:] = np.nanstd(feature_window, axis = 0, keepdims=True)

            # Take nan max for the window (one value per column (feature))
            sliding_window_max_current[j, :] = np.nanmax(feature_window, axis=0, keepdims=True)

            # Take nan range for the window (one value per column (feature))
            sliding_window_range_current[j, :] = np.nanmax(feature_window, axis=0, keepdims=True) - np.nanmin(feature_window, axis=0, keepdims=True)

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

    return sliding_window_stddev, sliding_window_max, sliding_window_range

def sliding_window_other_features(time_start, stride, window_size, feature_baseline, gloc_data_reduced, time_variable, number_windows, all_features):
    """
    This function creates the engineered features and gloc labels for the data. This includes a
    sliding window mean of the difference between left and right pupil and HbO/Hbd ratio.
    """

    # Find indices of left and right pupil
    index_left_pupil = all_features.index('Pupil diameter left [mm] - Tobii')
    index_right_pupil = all_features.index('Pupil diameter right [mm] - Tobii')

    # Find indices of HbO & Hbd
    index_Hbo = all_features.index('HbO2 - fNIRS')
    index_Hbd = all_features.index('Hbd - fNIRS')

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    sliding_window_pupil_difference = dict()
    sliding_window_ox_deox_ratio = dict()

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
        sliding_window_pupil_difference_current = np.zeros((number_windows_current, 1))
        sliding_window_ox_deox_ratio_current = np.zeros((number_windows_current, 1))

        # Define iteration time
        time_iteration = time_start

        # Iterate through all windows to compute relevant parameters
        for j in range(number_windows_current):

            # Find index for current window
            time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))

            # Find feature for current window
            feature_window = feature_baseline[trial_id_in_data[i]][time_period_feature]

            # Find pupil difference
            pupil_difference = feature_window[:,index_left_pupil] - feature_window[:,index_right_pupil]

            # Find Hbo/Hbd Ratio
            ox_deox_ratio = feature_window[:,index_Hbo] / feature_window[:,index_Hbd]

            # Take nan mean for the window-pupil difference (one value per column (feature))
            sliding_window_pupil_difference_current[j,:] = np.nanmean(pupil_difference, axis = 0, keepdims=True)

            # Take nan mean for the window-oxy/deoxy ratio
            sliding_window_ox_deox_ratio_current[j,:] = np.nanmean(ox_deox_ratio, axis = 0, keepdims=True)

            # Adjust iteration_time
            time_iteration = stride + time_iteration

        # Compute z-score to standardize-pupil difference
        sliding_window_pupil_difference_current_z_score = ((sliding_window_ox_deox_ratio_current - np.nanmean(sliding_window_ox_deox_ratio_current, axis = 0, keepdims=True))
                                               / np.nanstd(sliding_window_ox_deox_ratio_current, axis = 0, keepdims=True))

        # Compute z-score to standardize-ox/deox ratio
        sliding_window_ox_deox_ratio_current_z_score = ((sliding_window_pupil_difference_current - np.nanmean(sliding_window_pupil_difference_current, axis = 0, keepdims=True))
                                               / np.nanstd(sliding_window_pupil_difference_current, axis = 0, keepdims=True))

        # Define dictionary item for trial_id
        sliding_window_pupil_difference[trial_id_in_data[i]] = sliding_window_pupil_difference_current_z_score
        sliding_window_ox_deox_ratio[trial_id_in_data[i]] = sliding_window_ox_deox_ratio_current_z_score

    return sliding_window_pupil_difference, sliding_window_ox_deox_ratio