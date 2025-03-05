import numpy as np
import pandas as pd
import os

def create_v0_baseline(gloc_data_reduced, features, time_variable, all_features):
    """
    This function baselines the features with baseline method v0 (no baseline)
    and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """
    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v0 = dict()
    baseline_v0_derivative = dict()
    baseline_v0_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):
        # Find time window
        time_index = (gloc_data_reduced.trial_id == trial_id_in_data[i])
        time_array = np.array(gloc_data_reduced[time_variable])

        # Access all features for that trial id
        baseline_v0[trial_id_in_data[i]] = np.array(features[(gloc_data_reduced.trial_id == trial_id_in_data[i])])

        # Compute derivative
        time = time_array[time_index]
        baseline_v0_derivative[trial_id_in_data[i]] = np.gradient(baseline_v0[trial_id_in_data[i]], time, axis=0)

        # Compute 2nd derivative
        baseline_v0_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v0_derivative[trial_id_in_data[i]], time, axis=0)

        # Update baseline names
        all_features_updated = [s + '_v0' for s in all_features]

    return baseline_v0, baseline_v0_derivative, baseline_v0_second_derivative, all_features_updated

def create_v1_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys):
    """
    This function baselines the features with baseline method v1 (divide by mean)
    and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v1 = dict()
    baseline_v1_derivative = dict()
    baseline_v1_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Find time window
        index_window = ((gloc_data_reduced[time_variable]<baseline_window) & (gloc_data_reduced.trial_id == trial_id_in_data[i]))
        time_index = (gloc_data_reduced.trial_id == trial_id_in_data[i])
        time_array = np.array(gloc_data_reduced[time_variable])

        # Find baseline average based on specified baseline window
        baseline_feature = np.mean(features_phys[index_window], axis = 0)

        # Divide features for that trial by baselined data
        baseline_v1[trial_id_in_data[i]] = np.array(features_phys[(gloc_data_reduced.trial_id == trial_id_in_data[i])] /baseline_feature)

        # Compute derivative
        time = time_array[time_index]
        baseline_v1_derivative[trial_id_in_data[i]] = np.gradient(baseline_v1[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v1_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v1_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_phys_updated = [s + '_v1' for s in all_features_phys]

    return baseline_v1, baseline_v1_derivative, baseline_v1_second_derivative, all_features_phys_updated

def create_v2_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys):
    """
    This function baselines the features with baseline method v2 (subtract mean)
    and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v2 = dict()
    baseline_v2_derivative = dict()
    baseline_v2_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Find time window
        index_window = ((gloc_data_reduced[time_variable]<baseline_window) & (gloc_data_reduced.trial_id == trial_id_in_data[i]))
        time_index = (gloc_data_reduced.trial_id == trial_id_in_data[i])
        time_array = np.array(gloc_data_reduced[time_variable])

        # Find baseline average based on specified baseline window
        baseline_feature = np.mean(features_phys[index_window], axis = 0)

        # Subtract baseline data (baseline v2)
        baseline_v2[trial_id_in_data[i]] = np.array(features_phys[(gloc_data_reduced.trial_id == trial_id_in_data[i])] - baseline_feature)

        # Compute derivative
        time = time_array[time_index]
        baseline_v2_derivative[trial_id_in_data[i]] = np.gradient(baseline_v2[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v2_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v2_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_phys_updated = [s + '_v2' for s in all_features_phys]

    return baseline_v2, baseline_v2_derivative, baseline_v2_second_derivative, all_features_phys_updated

def create_v3_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys):
    """
    This function baselines the features with baseline method v3 (Each feature prior to ROR
    is divided by the first portion of the trial for baseline & the rest of the
    trial is divided by the portion prior to ROR for baseline) and puts them into a
    dictionary per trial. This dictionary is output. The first and second derivative of
    each baselined feature are computed and output in a dictionary.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v3 = dict()
    baseline_v3_derivative = dict()
    baseline_v3_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Define first baseline period (during initial specified baseline window)
        index_window1 = ((gloc_data_reduced[time_variable]<baseline_window) & (gloc_data_reduced.trial_id == trial_id_in_data[i]))

        # Find indices of other variables
        time_index = (gloc_data_reduced.trial_id == trial_id_in_data[i])
        time_array = np.array(gloc_data_reduced[time_variable])
        time = time_array[time_index]
        event_array = np.array(gloc_data_reduced['event_validated'])

        # Find baseline average based on specified baseline window
        baseline_period1_feature = np.mean(features_phys[index_window1], axis=0)

        # If ROR marked in trial, find when ROR begins and take the baseline directly before that (with window size defined in main)
        current_trial_event = event_array[time_index]
        if 'begin ROR' in current_trial_event:
            begin_ROR_index = np.argwhere(current_trial_event == 'begin ROR')
            time_ROR = time_array[begin_ROR_index][0,0]

            # Define second baseline period
            index_window2 = ((gloc_data_reduced[time_variable]>(time_ROR - baseline_window)) & (gloc_data_reduced[time_variable]<time_ROR)
                                                                                                & (gloc_data_reduced.trial_id == trial_id_in_data[i]))

            # Find baseline average based on specified baseline window
            baseline_period2_feature = np.mean(features_phys[index_window2], axis=0)

            # Divide features for that trial by correct baseline period
            first_baseline_period = (time_array < (time_ROR - baseline_window)) & (gloc_data_reduced.trial_id == trial_id_in_data[i])
            second_baseline_period = (time_array >= (time_ROR - baseline_window)) & (gloc_data_reduced.trial_id == trial_id_in_data[i])

            # Stack data from two windows of baselined data together
            baseline_v3[trial_id_in_data[i]] = np.array(features_phys[(first_baseline_period)] / baseline_period1_feature)
            baseline_v3[trial_id_in_data[i]] = np.vstack((baseline_v3[trial_id_in_data[i]], np.array(features_phys[(second_baseline_period)] / baseline_period2_feature)))

        # if no labeled begin of ROR, then just use the intial baseline period
        else:
            baseline_v3[trial_id_in_data[i]] = np.array(features_phys[(gloc_data_reduced.trial_id == trial_id_in_data[i])] / baseline_period1_feature)


        # Compute derivative
        baseline_v3_derivative[trial_id_in_data[i]] = np.gradient(baseline_v3[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v3_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v3_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_phys_updated = [s + '_v3' for s in all_features_phys]

    return baseline_v3, baseline_v3_derivative, baseline_v3_second_derivative, all_features_phys_updated

def create_v4_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys):
    """
    This function baselines the features with baseline method v4 (The first portion of
    the trial is subtracted from each feature prior to ROR & the period prior to ROR gets
    subtracted from the feature for the rest of the trial) and puts them into a
    dictionary per trial. This dictionary is output. The first and second derivative of
    each baselined feature are computed and output in a dictionary.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v4 = dict()
    baseline_v4_derivative = dict()
    baseline_v4_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Define first baseline period (during initial specified baseline window)
        index_window1 = ((gloc_data_reduced[time_variable]<baseline_window) & (gloc_data_reduced.trial_id == trial_id_in_data[i]))

        # Find indices of other variables
        time_index = (gloc_data_reduced.trial_id == trial_id_in_data[i])
        time_array = np.array(gloc_data_reduced[time_variable])
        time = time_array[time_index]
        event_array = np.array(gloc_data_reduced['event_validated'])

        # Find baseline average based on specified baseline window
        baseline_period1_feature = np.mean(features_phys[index_window1], axis=0)

        # If ROR marked in trial, find when ROR begins and take the baseline directly before that (with window size defined in main)
        current_trial_event = event_array[time_index]
        if 'begin ROR' in current_trial_event:
            begin_ROR_index = np.argwhere(current_trial_event == 'begin ROR')
            time_ROR = time_array[begin_ROR_index][0,0]

            # Define second baseline period
            index_window2 = ((gloc_data_reduced[time_variable]>(time_ROR - baseline_window)) & (gloc_data_reduced[time_variable]<time_ROR)
                                                                                                & (gloc_data_reduced.trial_id == trial_id_in_data[i]))

            # Find baseline average based on specified baseline window
            baseline_period2_feature = np.mean(features_phys[index_window2], axis=0)

            # Subtract features for that trial by correct baseline period
            first_baseline_period = (time_array < (time_ROR - baseline_window)) & (gloc_data_reduced.trial_id == trial_id_in_data[i])
            second_baseline_period = (time_array >= (time_ROR - baseline_window)) & (gloc_data_reduced.trial_id == trial_id_in_data[i])

            # Stack data from two windows of baselined data together
            baseline_v4[trial_id_in_data[i]] = np.array(features_phys[(first_baseline_period)] - baseline_period1_feature)
            baseline_v4[trial_id_in_data[i]] = np.vstack((baseline_v4[trial_id_in_data[i]], np.array(features_phys[(second_baseline_period)] - baseline_period2_feature)))

        # if no labeled begin of ROR, then just use the intial baseline period
        else:
            baseline_v4[trial_id_in_data[i]] = np.array(features_phys[(gloc_data_reduced.trial_id == trial_id_in_data[i])] / baseline_period1_feature)


        # Compute derivative
        baseline_v4_derivative[trial_id_in_data[i]] = np.gradient(baseline_v4[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v4_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v4_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_phys_updated = [s + '_v4' for s in all_features_phys]

    return baseline_v4, baseline_v4_derivative, baseline_v4_second_derivative, all_features_phys_updated

def create_v5_baseline(baseline_window, gloc_data_reduced, features_ecg, time_variable, participant_seated_rhr, all_features_ecg):
    """
    This function baselines the features with baseline method v5 (divide by RHR
    from another study) and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v5 = dict()
    baseline_v5_derivative = dict()
    baseline_v5_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Find time index and current participant
        time_index = (gloc_data_reduced.trial_id == trial_id_in_data[i])
        time_array = np.array(gloc_data_reduced[time_variable])
        participant_array = np.array(gloc_data_reduced.subject)

        # Find baseline for current participant
        current_participant = participant_array[time_index][0]
        baseline_feature = participant_seated_rhr[current_participant]

        # Divide features for that trial by baselined data
        baseline_v5[trial_id_in_data[i]] = np.array(features_ecg[(gloc_data_reduced.trial_id == trial_id_in_data[i])] /baseline_feature)

        # Compute derivative
        time = time_array[time_index]
        baseline_v5_derivative[trial_id_in_data[i]] = np.gradient(baseline_v5[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v5_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v5_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_ecg_updated = [s + '_v5' for s in all_features_ecg]

    return baseline_v5, baseline_v5_derivative, baseline_v5_second_derivative, all_features_ecg_updated

def create_v6_baseline(baseline_window, gloc_data_reduced, features_ecg, time_variable, participant_seated_rhr, all_features_ecg):
    """
    This function baselines the features with baseline method v6 (subtract RHR
    from another study) and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v6 = dict()
    baseline_v6_derivative = dict()
    baseline_v6_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Find time index and current participant
        time_index = (gloc_data_reduced.trial_id == trial_id_in_data[i])
        time_array = np.array(gloc_data_reduced[time_variable])
        participant_array = np.array(gloc_data_reduced.subject)

        # Find baseline for current participant
        current_participant = participant_array[time_index][0]
        baseline_feature = participant_seated_rhr[current_participant]

        # Divide features for that trial by baselined data
        baseline_v6[trial_id_in_data[i]] = np.array(features_ecg[(gloc_data_reduced.trial_id == trial_id_in_data[i])] - baseline_feature)

        # Compute derivative
        time = time_array[time_index]
        baseline_v6_derivative[trial_id_in_data[i]] = np.gradient(baseline_v6[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v6_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v6_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_ecg_updated = [s + '_v6' for s in all_features_ecg]

    return baseline_v6, baseline_v6_derivative, baseline_v6_second_derivative, all_features_ecg_updated

def create_v7_baseline(baseline_window, gloc_data_reduced, features_eeg, time_variable, eeg_baseline_delta,
                       eeg_baseline_theta, eeg_baseline_alpha, eeg_baseline_beta, all_features_eeg):
    """
    This function baselines the features with baseline method v7 (divide by processed EEG baseline)
    and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """
    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v7 = dict()
    baseline_v7_derivative = dict()
    baseline_v7_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Find time index and current participant
        time_index = (gloc_data_reduced.trial_id == trial_id_in_data[i])
        time_array = np.array(gloc_data_reduced[time_variable])
        participant_array = np.array(gloc_data_reduced.subject)

        # Find baseline for current participant
        current_participant = participant_array[time_index][0]
        baseline_feature_delta = eeg_baseline_delta.loc[current_participant]
        baseline_feature_theta = eeg_baseline_theta.loc[current_participant]
        baseline_feature_alpha = eeg_baseline_alpha.loc[current_participant]
        baseline_feature_beta = eeg_baseline_beta.loc[current_participant]

        # Set current trial
        current_trial_data = features_eeg[(gloc_data_reduced.trial_id == trial_id_in_data[i])]

        # Initialize np array to fill with each trial's data
        baselined_trial_data = np.zeros_like(current_trial_data)

        # Iterate through columns to divide by correct baseline channel
        for col in range(len(all_features_eeg)):
            # Get current column name
            column_name = all_features_eeg[col]

            # From column name in all_features_eeg, obtain the corresponding column name in
            # the baseline CSV file
            name_split = column_name.split('_')
            channel_name = name_split[0]

            # Find index of corresponding channel name in baseline file
            channel_index = baseline_feature_delta.index.get_loc(channel_name)

            if name_split[1] == 'delta - EEG':
                # Divide features for that trial by baselined data
                baselined_trial_data[:,col] = current_trial_data[:,col] / baseline_feature_delta[channel_index]
            elif name_split[1] == 'theta - EEG':
                # Divide features for that trial by baselined data
                baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_feature_theta[channel_index]
            elif name_split[1] == 'alpha - EEG':
                # Divide features for that trial by baselined data
                baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_feature_alpha[channel_index]
            elif name_split[1] == 'beta - EEG':
                # Divide features for that trial by baselined data
                baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_feature_beta[channel_index]

        # Once all columns have been iterated through, define baseline_v7
        baseline_v7[trial_id_in_data[i]] = baselined_trial_data

        # Compute derivative
        time = time_array[time_index]
        baseline_v7_derivative[trial_id_in_data[i]] = np.gradient(baseline_v7[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v7_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v7_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_eeg_updated = [s + '_v7' for s in all_features_eeg]

    return baseline_v7, baseline_v7_derivative, baseline_v7_second_derivative, all_features_eeg_updated

def create_v8_baseline(baseline_window, gloc_data_reduced, features_eeg, time_variable, eeg_baseline_delta,
                       eeg_baseline_theta, eeg_baseline_alpha, eeg_baseline_beta, all_features_eeg):
    """
    This function baselines the features with baseline method v8 (subtract EEG baseline)
    and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """
    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v8 = dict()
    baseline_v8_derivative = dict()
    baseline_v8_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Find time index and current participant
        time_index = (gloc_data_reduced.trial_id == trial_id_in_data[i])
        time_array = np.array(gloc_data_reduced[time_variable])
        participant_array = np.array(gloc_data_reduced.subject)

        # Find baseline for current participant
        current_participant = participant_array[time_index][0]
        baseline_feature_delta = eeg_baseline_delta.loc[current_participant]
        baseline_feature_theta = eeg_baseline_theta.loc[current_participant]
        baseline_feature_alpha = eeg_baseline_alpha.loc[current_participant]
        baseline_feature_beta = eeg_baseline_beta.loc[current_participant]

        # Set current trial
        current_trial_data = features_eeg[(gloc_data_reduced.trial_id == trial_id_in_data[i])]

        # Initialize np array to fill with each trial's data
        baselined_trial_data = np.zeros_like(current_trial_data)

        # Iterate through columns to divide by correct baseline channel
        for col in range(len(all_features_eeg)):
            # Get current column name
            column_name = all_features_eeg[col]

            # From column name in all_features_eeg, obtain the corresponding column name in
            # the baseline CSV file
            name_split = column_name.split('_')
            channel_name = name_split[0]

            # Find index of corresponding channel name in baseline file
            channel_index = baseline_feature_delta.index.get_loc(channel_name)

            if name_split[1] == 'delta - EEG':
                # Divide features for that trial by baselined data
                baselined_trial_data[:,col] = current_trial_data[:,col] - baseline_feature_delta[channel_index]
            elif name_split[1] == 'theta - EEG':
                # Divide features for that trial by baselined data
                baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_feature_theta[channel_index]
            elif name_split[1] == 'alpha - EEG':
                # Divide features for that trial by baselined data
                baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_feature_alpha[channel_index]
            elif name_split[1] == 'beta - EEG':
                # Divide features for that trial by baselined data
                baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_feature_beta[channel_index]

        # Once all columns have been iterated through, define baseline_v7
        baseline_v8[trial_id_in_data[i]] = baselined_trial_data

        # Compute derivative
        time = time_array[time_index]
        baseline_v8_derivative[trial_id_in_data[i]] = np.gradient(baseline_v8[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v8_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v8_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_eeg_updated = [s + '_v8' for s in all_features_eeg]

    return baseline_v8, baseline_v8_derivative, baseline_v8_second_derivative, all_features_eeg_updated