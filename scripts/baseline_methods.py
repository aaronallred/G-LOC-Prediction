import numpy as np
import pandas as pd
import warnings
from GLOC_data_processing import tabulateNaN

def baseline_data(baseline_methods_to_use, trial_column, time_column, event_validated_column, subject_column, features, all_features, gloc, baseline_window,
                  features_phys, all_features_phys, features_ecg, all_features_ecg, features_eeg, all_features_eeg,
                  baseline_data_filename, list_of_baseline_eeg_processed_files, model_type):

    """
    Baselines pre-feature data based on 'baseline_methods_to_use'
    """

    baseline, baseline_derivative, baseline_second_derivative, baseline_names = {}, {}, {}, {}

    # Load participant baseline data once if needed
    participant_seated_rhr = None
    if any(method in baseline_methods_to_use for method in ['v5', 'v6']):
        participant_baseline = pd.read_csv(baseline_data_filename)
        participant_seated_rhr = participant_baseline['resting HR [seated]'][:-1]
        participant_seated_rhr.index = [f"{i:02d}" for i in range(1, 14)]  # Index from '01' to '13'

    # Load EEG baseline data once if needed
    eeg_baseline_data = {}
    if any(method in baseline_methods_to_use for method in ['v7', 'v8']) and 'noAFE' in model_type:
        eeg_labels = ['delta', 'theta', 'alpha', 'beta']
        for idx, label in enumerate(eeg_labels):
            eeg_baseline_data[label] = pd.read_csv(list_of_baseline_eeg_processed_files[idx])
            eeg_baseline_data[label].index = [f"{i:02d}" for i in range(1, 14)]

    # Define baseline functions
    baseline_methods = {
        'v0': lambda: create_v0_baseline(trial_column, time_column, features, all_features),
        'v1': lambda: create_v1_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys),
        'v2': lambda: create_v2_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys),
        'v3': lambda: create_v3_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys),
        'v4': lambda: create_v4_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys),
        'v5': lambda: create_v5_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg),
        'v6': lambda: create_v6_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg),
        'v7': lambda: create_v7_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg,
                                         eeg_baseline_data['delta'], eeg_baseline_data['theta'],
                                         eeg_baseline_data['alpha'], eeg_baseline_data['beta'], all_features_eeg)
                if 'noAFE' in model_type else warnings.warn('EEG baseline methods not implemented for AFE conditions yet.'),
        'v8': lambda: create_v8_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg,
                                         eeg_baseline_data['delta'], eeg_baseline_data['theta'],
                                         eeg_baseline_data['alpha'], eeg_baseline_data['beta'], all_features_eeg)
                if 'noAFE' in model_type else warnings.warn('EEG baseline methods not implemented for AFE conditions yet.')
    }

    # Process only selected methods
    for method in baseline_methods_to_use:
        if method in baseline_methods:
            (baseline[method], baseline_derivative[method], baseline_second_derivative[method],
             baseline_names[method]) = baseline_methods[method]()

    # Combine all baseline methods into a large dictionary
    combined_baseline, combined_baseline_names = combine_all_baseline(trial_column, baseline, baseline_derivative,
                                                                      baseline_second_derivative, baseline_names)

    # baseline_v0 = baseline['v0']
    # baseline_names_v0 = baseline_names['v0']

    # Tabulate NaN
    # NaN_table, NaN_proportion, NaN_gloc_proportion = tabulateNaN(baseline['v0'], all_features, gloc,
    #                                                              gloc_data_reduced)

    return combined_baseline, combined_baseline_names, baseline['v0'], baseline_names['v0']

def create_v0_baseline(trial_column, time_column, features, all_features):
    """
    This function baselines the features with baseline method v0 (no baseline)
    and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """
    # Find Unique Trial ID
    trial_id_in_data = trial_column.unique()

    # Build Dictionary for each trial_id
    baseline_v0 = dict()
    baseline_v0_derivative = dict()
    baseline_v0_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):
        # Find time window
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)

        # Access all features for that trial id
        baseline_v0[trial_id_in_data[i]] = np.array(features[(trial_column == trial_id_in_data[i])])

        # Compute derivative
        time = time_array[time_index]
        baseline_v0_derivative[trial_id_in_data[i]] = np.gradient(baseline_v0[trial_id_in_data[i]], time, axis=0)

        # Compute 2nd derivative
        baseline_v0_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v0_derivative[trial_id_in_data[i]], time, axis=0)

        # Update baseline names
        all_features_updated = [s + '_v0' for s in all_features]

    return baseline_v0, baseline_v0_derivative, baseline_v0_second_derivative, all_features_updated

def create_v1_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys):
    """
    This function baselines the features with baseline method v1 (divide by mean)
    and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """

    # Find Unique Trial ID
    trial_id_in_data = trial_column.unique()

    # Build Dictionary for each trial_id
    baseline_v1 = dict()
    baseline_v1_derivative = dict()
    baseline_v1_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Find time window
        index_window = ((time_column<baseline_window) & (trial_column == trial_id_in_data[i]))
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)

        # Find baseline average based on specified baseline window
        baseline_feature = np.mean(features_phys[index_window], axis = 0)

        # Divide features for that trial by baselined data
        baseline_v1[trial_id_in_data[i]] = np.array(features_phys[(trial_column == trial_id_in_data[i])] /baseline_feature)

        # Compute derivative
        time = time_array[time_index]
        baseline_v1_derivative[trial_id_in_data[i]] = np.gradient(baseline_v1[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v1_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v1_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_phys_updated = [s + '_v1' for s in all_features_phys]

    return baseline_v1, baseline_v1_derivative, baseline_v1_second_derivative, all_features_phys_updated

def create_v2_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys):
    """
    This function baselines the features with baseline method v2 (subtract mean)
    and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """

    # Find Unique Trial ID
    trial_id_in_data = trial_column.unique()

    # Build Dictionary for each trial_id
    baseline_v2 = dict()
    baseline_v2_derivative = dict()
    baseline_v2_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Find time window
        index_window = ((time_column<baseline_window) & (trial_column == trial_id_in_data[i]))
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)

        # Find baseline average based on specified baseline window
        baseline_feature = np.mean(features_phys[index_window], axis = 0)

        # Subtract baseline data (baseline v2)
        baseline_v2[trial_id_in_data[i]] = np.array(features_phys[(trial_column == trial_id_in_data[i])] - baseline_feature)

        # Compute derivative
        time = time_array[time_index]
        baseline_v2_derivative[trial_id_in_data[i]] = np.gradient(baseline_v2[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v2_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v2_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_phys_updated = [s + '_v2' for s in all_features_phys]

    return baseline_v2, baseline_v2_derivative, baseline_v2_second_derivative, all_features_phys_updated

def create_v3_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys):
    """
    This function baselines the features with baseline method v3 (Each feature prior to ROR
    is divided by the first portion of the trial for baseline & the rest of the
    trial is divided by the portion prior to ROR for baseline) and puts them into a
    dictionary per trial. This dictionary is output. The first and second derivative of
    each baselined feature are computed and output in a dictionary.
    """

    # Find Unique Trial ID
    trial_id_in_data = trial_column.unique()

    # Build Dictionary for each trial_id
    baseline_v3 = dict()
    baseline_v3_derivative = dict()
    baseline_v3_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Define first baseline period (during initial specified baseline window)
        index_window1 = ((time_column<baseline_window) & (trial_column == trial_id_in_data[i]))

        # Find indices of other variables
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        time = time_array[time_index]
        event_array = np.array(event_validated_column)

        # Find baseline average based on specified baseline window
        baseline_period1_feature = np.mean(features_phys[index_window1], axis=0)

        # If ROR marked in trial, find when ROR begins and take the baseline directly before that (with window size defined in main)
        current_trial_event = event_array[time_index]
        if 'begin ROR' in current_trial_event:
            begin_ROR_index = np.argwhere(current_trial_event == 'begin ROR')
            time_ROR = time_array[begin_ROR_index][0,0]

            # Define second baseline period
            index_window2 = ((time_column>(time_ROR - baseline_window)) & (time_column<time_ROR) &
                             (trial_column == trial_id_in_data[i]))

            # Find baseline average based on specified baseline window
            baseline_period2_feature = np.mean(features_phys[index_window2], axis=0)

            # Divide features for that trial by correct baseline period
            first_baseline_period = (time_array < (time_ROR - baseline_window)) & (trial_column == trial_id_in_data[i])
            second_baseline_period = (time_array >= (time_ROR - baseline_window)) & (trial_column == trial_id_in_data[i])

            # Stack data from two windows of baselined data together
            baseline_v3[trial_id_in_data[i]] = np.array(features_phys[first_baseline_period] / baseline_period1_feature)
            baseline_v3[trial_id_in_data[i]] = np.vstack((baseline_v3[trial_id_in_data[i]], np.array(features_phys[second_baseline_period] / baseline_period2_feature)))

        # if no labeled begin of ROR, then just use the intial baseline period
        else:
            baseline_v3[trial_id_in_data[i]] = np.array(features_phys[(trial_column == trial_id_in_data[i])] / baseline_period1_feature)


        # Compute derivative
        baseline_v3_derivative[trial_id_in_data[i]] = np.gradient(baseline_v3[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v3_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v3_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_phys_updated = [s + '_v3' for s in all_features_phys]

    return baseline_v3, baseline_v3_derivative, baseline_v3_second_derivative, all_features_phys_updated

def create_v4_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys):
    """
    This function baselines the features with baseline method v4 (The first portion of
    the trial is subtracted from each feature prior to ROR & the period prior to ROR gets
    subtracted from the feature for the rest of the trial) and puts them into a
    dictionary per trial. This dictionary is output. The first and second derivative of
    each baselined feature are computed and output in a dictionary.
    """

    # Find Unique Trial ID
    trial_id_in_data = trial_column.unique()

    # Build Dictionary for each trial_id
    baseline_v4 = dict()
    baseline_v4_derivative = dict()
    baseline_v4_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Define first baseline period (during initial specified baseline window)
        index_window1 = ((time_column<baseline_window) & (trial_column == trial_id_in_data[i]))

        # Find indices of other variables
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        time = time_array[time_index]
        event_array = np.array(event_validated_column)

        # Find baseline average based on specified baseline window
        baseline_period1_feature = np.mean(features_phys[index_window1], axis=0)

        # If ROR marked in trial, find when ROR begins and take the baseline directly before that (with window size defined in main)
        current_trial_event = event_array[time_index]
        if 'begin ROR' in current_trial_event:
            begin_ROR_index = np.argwhere(current_trial_event == 'begin ROR')
            time_ROR = time_array[begin_ROR_index][0,0]

            # Define second baseline period
            index_window2 = ((time_column>(time_ROR - baseline_window)) & (time_column<time_ROR)
                             & (trial_column == trial_id_in_data[i]))

            # Find baseline average based on specified baseline window
            baseline_period2_feature = np.mean(features_phys[index_window2], axis=0)

            # Subtract features for that trial by correct baseline period
            first_baseline_period = (time_array < (time_ROR - baseline_window)) & (trial_column == trial_id_in_data[i])
            second_baseline_period = (time_array >= (time_ROR - baseline_window)) & (trial_column == trial_id_in_data[i])

            # Stack data from two windows of baselined data together
            baseline_v4[trial_id_in_data[i]] = np.array(features_phys[(first_baseline_period)] - baseline_period1_feature)
            baseline_v4[trial_id_in_data[i]] = np.vstack((baseline_v4[trial_id_in_data[i]], np.array(features_phys[(second_baseline_period)] - baseline_period2_feature)))

        # if no labeled begin of ROR, then just use the intial baseline period
        else:
            baseline_v4[trial_id_in_data[i]] = np.array(features_phys[(trial_column == trial_id_in_data[i])] / baseline_period1_feature)


        # Compute derivative
        baseline_v4_derivative[trial_id_in_data[i]] = np.gradient(baseline_v4[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v4_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v4_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_phys_updated = [s + '_v4' for s in all_features_phys]

    return baseline_v4, baseline_v4_derivative, baseline_v4_second_derivative, all_features_phys_updated

def create_v5_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg):
    """
    This function baselines the features with baseline method v5 (divide by RHR
    from another study) and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """

    # Find Unique Trial ID
    trial_id_in_data = trial_column.unique()

    # Build Dictionary for each trial_id
    baseline_v5 = dict()
    baseline_v5_derivative = dict()
    baseline_v5_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Find time index and current participant
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        participant_array = np.array(subject_column)

        # Find baseline for current participant
        current_participant = participant_array[time_index][0]
        baseline_feature = participant_seated_rhr[current_participant]

        # Divide features for that trial by baselined data
        baseline_v5[trial_id_in_data[i]] = np.array(features_ecg[(trial_column == trial_id_in_data[i])] /baseline_feature)

        # Compute derivative
        time = time_array[time_index]
        baseline_v5_derivative[trial_id_in_data[i]] = np.gradient(baseline_v5[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v5_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v5_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_ecg_updated = [s + '_v5' for s in all_features_ecg]

    return baseline_v5, baseline_v5_derivative, baseline_v5_second_derivative, all_features_ecg_updated

def create_v6_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg):
    """
    This function baselines the features with baseline method v6 (subtract RHR
    from another study) and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """

    # Find Unique Trial ID
    trial_id_in_data = trial_column.unique()

    # Build Dictionary for each trial_id
    baseline_v6 = dict()
    baseline_v6_derivative = dict()
    baseline_v6_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Find time index and current participant
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        participant_array = np.array(subject_column)

        # Find baseline for current participant
        current_participant = participant_array[time_index][0]
        baseline_feature = participant_seated_rhr[current_participant]

        # Divide features for that trial by baselined data
        baseline_v6[trial_id_in_data[i]] = np.array(features_ecg[(trial_column == trial_id_in_data[i])] - baseline_feature)

        # Compute derivative
        time = time_array[time_index]
        baseline_v6_derivative[trial_id_in_data[i]] = np.gradient(baseline_v6[trial_id_in_data[i]], time, axis = 0)

        # Compute 2nd derivative
        baseline_v6_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v6_derivative[trial_id_in_data[i]], time, axis = 0)

        # Update baseline names
        all_features_ecg_updated = [s + '_v6' for s in all_features_ecg]

    return baseline_v6, baseline_v6_derivative, baseline_v6_second_derivative, all_features_ecg_updated

def create_v7_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg, eeg_baseline_delta,
                       eeg_baseline_theta, eeg_baseline_alpha, eeg_baseline_beta, all_features_eeg):
    """
    This function baselines the features with baseline method v7 (divide by processed EEG baseline)
    and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """
    # Find Unique Trial ID
    trial_id_in_data = trial_column.unique()

    # Build Dictionary for each trial_id
    baseline_v7 = dict()
    baseline_v7_derivative = dict()
    baseline_v7_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Find time index and current participant
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        participant_array = np.array(subject_column)

        # Find baseline for current participant
        current_participant = participant_array[time_index][0]
        baseline_feature_delta = eeg_baseline_delta.loc[current_participant]
        baseline_feature_theta = eeg_baseline_theta.loc[current_participant]
        baseline_feature_alpha = eeg_baseline_alpha.loc[current_participant]
        baseline_feature_beta = eeg_baseline_beta.loc[current_participant]

        # Set current trial
        current_trial_data = features_eeg[time_index]

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
                if baseline_feature_delta.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:,col] = current_trial_data[:,col] / baseline_feature_delta.iloc[channel_index]
            elif name_split[1] == 'theta - EEG':
                # Divide features for that trial by baselined data
                if baseline_feature_theta.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_feature_theta.iloc[channel_index]
            elif name_split[1] == 'alpha - EEG':
                # Divide features for that trial by baselined data
                if baseline_feature_alpha.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_feature_alpha.iloc[channel_index]
            elif name_split[1] == 'beta - EEG':
                # Divide features for that trial by baselined data
                if baseline_feature_beta.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_feature_beta.iloc[channel_index]

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

def create_v8_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg, eeg_baseline_delta,
                       eeg_baseline_theta, eeg_baseline_alpha, eeg_baseline_beta, all_features_eeg):
    """
    This function baselines the features with baseline method v8 (subtract EEG baseline)
    and puts them into a dictionary per trial. This dictionary is output.
    The first and second derivative of each baselined feature are computed and
    output in a dictionary.
    """
    # Find Unique Trial ID
    trial_id_in_data = trial_column.unique()

    # Build Dictionary for each trial_id
    baseline_v8 = dict()
    baseline_v8_derivative = dict()
    baseline_v8_second_derivative = dict()

    # Iterate through trials
    for i in range(np.size(trial_id_in_data)):

        # Find time index and current participant
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        participant_array = np.array(subject_column)

        # Find baseline for current participant
        current_participant = participant_array[time_index][0]
        baseline_feature_delta = eeg_baseline_delta.loc[current_participant]
        baseline_feature_theta = eeg_baseline_theta.loc[current_participant]
        baseline_feature_alpha = eeg_baseline_alpha.loc[current_participant]
        baseline_feature_beta = eeg_baseline_beta.loc[current_participant]

        # Set current trial
        current_trial_data = features_eeg[time_index]

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
                # Subtract features for that trial by baselined data
                if baseline_feature_delta.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:,col] = current_trial_data[:,col] - baseline_feature_delta.iloc[channel_index]
            elif name_split[1] == 'theta - EEG':
                # Subtract features for that trial by baselined data
                if baseline_feature_theta.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_feature_theta.iloc[channel_index]
            elif name_split[1] == 'alpha - EEG':
                # Subtract features for that trial by baselined data
                if baseline_feature_alpha.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_feature_alpha.iloc[channel_index]
            elif name_split[1] == 'beta - EEG':
                # Subtract features for that trial by baselined data
                if baseline_feature_beta.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_feature_beta.iloc[channel_index]

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

def combine_all_baseline(trial_column, baseline, baseline_derivative, baseline_second_derivative, baseline_names):
    """
    This function combines the features, derivative of features, and second derivative of features into one np array.
    """

    # Find Unique Trial ID
    trial_id_in_data = trial_column.unique()

    # Preallocate the dictionary with NumPy arrays
    num_cols = 0
    for method in baseline.keys():
        num_cols += baseline[method][trial_id_in_data[0]].shape[1]*3
    combined_baseline = {trial: np.empty((baseline[list(baseline.keys())[0]][trial].shape[0], num_cols), dtype=np.float32) for trial in trial_id_in_data}
    # combined_baseline2 = {trial: np.empty((baseline[list(baseline.keys())[0]][trial].shape[0], 0)) for trial in
    #                      trial_id_in_data}

    # Iterate through all unique trial_id & combine the baseline, baseline derivative, and baseline second derivative
    for trial in trial_id_in_data:
        # all_baseline_data = []
        for i, method in enumerate(baseline.keys()):
            m,n = baseline[method][trial].shape
            start = i * n
            end = start + n
            combined_baseline[trial][:,start:end] = baseline[method][trial].astype(np.float32)
            combined_baseline[trial][:,start+n:end+n] = baseline_derivative[method][trial].astype(np.float32)
            combined_baseline[trial][:,start+2*n:end+2*n] = baseline_second_derivative[method][trial].astype(np.float32)
            # all_baseline_data.append(baseline[method][trial])
            # all_baseline_data.append(baseline_derivative[method][trial])
            # all_baseline_data.append(baseline_second_derivative[method][trial])

        # combined_baseline2[trial] = np.column_stack(tuple(all_baseline_data))

    combined_baseline_names = sum([baseline_names[method] + [s + '_derivative' for s in baseline_names[method]] +
                                       [s + '_2derivative' for s in baseline_names[method]] for method in baseline_names.keys()], [])

    return combined_baseline, combined_baseline_names


####  OLD CHUCKS AARON DOESN'T WANT TO DELETE
def baseline_data_old(baseline_methods_to_use, gloc_data_reduced, features,time_variable, all_features, gloc,baseline_window,
             features_phys, all_features_phys, features_ecg, all_features_ecg, features_eeg, all_features_eeg,
             baseline_data_filename, list_of_baseline_eeg_processed_files,model_type):

    """
    Baselines pre-feature data based on 'baseline_methods_to_use'
        combined_baseline: dict that specifics 'participant-trial' and their baseline data
        combined_baseline_names: specifies column names of data within combined_baseline['part-trial']
        baseline: dict that specifics 'method' and then 'participant-trial' and their baseline data
        baseline_names: dict that specifies column names of data within baseline['method']['part-trial']
    """

    baseline = dict()
    baseline_derivative = dict()
    baseline_second_derivative = dict()
    baseline_names = dict()

    for method in baseline_methods_to_use:
        if method == 'v0':
            # V0: No Baseline (feature categories: ECG, BR, temp, fnirs, eyetracking, AFE, G, cognitive, strain, EEG)
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[
                method] = (
                create_v0_baseline(gloc_data_reduced, features, time_variable, all_features))

            # Tabulate NaN
            NaN_table, NaN_proportion, NaN_gloc_proportion = tabulateNaN(baseline[method], all_features, gloc,
                                                                         gloc_data_reduced)
            # Nan_proportion_all = pd.read_pickle('../../NaN_proportion_all.pkl')

        if method == 'v1':
            # V1: Divide by Baseline Window (feature categories: ECG, BR, temp, fnirs, eyetracking, EEG)
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[
                method] = (
                create_v1_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys))

        if method == 'v2':
            # V2: Subtract Baseline Window (feature categories: ECG, BR, temp, fnirs, eyetracking, EEG)
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[
                method] = (
                create_v2_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys))

        if method == 'v3':
            # V3: pre ROR: divide by baseline window pre GOR, ROR: divide by baseline window pre ROR
            # feature categories: ECG, BR, temp, fnirs, eyetracking, EEG
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[
                method] = (
                create_v3_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys))

        if method == 'v4':
            # V4: pre ROR: subtract baseline window pre GOR, ROR: subtract baseline window pre ROR
            # feature categories: ECG, BR, temp, fnirs, eyetracking, EEG
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[
                method] = (
                create_v4_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys))

        if method == 'v5':
            # V5: Divide by seated resting HR (feature categories: ECG)
            # Import csv File
            participant_baseline = pd.read_csv(baseline_data_filename)
            participant_seated_rhr = participant_baseline['resting HR [seated]'][0:-1]
            participant_seated_rhr.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                                            '13']

            # V5: Divide by seated resting HR
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[
                method] = (
                create_v5_baseline(baseline_window, gloc_data_reduced, features_ecg, time_variable,
                                   participant_seated_rhr,
                                   all_features_ecg))

        if method == 'v6':
            # V6: Subtract seated resting HR (feature categories: ECG)
            # Import csv File (if not already imported from v5)
            if 'participant_baseline' not in locals() and 'participant_baseline' not in globals():
                participant_baseline = pd.read_csv(baseline_data_filename)
                participant_seated_rhr = participant_baseline['resting HR [seated]'][0:-1]
                participant_seated_rhr.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                                                '13']

            # V6: Subtract seated resting HR
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[
                method] = (
                create_v6_baseline(baseline_window, gloc_data_reduced, features_ecg, time_variable,
                                   participant_seated_rhr,
                                   all_features_ecg))

        if method == 'v7':
            # V7: Divide by resting EEG from first noAFE trial (feature categories: EEG)
            if 'noAFE' in model_type:
                # Import csv files
                eeg_baseline_delta = pd.read_csv(list_of_baseline_eeg_processed_files[0])
                eeg_baseline_delta.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                                            '13']

                eeg_baseline_theta = pd.read_csv(list_of_baseline_eeg_processed_files[1])
                eeg_baseline_theta.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                                            '13']

                eeg_baseline_alpha = pd.read_csv(list_of_baseline_eeg_processed_files[2])
                eeg_baseline_alpha.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                                            '13']

                eeg_baseline_beta = pd.read_csv(list_of_baseline_eeg_processed_files[3])
                eeg_baseline_beta.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

                # V7: Divide by resting EEG from first noAFE trial
                baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[
                    method] = (
                    create_v7_baseline(baseline_window, gloc_data_reduced, features_eeg, time_variable,
                                       eeg_baseline_delta,
                                       eeg_baseline_theta, eeg_baseline_alpha, eeg_baseline_beta, all_features_eeg))

            elif 'AFE' in model_type:
                # Output Warning
                warnings.warn('EEG baseline methods not implemented for AFE conditions yet. Waiting on data.')

        if method == 'v8':
            # V8: Subtract resting EEG from first noAFE trial (feature categories: EEG)
            # Import csv File (if not already imported from v5)
            if 'eeg_baseline_delta' not in locals() and 'eeg_baseline_delta' not in globals():
                if 'noAFE' in model_type:
                    # Import csv files
                    eeg_baseline_delta = pd.read_csv(list_of_baseline_eeg_processed_files[0])
                    eeg_baseline_delta.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                                                '13']

                    eeg_baseline_theta = pd.read_csv(list_of_baseline_eeg_processed_files[1])
                    eeg_baseline_theta.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                                                '13']

                    eeg_baseline_alpha = pd.read_csv(list_of_baseline_eeg_processed_files[2])
                    eeg_baseline_alpha.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                                                '13']

                    eeg_baseline_beta = pd.read_csv(list_of_baseline_eeg_processed_files[3])
                    eeg_baseline_beta.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                                               '13']

                elif 'AFE' in model_type:
                    # Output Warning
                    warnings.warn('EEG baseline methods not implemented for AFE conditions yet. Waiting on data.')
            else:
                # V8: Subtract resting EEG from first noAFE trial
                baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[
                    method] = (
                    create_v8_baseline(baseline_window, gloc_data_reduced, features_eeg, time_variable,
                                       eeg_baseline_delta,
                                       eeg_baseline_theta, eeg_baseline_alpha, eeg_baseline_beta, all_features_eeg))

    # Combine all baseline methods into a large dictionary
    combined_baseline, combined_baseline_names = combine_all_baseline(gloc_data_reduced, baseline, baseline_derivative,
                                                                      baseline_second_derivative, baseline_names)
    baseline_v0 = baseline['v0']
    baseline_names_v0 = baseline_names['v0']

    return combined_baseline, combined_baseline_names, baseline_v0, baseline_names_v0