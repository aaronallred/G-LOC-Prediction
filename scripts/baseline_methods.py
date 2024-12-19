import numpy as np
import pandas as pd
import os

def create_v0_baseline(gloc_data_reduced, features, time_variable, all_features):
    """
    This function takes features and puts them into a matrix with baseline method v0 (no baseline).
    The first and second derivative of each feature are computed and output.
    """
    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v0 = dict()
    baseline_v0_derivative = dict()
    baseline_v0_second_derivative = dict()

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
    This function baselines features for each trial being analyzed. The baseline window is specified
    in main. Each feature is defined by the average of the baseline feature for that trial.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v1 = dict()
    baseline_v1_derivative = dict()
    baseline_v1_second_derivative = dict()

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
    This function baselines features for each trial being analyzed. The baseline window is specified
    in main. Each feature is defined by the average of the baseline feature for that trial.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v2 = dict()
    baseline_v2_derivative = dict()
    baseline_v2_second_derivative = dict()

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
    This function baselines features for each trial being analyzed. The width of the baseline window is specified
    in main. Each feature prior to ROR is divided by the first portion of the trial for baseline. The rest of the
    trial is divided by the portion prior to ROR for baseline.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v3 = dict()
    baseline_v3_derivative = dict()
    baseline_v3_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):

        # Find time window
        index_window1 = ((gloc_data_reduced[time_variable]<baseline_window) & (gloc_data_reduced.trial_id == trial_id_in_data[i]))
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
            index_window2 = ((gloc_data_reduced[time_variable]>(time_ROR - baseline_window)) & (gloc_data_reduced[time_variable]<time_ROR)
                                                                                                & (gloc_data_reduced.trial_id == trial_id_in_data[i]))

            # Find baseline average based on specified baseline window
            baseline_period2_feature = np.mean(features_phys[index_window2], axis=0)

            # Divide features for that trial by correct baseline period
            first_baseline_period = (time_array < (time_ROR - baseline_window)) & (gloc_data_reduced.trial_id == trial_id_in_data[i])
            second_baseline_period = (time_array >= (time_ROR - baseline_window)) & (gloc_data_reduced.trial_id == trial_id_in_data[i])

            baseline_v3[trial_id_in_data[i]] = np.array(features_phys[(first_baseline_period)] / baseline_period1_feature)
            baseline_v3[trial_id_in_data[i]] = np.vstack((baseline_v3[trial_id_in_data[i]], np.array(features_phys[(second_baseline_period)] / baseline_period2_feature)))

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
    This function baselines features for each trial being analyzed. The width of the baseline window is specified
    in main. The first portion of the trial is subtracted from each feature prior to ROR for baseline. For the rest of the
    trial, the portion prior to ROR is subtracted from the feature to baseline.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v4 = dict()
    baseline_v4_derivative = dict()
    baseline_v4_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):

        # Find time window
        index_window1 = ((gloc_data_reduced[time_variable]<baseline_window) & (gloc_data_reduced.trial_id == trial_id_in_data[i]))
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
            index_window2 = ((gloc_data_reduced[time_variable]>(time_ROR - baseline_window)) & (gloc_data_reduced[time_variable]<time_ROR)
                                                                                                & (gloc_data_reduced.trial_id == trial_id_in_data[i]))

            # Find baseline average based on specified baseline window
            baseline_period2_feature = np.mean(features_phys[index_window2], axis=0)

            # Divide features for that trial by correct baseline period
            first_baseline_period = (time_array < (time_ROR - baseline_window)) & (gloc_data_reduced.trial_id == trial_id_in_data[i])
            second_baseline_period = (time_array >= (time_ROR - baseline_window)) & (gloc_data_reduced.trial_id == trial_id_in_data[i])

            baseline_v4[trial_id_in_data[i]] = np.array(features_phys[(first_baseline_period)] - baseline_period1_feature)
            baseline_v4[trial_id_in_data[i]] = np.vstack((baseline_v4[trial_id_in_data[i]], np.array(features_phys[(second_baseline_period)] - baseline_period2_feature)))

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
    This function baselines features for each trial being analyzed. Each feature is divided by the resting HR
    from another study.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v5 = dict()
    baseline_v5_derivative = dict()
    baseline_v5_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):

        # Find time window
        index_window = ((gloc_data_reduced[time_variable]<baseline_window) & (gloc_data_reduced.trial_id == trial_id_in_data[i]))
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
    This function baselines features for each trial being analyzed. The resting HR
    from another study is subtracted from every feature.
    """

    # Find Unique Trial ID
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Build Dictionary for each trial_id
    baseline_v6 = dict()
    baseline_v6_derivative = dict()
    baseline_v6_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):

        # Find time window
        index_window = ((gloc_data_reduced[time_variable]<baseline_window) & (gloc_data_reduced.trial_id == trial_id_in_data[i]))
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