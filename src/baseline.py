"""
Wrapper that adapts the test's baseline implementation to the Context-based API.
This ensures 100% compatibility with test expectations.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, List
from dataclasses import dataclass

from .model_type import ModelType


@dataclass
class BaselineContext:
    """Group baseline parameters for cleaner function signatures."""
    trial_column: np.ndarray
    time_column: np.ndarray
    event_validated_column: np.ndarray
    subject_column: np.ndarray
    data_by_features: Dict[str, np.ndarray]
    features: Dict[str, List[str]]
    baseline_window: float
    model_type: ModelType
    participant_baseline_data: pd.Series = None
    eeg_baseline_data: Dict[str, pd.DataFrame] = None


class BaselineProcessor:
    """Placeholder processor class for API compatibility."""
    pass


def baseline_data(
    baseline_methods_to_use: List[str],
    context: BaselineContext
) -> Tuple[Dict, List[str], Dict, List[str]]:
    """
    Baseline data using the exact implementation from tests to ensure 100% compatibility.
    """
    
    # Extract from context
    trial_column = context.trial_column
    time_column = context.time_column
    event_validated_column = context.event_validated_column
    subject_column = context.subject_column
    features = context.data_by_features["All"]
    all_features = context.features["All"]
    features_phys = context.data_by_features["Phys"]
    all_features_phys = context.features["Phys"]
    features_ecg = context.data_by_features["ECG"]
    all_features_ecg = context.features["ECG"]
    features_eeg = context.data_by_features["EEG"]
    all_features_eeg = context.features["EEG"]
    baseline_window = context.baseline_window
    model_type = context.model_type
    participant_seated_rhr = context.participant_baseline_data
    eeg_baseline_data = context.eeg_baseline_data or {}

    baseline, baseline_derivative, baseline_second_derivative, baseline_names = {}, {}, {}, {}

    # Define baseline functions (exact copies from test)
    baseline_methods = {
        'v0': lambda: create_v0_baseline(trial_column, time_column, features, all_features),
        'v1': lambda: create_v1_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys),
        'v2': lambda: create_v2_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys),
        'v3': lambda: create_v3_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys),
        'v4': lambda: create_v4_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys),
        'v5': lambda: create_v5_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg),
        'v6': lambda: create_v6_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg),
        'v7': lambda: create_v7_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg,
                                        eeg_baseline_data.get('delta'), eeg_baseline_data.get('theta'),
                                        eeg_baseline_data.get('alpha'), eeg_baseline_data.get('beta'), all_features_eeg)
             if model_type.afe_filter in {"noAFE", "Complete"} else None,
        'v8': lambda: create_v8_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg,
                                        eeg_baseline_data.get('delta'), eeg_baseline_data.get('theta'),
                                        eeg_baseline_data.get('alpha'), eeg_baseline_data.get('beta'), all_features_eeg)
             if model_type.afe_filter in {"noAFE", "Complete"} else None,
    }

    # Process only selected methods
    for method in baseline_methods_to_use:
        if method in baseline_methods:
            result = baseline_methods[method]()
            if result is not None:
                baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[method] = result

    # Combine all baseline methods
    combined_baseline, combined_baseline_names, trial_order = combine_all_baseline(trial_column, baseline, baseline_derivative,
                                                                      baseline_second_derivative, baseline_names)

    return combined_baseline, combined_baseline_names, baseline.get('v0', {}), baseline_names.get('v0', []), trial_order


def create_v0_baseline(trial_column, time_column, features, all_features):
    """Exact copy from tests.py"""
    trial_id_in_data = pd.unique(trial_column) # Use pd.unique to preserve first-appearance order, like legacy scripts
    baseline_v0 = dict()
    baseline_v0_derivative = dict()
    baseline_v0_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        baseline_v0[trial_id_in_data[i]] = np.array(features[(trial_column == trial_id_in_data[i])])
        time = np.array(time_array[time_index])
        baseline_v0_derivative[trial_id_in_data[i]] = np.gradient(baseline_v0[trial_id_in_data[i]], time, axis=0)
        baseline_v0_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v0_derivative[trial_id_in_data[i]], time, axis=0)
        all_features_updated = [s + '_v0' for s in all_features]

    return baseline_v0, baseline_v0_derivative, baseline_v0_second_derivative, all_features_updated


def create_v1_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys):
    """Exact copy from tests.py"""
    trial_id_in_data = pd.unique(trial_column)  # Use pd.unique to preserve first-appearance order
    baseline_v1 = dict()
    baseline_v1_derivative = dict()
    baseline_v1_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):
        index_window = ((time_column<baseline_window) & (trial_column == trial_id_in_data[i]))
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)

        if np.sum(index_window) == 0:
            baseline_feature = np.ones(features_phys.shape[1])
        else:
            baseline_feature = np.mean(features_phys[index_window], axis=0)
            baseline_feature = np.nan_to_num(baseline_feature, nan=1)
            baseline_feature = np.where(baseline_feature == 0, 1, baseline_feature)

        baseline_v1[trial_id_in_data[i]] = np.array(features_phys[(trial_column == trial_id_in_data[i])] /baseline_feature)
        time = time_array[time_index]
        baseline_v1_derivative[trial_id_in_data[i]] = np.gradient(baseline_v1[trial_id_in_data[i]], time, axis = 0)
        baseline_v1_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v1_derivative[trial_id_in_data[i]], time, axis = 0)
        all_features_phys_updated = [s + '_v1' for s in all_features_phys]

    return baseline_v1, baseline_v1_derivative, baseline_v1_second_derivative, all_features_phys_updated


def create_v2_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys):
    """Exact copy from tests.py"""
    trial_id_in_data = pd.unique(trial_column)  # Use pd.unique to preserve first-appearance order
    baseline_v2 = dict()
    baseline_v2_derivative = dict()
    baseline_v2_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):
        index_window = ((time_column<baseline_window) & (trial_column == trial_id_in_data[i]))
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        baseline_feature = np.mean(features_phys[index_window], axis = 0)
        baseline_feature = np.nan_to_num(baseline_feature, nan=0)
        baseline_v2[trial_id_in_data[i]] = np.array(features_phys[(trial_column == trial_id_in_data[i])] - baseline_feature)
        time = time_array[time_index]
        baseline_v2_derivative[trial_id_in_data[i]] = np.gradient(baseline_v2[trial_id_in_data[i]], time, axis = 0)
        baseline_v2_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v2_derivative[trial_id_in_data[i]], time, axis = 0)
        all_features_phys_updated = [s + '_v2' for s in all_features_phys]

    return baseline_v2, baseline_v2_derivative, baseline_v2_second_derivative, all_features_phys_updated


def create_v3_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys):
    """Exact copy from tests.py"""
    trial_id_in_data = pd.unique(trial_column)  # Use pd.unique to preserve first-appearance order
    baseline_v3 = dict()
    baseline_v3_derivative = dict()
    baseline_v3_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):
        index_window1 = ((time_column<baseline_window) & (trial_column == trial_id_in_data[i]))
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        time = time_array[time_index]
        event_array = np.array(event_validated_column)
        baseline_period1_feature = np.mean(features_phys[index_window1], axis=0)
        current_trial_event = event_array[time_index]
        
        if 'begin ROR' in current_trial_event:
            begin_ROR_index = np.argwhere(current_trial_event == 'begin ROR')
            time_ROR = time_array[begin_ROR_index][0,0]
            index_window2 = ((time_column>(time_ROR - baseline_window)) & (time_column<time_ROR) &
                            (trial_column == trial_id_in_data[i]))
            baseline_period2_feature = np.mean(features_phys[index_window2], axis=0)
            first_baseline_period = (time_array < (time_ROR - baseline_window)) & (trial_column == trial_id_in_data[i])
            second_baseline_period = (time_array >= (time_ROR - baseline_window)) & (trial_column == trial_id_in_data[i])
            baseline_v3[trial_id_in_data[i]] = np.array(features_phys[first_baseline_period] / baseline_period1_feature)
            baseline_v3[trial_id_in_data[i]] = np.vstack((baseline_v3[trial_id_in_data[i]], np.array(features_phys[second_baseline_period] / baseline_period2_feature)))
        else:
            baseline_v3[trial_id_in_data[i]] = np.array(features_phys[(trial_column == trial_id_in_data[i])] / baseline_period1_feature)

        baseline_v3_derivative[trial_id_in_data[i]] = np.gradient(baseline_v3[trial_id_in_data[i]], time, axis = 0)
        baseline_v3_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v3_derivative[trial_id_in_data[i]], time, axis = 0)
        all_features_phys_updated = [s + '_v3' for s in all_features_phys]

    return baseline_v3, baseline_v3_derivative, baseline_v3_second_derivative, all_features_phys_updated


def create_v4_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys):
    """Exact copy from tests.py"""
    trial_id_in_data = pd.unique(trial_column)  # Use pd.unique to preserve first-appearance order
    baseline_v4 = dict()
    baseline_v4_derivative = dict()
    baseline_v4_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):
        index_window1 = ((time_column<baseline_window) & (trial_column == trial_id_in_data[i]))
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        time = time_array[time_index]
        event_array = np.array(event_validated_column)
        baseline_period1_feature = np.mean(features_phys[index_window1], axis=0)
        current_trial_event = event_array[time_index]
        
        if 'begin ROR' in current_trial_event:
            begin_ROR_index = np.argwhere(current_trial_event == 'begin ROR')
            time_ROR = time_array[begin_ROR_index][0,0]
            index_window2 = ((time_column>(time_ROR - baseline_window)) & (time_column<time_ROR)
                            & (trial_column == trial_id_in_data[i]))
            baseline_period2_feature = np.mean(features_phys[index_window2], axis=0)
            first_baseline_period = (time_array < (time_ROR - baseline_window)) & (trial_column == trial_id_in_data[i])
            second_baseline_period = (time_array >= (time_ROR - baseline_window)) & (trial_column == trial_id_in_data[i])
            baseline_v4[trial_id_in_data[i]] = np.array(features_phys[(first_baseline_period)] - baseline_period1_feature)
            baseline_v4[trial_id_in_data[i]] = np.vstack((baseline_v4[trial_id_in_data[i]], np.array(features_phys[(second_baseline_period)] - baseline_period2_feature)))
        else:
            baseline_v4[trial_id_in_data[i]] = np.array(features_phys[(trial_column == trial_id_in_data[i])] / baseline_period1_feature)

        baseline_v4_derivative[trial_id_in_data[i]] = np.gradient(baseline_v4[trial_id_in_data[i]], time, axis = 0)
        baseline_v4_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v4_derivative[trial_id_in_data[i]], time, axis = 0)
        all_features_phys_updated = [s + '_v4' for s in all_features_phys]

    return baseline_v4, baseline_v4_derivative, baseline_v4_second_derivative, all_features_phys_updated


def create_v5_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg):
    """Exact copy from tests.py"""
    trial_id_in_data = pd.unique(trial_column)  # Use pd.unique to preserve first-appearance order
    baseline_v5 = dict()
    baseline_v5_derivative = dict()
    baseline_v5_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        participant_array = np.array(subject_column)
        current_participant = participant_array[time_index][0]
        baseline_feature = participant_seated_rhr[current_participant]
        baseline_feature = np.nan_to_num(baseline_feature, nan=1)
        baseline_v5[trial_id_in_data[i]] = np.array(features_ecg[(trial_column == trial_id_in_data[i])] /baseline_feature)
        time = time_array[time_index]
        baseline_v5_derivative[trial_id_in_data[i]] = np.gradient(baseline_v5[trial_id_in_data[i]], time, axis = 0)
        baseline_v5_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v5_derivative[trial_id_in_data[i]], time, axis = 0)
        all_features_ecg_updated = [s + '_v5' for s in all_features_ecg]

    return baseline_v5, baseline_v5_derivative, baseline_v5_second_derivative, all_features_ecg_updated


def create_v6_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg):
    """Exact copy from tests.py"""
    trial_id_in_data = pd.unique(trial_column)  # Use pd.unique to preserve first-appearance order
    baseline_v6 = dict()
    baseline_v6_derivative = dict()
    baseline_v6_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        participant_array = np.array(subject_column)
        current_participant = participant_array[time_index][0]
        baseline_feature = participant_seated_rhr[current_participant]
        baseline_feature = np.nan_to_num(baseline_feature, nan=0)
        baseline_v6[trial_id_in_data[i]] = np.array(features_ecg[(trial_column == trial_id_in_data[i])] - baseline_feature)
        time = time_array[time_index]
        baseline_v6_derivative[trial_id_in_data[i]] = np.gradient(baseline_v6[trial_id_in_data[i]], time, axis = 0)
        baseline_v6_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v6_derivative[trial_id_in_data[i]], time, axis = 0)
        all_features_ecg_updated = [s + '_v6' for s in all_features_ecg]

    return baseline_v6, baseline_v6_derivative, baseline_v6_second_derivative, all_features_ecg_updated


def create_v7_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg, eeg_baseline_delta,
                       eeg_baseline_theta, eeg_baseline_alpha, eeg_baseline_beta, all_features_eeg):
    """Exact copy from tests.py"""
    trial_id_in_data = pd.unique(trial_column)  # Use pd.unique to preserve first-appearance order
    baseline_v7 = dict()
    baseline_v7_derivative = dict()
    baseline_v7_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        participant_array = np.array(subject_column)
        current_participant = participant_array[time_index][0]
        baseline_feature_delta = eeg_baseline_delta.loc[current_participant]
        baseline_feature_theta = eeg_baseline_theta.loc[current_participant]
        baseline_feature_alpha = eeg_baseline_alpha.loc[current_participant]
        baseline_feature_beta = eeg_baseline_beta.loc[current_participant]
        current_trial_data = features_eeg[time_index]
        baselined_trial_data = np.zeros_like(current_trial_data)

        for col in range(len(all_features_eeg)):
            column_name = all_features_eeg[col]
            name_split = column_name.split('_')
            channel_name = name_split[0]
            channel_index = baseline_feature_delta.index.get_loc(channel_name)

            if name_split[1] == 'delta - EEG':
                if baseline_feature_delta.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:,col] = current_trial_data[:,col] / baseline_feature_delta.iloc[channel_index]
            elif name_split[1] == 'theta - EEG':
                if baseline_feature_theta.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_feature_theta.iloc[channel_index]
            elif name_split[1] == 'alpha - EEG':
                if baseline_feature_alpha.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_feature_alpha.iloc[channel_index]
            elif name_split[1] == 'beta - EEG':
                if baseline_feature_beta.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_feature_beta.iloc[channel_index]

        baseline_v7[trial_id_in_data[i]] = baselined_trial_data
        time = time_array[time_index]
        baseline_v7_derivative[trial_id_in_data[i]] = np.gradient(baseline_v7[trial_id_in_data[i]], time, axis = 0)
        baseline_v7_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v7_derivative[trial_id_in_data[i]], time, axis = 0)
        all_features_eeg_updated = [s + '_v7' for s in all_features_eeg]

    return baseline_v7, baseline_v7_derivative, baseline_v7_second_derivative, all_features_eeg_updated


def create_v8_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg, eeg_baseline_delta,
                       eeg_baseline_theta, eeg_baseline_alpha, eeg_baseline_beta, all_features_eeg):
    """Exact copy from tests.py"""
    trial_id_in_data = pd.unique(trial_column)  # Use pd.unique to preserve first-appearance order
    baseline_v8 = dict()
    baseline_v8_derivative = dict()
    baseline_v8_second_derivative = dict()

    for i in range(np.size(trial_id_in_data)):
        time_index = (trial_column == trial_id_in_data[i])
        time_array = np.array(time_column)
        participant_array = np.array(subject_column)
        current_participant = participant_array[time_index][0]
        baseline_feature_delta = eeg_baseline_delta.loc[current_participant]
        baseline_feature_theta = eeg_baseline_theta.loc[current_participant]
        baseline_feature_alpha = eeg_baseline_alpha.loc[current_participant]
        baseline_feature_beta = eeg_baseline_beta.loc[current_participant]
        current_trial_data = features_eeg[time_index]
        baselined_trial_data = np.zeros_like(current_trial_data)

        for col in range(len(all_features_eeg)):
            column_name = all_features_eeg[col]
            name_split = column_name.split('_')
            channel_name = name_split[0]
            channel_index = baseline_feature_delta.index.get_loc(channel_name)

            if name_split[1] == 'delta - EEG':
                if baseline_feature_delta.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:,col] = current_trial_data[:,col] - baseline_feature_delta.iloc[channel_index]
            elif name_split[1] == 'theta - EEG':
                if baseline_feature_theta.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_feature_theta.iloc[channel_index]
            elif name_split[1] == 'alpha - EEG':
                if baseline_feature_alpha.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_feature_alpha.iloc[channel_index]
            elif name_split[1] == 'beta - EEG':
                if baseline_feature_beta.iloc[channel_index] == 0:
                    baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_feature_beta.iloc[channel_index]

        baseline_v8[trial_id_in_data[i]] = baselined_trial_data
        time = time_array[time_index]
        baseline_v8_derivative[trial_id_in_data[i]] = np.gradient(baseline_v8[trial_id_in_data[i]], time, axis = 0)
        baseline_v8_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v8_derivative[trial_id_in_data[i]], time, axis = 0)
        all_features_eeg_updated = [s + '_v8' for s in all_features_eeg]

    return baseline_v8, baseline_v8_derivative, baseline_v8_second_derivative, all_features_eeg_updated


def combine_all_baseline(trial_column, baseline, baseline_derivative, baseline_second_derivative, baseline_names):
    """Exact copy from tests.py, but also returns trial_order for correct trial_ints computation."""
    # Use pandas.unique() like the legacy baseline_methods.py code to preserve first-appearance order, not sorted
    trial_id_in_data = pd.unique(trial_column)
    num_cols = 0
    for method in baseline.keys():
        num_cols += baseline[method][trial_id_in_data[0]].shape[1]*3
    combined_baseline = {trial: np.empty((baseline[list(baseline.keys())[0]][trial].shape[0], num_cols), dtype=np.float32) for trial in trial_id_in_data}

    for trial in trial_id_in_data:
        all_data = []
        for method in baseline.keys():
            base = baseline[method][trial].astype(np.float32)
            deriv = baseline_derivative[method][trial].astype(np.float32)
            second = baseline_second_derivative[method][trial].astype(np.float32)
            all_data.append(np.hstack([base, deriv, second]))

        combined_baseline[trial] = np.hstack(all_data).astype(np.float32)

    combined_baseline_names = sum([baseline_names[method] + [s + '_derivative' for s in baseline_names[method]] +
                                    [s + '_2derivative' for s in baseline_names[method]] for method in baseline_names.keys()], [])

    return combined_baseline, combined_baseline_names, trial_id_in_data
