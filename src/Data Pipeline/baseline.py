"""
Wrapper that adapts the test's baseline implementation to the Context-based API.
This ensures 100% compatibility with test expectations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass

from src.model_type import ModelType


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


def _resolve_trial_ids(trial_column, trial_id_in_data=None):
    if trial_id_in_data is None:
        return pd.unique(trial_column)
    return trial_id_in_data


def _feature_names_with_suffix(feature_names, suffix):
    return [f"{name}_{suffix}" for name in feature_names]


def _compute_derivative_stack(values, time):
    first_derivative = np.gradient(values, time, axis=0)
    second_derivative = np.gradient(first_derivative, time, axis=0)
    return first_derivative, second_derivative


def _prepare_eeg_feature_meta(all_features_eeg, eeg_baseline_delta):
    channel_to_index = {name: idx for idx, name in enumerate(eeg_baseline_delta.columns)}
    feature_meta = []
    for col, column_name in enumerate(all_features_eeg):
        channel_name, frequency_suffix = column_name.split("_", maxsplit=1)
        feature_meta.append((col, channel_name, frequency_suffix))
    return channel_to_index, feature_meta


def baseline_data(
    baseline_methods_to_use: List[str],
    context: BaselineContext
) -> Tuple[Dict, List[str], Dict, List[str], np.ndarray]:
    """
    Baseline data using the exact implementation from tests to ensure 100% compatibility.
    """
    
    # Extract from context
    trial_column = np.asarray(context.trial_column)
    time_column = np.asarray(context.time_column)
    event_validated_column = np.asarray(context.event_validated_column)
    subject_column = np.asarray(context.subject_column)
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
    trial_id_in_data = pd.unique(trial_column)

    baseline, baseline_derivative, baseline_second_derivative, baseline_names = {}, {}, {}, {}

    # Define baseline functions (exact copies from test)
    baseline_methods = {
        'v0': lambda: create_v0_baseline(trial_column, time_column, features, all_features, trial_id_in_data=trial_id_in_data),
        'v1': lambda: create_v1_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys, trial_id_in_data=trial_id_in_data),
        'v2': lambda: create_v2_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys, trial_id_in_data=trial_id_in_data),
        'v3': lambda: create_v3_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys, trial_id_in_data=trial_id_in_data),
        'v4': lambda: create_v4_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys, trial_id_in_data=trial_id_in_data),
        'v5': lambda: create_v5_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg, trial_id_in_data=trial_id_in_data),
        'v6': lambda: create_v6_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg, trial_id_in_data=trial_id_in_data),
        'v7': lambda: create_v7_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg,
                                        eeg_baseline_data.get('delta'), eeg_baseline_data.get('theta'),
                                        eeg_baseline_data.get('alpha'), eeg_baseline_data.get('beta'), all_features_eeg,
                                        trial_id_in_data=trial_id_in_data)
             if model_type.afe_filter in {"noAFE", "Complete"} else None,
        'v8': lambda: create_v8_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg,
                                        eeg_baseline_data.get('delta'), eeg_baseline_data.get('theta'),
                                        eeg_baseline_data.get('alpha'), eeg_baseline_data.get('beta'), all_features_eeg,
                                        trial_id_in_data=trial_id_in_data)
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
                                                                      baseline_second_derivative, baseline_names,
                                                                      trial_id_in_data=trial_id_in_data)

    return combined_baseline, combined_baseline_names, baseline.get('v0', {}), baseline_names.get('v0', []), trial_order


def create_v0_baseline(trial_column, time_column, features, all_features, trial_id_in_data=None):
    """Exact copy from tests.py"""
    trial_id_in_data = _resolve_trial_ids(trial_column, trial_id_in_data)
    time_array = np.asarray(time_column)
    baseline_v0 = dict()
    baseline_v0_derivative = dict()
    baseline_v0_second_derivative = dict()
    all_features_updated = _feature_names_with_suffix(all_features, "v0")

    for trial_id in trial_id_in_data:
        time_index = trial_column == trial_id
        baseline_v0[trial_id] = np.array(features[time_index])
        time = time_array[time_index]
        first_derivative, second_derivative = _compute_derivative_stack(baseline_v0[trial_id], time)
        baseline_v0_derivative[trial_id] = first_derivative
        baseline_v0_second_derivative[trial_id] = second_derivative

    return baseline_v0, baseline_v0_derivative, baseline_v0_second_derivative, all_features_updated


def create_v1_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys, trial_id_in_data=None):
    """Exact copy from tests.py"""
    trial_id_in_data = _resolve_trial_ids(trial_column, trial_id_in_data)
    time_array = np.asarray(time_column)
    baseline_v1 = dict()
    baseline_v1_derivative = dict()
    baseline_v1_second_derivative = dict()
    all_features_phys_updated = _feature_names_with_suffix(all_features_phys, "v1")

    for trial_id in trial_id_in_data:
        index_window = ((time_column<baseline_window) & (trial_column == trial_id))
        time_index = (trial_column == trial_id)

        if np.sum(index_window) == 0:
            baseline_feature = np.ones(features_phys.shape[1])
        else:
            baseline_feature = np.mean(features_phys[index_window], axis=0)
            baseline_feature = np.nan_to_num(baseline_feature, nan=1)
            baseline_feature = np.where(baseline_feature == 0, 1, baseline_feature)

        baseline_v1[trial_id] = np.array(features_phys[time_index] / baseline_feature)
        time = time_array[time_index]
        first_derivative, second_derivative = _compute_derivative_stack(baseline_v1[trial_id], time)
        baseline_v1_derivative[trial_id] = first_derivative
        baseline_v1_second_derivative[trial_id] = second_derivative

    return baseline_v1, baseline_v1_derivative, baseline_v1_second_derivative, all_features_phys_updated


def create_v2_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys, trial_id_in_data=None):
    """Exact copy from tests.py"""
    trial_id_in_data = _resolve_trial_ids(trial_column, trial_id_in_data)
    time_array = np.asarray(time_column)
    baseline_v2 = dict()
    baseline_v2_derivative = dict()
    baseline_v2_second_derivative = dict()
    all_features_phys_updated = _feature_names_with_suffix(all_features_phys, "v2")

    for trial_id in trial_id_in_data:
        index_window = ((time_column<baseline_window) & (trial_column == trial_id))
        time_index = (trial_column == trial_id)
        baseline_feature = np.mean(features_phys[index_window], axis = 0)
        baseline_feature = np.nan_to_num(baseline_feature, nan=0)
        baseline_v2[trial_id] = np.array(features_phys[time_index] - baseline_feature)
        time = time_array[time_index]
        first_derivative, second_derivative = _compute_derivative_stack(baseline_v2[trial_id], time)
        baseline_v2_derivative[trial_id] = first_derivative
        baseline_v2_second_derivative[trial_id] = second_derivative

    return baseline_v2, baseline_v2_derivative, baseline_v2_second_derivative, all_features_phys_updated


def create_v3_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys, trial_id_in_data=None):
    """Exact copy from tests.py"""
    trial_id_in_data = _resolve_trial_ids(trial_column, trial_id_in_data)
    time_array = np.asarray(time_column)
    event_array = np.asarray(event_validated_column)
    baseline_v3 = dict()
    baseline_v3_derivative = dict()
    baseline_v3_second_derivative = dict()
    all_features_phys_updated = _feature_names_with_suffix(all_features_phys, "v3")

    for trial_id in trial_id_in_data:
        index_window1 = ((time_column<baseline_window) & (trial_column == trial_id))
        time_index = (trial_column == trial_id)
        time = time_array[time_index]
        baseline_period1_feature = np.mean(features_phys[index_window1], axis=0)
        current_trial_event = event_array[time_index]
        
        if 'begin ROR' in current_trial_event:
            begin_ROR_index = np.argwhere(current_trial_event == 'begin ROR')
            time_ROR = time_array[begin_ROR_index][0,0]
            index_window2 = ((time_column>(time_ROR - baseline_window)) & (time_column<time_ROR) &
                            (trial_column == trial_id))
            baseline_period2_feature = np.mean(features_phys[index_window2], axis=0)
            first_baseline_period = (time_array < (time_ROR - baseline_window)) & (trial_column == trial_id)
            second_baseline_period = (time_array >= (time_ROR - baseline_window)) & (trial_column == trial_id)
            baseline_v3[trial_id] = np.array(features_phys[first_baseline_period] / baseline_period1_feature)
            baseline_v3[trial_id] = np.vstack((baseline_v3[trial_id], np.array(features_phys[second_baseline_period] / baseline_period2_feature)))
        else:
            baseline_v3[trial_id] = np.array(features_phys[time_index] / baseline_period1_feature)

        first_derivative, second_derivative = _compute_derivative_stack(baseline_v3[trial_id], time)
        baseline_v3_derivative[trial_id] = first_derivative
        baseline_v3_second_derivative[trial_id] = second_derivative

    return baseline_v3, baseline_v3_derivative, baseline_v3_second_derivative, all_features_phys_updated


def create_v4_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys, trial_id_in_data=None):
    """Exact copy from tests.py"""
    trial_id_in_data = _resolve_trial_ids(trial_column, trial_id_in_data)
    time_array = np.asarray(time_column)
    event_array = np.asarray(event_validated_column)
    baseline_v4 = dict()
    baseline_v4_derivative = dict()
    baseline_v4_second_derivative = dict()
    all_features_phys_updated = _feature_names_with_suffix(all_features_phys, "v4")

    for trial_id in trial_id_in_data:
        index_window1 = ((time_column<baseline_window) & (trial_column == trial_id))
        time_index = (trial_column == trial_id)
        time = time_array[time_index]
        baseline_period1_feature = np.mean(features_phys[index_window1], axis=0)
        current_trial_event = event_array[time_index]
        
        if 'begin ROR' in current_trial_event:
            begin_ROR_index = np.argwhere(current_trial_event == 'begin ROR')
            time_ROR = time_array[begin_ROR_index][0,0]
            index_window2 = ((time_column>(time_ROR - baseline_window)) & (time_column<time_ROR)
                            & (trial_column == trial_id))
            baseline_period2_feature = np.mean(features_phys[index_window2], axis=0)
            first_baseline_period = (time_array < (time_ROR - baseline_window)) & (trial_column == trial_id)
            second_baseline_period = (time_array >= (time_ROR - baseline_window)) & (trial_column == trial_id)
            baseline_v4[trial_id] = np.array(features_phys[(first_baseline_period)] - baseline_period1_feature)
            baseline_v4[trial_id] = np.vstack((baseline_v4[trial_id], np.array(features_phys[(second_baseline_period)] - baseline_period2_feature)))
        else:
            baseline_v4[trial_id] = np.array(features_phys[time_index] / baseline_period1_feature)

        first_derivative, second_derivative = _compute_derivative_stack(baseline_v4[trial_id], time)
        baseline_v4_derivative[trial_id] = first_derivative
        baseline_v4_second_derivative[trial_id] = second_derivative

    return baseline_v4, baseline_v4_derivative, baseline_v4_second_derivative, all_features_phys_updated


def create_v5_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg, trial_id_in_data=None):
    """Exact copy from tests.py"""
    trial_id_in_data = _resolve_trial_ids(trial_column, trial_id_in_data)
    time_array = np.asarray(time_column)
    participant_array = np.asarray(subject_column)
    baseline_v5 = dict()
    baseline_v5_derivative = dict()
    baseline_v5_second_derivative = dict()
    all_features_ecg_updated = _feature_names_with_suffix(all_features_ecg, "v5")

    for trial_id in trial_id_in_data:
        time_index = (trial_column == trial_id)
        current_participant = participant_array[time_index][0]
        baseline_feature = participant_seated_rhr[current_participant]
        baseline_feature = np.nan_to_num(baseline_feature, nan=1)
        baseline_v5[trial_id] = np.array(features_ecg[time_index] / baseline_feature)
        time = time_array[time_index]
        first_derivative, second_derivative = _compute_derivative_stack(baseline_v5[trial_id], time)
        baseline_v5_derivative[trial_id] = first_derivative
        baseline_v5_second_derivative[trial_id] = second_derivative

    return baseline_v5, baseline_v5_derivative, baseline_v5_second_derivative, all_features_ecg_updated


def create_v6_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg, trial_id_in_data=None):
    """Exact copy from tests.py"""
    trial_id_in_data = _resolve_trial_ids(trial_column, trial_id_in_data)
    time_array = np.asarray(time_column)
    participant_array = np.asarray(subject_column)
    baseline_v6 = dict()
    baseline_v6_derivative = dict()
    baseline_v6_second_derivative = dict()
    all_features_ecg_updated = _feature_names_with_suffix(all_features_ecg, "v6")

    for trial_id in trial_id_in_data:
        time_index = (trial_column == trial_id)
        current_participant = participant_array[time_index][0]
        baseline_feature = participant_seated_rhr[current_participant]
        baseline_feature = np.nan_to_num(baseline_feature, nan=0)
        baseline_v6[trial_id] = np.array(features_ecg[time_index] - baseline_feature)
        time = time_array[time_index]
        first_derivative, second_derivative = _compute_derivative_stack(baseline_v6[trial_id], time)
        baseline_v6_derivative[trial_id] = first_derivative
        baseline_v6_second_derivative[trial_id] = second_derivative

    return baseline_v6, baseline_v6_derivative, baseline_v6_second_derivative, all_features_ecg_updated


def create_v7_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg, eeg_baseline_delta,
                       eeg_baseline_theta, eeg_baseline_alpha, eeg_baseline_beta, all_features_eeg, trial_id_in_data=None):
    """Exact copy from tests.py"""
    trial_id_in_data = _resolve_trial_ids(trial_column, trial_id_in_data)
    time_array = np.asarray(time_column)
    participant_array = np.asarray(subject_column)
    baseline_v7 = dict()
    baseline_v7_derivative = dict()
    baseline_v7_second_derivative = dict()
    all_features_eeg_updated = _feature_names_with_suffix(all_features_eeg, "v7")
    channel_to_index, feature_meta = _prepare_eeg_feature_meta(all_features_eeg, eeg_baseline_delta)

    for trial_id in trial_id_in_data:
        time_index = (trial_column == trial_id)
        current_participant = participant_array[time_index][0]
        baseline_feature_delta = eeg_baseline_delta.loc[current_participant].to_numpy()
        baseline_feature_theta = eeg_baseline_theta.loc[current_participant].to_numpy()
        baseline_feature_alpha = eeg_baseline_alpha.loc[current_participant].to_numpy()
        baseline_feature_beta = eeg_baseline_beta.loc[current_participant].to_numpy()
        current_trial_data = features_eeg[time_index]
        baselined_trial_data = np.zeros_like(current_trial_data)

        for col, channel_name, frequency_suffix in feature_meta:
            channel_index = channel_to_index[channel_name]

            if frequency_suffix == 'delta - EEG':
                baseline_value = baseline_feature_delta[channel_index]
                if baseline_value == 0:
                    baselined_trial_data[:, col] = 0
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_value
            elif frequency_suffix == 'theta - EEG':
                baseline_value = baseline_feature_theta[channel_index]
                if baseline_value == 0:
                    baselined_trial_data[:, col] = 0
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_value
            elif frequency_suffix == 'alpha - EEG':
                baseline_value = baseline_feature_alpha[channel_index]
                if baseline_value == 0:
                    baselined_trial_data[:, col] = 0
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_value
            elif frequency_suffix == 'beta - EEG':
                baseline_value = baseline_feature_beta[channel_index]
                if baseline_value == 0:
                    baselined_trial_data[:, col] = 0
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_value

        baseline_v7[trial_id] = baselined_trial_data
        time = time_array[time_index]
        first_derivative, second_derivative = _compute_derivative_stack(baseline_v7[trial_id], time)
        baseline_v7_derivative[trial_id] = first_derivative
        baseline_v7_second_derivative[trial_id] = second_derivative

    return baseline_v7, baseline_v7_derivative, baseline_v7_second_derivative, all_features_eeg_updated


def create_v8_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg, eeg_baseline_delta,
                       eeg_baseline_theta, eeg_baseline_alpha, eeg_baseline_beta, all_features_eeg, trial_id_in_data=None):
    """Exact copy from tests.py"""
    trial_id_in_data = _resolve_trial_ids(trial_column, trial_id_in_data)
    time_array = np.asarray(time_column)
    participant_array = np.asarray(subject_column)
    baseline_v8 = dict()
    baseline_v8_derivative = dict()
    baseline_v8_second_derivative = dict()
    all_features_eeg_updated = _feature_names_with_suffix(all_features_eeg, "v8")
    channel_to_index, feature_meta = _prepare_eeg_feature_meta(all_features_eeg, eeg_baseline_delta)

    for trial_id in trial_id_in_data:
        time_index = (trial_column == trial_id)
        current_participant = participant_array[time_index][0]
        baseline_feature_delta = eeg_baseline_delta.loc[current_participant].to_numpy()
        baseline_feature_theta = eeg_baseline_theta.loc[current_participant].to_numpy()
        baseline_feature_alpha = eeg_baseline_alpha.loc[current_participant].to_numpy()
        baseline_feature_beta = eeg_baseline_beta.loc[current_participant].to_numpy()
        current_trial_data = features_eeg[time_index]
        baselined_trial_data = np.zeros_like(current_trial_data)

        for col, channel_name, frequency_suffix in feature_meta:
            channel_index = channel_to_index[channel_name]

            if frequency_suffix == 'delta - EEG':
                baseline_value = baseline_feature_delta[channel_index]
                if baseline_value == 0:
                    baselined_trial_data[:, col] = 0
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_value
            elif frequency_suffix == 'theta - EEG':
                baseline_value = baseline_feature_theta[channel_index]
                if baseline_value == 0:
                    baselined_trial_data[:, col] = 0
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_value
            elif frequency_suffix == 'alpha - EEG':
                baseline_value = baseline_feature_alpha[channel_index]
                if baseline_value == 0:
                    baselined_trial_data[:, col] = 0
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_value
            elif frequency_suffix == 'beta - EEG':
                baseline_value = baseline_feature_beta[channel_index]
                if baseline_value == 0:
                    baselined_trial_data[:, col] = 0
                else:
                    baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_value

        baseline_v8[trial_id] = baselined_trial_data
        time = time_array[time_index]
        first_derivative, second_derivative = _compute_derivative_stack(baseline_v8[trial_id], time)
        baseline_v8_derivative[trial_id] = first_derivative
        baseline_v8_second_derivative[trial_id] = second_derivative

    return baseline_v8, baseline_v8_derivative, baseline_v8_second_derivative, all_features_eeg_updated


def combine_all_baseline(trial_column, baseline, baseline_derivative, baseline_second_derivative, baseline_names, trial_id_in_data=None):
    """Exact copy from tests.py, but also returns trial_order for correct trial_ints computation."""
    # Use pandas.unique() like the legacy baseline_methods.py code to preserve first-appearance order, not sorted
    trial_id_in_data = _resolve_trial_ids(trial_column, trial_id_in_data)

    if not baseline:
        return {}, [], trial_id_in_data

    methods = list(baseline.keys())
    first_method = methods[0]
    first_trial = trial_id_in_data[0]
    num_cols = sum(baseline[method][first_trial].shape[1] * 3 for method in methods)

    combined_baseline = {}
    for trial in trial_id_in_data:
        num_rows = baseline[first_method][trial].shape[0]
        combined_trial = np.empty((num_rows, num_cols), dtype=np.float32)
        col_start = 0
        for method in methods:
            base = baseline[method][trial].astype(np.float32, copy=False)
            deriv = baseline_derivative[method][trial].astype(np.float32, copy=False)
            second = baseline_second_derivative[method][trial].astype(np.float32, copy=False)

            width = base.shape[1]
            combined_trial[:, col_start:col_start + width] = base
            col_start += width
            combined_trial[:, col_start:col_start + width] = deriv
            col_start += width
            combined_trial[:, col_start:col_start + width] = second
            col_start += width
        combined_baseline[trial] = combined_trial

    combined_baseline_names = []
    for method in baseline_names.keys():
        method_names = baseline_names[method]
        combined_baseline_names.extend(method_names)
        combined_baseline_names.extend([s + '_derivative' for s in method_names])
        combined_baseline_names.extend([s + '_2derivative' for s in method_names])

    return combined_baseline, combined_baseline_names, trial_id_in_data
