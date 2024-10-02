import numpy as np
import pandas as pd
import os

def load_and_process_csv(filename, feature_to_analyze, time_variable):
    # pickle file name
    global ecg_features, br_features, temp_features, fnirs_features, eeg_features, eyetracking_features
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

    # Separate Subject/Trial Column
    trial_id = gloc_data['trial_id'].to_numpy().astype('str')
    trial_id = np.array(np.char.split(trial_id, '-').tolist())
    subject = trial_id[:,0]
    trial = trial_id[:,1]
    time = gloc_data[time_variable].to_numpy()

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

    feature = gloc_data[all_features].to_numpy()

    return gloc_data, subject, trial, time, feature, all_features

def baseline_features(baseline_window, subject_to_plot, trial_to_plot, time, feature, subject, trial):
    # Baseline Feature
    baseline_feature = np.mean(feature[(time<baseline_window) & \
                                       (subject == subject_to_plot) & \
                                       (trial == trial_to_plot)], axis = 0)
    feature_baseline = feature[(subject == subject_to_plot) & \
                                       (trial == trial_to_plot)]/baseline_feature

    time_trimmed = time[(subject == subject_to_plot) & (trial == trial_to_plot)]

    return feature_baseline, time_trimmed

def sliding_window_mean_calc(time_trimmed, time_start, time_end, offset, stride, window_size, subject, subject_to_analyze, trial, trial_to_analyze, feature_baseline, gloc):
    number_windows = np.int32(((time_end - offset) // stride) - (window_size // stride - 1))

    # Pre-allocate
    sliding_window_mean = np.zeros((number_windows, np.size(feature_baseline,1)))
    gloc_window = np.zeros((number_windows, 1))

    # Create trimmed gloc data for the specific
    gloc_trimmed = gloc[(subject == subject_to_analyze) & (trial == trial_to_analyze)]

    for i in range(number_windows):
        time_period_feature = (time_start <= time_trimmed) & (time_trimmed < (time_start + window_size))
        sliding_window_mean[i,:] = np.mean(feature_baseline[time_period_feature], axis = 0, keepdims=True)

        time_period_gloc = ((time_start + offset) <= time_trimmed) & (
                    time_trimmed < (time_start + offset + window_size))
        gloc_window[i] = np.any(gloc_trimmed[time_period_gloc])

        time_start = stride + time_start

    return gloc_window, sliding_window_mean, number_windows