import numpy as np
import pandas as pd

def load_and_process_csv(filename, feature_to_analyze, time_variable):
    # Load CSV (can read just a chunk of the file)
    # chunksize = 10**3
    gloc_data = pd.read_csv(filename)#, chunksize = chunksize)

    # Separate Subject/Trial Column
    trial_id = gloc_data['trial_id'].to_numpy().astype('str')
    trial_id = np.array(np.char.split(trial_id, '-').tolist())
    subject = trial_id[:,0]
    trial = trial_id[:,1]
    time = gloc_data[time_variable].to_numpy()
    feature = gloc_data[feature_to_analyze].to_numpy()

    return gloc_data, subject, trial, time, feature

def baseline_features(baseline_window, subject_to_plot, trial_to_plot, time, feature, subject, trial):
    # Baseline Feature
    baseline_feature = np.mean(feature[(time<baseline_window) & \
                                       (subject == subject_to_plot) & \
                                       (trial == trial_to_plot)])
    feature_baseline = feature[(subject == subject_to_plot) & \
                                       (trial == trial_to_plot)]/baseline_feature

    time_trimmed = time[(subject == subject_to_plot) & (trial == trial_to_plot)]

    return feature_baseline, time_trimmed

def sliding_window_mean_calc(time_trimmed, time_start, time_end, offset, stride, window_size, subject, subject_to_analyze, trial, trial_to_analyze, feature_baseline, gloc):
    number_windows = np.int32(((time_end - offset) // stride) - (window_size // stride - 1))

    # Pre-allocate
    sliding_window_mean = np.zeros((number_windows, 1))
    gloc_window = np.zeros((number_windows, 1))

    # Create trimmed gloc data for the specific
    gloc_trimmed = gloc[(subject == subject_to_analyze) & (trial == trial_to_analyze)]

    for i in range(number_windows):
        time_period_feature = (time_start <= time_trimmed) & (time_trimmed < (time_start + window_size))
        sliding_window_mean[i] = np.mean(feature_baseline[time_period_feature])

        time_period_gloc = ((time_start + offset) <= time_trimmed) & (
                    time_trimmed < (time_start + offset + window_size))
        gloc_window[i] = np.any(gloc_trimmed[time_period_gloc])

        time_start = stride + time_start

    return gloc_window, sliding_window_mean