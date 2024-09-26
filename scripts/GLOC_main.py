from GLOC_data_processing import *
from GLOC_visualization import *
from GLOC_classifier import *

if __name__ == "__main__":
    # File Name Def
    filename = '../../all_trials_25_hz_stacked_null_str_filled.csv'

    # Plot Flag
    plot_data = 1

    # Feature Info
    feature_to_analyze = 'HR (bpm) - Equivital' # currently only one feature, coming soon: ability to iterate through all features
    # feature_to_analyze = 'BR (rpm) - Equivital'
    time_variable ='Time (s)'

    # Data Parameters
    subject_to_analyze = '01' # currently only one subject, coming soon: ability to iterate through all subjects
    trial_to_analyze = '01' # currently only one trial, coming soon: ability to iterate through all trials

    baseline_window = 10 # seconds
    window_size = 10     # seconds
    stride = 1           # seconds
    offset = 10          # seconds
    time_start = 0       # seconds

    # ML Splits
    training_ratio = 0.8

    # Process CSV
    gloc_data, subject, trial, time, feature = load_and_process_csv(filename, feature_to_analyze, time_variable)

    # Create GLOC Categorical Vector
    gloc = categorize_gloc(gloc_data)

    # Baseline Feature for Subject/Trial
    feature_baseline, time_trimmed = baseline_features(baseline_window, subject_to_analyze, trial_to_analyze, time, feature, subject, trial)

    # Visualization
    if plot_data == 1:
        initial_visualization(subject_to_analyze, trial_to_analyze, time, gloc, feature_baseline, subject, trial, feature_to_analyze, time_variable)

    # Sliding Window Mean
    time_end = np.max(time_trimmed)

    gloc_window, sliding_window_mean, number_windows = sliding_window_mean_calc(time_trimmed, time_start, time_end, offset, stride, window_size, subject,
                            subject_to_analyze, trial, trial_to_analyze, feature_baseline, gloc)

    # Visualize sliding window mean
    sliding_window_visualization(gloc_window, sliding_window_mean, number_windows)

    # Call functions for ML classification

    # Logistic Regression
    classify_logistic_regression(gloc_window, sliding_window_mean, training_ratio)

    # RF

    # KNN

    # Breakpoint for troubleshooting
    x = 1
