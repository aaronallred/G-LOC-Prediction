from GLOC_data_processing import *
from GLOC_visualization import *
from GLOC_classifier import *

if __name__ == "__main__":
    # File Name Def
    filename = '../../all_trials_25_hz_stacked_null_str_filled.csv'

    # Plot Flag
    plot_data = 1
    plot_pairwise = 1

    # Feature Info
    # Options:
    # ECG ('HR (bpm) - Equivital','ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital', 'HR_instant - Equivital', 'HR_average - Equivital', 'HR_w_average - Equivital')
    # BR ('BR (rpm) - Equivital')
    # temp ('Skin Temperature - IR Thermometer (°C) - Equivital')
    # fnirs ('HbO2 - fNIRS', 'Hbd - fNIRS')
    # eyetracking ('Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii', 'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii'
            # 'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii', 'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii')
    # EEG (coming soon!!!)
    feature_to_analyze = ['ECG','BR', 'temp', 'fnirs', 'eyetracking']

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
    gloc_data, subject, trial, time, feature, all_features, g = load_and_process_csv(filename, feature_to_analyze, time_variable)

    # Create GLOC Categorical Vector
    gloc = categorize_gloc(gloc_data)

    # Check for A-LOC
    other_vals_event, other_vals_event_validated = check_for_aloc(gloc_data)

    # Baseline Feature for Subject/Trial
    feature_baseline, time_trimmed = baseline_features(baseline_window, subject_to_analyze, trial_to_analyze, time, feature, subject, trial)

    # Visualization of feature throughout trial
    if plot_data == 1:
        initial_visualization(subject_to_analyze, trial_to_analyze, time, gloc, feature_baseline, subject, trial, feature_to_analyze, time_variable, all_features, g)

    # Sliding Window Mean
    time_end = np.max(time_trimmed)

    gloc_window, sliding_window_mean, number_windows = sliding_window_mean_calc(time_trimmed, time_start, time_end, offset, stride, window_size, subject,
                            subject_to_analyze, trial, trial_to_analyze, feature_baseline, gloc)

    # Visualize sliding window mean
    if plot_data == 1:
        sliding_window_visualization(gloc_window, sliding_window_mean, number_windows, all_features)

    # Visualization of pairwise features
    if plot_pairwise == 1:
        pairwise_visualization(gloc_window, sliding_window_mean, all_features)

    ## Call functions for ML classification ##

    # Logistic Regression
    classify_logistic_regression(gloc_window, sliding_window_mean, training_ratio, all_features)

    # RF
    classify_random_forest(gloc_window, sliding_window_mean, training_ratio, all_features)

    # LDA
    classify_lda(gloc_window, sliding_window_mean, training_ratio, all_features)

    # KNN
    classify_knn(gloc_window, sliding_window_mean, training_ratio)

    # SVM
    classify_svm(gloc_window, sliding_window_mean, training_ratio)

    # Ensemble with Gradient Boosting
    classify_ensemble_with_gradboost(gloc_window, sliding_window_mean, training_ratio)

    # Breakpoint for troubleshooting
    x = 1
