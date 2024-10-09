from GLOC_data_processing import *
from GLOC_visualization import *
from GLOC_classifier import *
import numpy as np

if __name__ == "__main__":
    # File Name Def
    filename = '../../all_trials_25_hz_stacked_null_str_filled.csv'

    # Plot Flags
    plot_data = 0
    plot_pairwise = 0

    # Feature Info
    # feature_to_analyze options:
        # ECG ('HR (bpm) - Equivital','ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital', 'HR_instant - Equivital', 'HR_average - Equivital', 'HR_w_average - Equivital')
        # BR ('BR (rpm) - Equivital')
        # temp ('Skin Temperature - IR Thermometer (°C) - Equivital')
        # fnirs ('HbO2 - fNIRS', 'Hbd - fNIRS')
        # eyetracking ('Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii', 'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii'
            # 'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii', 'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii')
        # EEG (coming soon!!!)
    feature_to_analyze = ['ECG','BR', 'temp', 'fnirs', 'eyetracking']

    time_variable = 'Time (s)'
    g_variable = 'magnitude - Centrifuge'

    # Data Parameters
    analysis_type = 2   # flag to set which data should be analyzed
                        # analysis_type = 0: analyze one trial from a subject
                            # if analysis_type = 0, then set subject_to_analyze and trial_to_analyze parameters below
                        # analysis_type = 1: analyze subject data (all trials for a subject)
                            # if analysis_type = 1, then set subject_to_analyze parameter below
                        # analysis_type = 2: analyze cohort data (all subjects, all trials)
                            # if analysis_type = 2, then no extra parameters need to be set

    subject_to_analyze = '02'
    trial_to_analyze = '01'

    baseline_window = 10 # seconds
    window_size = 10     # seconds
    stride = 1           # seconds
    offset = 10          # seconds
    time_start = 0       # seconds

    # ML Splits
    training_ratio = 0.8

    # Process CSV
    if analysis_type == 0:  # One Trial / One Subject
        gloc_data_reduced, features, all_features = load_and_process_csv(filename, analysis_type, feature_to_analyze, time_variable,
                                                                                         trial_to_analyze=trial_to_analyze,
                                                                                         subject_to_analyze = subject_to_analyze)
    elif analysis_type == 1:  # All Trials for One Subject
        gloc_data_reduced, features, all_features = load_and_process_csv(filename, analysis_type, feature_to_analyze, time_variable,
                                                                                         subject_to_analyze = subject_to_analyze)
    elif analysis_type == 2: # All Trials for All Subjects
        gloc_data_reduced, features, all_features = load_and_process_csv(filename, analysis_type, feature_to_analyze, time_variable)

    # Create GLOC Categorical Vector
    gloc = categorize_gloc(gloc_data_reduced)

    # Check for A-LOC
    other_vals_event, other_vals_event_validated = check_for_aloc(gloc_data_reduced)

    # Baseline Feature for Subject/Trial
    feature_baseline = baseline_features(baseline_window, gloc_data_reduced, features, time_variable)

    # Visualization of feature throughout trial
    if plot_data == 1:
        initial_visualization(gloc_data_reduced, gloc, feature_baseline, all_features, time_variable, g_variable)

    # Sliding Window Mean
    gloc_window, sliding_window_mean, number_windows = sliding_window_mean_calc(time_start, offset, stride, window_size, feature_baseline, gloc, gloc_data_reduced, time_variable)

    # Visualize sliding window mean
    if plot_data == 1:
        sliding_window_visualization(gloc_window, sliding_window_mean, number_windows, all_features, gloc_data_reduced)

    # Visualization of pairwise features
    if plot_pairwise == 1:
        pairwise_visualization(gloc_window, sliding_window_mean, all_features, gloc_data_reduced)

    # Unpack Dictionary into Array
    y_gloc_labels, x_feature_matrix = unpack_dict(gloc_window, sliding_window_mean, number_windows)

    # Remove rows with NaN (temporary solution-should replace with other method eventually)
    y_gloc_labels_noNaN, x_feature_matrix_noNaN = process_NaN(y_gloc_labels, x_feature_matrix)

    ## Call functions for ML classification ##

    # Logistic Regression
    classify_logistic_regression(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio, all_features)

    # RF
    classify_random_forest(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio, all_features)

    # LDA
    classify_lda(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio, all_features)

    # KNN
    classify_knn(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio)

    # SVM
    classify_svm(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio)

    # Ensemble with Gradient Boosting
    classify_ensemble_with_gradboost(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio)

    # Breakpoint for troubleshooting
    x = 1
