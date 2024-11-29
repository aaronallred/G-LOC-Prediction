from GLOC_data_processing import *
from GLOC_visualization import *
from GLOC_classifier import *
from features import *
from feature_selection import *
import numpy as np

if __name__ == "__main__":
    # File Name & Path
    filename = '../../all_trials_25_hz_stacked_null_str_filled.csv'

    # Plot Flags
    plot_data = 0       # flag to set whether plots should be generated (0 = no, 1 = yes)
    plot_pairwise = 0   # flag to set whether pairwise plots should be generated (0 = no, 1 = yes)

    # Feature Info
    # feature_to_analyze options:
        # ECG ('HR (bpm) - Equivital','ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital', 'HR_instant - Equivital', 'HR_average - Equivital', 'HR_w_average - Equivital')
        # BR ('BR (rpm) - Equivital')
        # temp ('Skin Temperature - IR Thermometer (°C) - Equivital')
        # fnirs ('HbO2 - fNIRS', 'Hbd - fNIRS')
        # eyetracking ('Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii', 'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii'
            # 'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii', 'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii')
        # EEG (coming soon!!!)
        # AFE (coming soon!!!)
        # G (coming soon!!!)
        # strain (coming soon!!!)

    feature_to_analyze = ['ECG','BR', 'temp', 'fnirs', 'eyetracking']

    time_variable = 'Time (s)'

    # Data Parameters
    analysis_type = 2   # flag to set which data should be analyzed
                        # analysis_type = 0: analyze one trial from a subject
                            # if analysis_type = 0, then set subject_to_analyze and trial_to_analyze parameters below
                        # analysis_type = 1: analyze subject data (all trials for a subject)
                            # if analysis_type = 1, then set subject_to_analyze parameter below
                        # analysis_type = 2: analyze cohort data (all subjects, all trials)
                            # if analysis_type = 2, then no extra parameters need to be set

    baseline_window = 10 # seconds
    window_size = 10     # seconds
    stride = 1           # seconds
    offset = 30          # seconds
    time_start = 0       # seconds

    # ML Splits
    training_ratio = 0.8

    # Subject & Trial Information
    subject_to_analyze = '01'
    trial_to_analyze = '02'

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
    gloc = label_gloc_events(gloc_data_reduced)

    # Find time window after acceleration before GLOC
    # find_prediction_window(gloc_data_reduced, gloc, time_variable)

    # Baseline Feature for Subject/Trial
    feature_baseline, feature_baseline_derivative, feature_baseline_second_derivative = baseline_features(baseline_window, gloc_data_reduced, features, time_variable)

    # Non Baselined Features (in Dictionary)
    feature_no_baseline, feature_no_baseline_derivative = non_baseline_features(baseline_window, gloc_data_reduced, features, time_variable)

    # Tabulate NaN
    # NaN_table, NaN_proportion, NaN_gloc_proportion = tabulateNaN(feature_baseline, all_features, gloc, gloc_data_reduced)
    # Nan_proportion_all = pd.read_pickle('../../NaN_proportion_all.pkl')

    # Visualization of feature throughout trial
    if plot_data == 1:
        initial_visualization(gloc_data_reduced, gloc, feature_baseline, all_features, time_variable)

    # Sliding Window Mean
    gloc_window, sliding_window_mean, number_windows = sliding_window_mean_calc(time_start, offset, stride, window_size, feature_baseline, feature_baseline_derivative, feature_baseline_second_derivative, gloc, gloc_data_reduced, time_variable)

    # Sliding Window Standard Deviation, Max, Range
    sliding_window_stddev, sliding_window_max, sliding_window_range = sliding_window_calc(time_start, stride, window_size, feature_baseline, feature_baseline_derivative, feature_baseline_second_derivative, gloc_data_reduced, time_variable, number_windows)

    # Additional Features
    (sliding_window_pupil_difference, sliding_window_ox_deox_ratio, sliding_window_integral_left_pupil,
     sliding_window_integral_right_pupil, sliding_window_consecutive_elements_mean_left_pupil,
     sliding_window_consecutive_elements_mean_right_pupil, sliding_window_consecutive_elements_max_left_pupil,
     sliding_window_consecutive_elements_max_right_pupil, sliding_window_hrv_sdnn, sliding_window_hrv_rmssd) = sliding_window_other_features(time_start, stride, window_size,
                                                                                          feature_baseline, gloc_data_reduced,
                                                                                          time_variable, number_windows,
                                                                                          all_features, feature_no_baseline)


    # Visualize sliding window mean
    if plot_data == 1:
        sliding_window_visualization(gloc_window, sliding_window_mean, number_windows, all_features, gloc_data_reduced)

    # Visualization of pairwise features
    if plot_pairwise == 1:
        pairwise_visualization(gloc_window, sliding_window_mean, all_features, gloc_data_reduced)

    # Unpack Dictionary into Array & combine features into one feature array
    y_gloc_labels, x_feature_matrix = unpack_dict(gloc_window, sliding_window_mean, number_windows, sliding_window_stddev,
                                                  sliding_window_max, sliding_window_range,
                                                  sliding_window_pupil_difference, sliding_window_ox_deox_ratio,
                                                  sliding_window_integral_left_pupil,sliding_window_integral_right_pupil,
                                                  sliding_window_consecutive_elements_mean_left_pupil, sliding_window_consecutive_elements_mean_right_pupil,
                                                  sliding_window_consecutive_elements_max_left_pupil, sliding_window_consecutive_elements_max_right_pupil, sliding_window_hrv_sdnn, sliding_window_hrv_rmssd)

    # Remove rows with NaN (temporary solution-should replace with other method eventually)
    y_gloc_labels_noNaN, x_feature_matrix_noNaN = process_NaN(y_gloc_labels, x_feature_matrix)

    # Update all features array
    all_features_mean = [s + '_mean' for s in all_features] + [s + '_mean_derivative' for s in all_features] + [s + '_mean_second_derivative' for s in all_features]
    all_features_stddev = [s + '_stddev' for s in all_features] + [s + '_stddev_derivative' for s in all_features] + [s + '_stddev_second_derivative' for s in all_features]
    all_features_max = [s + '_max' for s in all_features] + [s + '_max_derivative' for s in all_features] + [s + '_max_second_derivative' for s in all_features]
    all_features_range = [s + '_range' for s in all_features] + [s + '_range_derivative' for s in all_features] + [s + '_range_second_derivative' for s in all_features]
    additional_features = ['Pupil Difference', 'Hbo/Hbd', 'Left Pupil Integral (Non-Baseline)',
                           'Right Pupil Integral (Non-Baseline)', 'Left Pupil Mean of Consecutive Difference (Non-Baseline)',
                           'Right Pupil Mean of Consecutive Difference (Non-Baseline)', 'Left Pupil Max of Consecutive Difference (Non-Baseline)',
                           'Right Pupil Max of Consecutive Difference (Non-Baseline)', 'HRV (SDNN)', 'HRV (RMSSD)']

    all_features = all_features_mean + all_features_stddev + all_features_max +  all_features_range + additional_features


    # Training/Test Split
    x_train, x_test, y_train, y_test = pre_classification_training_test_split(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio)

    # Feature Selection
    selected_features_lasso = feature_selection_lasso(x_train, y_train, all_features)
    selected_features_enet = feature_selection_elastic_net(x_train, y_train, all_features)
    selected_features_ridge = feature_selection_ridge(x_train, y_train, all_features)
    selected_features_mrmr = feature_selection_mrmr(x_train, y_train, all_features)

    ## Call functions for ML classification ##

    # Logistic Regression
    accuracy_logreg, precision_logreg, recall_logreg, f1_logreg, specificity_logreg = classify_logistic_regression(x_train, x_test, y_train, y_test, all_features)

    # RF
    accuracy_rf, precision_rf, recall_rf, f1_rf, tree_depth, specificity_rf = classify_random_forest(x_train, x_test, y_train, y_test, all_features)

    # LDA
    accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda = classify_lda(x_train, x_test, y_train, y_test, all_features)

    # KNN
    accuracy_knn, precision_knn, recall_knn, f1_knn, specificity_knn = classify_knn(x_train, x_test, y_train, y_test)

    # SVM
    accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm = classify_svm(x_train, x_test, y_train, y_test)

    # Ensemble with Gradient Boosting
    accuracy_gb, precision_gb, recall_gb, f1_gb, specificity_gb = classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test)

    # Build Performance Metric Summary Tables
    performance_metric_summary = summarize_performance_metrics(accuracy_logreg, accuracy_rf, accuracy_lda, accuracy_knn, accuracy_svm, accuracy_gb,
                                                     precision_logreg, precision_rf, precision_lda, precision_knn, precision_svm, precision_gb,
                                                     recall_logreg, recall_rf, recall_lda, recall_knn, recall_svm, recall_gb,
                                                     f1_logreg, f1_rf, f1_lda, f1_knn, f1_svm, f1_gb,
                                                     specificity_logreg, specificity_rf, specificity_lda, specificity_knn, specificity_svm, specificity_gb)


    # Breakpoint for troubleshooting
    x = 1
