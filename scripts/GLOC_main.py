from GLOC_data_processing import *
from GLOC_visualization import *
from GLOC_classifier import *
from features import *
from feature_selection import *
from baseline_methods import *
import numpy as np
import pandas as pd

if __name__ == "__main__":

    ##################################################### SETUP  #####################################################
    ## File Name & Path
    # Data CSV
    filename = '../../all_trials_25_hz_stacked_null_str_filled.csv'

    # Baseline Data (HR)
    baseline_data_filename = "../../ParticipantBaseline.csv"

    # Modified Demographic Data (put in order of participant 1-13, removed excess calculations, and converted from .xlsx to .csv)
    demographic_data_filename = "../../GLOC_Effectiveness_Final.csv"

    # Feature Info
        # Example with all feature groups:
            # feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'fnirs', 'eyetracking', 'AFE', 'G', 'cognitive',
                # 'rawEEG_shared', 'processedEEG_shared', 'strain', 'demographics']
        # NOTES:
            # Update from GLOC tagup 01/15/25: Chris said the FNIRS data should not be trusted and should not be used.
            # The light from the eye tracking glasses washed out this data.
    feature_groups_to_analyze = ['ECG']

    # Baseline Method
        # baseline_methods_to_use options:
            # V0: no baseline
            # V1: divide by baseline window
            # V2: subtract baseline window
            # V3: pre ROR: divide by baseline window pre GOR, ROR: divide by baseline window pre ROR
            # V4: pre ROR: subtract baseline window pre GOR, ROR: subtract baseline window pre ROR
            # V5: divide by seated resting HR
            # V6: subtract seated resting HR
        # Example with all feature groups:
            # baseline_methods_to_use = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        # NOTES:
            # ALWAYS use v0- it is needed for several additional features that get computed
    # baseline_methods_to_use = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    baseline_methods_to_use = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

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
    offset = 0           # seconds
    time_start = 0       # seconds

    # ML Splits (Training/Test Split, specify proportion of training data 0-1)
    training_ratio = 0.8

    # Subject & Trial Information (only need to adjust this if doing analysis type 0,1)
    subject_to_analyze = '01'
    trial_to_analyze = '02'

    ############################################# LOAD AND PROCESS DATA #############################################

    # Process CSV
    if analysis_type == 0:  # One Trial / One Subject
        gloc_data_reduced, features, features_phys, features_ecg, all_features, all_features_phys, all_features_ecg = (
            load_and_process_csv(filename, analysis_type, feature_groups_to_analyze, demographic_data_filename,
                                 trial_to_analyze=trial_to_analyze,subject_to_analyze = subject_to_analyze))

    elif analysis_type == 1:  # All Trials for One Subject
        gloc_data_reduced, features, features_phys, features_ecg, all_features, all_features_phys, all_features_ecg = (
            load_and_process_csv(filename, analysis_type, feature_groups_to_analyze, demographic_data_filename,
                                 subject_to_analyze = subject_to_analyze))

    elif analysis_type == 2: # All Trials for All Subjects
        gloc_data_reduced, features, features_phys, features_ecg, all_features, all_features_phys, all_features_ecg = (
            load_and_process_csv(filename, analysis_type, feature_groups_to_analyze, demographic_data_filename))

    # Create GLOC Categorical Vector
    gloc = label_gloc_events(gloc_data_reduced)

    # Find time window after acceleration before GLOC (to compare our data to LOCINDTI)
    # find_prediction_window(gloc_data_reduced, gloc, time_variable)

    ################################################## BASELINE DATA ##################################################
    baseline = dict()
    baseline_derivative = dict()
    baseline_second_derivative = dict()
    baseline_names = dict()

    for method in baseline_methods_to_use:
        if method == 'v0':
            # V0: No Baseline (feature categories: ECG, BR, temp, fnirs, eyetracking, AFE, G, cognitive)
                # to be implemented: EEG, strain
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[method] = (
                create_v0_baseline(gloc_data_reduced, features, time_variable, all_features))

            # Tabulate NaN
            NaN_table, NaN_proportion, NaN_gloc_proportion = tabulateNaN(baseline[method], all_features, gloc, gloc_data_reduced)
            # Nan_proportion_all = pd.read_pickle('../../NaN_proportion_all.pkl')

        if method == 'v1':
            # V1: Divide by Baseline Window (feature categories: ECG, BR, temp, fnirs, eyetracking)
                # to be implemented: EEG
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[method] = (
                create_v1_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys))
    
        if method == 'v2':
            # V2: Subtract Baseline Window (feature categories: ECG, BR, temp, fnirs, eyetracking)
                # to be implemented: EEG
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[method] = (
                create_v2_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys))
    
        if method == 'v3':
            # V3: pre ROR: divide by baseline window pre GOR, ROR: divide by baseline window pre ROR
            # feature categories: ECG, BR, temp, fnirs, eyetracking
                # to be implemented: EEG
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[method] = (
                create_v3_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys))
    
        if method == 'v4':
            # V4: pre ROR: subtract baseline window pre GOR, ROR: subtract baseline window pre ROR
            # feature categories: ECG, BR, temp, fnirs, eyetracking
                # to be implemented: EEG
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[method] = (
                create_v4_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys))
    
        if method == 'v5':
            # Import xlsx File
            participant_baseline = pd.read_csv(baseline_data_filename)
            participant_seated_rhr = participant_baseline['resting HR [seated]'][0:-1]
            participant_seated_rhr.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
    
            # V5: Divide by seated resting HR (feature categories: ECG)
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[method] = (
                create_v5_baseline(baseline_window, gloc_data_reduced, features_ecg, time_variable, participant_seated_rhr,
                                   all_features_ecg))
    
        if method == 'v6':
            # Import xlsx File (if not already imported from v5)
            if 'participant_baseline' not in locals() and 'participant_baseline' not in globals():
                participant_baseline = pd.read_csv(baseline_data_filename)
                participant_seated_rhr = participant_baseline['resting HR [seated]'][0:-1]
                participant_seated_rhr.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
    
            # V6: Subtract seated resting HR (feature categories: ECG)
            baseline[method], baseline_derivative[method], baseline_second_derivative[method], baseline_names[method] = (
                create_v6_baseline(baseline_window, gloc_data_reduced, features_ecg, time_variable, participant_seated_rhr,
                                   all_features_ecg))

    # Combine all baseline methods into a large dictionary
    combined_baseline, combined_baseline_names = combine_all_baseline(gloc_data_reduced, baseline, baseline_derivative,
                                                                      baseline_second_derivative, baseline_names)

    ################################################ GENERATE FEATURES ################################################

    # Sliding Window Mean (Intra-Trial Standardization)
    gloc_window, sliding_window_mean_s1, number_windows, all_features_mean_s1, sliding_window_mean_s2, all_features_mean_s2= (
        sliding_window_mean_calc(time_start, offset, stride, window_size, combined_baseline, gloc, gloc_data_reduced,
                                 time_variable, combined_baseline_names))

    # Sliding Window Standard Deviation, Max, Range
    (sliding_window_stddev_s1, sliding_window_max_s1, sliding_window_range_s1, all_features_stddev_s1, all_features_max_s1,
     all_features_range_s1, sliding_window_stddev_s2, sliding_window_max_s2, sliding_window_range_s2, all_features_stddev_s2, all_features_max_s2,
     all_features_range_s2) = (sliding_window_calc(time_start, stride, window_size, combined_baseline, gloc_data_reduced, time_variable,
                            number_windows, combined_baseline_names))

    # Additional Features
    (all_features_additional_s1, sliding_window_integral_left_pupil_s1, sliding_window_integral_right_pupil_s1,
     sliding_window_consecutive_elements_mean_left_pupil_s1, sliding_window_consecutive_elements_mean_right_pupil_s1,
     sliding_window_consecutive_elements_max_left_pupil_s1, sliding_window_consecutive_elements_max_right_pupil_s1,
     sliding_window_consecutive_elements_sum_left_pupil_s1, sliding_window_consecutive_elements_sum_right_pupil_s1,
     sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1, sliding_window_hrv_pnn50_s1, sliding_window_cognitive_IES_s1,
     all_features_additional_s2, sliding_window_integral_left_pupil_s2, sliding_window_integral_right_pupil_s2,
     sliding_window_consecutive_elements_mean_left_pupil_s2, sliding_window_consecutive_elements_mean_right_pupil_s2,
     sliding_window_consecutive_elements_max_left_pupil_s2, sliding_window_consecutive_elements_max_right_pupil_s2,
     sliding_window_consecutive_elements_sum_left_pupil_s2, sliding_window_consecutive_elements_sum_right_pupil_s2,
     sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2, sliding_window_hrv_pnn50_s2,
     sliding_window_cognitive_IES_s2) = \
        (sliding_window_other_features(time_start, stride, window_size, gloc_data_reduced,time_variable, number_windows,
                                      baseline_names['v0'], baseline['v0'], feature_groups_to_analyze))

    # Unpack Dictionary into Array & combine features into one feature array
    y_gloc_labels, x_feature_matrix = unpack_dict(gloc_window, sliding_window_mean_s1, number_windows, sliding_window_stddev_s1,
                                                  sliding_window_max_s1, sliding_window_range_s1,
                                                  sliding_window_integral_left_pupil_s1,sliding_window_integral_right_pupil_s1,
                                                  sliding_window_consecutive_elements_mean_left_pupil_s1,
                                                  sliding_window_consecutive_elements_mean_right_pupil_s1,
                                                  sliding_window_consecutive_elements_max_left_pupil_s1,
                                                  sliding_window_consecutive_elements_max_right_pupil_s1,
                                                  sliding_window_consecutive_elements_sum_left_pupil_s1,
                                                  sliding_window_consecutive_elements_sum_right_pupil_s1,
                                                  sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1,
                                                  sliding_window_hrv_pnn50_s1, sliding_window_cognitive_IES_s1,
                                                  sliding_window_mean_s2, sliding_window_stddev_s2,
                                                  sliding_window_max_s2, sliding_window_range_s2,
                                                  sliding_window_integral_left_pupil_s2,
                                                  sliding_window_integral_right_pupil_s2,
                                                  sliding_window_consecutive_elements_mean_left_pupil_s2,
                                                  sliding_window_consecutive_elements_mean_right_pupil_s2,
                                                  sliding_window_consecutive_elements_max_left_pupil_s2,
                                                  sliding_window_consecutive_elements_max_right_pupil_s2,
                                                  sliding_window_consecutive_elements_sum_left_pupil_s2,
                                                  sliding_window_consecutive_elements_sum_right_pupil_s2,
                                                  sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2,
                                                  sliding_window_hrv_pnn50_s2, sliding_window_cognitive_IES_s2)

    # Remove rows with NaN (temporary solution-should replace with other method eventually)
    y_gloc_labels_noNaN, x_feature_matrix_noNaN = process_NaN(y_gloc_labels, x_feature_matrix)

    all_features = (all_features_mean_s1 + all_features_stddev_s1 + all_features_max_s1 +  all_features_range_s1 + all_features_additional_s1 +
                    all_features_mean_s2 + all_features_stddev_s2 + all_features_max_s2 + all_features_range_s2 + all_features_additional_s2)

    ################################################ FEATURE SELECTION ################################################

    # Training/Test Split
    x_train, x_test, y_train, y_test = pre_classification_training_test_split(y_gloc_labels_noNaN, x_feature_matrix_noNaN,
                                                                              training_ratio)

    # Feature Selection
    selected_features_lasso = feature_selection_lasso(x_train, y_train, all_features)
    selected_features_enet = feature_selection_elastic_net(x_train, y_train, all_features)
    selected_features_ridge = feature_selection_ridge(x_train, y_train, all_features)
    selected_features_mrmr = feature_selection_mrmr(x_train, y_train, all_features)

    ################################################ MACHINE LEARNING ################################################

    # Logistic Regression
    accuracy_logreg, precision_logreg, recall_logreg, f1_logreg, specificity_logreg = (
        classify_logistic_regression(x_train, x_test, y_train, y_test, all_features))

    # RF
    accuracy_rf, precision_rf, recall_rf, f1_rf, tree_depth, specificity_rf = (
        classify_random_forest(x_train, x_test, y_train, y_test, all_features))

    # LDA
    accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda = (
        classify_lda(x_train, x_test, y_train, y_test, all_features))

    # KNN
    accuracy_knn, precision_knn, recall_knn, f1_knn, specificity_knn = classify_knn(x_train, x_test, y_train, y_test)

    # SVM
    accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm = classify_svm(x_train, x_test, y_train, y_test)

    # Ensemble with Gradient Boosting
    accuracy_gb, precision_gb, recall_gb, f1_gb, specificity_gb = (
        classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test))

    # Build Performance Metric Summary Tables
    performance_metric_summary = (
        summarize_performance_metrics(accuracy_logreg, accuracy_rf, accuracy_lda, accuracy_knn, accuracy_svm, accuracy_gb,
                                      precision_logreg, precision_rf, precision_lda, precision_knn, precision_svm, precision_gb,
                                      recall_logreg, recall_rf, recall_lda, recall_knn, recall_svm, recall_gb,
                                      f1_logreg, f1_rf, f1_lda, f1_knn, f1_svm, f1_gb,
                                      specificity_logreg, specificity_rf, specificity_lda, specificity_knn, specificity_svm, specificity_gb))


    # Breakpoint for troubleshooting
    x = 1
