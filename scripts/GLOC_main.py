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
    # File Name & Path
    filename = '../../all_trials_25_hz_stacked_null_str_filled.csv'

    baseline_data_filename = "../../ParticipantBaseline.csv"

    # Tabulate NaN
    # NaN_table, NaN_proportion, NaN_gloc_proportion = tabulateNaN(baseline_v1, all_features, gloc, gloc_data_reduced)
    Nan_proportion_all = pd.read_pickle('../../NaN_proportion_all.pkl')

    # Feature Info
    # feature_to_analyze options:
        # ECG ('HR (bpm) - Equivital','ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital', 'HR_instant - Equivital', 'HR_average - Equivital', 'HR_w_average - Equivital')
        # BR ('BR (rpm) - Equivital')
        # temp ('Skin Temperature - IR Thermometer (°C) - Equivital')
        # fnirs ('HbO2 - fNIRS', 'Hbd - fNIRS')
        # eyetracking ('Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii', 'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii'
            # 'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii', 'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii')
        # AFE ('condition')
        # G ('magnitude - Centrifuge')
        # cognitive ('deviation - Cog', 'RespTime - Cog', 'Correct - Cog', 'joystickVelMag - Cog')
            #coming soon:'tgtposX - Cog', 'tgtposY - Cog' related to deviation and other combo metrics (accuracy & speed combo)
        # EEG (coming soon!!!- Waiting on more info)
        # strain (coming soon!!!- Waiting on more info)

    feature_to_analyze = ['ECG','BR', 'temp', 'fnirs', 'eyetracking', 'AFE', 'G', 'cognitive']

    # Baseline Method Info
    baseline_methods_to_use = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    # baseline_methods_to_use options:
        # V0: no baseline
        # V1: divide by baseline window
        # V2: subtract baseline window
        # V3: pre ROR: divide by baseline window pre GOR, ROR: divide by baseline window pre ROR
        # V4: pre ROR: subtract baseline window pre GOR, ROR: subtract baseline window pre ROR
        # V5: divide by seated resting HR
        # V6: subtract seated resting HR

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

    ################################################ LOAD AND PROCESS ################################################

    # Process CSV
    if analysis_type == 0:  # One Trial / One Subject
        gloc_data_reduced, features, features_phys, features_ecg, all_features, all_features_phys, all_features_ecg = (
            load_and_process_csv(filename, analysis_type, feature_to_analyze, time_variable,
                                 trial_to_analyze=trial_to_analyze,subject_to_analyze = subject_to_analyze))

    elif analysis_type == 1:  # All Trials for One Subject
        gloc_data_reduced, features, features_phys, features_ecg, all_features, all_features_phys, all_features_ecg = (
            load_and_process_csv(filename, analysis_type, feature_to_analyze, time_variable,
                                 subject_to_analyze = subject_to_analyze))

    elif analysis_type == 2: # All Trials for All Subjects
        gloc_data_reduced, features, features_phys, features_ecg, all_features, all_features_phys, all_features_ecg = (
            load_and_process_csv(filename, analysis_type, feature_to_analyze, time_variable))

    # Create GLOC Categorical Vector
    gloc = label_gloc_events(gloc_data_reduced)

    # Find time window after acceleration before GLOC
    # find_prediction_window(gloc_data_reduced, gloc, time_variable)

    ################################################## BASELINE DATA ##################################################

    if 'v0' in baseline_methods_to_use:
        # V0: No Baseline (feature categories: ECG, BR, temp, fnirs, eyetracking, AFE, G, cognitive)
            # to be implemented: EEG, strain
        baseline_v0, baseline_v0_derivative, baseline_v0_second_derivative, baseline_names_v0 = (
            create_v0_baseline(gloc_data_reduced, features, time_variable, all_features))
    else:
        baseline_v0 = []
        baseline_v0_derivative = []
        baseline_v0_second_derivative = []
        baseline_names_v0 = []

    if 'v1' in baseline_methods_to_use:
        # V1: Divide by Baseline Window (feature categories: ECG, BR, temp, fnirs, eyetracking)
            # to be implemented: EEG
        baseline_v1, baseline_v1_derivative, baseline_v1_second_derivative, baseline_names_v1 = (
            create_v1_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys))
    else:
        baseline_v1 = []
        baseline_v1_derivative = []
        baseline_v1_second_derivative = []
        baseline_names_v1 = []

    if 'v2' in baseline_methods_to_use:
        # V2: Subtract Baseline Window (feature categories: ECG, BR, temp, fnirs, eyetracking)
            # to be implemented: EEG
        baseline_v2, baseline_v2_derivative, baseline_v2_second_derivative, baseline_names_v2 = (
            create_v2_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys))
    else:
        baseline_v2 = []
        baseline_v2_derivative = []
        baseline_v2_second_derivative = []
        baseline_names_v2 = []

    if 'v3' in baseline_methods_to_use:
        # V3: pre ROR: divide by baseline window pre GOR, ROR: divide by baseline window pre ROR (feature categories: ECG, BR, temp, fnirs, eyetracking)
            # to be implemented: EEG
        baseline_v3, baseline_v3_derivative, baseline_v3_second_derivative, baseline_names_v3 = (
            create_v3_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys))
    else:
        baseline_v3 = []
        baseline_v3_derivative = []
        baseline_v3_second_derivative = []
        baseline_names_v3 =[]

    if 'v4' in baseline_methods_to_use:
        # V4: pre ROR: subtract baseline window pre GOR, ROR: subtract baseline window pre ROR (feature categories: ECG, BR, temp, fnirs, eyetracking)
            # to be implemented: EEG
        baseline_v4, baseline_v4_derivative, baseline_v4_second_derivative, baseline_names_v4 = (
            create_v4_baseline(baseline_window, gloc_data_reduced, features_phys, time_variable, all_features_phys))
    else:
        baseline_v4 = []
        baseline_v4_derivative = []
        baseline_v4_second_derivative = []
        baseline_names_v4 = []

    if 'v5' in baseline_methods_to_use:
        # Import xlsx File
        participant_baseline = pd.read_csv(baseline_data_filename)
        participant_seated_rhr = participant_baseline['resting HR [seated]'][0:-1]
        participant_seated_rhr.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

        # V5: Divide by seated resting HR (feature categories: ECG)
        baseline_v5, baseline_v5_derivative, baseline_v5_second_derivative, baseline_names_v5 = (
            create_v5_baseline(baseline_window, gloc_data_reduced, features_ecg, time_variable, participant_seated_rhr, all_features_ecg))
    else:
        baseline_v5 = []
        baseline_v5_derivative = []
        baseline_v5_second_derivative = []
        baseline_names_v5 = []

    if 'v6' in baseline_methods_to_use:
        # Import xlsx File (if not already imported from v5)
        if 'participant_baseline' not in locals() and 'participant_baseline' not in globals():
            participant_baseline = pd.read_csv(baseline_data_filename)
            participant_seated_rhr = participant_baseline['resting HR [seated]'][0:-1]
            participant_seated_rhr.index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

        # V6: Subtract seated resting HR (feature categories: ECG)
        baseline_v6, baseline_v6_derivative, baseline_v6_second_derivative, baseline_names_v6 = (
            create_v6_baseline(baseline_window, gloc_data_reduced, features_ecg, time_variable, participant_seated_rhr, all_features_ecg))
    else:
        baseline_v6 = []
        baseline_v6_derivative = []
        baseline_v6_second_derivative = []
        baseline_names_v6 =[]

    # Combine all baseline methods into a large dictionary
    combined_baseline, combined_baseline_names = combine_all_baseline(gloc_data_reduced, baseline_v0, baseline_v0_derivative, baseline_v0_second_derivative,
                                             baseline_v1, baseline_v1_derivative, baseline_v1_second_derivative,
                                             baseline_v2, baseline_v2_derivative, baseline_v2_second_derivative,
                                             baseline_v3, baseline_v3_derivative, baseline_v3_second_derivative,
                                             baseline_v4, baseline_v4_derivative, baseline_v4_second_derivative,
                                             baseline_v5, baseline_v5_derivative, baseline_v5_second_derivative,
                                             baseline_v6, baseline_v6_derivative, baseline_v6_second_derivative,
                                             baseline_names_v0, baseline_names_v1, baseline_names_v2, baseline_names_v3,
                                             baseline_names_v4, baseline_names_v5, baseline_names_v6)

    ################################################ GENERATE FEATURES ################################################

    # Sliding Window Mean
    gloc_window, sliding_window_mean, number_windows, all_features_mean = (
        sliding_window_mean_calc(time_start, offset, stride, window_size, combined_baseline, gloc, gloc_data_reduced, time_variable, combined_baseline_names))

    # Sliding Window Standard Deviation, Max, Range
    sliding_window_stddev, sliding_window_max, sliding_window_range, all_features_stddev, all_features_max, all_features_range = (
        sliding_window_calc(time_start, stride, window_size, combined_baseline, gloc_data_reduced, time_variable, number_windows, combined_baseline_names))

    # Additional Features
    (sliding_window_integral_left_pupil, sliding_window_integral_right_pupil,
     sliding_window_consecutive_elements_mean_left_pupil, sliding_window_consecutive_elements_mean_right_pupil,
     sliding_window_consecutive_elements_max_left_pupil, sliding_window_consecutive_elements_max_right_pupil,
     sliding_window_consecutive_elements_sum_left_pupil, sliding_window_consecutive_elements_sum_right_pupil,
     sliding_window_hrv_sdnn, sliding_window_hrv_rmssd, all_features_additional) = (
        sliding_window_other_features(time_start, stride, window_size, gloc_data_reduced,time_variable, number_windows,baseline_names_v0, baseline_v0))

    # Unpack Dictionary into Array & combine features into one feature array
    y_gloc_labels, x_feature_matrix = unpack_dict(gloc_window, sliding_window_mean, number_windows, sliding_window_stddev,
                                                  sliding_window_max, sliding_window_range,
                                                  sliding_window_integral_left_pupil,sliding_window_integral_right_pupil,
                                                  sliding_window_consecutive_elements_mean_left_pupil, sliding_window_consecutive_elements_mean_right_pupil,
                                                  sliding_window_consecutive_elements_max_left_pupil, sliding_window_consecutive_elements_max_right_pupil,
                                                  sliding_window_consecutive_elements_sum_left_pupil, sliding_window_consecutive_elements_sum_right_pupil,
                                                  sliding_window_hrv_sdnn, sliding_window_hrv_rmssd)

    # Remove rows with NaN (temporary solution-should replace with other method eventually)
    y_gloc_labels_noNaN, x_feature_matrix_noNaN = process_NaN(y_gloc_labels, x_feature_matrix)

    all_features = all_features_mean + all_features_stddev + all_features_max +  all_features_range + all_features_additional

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
