from threading import main_thread

from GLOC_data_processing import *
from GLOC_visualization import *
from GLOC_classifier import *
from features import *
import numpy as np
import pandas as pd

def main_analysis_loop(filename, plot_data, plot_pairwise, feature_to_analyze, time_variable, analysis_type,
                       baseline_window, window_size, stride, offset, time_start, training_ratio, subject_to_analyze):

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

    # Baseline Feature for Subject/Trial
    feature_baseline = baseline_features(baseline_window, gloc_data_reduced, features, time_variable)

    # Tabulate NaN
    NaN_table, NaN_proportion = tabulateNaN(feature_baseline, all_features)

    # Visualization of feature throughout trial
    if plot_data == 1:
        initial_visualization(gloc_data_reduced, gloc, feature_baseline, all_features, time_variable)

    # Sliding Window Mean
    gloc_window, sliding_window_mean, number_windows = sliding_window_mean_calc(time_start, offset, stride, window_size, feature_baseline, gloc, gloc_data_reduced, time_variable)

    # Sliding Window Standard Deviation, Max, Range
    sliding_window_stddev, sliding_window_max, sliding_window_range = sliding_window_calc(time_start, stride, window_size, feature_baseline, gloc_data_reduced, time_variable, number_windows)

    # Visualize sliding window mean
    if plot_data == 1:
        sliding_window_visualization(gloc_window, sliding_window_mean, number_windows, all_features, gloc_data_reduced)

    # Visualization of pairwise features
    if plot_pairwise == 1:
        pairwise_visualization(gloc_window, sliding_window_mean, all_features, gloc_data_reduced)

    # Unpack Dictionary into Array & combine features into one feature array
    y_gloc_labels, x_feature_matrix = unpack_dict(gloc_window, sliding_window_mean, number_windows, sliding_window_stddev, sliding_window_max, sliding_window_range)

    # Remove rows with NaN (temporary solution-should replace with other method eventually)
    y_gloc_labels_noNaN, x_feature_matrix_noNaN = process_NaN(y_gloc_labels, x_feature_matrix)

    # Update all features array
    all_features_mean = [s + '_mean' for s in all_features]
    all_features_stddev = [s + '_stddev' for s in all_features]
    all_features_max = [s + '_max' for s in all_features]
    all_features_range = [s + '_range' for s in all_features]
    all_features = all_features_mean + all_features_stddev + all_features_max + all_features_range

    ## Call functions for ML classification ##

    # Logistic Regression
    accuracy_logreg, precision_logreg, recall_logreg, f1_logreg = classify_logistic_regression(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio, all_features)

    # RF
    accuracy_rf, precision_rf, recall_rf, f1_rf = classify_random_forest(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio, all_features)

    # LDA
    accuracy_lda, precision_lda, recall_lda, f1_lda = classify_lda(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio, all_features)

    # KNN
    accuracy_knn, precision_knn, recall_knn, f1_knn = classify_knn(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio)

    # SVM
    accuracy_svm, precision_svm, recall_svm, f1_svm = classify_svm(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio)

    # Ensemble with Gradient Boosting
    accuracy_gb, precision_gb, recall_gb, f1_gb = classify_ensemble_with_gradboost(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio)

    # Define performance metrics
    accuracy = np.array([accuracy_logreg, accuracy_rf, accuracy_lda, accuracy_knn, accuracy_svm, accuracy_gb])
    precision = np.array([precision_logreg, precision_rf, precision_lda, precision_knn, precision_svm, precision_gb])
    recall = np.array([recall_logreg, recall_rf, recall_lda, recall_knn, recall_svm, recall_gb])
    f1 = np.array([f1_logreg, f1_rf, f1_lda, f1_knn, f1_svm, f1_gb])

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    # File Name & Path
    filename = '../data/all_trials_25_hz_stacked_null_str_filled.csv'

    # Plot Flags
    plot_data = 0  # flag to set whether plots should be generated (0 = no, 1 = yes)
    plot_pairwise = 0  # flag to set whether pairwise plots should be generated (0 = no, 1 = yes)

    # Feature Info
    # feature_to_analyze options:
    # ECG ('HR (bpm) - Equivital','ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital', 'HR_instant - Equivital', 'HR_average - Equivital', 'HR_w_average - Equivital')
    # BR ('BR (rpm) - Equivital')
    # temp ('Skin Temperature - IR Thermometer (°C) - Equivital')
    # fnirs ('HbO2 - fNIRS', 'Hbd - fNIRS')
    # eyetracking ('Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii', 'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii'
    # 'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii', 'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii')
    # EEG (coming soon!!!)
    feature_to_analyze = ['ECG', 'BR', 'temp', 'fnirs', 'eyetracking']

    time_variable = 'Time (s)'

    # Data Parameters
    analysis_type = 1  # flag to set which data should be analyzed
    # analysis_type = 0: analyze one trial from a subject
    # if analysis_type = 0, then set subject_to_analyze and trial_to_analyze parameters below
    # analysis_type = 1: analyze subject data (all trials for a subject)
    # if analysis_type = 1, then set subject_to_analyze parameter below
    # analysis_type = 2: analyze cohort data (all subjects, all trials)
    # if analysis_type = 2, then no extra parameters need to be set

    baseline_window = 10 # seconds
    window_size = 10     # seconds
    stride = 1           # seconds
    offset = 10          # seconds
    time_start = 0       # seconds

    # ML Splits
    training_ratio = 0.8

    subjects_to_analyze = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

    classifiers = ['Log Reg', 'RF', 'LDA', 'KNN', 'SVM', 'Ensemble w/ GB']

    # Initialize arrays
    accuracy = np.zeros((len(subjects_to_analyze), len(classifiers)))
    precision = np.zeros((len(subjects_to_analyze), len(classifiers)))
    recall = np.zeros((len(subjects_to_analyze), len(classifiers)))
    f1 = np.zeros((len(subjects_to_analyze), len(classifiers)))

    for subject in range(len(subjects_to_analyze)):
        subject_to_analyze = subjects_to_analyze[subject]

        accuracy[subject,:], precision[subject,:], recall[subject,:], f1[subject,:] = main_analysis_loop(filename, plot_data, plot_pairwise, feature_to_analyze, time_variable, analysis_type,
                       baseline_window, window_size, stride, offset, time_start, training_ratio, subject_to_analyze)


    accuracy_all = pd.DataFrame(accuracy, index = subjects_to_analyze, columns = classifiers)
    precision_all = pd.DataFrame(precision, index = subjects_to_analyze, columns = classifiers)
    recall_all = pd.DataFrame(recall, index = subjects_to_analyze, columns = classifiers)
    f1_all = pd.DataFrame(f1, index = subjects_to_analyze, columns = classifiers)

    x = 1