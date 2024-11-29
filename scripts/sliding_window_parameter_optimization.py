from GLOC_data_processing import *
from GLOC_visualization import *
from GLOC_classifier import *
from features import *
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


def main_analysis_loop(filename, plot_data, plot_pairwise, feature_to_analyze, time_variable, analysis_type,
                       baseline_window, window_size, stride, offset, time_start, training_ratio, subject_to_analyze):

    # Process CSV
    if analysis_type == 0:  # One Trial / One Subject
        gloc_data_reduced, features, all_features = load_and_process_csv(filename, analysis_type, feature_to_analyze,
                                                                         time_variable,
                                                                         trial_to_analyze=trial_to_analyze,
                                                                         subject_to_analyze=subject_to_analyze)
    elif analysis_type == 1:  # All Trials for One Subject
        gloc_data_reduced, features, all_features = load_and_process_csv(filename, analysis_type, feature_to_analyze,
                                                                         time_variable,
                                                                         subject_to_analyze=subject_to_analyze)
    elif analysis_type == 2:  # All Trials for All Subjects
        gloc_data_reduced, features, all_features = load_and_process_csv(filename, analysis_type, feature_to_analyze,
                                                                         time_variable)

    # Create GLOC Categorical Vector
    gloc = label_gloc_events(gloc_data_reduced)

    # Find time window after acceleration before GLOC
    # find_prediction_window(gloc_data_reduced, gloc, time_variable)

    # Baseline Feature for Subject/Trial
    feature_baseline = baseline_features(baseline_window, gloc_data_reduced, features, time_variable)

    # Tabulate NaN
    # NaN_table, NaN_proportion = tabulateNaN(feature_baseline, all_features)

    # Visualization of feature throughout trial
    if plot_data == 1:
        initial_visualization(gloc_data_reduced, gloc, feature_baseline, all_features, time_variable)

    # Sliding Window Mean
    gloc_window, sliding_window_mean, number_windows = sliding_window_mean_calc(time_start, offset, stride, window_size,
                                                                                feature_baseline, gloc,
                                                                                gloc_data_reduced, time_variable)

    # Sliding Window Standard Deviation, Max, Range
    sliding_window_stddev, sliding_window_max, sliding_window_range = sliding_window_calc(time_start, stride,
                                                                                          window_size, feature_baseline,
                                                                                          gloc_data_reduced,
                                                                                          time_variable, number_windows)

    # Additional Features
    sliding_window_pupil_difference, sliding_window_ox_deox_ratio = sliding_window_other_features(time_start, stride,
                                                                                                  window_size,
                                                                                                  feature_baseline,
                                                                                                  gloc_data_reduced,
                                                                                                  time_variable,
                                                                                                  number_windows,
                                                                                                  all_features)

    # Visualize sliding window mean
    if plot_data == 1:
        sliding_window_visualization(gloc_window, sliding_window_mean, number_windows, all_features, gloc_data_reduced)

    # Visualization of pairwise features
    if plot_pairwise == 1:
        pairwise_visualization(gloc_window, sliding_window_mean, all_features, gloc_data_reduced)

    # Unpack Dictionary into Array & combine features into one feature array
    y_gloc_labels, x_feature_matrix = unpack_dict(gloc_window, sliding_window_mean, number_windows,
                                                  sliding_window_stddev, sliding_window_max, sliding_window_range,
                                                  sliding_window_pupil_difference, sliding_window_ox_deox_ratio)

    # Remove rows with NaN (temporary solution-should replace with other method eventually)
    y_gloc_labels_noNaN, x_feature_matrix_noNaN = process_NaN(y_gloc_labels, x_feature_matrix)

    # Update all features array
    all_features_mean = [s + '_mean' for s in all_features]
    all_features_stddev = [s + '_stddev' for s in all_features]
    all_features_max = [s + '_max' for s in all_features]
    all_features_range = [s + '_range' for s in all_features]
    pupil_features = ['Pupil Difference']
    fnirs_features = ['Hbo/Hbd']
    all_features = all_features_mean + all_features_stddev + all_features_max + all_features_range + pupil_features + fnirs_features

    ## Call functions for ML classification ##

    # Logistic Regression
    # accuracy_logreg, precision_logreg, recall_logreg, f1_logreg, specificity_logreg = classify_logistic_regression(
    #    y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio, all_features)

    # RF
    accuracy_rf, precision_rf, recall_rf, f1_rf, tree_depth, specificity_rf = classify_random_forest(
        y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio, all_features)

    # LDA
    # accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda = classify_lda(y_gloc_labels_noNaN,
    #                                                                                x_feature_matrix_noNaN,
    #                                                                                training_ratio, all_features)

    # KNN
    # accuracy_knn, precision_knn, recall_knn, f1_knn, specificity_knn = classify_knn(y_gloc_labels_noNaN,
    #                                                                                x_feature_matrix_noNaN,
    #                                                                                training_ratio)

    # SVM
    # accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm = classify_svm(y_gloc_labels_noNaN,
    #                                                                                x_feature_matrix_noNaN,
    #                                                                                training_ratio)

    # Ensemble with Gradient Boosting
    #accuracy_gb, precision_gb, recall_gb, f1_gb, specificity_gb = classify_ensemble_with_gradboost(y_gloc_labels_noNaN,
    #                                                                                               x_feature_matrix_noNaN,
    #                                                                                               training_ratio)

    # Build Performance Metric Summary Tables
    # performance_metric_summary = summarize_performance_metrics(accuracy_logreg, accuracy_rf, accuracy_lda, accuracy_knn,
    #                                                           accuracy_svm, accuracy_gb,
    #                                                           precision_logreg, precision_rf, precision_lda,
    #                                                           precision_knn, precision_svm, precision_gb,
    #                                                           recall_logreg, recall_rf, recall_lda, recall_knn,
    #                                                           recall_svm, recall_gb,
    #                                                           f1_logreg, f1_rf, f1_lda, f1_knn, f1_svm, f1_gb,
    #                                                           specificity_logreg, specificity_rf, specificity_lda,
    #                                                           specificity_knn, specificity_svm, specificity_gb)


    return accuracy_rf, precision_rf, recall_rf, f1_rf, specificity_rf

if __name__ == "__main__":
    # File Name & Path
    filename = '../../all_trials_25_hz_stacked_null_str_filled.csv'

    # Plot Flags
    plot_data = 0        # flag to set whether plots should be generated (0 = no, 1 = yes)
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

    baseline_window = 10  # seconds
    time_start = 0  # seconds

    # ML Splits
    training_ratio = 0.8

    # Subject & Trial Information
    subject_to_analyze = '04'
    trial_to_analyze = '01'

    subjects_to_analyze = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
    classifiers = ['Log Reg', 'RF', 'LDA', 'KNN', 'SVM', 'Ensemble w/ GB']

    window_sizes_to_analyze = np.linspace(1, 20, 10)
    strides_to_analyze = np.linspace(0.25, 4, 10)
    offsets_to_analyze = np.linspace(0, 25, 6)

    # window_sizes_to_analyze = np.linspace(5, 15, 2)
    # strides_to_analyze = np.linspace(0.25, 15, 2)
    # offsets_to_analyze = np.linspace(0, 30, 2)

    # Initialize arrays
    accuracy = dict()
    precision = dict()
    recall = dict()
    specificity = dict()
    f1 = dict()

    # Initialize arrays
    accuracy_single = np.zeros((len(strides_to_analyze), len(window_sizes_to_analyze)))
    precision_single = np.zeros((len(strides_to_analyze), len(window_sizes_to_analyze)))
    recall_single = np.zeros((len(strides_to_analyze), len(window_sizes_to_analyze)))
    f1_single = np.zeros((len(strides_to_analyze), len(window_sizes_to_analyze)))
    specificity_single = np.zeros((len(strides_to_analyze), len(window_sizes_to_analyze)))

    for i in range(len(offsets_to_analyze)):
        offset = offsets_to_analyze[i]
        for j in range(len(window_sizes_to_analyze)):
            window_size = window_sizes_to_analyze[j]
            for k in range(len(strides_to_analyze)):
                stride = strides_to_analyze[k]

                accuracy_single[k,j], precision_single[k,j], recall_single[k,j], f1_single[k,j], specificity_single[k,j] = main_analysis_loop(filename,plot_data,plot_pairwise,feature_to_analyze,time_variable,analysis_type,
                                                                                                                     baseline_window,window_size,stride,offset,time_start,training_ratio,subject_to_analyze)

        accuracy_all = pd.DataFrame(accuracy_single, index=strides_to_analyze, columns=window_sizes_to_analyze)
        precision_all = pd.DataFrame(precision_single, index=strides_to_analyze, columns=window_sizes_to_analyze)
        recall_all = pd.DataFrame(recall_single, index=strides_to_analyze, columns=window_sizes_to_analyze)
        specificity_all = pd.DataFrame(specificity_single, index=strides_to_analyze, columns=window_sizes_to_analyze)
        f1_all = pd.DataFrame(f1_single, index=strides_to_analyze, columns=window_sizes_to_analyze)

        accuracy[offset] = accuracy_all
        precision[offset] = precision_all
        recall[offset] = recall_all
        specificity[offset] = specificity_all
        f1[offset] = f1_all

        ## Matplotlib Sample Code using 2D arrays via meshgrid
        X = window_sizes_to_analyze
        Y = strides_to_analyze
        XX, YY = np.meshgrid(X, Y)

        reshaped_x = XX.reshape(-1)
        reshaped_y = YY.reshape(-1)

        z = np.array(f1[offset])
        reshaped_z = z.reshape(-1)

        fig = plt.figure()

        # syntax for 3-D projection
        ax = plt.axes(projection='3d')

        # Plot the surface
        ax.plot_surface(XX, YY, z, cmap='cool')

        # Set labels
        ax.set_xlabel('Window Size [s]')
        ax.set_ylabel('Stride [s]')
        ax.set_zlabel('F1 Score')

        plt.title(f'Offset = {offset} [s]')

        # Show the plot
        plt.show()



        ## Matplotlib Sample Code using 2D arrays via meshgrid
        X = window_sizes_to_analyze
        Y = strides_to_analyze
        XX, YY = np.meshgrid(X, Y)

        reshaped_x = XX.reshape(-1)
        reshaped_y = YY.reshape(-1)

        z = np.array(f1[offset])
        reshaped_z = z.reshape(-1)

        fig = plt.figure()

        # syntax for 3-D projection
        ax = plt.axes(projection='3d')

        # Plot the surface
        ax.scatter(reshaped_x, reshaped_y, reshaped_z)

        # Set labels
        ax.set_xlabel('Window Size [s]')
        ax.set_ylabel('Stride [s]')
        ax.set_zlabel('F1 Score')

        plt.title(f'Offset = {offset} [s]')

        # Show the plot
        plt.show()

    x = 1
