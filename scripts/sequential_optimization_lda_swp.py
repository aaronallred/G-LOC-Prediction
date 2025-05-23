from GLOC_data_processing import *
from imputation import *
from baseline_methods import *
from features import *
from feature_selection import *
from GLOC_classifier import *
from GLOC_visualization import *
from imbalance_techniques import *
import pickle
import time
from numpy import number
from openpyxl.styles.builtins import percent
import os
from datetime import datetime

def main_loop(impute_type, n_neighbors, timestamp, model_type, method_key, baseline_window, window_size, stride, classifier_type,
              gloc,time_start, offset, training_ratio, random_state, class_weight_imb, trial_column, time_column, all_features,
              baseline_methods_to_use, event_validated_column, subject_column, features, features_phys,
              all_features_phys, features_ecg, all_features_ecg, features_eeg, all_features_eeg,
              baseline_data_filename, list_of_baseline_eeg_processed_files):

    # Define Loop Time
    loop_time = time.time()

    ################################################# GRAB USER INPUTS  ################################################

    # File names to save data .pkl as

    output_file = ('all_data_'  + str(classifier_type) +
                         '_baseline_' + str(round(baseline_window, 0)) +
                         '_window_size_' + str(round(window_size, 0)) + '_stride_' + str(round(stride, 0)) +
                         '_model_type_' + str(model_type[1]) +'.pkl')

    # # Specify Folder to Save X_matrix, Y-label and Feature Names
    # pkl_save_folder = os.path.join("../PklSave/SequentialOptimizationSlidingWindow", classifier_type, timestamp_outerloop)
    # if not os.path.exists(pkl_save_folder):
    #     os.makedirs(pkl_save_folder)

    ################################################## BASELINE DATA ##################################################
    """ 
        Baselines pre-feature data based on 'baseline_methods_to_use'
    """

    combined_baseline, combined_baseline_names, baseline_v0, baseline_names_v0= (
        baseline_data(baseline_methods_to_use, trial_column, time_column, event_validated_column, subject_column, features, all_features,
                      gloc,baseline_window, features_phys, all_features_phys, features_ecg, all_features_ecg,
                      features_eeg, all_features_eeg, baseline_data_filename, list_of_baseline_eeg_processed_files,
                      model_type))

    ################################################ GENERATE FEATURES ################################################
    """
        Generates features from baseline data
    """

    y_gloc_labels, x_feature_matrix, all_features = (
        feature_generation(time_start, offset, stride, window_size, combined_baseline, gloc, trial_column, time_column,
                           combined_baseline_names,baseline_names_v0, baseline_v0, feature_groups_to_analyze))

    # Remove constant columns
    x_feature_matrix, all_features = remove_constant_columns(x_feature_matrix, all_features)

    # # Save pkl
    # with open(os.path.join(pkl_save_folder, y_label_name), 'wb') as file:
    #     pickle.dump(y_gloc_labels, file)
    #
    # with open(os.path.join(pkl_save_folder, feature_matrix_name), 'wb') as file:
    #     pickle.dump(x_feature_matrix, file)
    #
    # with open(os.path.join(pkl_save_folder, all_features_name), 'wb') as file:
    #     pickle.dump(all_features, file)

    ################################################ TRAIN/TEST SPLIT  ################################################
    """ 
          Split data into training/test for optimization loop of sequential optimization framework.
    """

    # Remove all NaN rows from x matrix before train/test split for method 2 & method 1 if there are remaining NaNs
    if impute_type == 2 or impute_type == 1:
        y_gloc_labels, x_feature_matrix, all_features = process_NaN(y_gloc_labels, x_feature_matrix, all_features)


    # Training/Test Split
    x_train_NaN, x_test_NaN, y_train_NaN, y_test_NaN = pre_classification_training_test_split(y_gloc_labels,
                                                                                              x_feature_matrix,
                                                                                              training_ratio,
                                                                                              random_state)

    ################################################# IMPUTATION ####################################################
    """ 
          Remove NaNs from data if method 2, impute using kNN imputation if method 3. Remove all remaining rows with NaN
          for method 1. Otherwise, do nothing.
    """
    # If method 3, apply knn imputation to x matrix on train/test separately
    if impute_type == 3:
        # Leave y-labels as-is
        y_train = y_train_NaN
        y_test = y_test_NaN

        # Impute Train Data Independently
        x_train = faster_knn_impute(x_train_NaN, k=n_neighbors)

        # Impute Test Data
        x_test = faster_knn_impute(x_test_NaN, k=n_neighbors)

    # if method 1 or 2, NaN rows have already been removed, so just need to relabel variables
    else:
        # Leave train/test matrix as is
        y_train, x_train = y_train_NaN, x_train_NaN
        y_test, x_test = y_test_NaN, x_test_NaN

    # # Save pkl
    # output_data = (y_train, y_test, x_train, x_test)
    # with open(os.path.join(pkl_save_folder, output_file), 'wb') as file:
    #     pickle.dump(output_data, file)
    ################################################ MACHINE LEARNING ################################################

    # Linear discriminant analysis HPO | LDA_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'LDA_hpo':
        accuracy_lda_hpo, precision_lda_hpo, recall_lda_hpo, f1_lda_hpo, specificity_lda_hpo, g_mean_lda_hpo = (
            classify_lda_hpo(x_train, x_test, y_train, y_test, random_state,
                             save_folder=os.path.join("../ModelSave/SequentialOptimizationSlidingWindow", timestamp, method_key), retrain=train_class))

        performance_metric_summary = single_classifier_performance_summary(accuracy_lda_hpo, precision_lda_hpo,
                                                                           recall_lda_hpo, f1_lda_hpo,
                                                                           specificity_lda_hpo, g_mean_lda_hpo)
    
    # Linear discriminant analysis | LDA
    if classifier_type == 'all' or classifier_type == 'LDA':
        accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda, g_mean_lda = (
            classify_lda(x_train, x_test, y_train, y_test, random_state,
                         save_folder=os.path.join("../ModelSave/SequentialOptimizationSlidingWindow", timestamp, method_key), retrain=train_class))

        performance_metric_summary = single_classifier_performance_summary(accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda, g_mean_lda)

    loop_duration = time.time() - loop_time
    print(loop_duration)

    return performance_metric_summary


if __name__ == "__main__":
    # Get start time
    start_time = time.time()

    # Get time stamp for saving models
    timestamp_outerloop = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ################################################### USER INPUTS  ###################################################
    ## Data Folder Location
    datafolder = '../../'

    # Random State | 42 - Debug mode
    random_state = 42

    # Data Handling Options
    remove_NaN_trials = True

    # Imputation Methods to Test
    impute_type = 1

    # N-Neighbors Definition for Imputation Methods 1 & 3
    n_neighbors = 5

    # Specify Model Type
    model_type = ['noAFE', 'explicit']

    ## Model Parameters
    if 'noAFE' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                 'rawEEG', 'processedEEG', 'strain', 'demographics']
    if 'noAFE' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking','rawEEG', 'processedEEG']

    baseline_methods_to_use = ['v0','v1','v2','v5','v6','v7','v8']

    analysis_type = 2

    # Subject & Trial Information (only need to adjust this if doing analysis type 0,1)
    subject_to_analyze = '01'
    trial_to_analyze = '02'

    # Classifier Type
    classifier_to_use = 'LDA'
    train_class = True
    class_weight_imb = None

    # Sliding Window Parameters
    offset = 0  # seconds
    time_start = 0  # seconds
    training_ratio = 0.8

    ############################################# LOAD AND PROCESS DATA #############################################
    """ 
       Grabs GLOC event and predictor data, depending on 'analysis_type' and 'feature_groups_to_analyze'
    """

    # Grab Data File Locations
    (filename, baseline_data_filename, demographic_data_filename,
     list_of_eeg_data_files, list_of_baseline_eeg_processed_files) = data_locations(datafolder)

    # Load Data
    (gloc_data_reduced, features, features_phys, features_ecg, features_eeg, all_features, all_features_phys,
     all_features_ecg, all_features_eeg) = (
        analysis_driven_csv_processing(analysis_type, filename, feature_groups_to_analyze, demographic_data_filename,
                                       model_type,list_of_eeg_data_files,trial_to_analyze,subject_to_analyze))

    # Create GLOC Categorical Vector
    gloc = label_gloc_events(gloc_data_reduced)

    # Reduce Dataset based on AFE / nonAFE condition
    gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc = (
        afe_subset(model_type, gloc_data_reduced,all_features,
                   features,features_phys, features_ecg, features_eeg, gloc))

    ### Remove full trials with NaN
    if remove_NaN_trials:
        gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc, nan_proportion_df = (
            remove_all_nan_trials(gloc_data_reduced, all_features,
                                  features,features_phys, features_ecg, features_eeg, gloc))

    ############################################### DATA CLEAN AND PREP ###############################################
    """ 
       Optional handling of raw NaN data, depending on 'remove_NaN_trials' 'impute_type' <= 1
    """

    ### Impute missing row data
    if impute_type == 1:
        features = faster_knn_impute(features, k=n_neighbors)

    ################################################## REDUCE MEMORY ##################################################

    # Grab columns from gloc_data_reduced and remove gloc_data_reduced variable from memory
    trial_column = gloc_data_reduced['trial_id']
    time_column = gloc_data_reduced['Time (s)']
    event_validated_column = gloc_data_reduced['event_validated']
    subject_column = gloc_data_reduced['subject']

    del gloc_data_reduced

    ############################################# SWP ASSESSMENT LOOP #############################################

    # Pre-Allocate Performance Summary Dictionary
    sliding_window_performance_summary = dict()

    # Define Range of Values for Sliding Window Parameter Optimization
    window_sizes_to_analyze = np.linspace(5, 15, 5)
    strides_to_analyze = np.linspace(0.25, 5, 5)
    baseline_windows_to_analyze = np.linspace(5, 60, 5)

    for i in range(len(baseline_windows_to_analyze)):
        current_baseline_window = baseline_windows_to_analyze[i]
        for j in range(len(window_sizes_to_analyze)):
            current_window_size = window_sizes_to_analyze[j]
            for k in range(len(strides_to_analyze)):
                current_stride = strides_to_analyze[k]
                method_key_sliding_window = ('baseline_' + str(round(current_baseline_window,0)) + '_window_size_' +
                                             str(round(current_window_size, 0)) + '_stride_' + str(round(current_stride, 0)))
                sliding_window_performance_summary[method_key_sliding_window] = main_loop(impute_type, n_neighbors,
                                                                                          timestamp_outerloop, model_type,
                                                                                          method_key_sliding_window,
                                                                                          current_baseline_window,
                                                                                          current_window_size,
                                                                                          current_stride, classifier_to_use,
                                                                                          gloc, time_start, offset,
                                                                                          training_ratio, random_state,
                                                                                          class_weight_imb,
                                                                                          trial_column, time_column,
                                                                                          all_features,
                                                                                          baseline_methods_to_use,
                                                                                          event_validated_column,
                                                                                          subject_column, features,
                                                                                          features_phys,
                                                                                          all_features_phys,
                                                                                          features_ecg, all_features_ecg,
                                                                                          features_eeg, all_features_eeg,
                                                                                          baseline_data_filename,
                                                                                          list_of_baseline_eeg_processed_files
                                                                                          )

    # Save pkl summary
    save_folder = os.path.join("../PerformanceSave/SequentialOptimizationSlidingWindow", classifier_to_use, timestamp_outerloop)
    save_file = 'Sequential_Optimization_Sliding_Window_dictionary_model_type_' + str(model_type[1]) + '.pkl'

    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(save_folder, save_file)

    with open(save_path, 'wb') as file:
        pickle.dump(sliding_window_performance_summary, file)

    duration = time.time() - start_time
    print(duration)