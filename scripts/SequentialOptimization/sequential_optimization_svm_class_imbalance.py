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

def main_loop(x_train, x_test, y_train, y_test, random_state, class_weight_imb, classifier_type,
              imbalance_technique, timestamp, method_key, train_class):

    # Define Loop Time
    loop_time = time.time()

    ################################################  CLASS IMBALANCE  ################################################
    """ 
          Explore Class Imbalance Section of Sequential Optimization Framework
    """

    # Random Under Sampling | rus
    if imbalance_technique == 'all' or imbalance_technique == 'rus':
        # Implement Imbalance Sampling Technique
        x_train, y_train = resample_rus(x_train, y_train, random_state)

    # Random Over Sampling | ros
    if imbalance_technique == 'all' or imbalance_technique == 'ros':
        # Implement Imbalance Sampling Technique
        x_train, y_train = resample_ros(x_train, y_train, random_state)

    # Synthetic Minority Over Sampling Technique | smote
    if imbalance_technique == 'all' or imbalance_technique == 'smote':
        # Implement Imbalance Sampling Technique
        x_train, y_train = simple_smote(x_train, y_train, random_state)

    # Modify Cost Function | cost_function
    if imbalance_technique == 'all' or imbalance_technique == 'cost_function':
        # Implement Imbalance Cost Function Technique
        class_weight_imb = 'balanced'

    # Random Under Sampling & Modify Cost Function | rus_cf
    if imbalance_technique == 'all' or imbalance_technique == 'rus_cf':
        # Implement Imbalance Hybrid Technique
        class_weight_imb = 'balanced'
        x_train, y_train = resample_rus(x_train, y_train, random_state)

    # Random Over Sampling & Modify Cost Function | rus_cf
    if imbalance_technique == 'all' or imbalance_technique == 'ros_cf':
        # Implement Imbalance Hybrid Technique
        class_weight_imb = 'balanced'
        x_train, y_train = resample_ros(x_train, y_train, random_state)

    # Synthetic Minority Over Sampling Technique & Modify Cost Function | smote_cf
    if imbalance_technique == 'all' or imbalance_technique == 'smote_cf':
        # Implement Imbalance Hybrid Technique
        class_weight_imb = 'balanced'
        x_train, y_train = simple_smote(x_train, y_train, random_state)

    # No imbalance technique | none
    if imbalance_technique == 'all' or imbalance_technique == 'none':
        # Implement no Class Imbalance Techniques
        x_train, y_train = x_train, y_train

    # Support Vector Machine | SVM
    if classifier_type == 'all' or classifier_type == 'SVM':
        accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm, g_mean_svm = (
            classify_svm(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                         save_folder=os.path.join("../ModelSave/SequentialOptimizationClassImbalance",
                                                  classifier_type, timestamp,
                                                  method_key), retrain=train_class))

    performance_metric_summary = single_classifier_performance_summary(accuracy_svm, precision_svm, recall_svm,
                                                                              f1_svm, specificity_svm, g_mean_svm)


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
    n_neighbors = 3

    # Specify Model Type
    model_type = ['noAFE', 'explicit']

    ## Model Parameters
    if 'noAFE' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                 'rawEEG', 'processedEEG', 'strain', 'demographics']
    if 'noAFE' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking','rawEEG', 'processedEEG']

    baseline_methods_to_use = ['v0','v1','v2']

    analysis_type = 2

    # Subject & Trial Information (only need to adjust this if doing analysis type 0,1)
    subject_to_analyze = '01'
    trial_to_analyze = '02'

    # Classifier Type
    classifier_to_use = 'SVM'
    train_class = True
    class_weight_imb = None

    # Sliding Window Parameters
    offset = 0  # seconds
    time_start = 0  # seconds
    training_ratio = 0.8

    # Define Sliding Window Parameters to Use
    baseline_window = 32.5
    window_size = 15
    stride = 0.25


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

    ################################################ FEATURE REDUCTION ################################################
    """ 
          Explore Feature Reduction Section of Sequential Optimization Framework
    """
    # Ridge Regression Feature Selection
    # Set threshold range
    percentile_threshold = 10

    # Determine reduced feature set & transform x_train and x_test
    x_train, x_test, selected_features = feature_selection_ridge(x_train, x_test,
                                                                 y_train, all_features,
                                                                 percentile_threshold,
                                                                 random_state)

    ############################################## CLASS IMBALANCE LOOP ##############################################

    # Pre-Allocate Performance Summary Dictionary
    class_imbalance_summary = dict()

    # Class Imbalance Methods to Implement
    class_imbalance_methods = ['rus', 'ros', 'smote', 'cost_function', 'rus_cf', 'ros_cf', 'smote_cf', 'none']

    for i in range(len(class_imbalance_methods)):
        current_class_imbalance_method = class_imbalance_methods[i]
        method_key_class_imbalance = 'class_imbalance_method_' + current_class_imbalance_method
        class_imbalance_summary[method_key_class_imbalance] = main_loop(x_train.copy(), x_test.copy(),
                                                                        y_train.copy(), y_test.copy(),
                                                                        random_state, class_weight_imb,
                                                                        classifier_to_use,
                                                                        current_class_imbalance_method,
                                                                        timestamp_outerloop,
                                                                        method_key_class_imbalance,
                                                                        train_class)

    # Save pkl summary
    save_folder = os.path.join("../PerformanceSave/SequentialOptimizationClassImbalance", classifier_to_use, timestamp_outerloop)
    save_file = 'Sequential_Optimization_ClassImbalance_dictionary_model_type_' + str(model_type[1]) + '.pkl'

    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(save_folder, save_file)

    with open(save_path, 'wb') as file:
        pickle.dump(class_imbalance_summary, file)

    duration = time.time() - start_time
    print(duration)