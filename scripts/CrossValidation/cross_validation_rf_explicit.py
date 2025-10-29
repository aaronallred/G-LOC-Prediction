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

def main_loop(kfold_ID, num_splits, runname, y_gloc_labels, x_feature_matrix, random_state, all_features,
              classifier_type, train_class, class_weight_imb):

    # Define Loop Time
    loop_time = time.time()

    ################################################ TRAIN/TEST SPLIT  ################################################
    """ 
          Split data into training/test for optimization loop of sequential optimization framework.
    """

    # Training/Test Split
    x_train, x_test, y_train, y_test = stratified_kfold_split(y_gloc_labels,x_feature_matrix,
                                                              num_splits, kfold_ID)

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
        y_train, x_train = y_train, x_train
        y_test, x_test = y_test, x_test

    ################################################ FEATURE REDUCTION ################################################
    """ 
          Explore Feature Reduction Section of Sequential Optimization Framework
    """
    # None

    ################################################ CLASS IMBALANCE ################################################

    # No imbalance technique | none
    x_train, y_train = x_train, y_train

    ################################################ MACHINE LEARNING ################################################
    save_folder = os.path.join("../ModelSave/CV", runname, str(kfold_ID))

    # Random Forest HPO | rf_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'rf_hpo':
        accuracy_rf_hpo, precision_rf_hpo, recall_rf_hpo, f1_rf_hpo, tree_depth_hpo, specificity_rf_hpo, g_mean_rf_hpo  = (
            classify_random_forest_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                                       save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_rf_hpo, precision_rf_hpo, recall_rf_hpo, f1_rf_hpo,
            specificity_rf_hpo, g_mean_rf_hpo,['RF'])


    loop_duration = time.time() - loop_time
    print(loop_duration)

    if classifier_type == 'all':
        return performance_metric_summary
    if classifier_type == 'all_hpo':
        return performance_metric_summary_hpo
    else:
        return performance_metric_summary_single

if __name__ == "__main__":
    # Get start time
    start_time = time.time()

    ################################################### USER INPUTS  ###################################################
    ## Data Folder Location
    datafolder = '../../'
    # datafolder = '../data/'

    # Random State | 42 - Debug mode
    random_state = 42

    ## Classifier | Pick 'logreg' 'rf' 'LDA' 'KNN' 'SVM' 'EGB' or 'all'
    classifier_type = 'rf_hpo'
    train_class = True
    class_weight_imb = None

    # Data Handling Options
    remove_NaN_trials = True

    impute_type = 1
    n_neighbors = 3

    ## Model Parameters
    model_type = ['noAFE', 'explicit']
    if 'noAFE' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                     'rawEEG', 'processedEEG', 'strain', 'demographics']
    if 'noAFE' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'rawEEG', 'processedEEG']
    if 'combined' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                     'rawEEG', 'processedEEG', 'strain', 'demographics']
    if 'combined' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'rawEEG', 'processedEEG', 'AFE']

    baseline_methods_to_use = ['v0','v1','v2','v5','v6','v7','v8']

    analysis_type = 2

    # Define Sliding Window Parameters to Use
    baseline_window = 18.75
    window_size = 7.5
    stride = 0.25
    offset = 0  # seconds
    time_start = 0  # seconds

    # Subject & Trial Information (only need to adjust this if doing analysis type 0,1)
    subject_to_analyze = '01'
    trial_to_analyze = '02'

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
                                       model_type, list_of_eeg_data_files, trial_to_analyze, subject_to_analyze))

    # Create GLOC Categorical Vector
    gloc = label_gloc_events(gloc_data_reduced)

    # Reduce Dataset based on AFE / nonAFE condition
    gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc = (
        afe_subset(model_type, gloc_data_reduced, all_features,
                   features, features_phys, features_ecg, features_eeg, gloc))

    ############################################### DATA CLEAN AND PREP ###############################################
    """ 
       Optional handling of raw NaN data, depending on 'remove_NaN_trials' 'impute_type' <= 1
    """

    ### Remove full trials with NaN
    if remove_NaN_trials:
        gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc, nan_proportion_df = (
            remove_all_nan_trials(gloc_data_reduced, all_features,
                                  features, features_phys, features_ecg, features_eeg, gloc))

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

    combined_baseline, combined_baseline_names, baseline_v0, baseline_names_v0 = (
        baseline_data(baseline_methods_to_use, trial_column, time_column, event_validated_column, subject_column,
                      features, all_features,
                      gloc, baseline_window, features_phys, all_features_phys, features_ecg, all_features_ecg,
                      features_eeg, all_features_eeg, baseline_data_filename, list_of_baseline_eeg_processed_files,
                      model_type))

    ################################################ GENERATE FEATURES ################################################
    """
        Generates features from baseline data
    """

    y_gloc_labels, x_feature_matrix, all_features = (
        feature_generation(time_start, offset, stride, window_size, combined_baseline, gloc, trial_column, time_column,
                           combined_baseline_names, baseline_names_v0, baseline_v0, feature_groups_to_analyze))

    ############################################# FEATURE CLEAN AND PREP ##############################################
    """ 
          Optional handling of raw NaN data, depending on 'impute_type' >= 2
    """

    # Remove constant columns
    x_feature_matrix, all_features = remove_constant_columns(x_feature_matrix, all_features)

    # Remove all NaN rows from x matrix before train/test split for method 2 & method 1 if there are remaining NaNs
    if impute_type == 2 or impute_type == 1:
        y_gloc_labels, x_feature_matrix, all_features = process_NaN(y_gloc_labels, x_feature_matrix, all_features)


    #################################################### CV LOOP #####################################################
    # Get time stamp for saving models
    # runname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runname = 'Explicit'

    # Test set identifier for 10-fold Model Validation
    num_splits = 10
    kfold_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Pre-Allocate Performance Summary Dictionary
    kfold_performance_summary = dict()

    # Loop through Imputation Methods
    for i in range(len(kfold_ID)):
        # Loop through all train-test splits
        method_key = str(kfold_ID[i])
        kfold_performance_summary[method_key] = main_loop(kfold_ID[i], num_splits, runname, y_gloc_labels.copy(),
                                                          x_feature_matrix.copy(), random_state, all_features.copy(),
                                                          classifier_type, train_class, class_weight_imb)

        # Save pkl summary for this iteration
        save_folder = os.path.join("../PerformanceSave/CrossValidation", classifier_type, runname, method_key)
        save_file = 'FoldSummary.pkl'
        save_path = os.path.join(save_folder, save_file)

        # Ensure the save folder exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        with open(save_path, 'wb') as file:
            pickle.dump(kfold_performance_summary, file)

    # Save pkl summary
    save_folder = os.path.join("../PerformanceSave/CrossValidation", classifier_type, runname)
    save_file = 'CrossValidation.pkl'
    save_path = os.path.join(save_folder, save_file)

    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(save_path, 'wb') as file:
        pickle.dump(kfold_performance_summary, file)

    plot_cross_val(kfold_performance_summary)
	
    duration = time.time() - start_time
    print(duration)