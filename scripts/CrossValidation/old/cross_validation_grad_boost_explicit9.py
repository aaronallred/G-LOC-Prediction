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

def main_loop(kfold_ID, num_splits, runname, y_gloc_labels_noNaN, x_feature_matrix_noNaN, random_state, all_features,
              classifier_type, train_class):

    # Define Loop Time
    loop_time = time.time()

    ################################################ TRAIN/TEST SPLIT  ################################################
    """ 
          Split data into training/test for optimization loop of sequential optimization framework.
    """

    # Training/Test Split
    x_train, x_test, y_train, y_test = stratified_kfold_split(y_gloc_labels_noNaN,x_feature_matrix_noNaN,
                                                              num_splits, kfold_ID)

    ################################################ FEATURE REDUCTION ################################################
    """ 
          Explore Feature Reduction Section of Sequential Optimization Framework
    """
    # Ridge Regression Feature Selection
    # Set threshold range
    percentile_threshold = 60

    # Determine reduced feature set & transform x_train and x_test
    x_train, x_test, selected_features = feature_selection_ridge(x_train, x_test,
                                                                 y_train, all_features,
                                                                 percentile_threshold,
                                                                 random_state)

    ################################################ CLASS IMBALANCE ################################################

    # No imbalance technique | none
    x_train, y_train = x_train, y_train

    ################################################ MACHINE LEARNING ################################################
    save_folder = os.path.join("../ModelSave/CV", runname, str(kfold_ID))

    # Ensemble with Gradient Boosting HPO | EGB_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'EGB_hpo':
        accuracy_gb_hpo, precision_gb_hpo, recall_gb_hpo, f1_gb_hpo, specificity_gb_hpo, g_mean_gb_hpo = (
            classify_ensemble_with_gradboost_hpo(x_train, x_test, y_train, y_test, random_state,
                                                 save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_gb_hpo, precision_gb_hpo, recall_gb_hpo, f1_gb_hpo,
            specificity_gb_hpo, g_mean_gb_hpo, ['Ensemble w/ GB'])

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
    classifier_type = 'EGB_hpo'
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

    baseline_methods_to_use = ['v0','v1','v2','v5','v6','v7','v8']

    analysis_type = 2

    # Define Sliding Window Parameters to Use
    baseline_window = 32.5
    window_size = 15
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

    if impute_type == 2 or impute_type == 1:
        # Remove rows with NaN (temporary solution-should replace with other method eventually)
        y_gloc_labels_noNaN, x_feature_matrix_noNaN, all_features = process_NaN(y_gloc_labels, x_feature_matrix,
                                                                                all_features)
    # elif impute_type == 3:
    #     y_gloc_labels_noNaN = y_gloc_labels
    #     x_feature_matrix_noNaN, indicator_matrix = knn_impute(x_feature_matrix, n_neighbors)
    else:
        y_gloc_labels_noNaN, x_feature_matrix_noNaN = y_gloc_labels, x_feature_matrix

    #################################################### CV LOOP #####################################################
    # Get time stamp for saving models
    # runname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runname = 'Explicit'

    # Test set identifier for 10-fold Model Validation
    num_splits = 10
    kfold_ID = [8]

    # Pre-Allocate Performance Summary Dictionary
    kfold_performance_summary = dict()

    # Loop through Imputation Methods
    for i in range(len(kfold_ID)):
        # Loop through all train-test splits
        method_key = str(kfold_ID[i])
        kfold_performance_summary[method_key] = main_loop(kfold_ID[i], num_splits, runname, y_gloc_labels_noNaN.copy(),
                                                          x_feature_matrix_noNaN.copy(), random_state, all_features.copy(),
                                                          classifier_type, train_class)

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