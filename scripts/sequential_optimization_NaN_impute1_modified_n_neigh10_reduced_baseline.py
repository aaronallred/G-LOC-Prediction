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

def main_loop(impute_type, n_neighbors, timestamp, model_type, method_key):
    start_time = time.time()

    ################################################### USER INPUTS  ###################################################
    ## Data Folder Location
    datafolder = '../../'
    # datafolder = '../data/'

    # Random State | 42 - Debug mode
    random_state = 42

    ## Classifier | Pick 'logreg' 'rf' 'LDA' 'KNN' 'SVM' 'EGB' or 'all'
    classifier_type = 'all'
    train_class = True
    class_weight_imb = None

    # Data Handling Options
    remove_NaN_trials = True
    impute_type = impute_type

    ## Model Parameters
    model_type = model_type
    if 'noAFE' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                 'rawEEG', 'processedEEG', 'strain', 'demographics']
    if 'noAFE' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking','rawEEG', 'processedEEG']

    baseline_methods_to_use = ['v0','v1','v2','v5','v6','v7','v8']

    analysis_type = 2

    baseline_window = 10  # seconds
    window_size = 10  # seconds
    stride = 1  # seconds
    offset = 0  # seconds
    time_start = 0  # seconds
    training_ratio = 0.8

    # Subject & Trial Information (only need to adjust this if doing analysis type 0,1)
    subject_to_analyze = '01'
    trial_to_analyze = '02'

    # File names to save data .pkl as
    output_file = ('all_data_' + 'impute_type_' + str(impute_type) +'.pkl')


    # Specify Folder to Save X_matrix, Y-label and Feature Names
    pkl_save_folder = os.path.join("../PklSave/SequentialOptimizationNaN", timestamp_outerloop)
    if not os.path.exists(pkl_save_folder):
        os.makedirs(pkl_save_folder)


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


    ############################################### DATA CLEAN AND PREP ###############################################
    """ 
       Optional handling of raw NaN data, depending on 'remove_NaN_trials' 'impute_type' <= 1
    """

    ### Remove full trials with NaN
    if remove_NaN_trials:
        gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc, nan_proportion_df = (
            remove_all_nan_trials(gloc_data_reduced, all_features,
                                  features,features_phys, features_ecg, features_eeg, gloc))

    ### Impute missing row data
    if impute_type == 1:
        features, indicator_matrix = knn_impute(features, n_neighbors)

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

   ################################################## IMPUTATION ####################################################
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
        x_train, indicator_matrix_train = knn_impute(x_train_NaN, n_neighbors)

        # Impute Test Data
        x_test, indicator_matrix_test = knn_impute(x_test_NaN, n_neighbors)

    # if method 1 or 2, NaN rows have already been removed, so just need to relabel variables
    else:
        # Leave train/test matrix as is
        y_train, x_train = y_train_NaN, x_train_NaN
        y_test, x_test = y_test_NaN, x_test_NaN

    # Save pkl
    output_data = (y_train, y_test, x_train, x_test)
    with open(os.path.join(pkl_save_folder, output_file), 'wb') as file:
        pickle.dump(output_data, file)

 ################################################ MACHINE LEARNING ################################################

    # Logistic Regression HPO | logreg_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'logreg_hpo':
        accuracy_logreg_hpo, precision_logreg_hpo, recall_logreg_hpo, f1_logreg_hpo, specificity_logreg_hpo, g_mean_logreg_hpo = (
            classify_logistic_regression_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                                             save_folder=os.path.join("../ModelSave/SequentialOptimizationNaN", timestamp, method_key), retrain=train_class))

    # Logistic Regression | logreg
    if classifier_type == 'all' or classifier_type == 'logreg':
        accuracy_logreg, precision_logreg, recall_logreg, f1_logreg, specificity_logreg, g_mean_logreg = (
            classify_logistic_regression(x_train, x_test, y_train, y_test, class_weight_imb,random_state,
                                         save_folder=os.path.join("../ModelSave/SequentialOptimizationNaN", timestamp, method_key), retrain=train_class))

    # Random Forest HPO | rf_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'rf_hpo':
        accuracy_rf_hpo, precision_rf_hpo, recall_rf_hpo, f1_rf_hpo, tree_depth_hpo, specificity_rf_hpo, g_mean_rf_hpo  = (
            classify_random_forest_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                                       save_folder=os.path.join("../ModelSave/SequentialOptimizationNaN", timestamp, method_key), retrain=train_class))

    # Random Forrest | rf
    if classifier_type == 'all' or classifier_type == 'rf':
        accuracy_rf, precision_rf, recall_rf, f1_rf, tree_depth, specificity_rf, g_mean_rf = (
            classify_random_forest(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                                   save_folder=os.path.join("../ModelSave/SequentialOptimizationNaN", timestamp, method_key), retrain=train_class))

    # Linear discriminant analysis HPO | LDA_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'LDA_hpo':
        accuracy_lda_hpo, precision_lda_hpo, recall_lda_hpo, f1_lda_hpo, specificity_lda_hpo, g_mean_lda_hpo = (
            classify_lda_hpo(x_train, x_test, y_train, y_test, random_state,
                             save_folder=os.path.join("../ModelSave/SequentialOptimizationNaN", timestamp, method_key), retrain=train_class))

    # Linear discriminant analysis | LDA
    if classifier_type == 'all' or classifier_type == 'LDA':
        accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda, g_mean_lda = (
            classify_lda(x_train, x_test, y_train, y_test, random_state,
                         save_folder=os.path.join("../ModelSave/SequentialOptimizationNaN", timestamp, method_key), retrain=train_class))

    # K Nearest Neighbors HPO | KNN_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'KNN_hpo':
        accuracy_knn_hpo, precision_knn_hpo, recall_knn_hpo, f1_knn_hpo, specificity_knn_hpo, g_mean_knn_hpo = (
            classify_knn_hpo(x_train, x_test, y_train, y_test, random_state,
                             save_folder=os.path.join("../ModelSave/SequentialOptimizationNaN", timestamp, method_key), retrain=train_class))

    # K Nearest Neighbors | KNN
    if classifier_type == 'all' or classifier_type == 'KNN':
        accuracy_knn, precision_knn, recall_knn, f1_knn, specificity_knn, g_mean_knn = (
            classify_knn(x_train, x_test, y_train, y_test, random_state,
                         save_folder=os.path.join("../ModelSave/SequentialOptimizationNaN", timestamp, method_key), retrain=train_class))

    # Support Vector Machine HPO | SVM_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'SVM_hpo':
        accuracy_svm_hpo, precision_svm_hpo, recall_svm_hpo, f1_svm_hpo, specificity_svm_hpo, g_mean_svm_hpo = (
            classify_svm_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                             save_folder=os.path.join("../ModelSave/SequentialOptimizationNaN", timestamp, method_key), retrain=train_class))

    # Support Vector Machine | SVM
    if classifier_type == 'all' or classifier_type == 'SVM':
        accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm, g_mean_svm = (
            classify_svm(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                         save_folder=os.path.join("../ModelSave/SequentialOptimizationNaN", timestamp, method_key), retrain=train_class))

    # Ensemble with Gradient Boosting HPO | EGB_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'EGB_hpo':
        accuracy_gb_hpo, precision_gb_hpo, recall_gb_hpo, f1_gb_hpo, specificity_gb_hpo, g_mean_gb_hpo = (
            classify_ensemble_with_gradboost_hpo(x_train, x_test, y_train, y_test, random_state,
                                                 save_folder=os.path.join("../ModelSave/SequentialOptimizationNaN", timestamp, method_key), retrain=train_class))

    # Ensemble with Gradient Boosting | EGB
    if classifier_type == 'all' or classifier_type == 'EGB':
        accuracy_gb, precision_gb, recall_gb, f1_gb, specificity_gb, g_mean_gb = (
            classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test, random_state,
                                             save_folder=os.path.join("../ModelSave/SequentialOptimizationNaN", timestamp, method_key), retrain=train_class))

    # Build Performance Metric Summary Tables
    if classifier_type == 'all':
        performance_metric_summary = (summarize_performance_metrics(accuracy_logreg, accuracy_rf, accuracy_lda,
                                                                    accuracy_knn, accuracy_svm, accuracy_gb,
                                                                    precision_logreg, precision_rf, precision_lda,
                                                                    precision_knn, precision_svm, precision_gb,
                                                                    recall_logreg, recall_rf, recall_lda, recall_knn,
                                                                    recall_svm, recall_gb, f1_logreg, f1_rf, f1_lda,
                                                                    f1_knn, f1_svm, f1_gb,specificity_logreg,
                                                                    specificity_rf, specificity_lda, specificity_knn,
                                                                    specificity_svm, specificity_gb, g_mean_logreg,
                                                                    g_mean_rf, g_mean_lda, g_mean_knn,
                                                                    g_mean_svm, g_mean_gb))

    if classifier_type == 'all_hpo':
        performance_metric_summary_hpo = (summarize_performance_metrics(accuracy_logreg_hpo, accuracy_rf_hpo, accuracy_lda_hpo,
                                                                    accuracy_knn_hpo, accuracy_svm_hpo, accuracy_gb_hpo,
                                                                    precision_logreg_hpo, precision_rf_hpo, precision_lda_hpo,
                                                                    precision_knn_hpo, precision_svm_hpo, precision_gb_hpo,
                                                                    recall_logreg_hpo, recall_rf_hpo, recall_lda_hpo, recall_knn_hpo,
                                                                    recall_svm_hpo, recall_gb_hpo, f1_logreg_hpo, f1_rf_hpo, f1_lda_hpo,
                                                                    f1_knn_hpo, f1_svm_hpo, f1_gb_hpo,specificity_logreg_hpo,
                                                                    specificity_rf_hpo, specificity_lda_hpo, specificity_knn_hpo,
                                                                    specificity_svm_hpo, specificity_gb_hpo, g_mean_logreg_hpo,
                                                                    g_mean_rf_hpo, g_mean_lda_hpo, g_mean_knn_hpo,
                                                                    g_mean_svm_hpo, g_mean_gb_hpo))



    duration = time.time() - start_time
    print(duration)

    if classifier_type == 'all':
        return performance_metric_summary
    if classifier_type == 'all_hpo':
        return performance_metric_summary_hpo

if __name__ == "__main__":

    # Get time stamp for saving models
    timestamp_outerloop = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Imputation Methods to Test
    impute_type_all = [1]

    # N-Neighbors Definition for Imputation Methods 1 & 3
    n_neighbors_all = [10]

    # Specify Model Type
    model_type_outerloop = ['noAFE', 'explicit']

    # Pre-Allocate Performance Summary Dictionary
    imputation_performance_summary = dict()

    # Loop through Imputation Methods
    for i in range(len(impute_type_all)):
        # Loop through all n-neighbors for Imputation Methods 1 & 3
        if impute_type_all[i] == 1 or impute_type_all[i] == 3:
            for j in range(len(n_neighbors_all)):
                # Define Key for each Element in Dictionary
                if impute_type_all[i] == 1:
                    method_key_impute = 'knn_raw_n_neighbors_' + str(n_neighbors_all[j])
                    imputation_performance_summary[method_key_impute] = main_loop(impute_type_all[i], n_neighbors_all[j], timestamp_outerloop, model_type_outerloop, method_key_impute)
                elif impute_type_all[i] == 3:
                    method_key_impute = 'knn_features_n_neighbors_' + str(n_neighbors_all[j])
                    imputation_performance_summary[method_key_impute] = main_loop(impute_type_all[i], n_neighbors_all[j], timestamp_outerloop, model_type_outerloop, method_key_impute)

        elif impute_type_all[i] == 2:
            method_key_impute = 'listwise_deletion'
            imputation_performance_summary[method_key_impute] = main_loop(impute_type_all[i], 0, timestamp_outerloop, model_type_outerloop, method_key_impute)

    # Save pkl summary
    save_folder = os.path.join("../PerformanceSave/SequentialOptimizationNaN", timestamp_outerloop)
    save_file = 'Sequential_Optimization_NaN_dictionary_model_type_' + str(model_type_outerloop[1]) + '.pkl'
    save_path = os.path.join(save_folder, save_file)

    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(save_path, 'wb') as file:
        pickle.dump(imputation_performance_summary, file)