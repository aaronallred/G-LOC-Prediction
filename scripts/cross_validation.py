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
from sklearn.preprocessing import StandardScaler

def main_loop(kfold_ID, num_splits, runname):
    start_time = time.time()

    ################################################### USER INPUTS  ###################################################
    ## Data Folder Location
    # datafolder = '../../'
    datafolder = '../data/'

    # Random State | 42 - Debug mode
    random_state = 42

    ## Classifier | Pick 'logreg' 'rf' 'LDA' 'KNN' 'SVM' 'EGB' or 'all'
    classifier_type = 'LSTM'
    train_class = True
    class_weight_imb = 'balanced'

    # Data Handling Options
    remove_NaN_trials = True
    impute_type = 2
    n_neighbors = 3

    ## Model Parameters
    model_type = ['noAFE', 'implicit']
    if 'noAFE' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                 'rawEEG', 'processedEEG', 'strain', 'demographics']
    if 'noAFE' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking','rawEEG', 'processedEEG']
        feature_groups_to_analyze = ['ECG']

    # baseline_methods_to_use = ['v0','v1','v2','v3','v4','v5','v6','v7','v8']
    baseline_methods_to_use = ['v0','v1','v2','v5','v6','v7','v8']
    baseline_methods_to_use = ['v0']

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
    feature_matrix_name = 'x_feature_matrix_imputation_method' + str(impute_type) +'.pkl'
    y_label_name = 'y_gloc_labels_imputation_method' + str(impute_type) +'.pkl'
    all_features_name = 'all_features_imputation_method' + str(impute_type) +'.pkl'

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
    elif impute_type == 1:
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
        Generates unengineered features from baseline data using same naming convention as traditional models
    """

    # Unpack without feature generation
    x_feature_matrix = np.vstack([
        combined_baseline[trial_id] for trial_id in combined_baseline
    ]).astype(np.float32)

    y_gloc_labels = gloc

    all_features = combined_baseline_names

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
    elif impute_type == 3:
        y_gloc_labels_noNaN = y_gloc_labels
        x_feature_matrix_noNaN, indicator_matrix = knn_impute(x_feature_matrix, n_neighbors)
    else:
        y_gloc_labels_noNaN, x_feature_matrix_noNaN = y_gloc_labels, x_feature_matrix


    # Save pkl
    with open (y_label_name, 'wb') as file:
        pickle.dump(y_gloc_labels_noNaN, file)

    with open (feature_matrix_name, 'wb') as file:
        pickle.dump(x_feature_matrix_noNaN, file)

    with open (all_features_name, 'wb') as file:
        pickle.dump(all_features, file)

    ################################################ TRAIN/TEST SPLIT  ################################################
    """ 
          Split data into training/test for optimization loop of sequential optimization framework.
    """

    # Training/Test Split
    x_train, x_test, y_train, y_test = stratified_kfold_split(y_gloc_labels_noNaN,x_feature_matrix_noNaN,
                                                              num_splits, kfold_ID)

    # And standardize based on training data
    scaler  = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test  = scaler.transform(x_test)

    ################################################ Feature Selection ################################################

    # x_train, x_test, selected_features =  feature_selection_lasso(x_train, x_test, y_train, all_features, random_state)

    ################################################ Class Imbalance ################################################

    # x_train, y_train =  simple_smote(x_train, y_train, random_state)

    ################################################ MACHINE LEARNING ################################################
    save_folder = os.path.join("../ModelSave/CV", runname, str(kfold_ID))

    # Long Short Term Memory RNN
    if classifier_type == 'LSTM' or classifier_type == 'all':
        accuracy, precision, recall, f1, specificity, g_mean = (
            lstm_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                                             save_folder=save_folder))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy, precision, recall, f1, specificity, g_mean,['LSTM'])


    duration = time.time() - start_time
    print(duration)

    if classifier_type == 'all':
        return performance_metric_summary
    else:
        return performance_metric_summary_single

if __name__ == "__main__":

    # Get time stamp for saving models
    # runname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runname = 'ImplicitV0-8HPO_noNAN_LSTM'

    # Test set identifier for 10-fold Model Validation
    num_splits = 10
    kfold_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Pre-Allocate Performance Summary Dictionary
    kfold_performance_summary = dict()

    # Loop through Imputation Methods
    for i in range(len(kfold_ID)):
        # Loop through all train-test splits
        method_key = str(kfold_ID[i])
        kfold_performance_summary[method_key] = main_loop(kfold_ID[i], num_splits, runname)

        # Save pkl summary for this iteration
        save_folder = os.path.join("../PerformanceSave/CrossValidation", runname, method_key)
        save_file = 'FoldSummary.pkl'
        save_path = os.path.join(save_folder, save_file)

        # Ensure the save folder exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        with open(save_path, 'wb') as file:
            pickle.dump(kfold_performance_summary, file)

    # Save pkl summary
    save_folder = os.path.join("../PerformanceSave/CrossValidation", runname)
    save_file = 'CrossValidation.pkl'
    save_path = os.path.join(save_folder, save_file)

    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(save_path, 'wb') as file:
        pickle.dump(kfold_performance_summary, file)

    plot_cross_val(kfold_performance_summary)