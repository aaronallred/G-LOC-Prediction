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
import pandas as pd
from numpy import number
from openpyxl.styles.builtins import percent
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from LogRegTS_supporting import lrts_binary_class
from LSTM_supporting import lstm_binary_class
from Transformer_supporting import transformer_class
from TCN_supporting import tcn_binary_class

def main_loop(kfold_ID, num_splits, runname):
    start_time = time.time()

    ################################################### USER INPUTS  ###################################################
    ## Data Folder Location
    # datafolder = '../../'
    datafolder = '../data/'

    # Random State | 42 - Debug mode
    random_state = 42

    ## Classifier | Pick 'LogRegTS', 'LSTM', 'TCN', 'Trans', or 'all'
    classifier_type = 'all'
    train_class = True
    class_weight_imb = 'balanced'

    # Data Handling Options
    remove_NaN_trials = True
    impute_type = 1
    n_neighbors = 3

    ## Model Parameters
    model_type = ['noAFE', 'implicit']
    if 'noAFE' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                 'rawEEG', 'strain', 'demographics']

    if 'noAFE' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG','BR','temp', 'eyetracking','rawEEG']

    # baseline_methods_to_use = ['v0','v1','v2','v3','v4','v5','v6','v7','v8']
    baseline_methods_to_use = ['v0','v1','v2','v5','v6','v7','v8']
    baseline_methods_to_use = ['v0','v1','v2','v5','v6']

    baseline_window = 32.5  # seconds

    analysis_type = 2
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


    ################################################## REDUCE MEMORY ##################################################

    # Grab columns from gloc_data_reduced and remove gloc_data_reduced variable from memory
    trial_column = gloc_data_reduced['trial_id']
    trial_ints = convert_to_unique_ordered_integers(trial_column)
    time_column = gloc_data_reduced['Time (s)']
    event_validated_column = gloc_data_reduced['event_validated']
    subject_column = gloc_data_reduced['subject']

    del gloc_data_reduced


    ################################################## Impute Missing ##################################################
    """
        Imputes data using train / test split within imputation to prevent data leakage
    """
    ### Impute missing row data
    if impute_type == 1:
        # Grab train and test indices
        _, _, _, _, train_ind, test_ind = groupedtrial_kfold_split(gloc, features,
                                                                   trial_ints,
                                                                   num_splits, kfold_ID)

        # Impute the missing data
        features = faster_knn_impute_train_test(features, train_ind, test_ind, n_neighbors)

        # Calculate new subfeature arrays
        phys_indices = [i for i, feature in enumerate(all_features) if (feature in all_features_phys)]
        features_phys = features[:,phys_indices]

        ecg_indices = [i for i, feature in enumerate(all_features) if (feature in all_features_ecg)]
        features_ecg = features[:,ecg_indices]

        eeg_indices = [i for i, feature in enumerate(all_features) if (feature in all_features_eeg)]
        features_eeg = features[:,eeg_indices]



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
    x_feature_matrix = np.vstack([combined_baseline[trial_id] for trial_id in combined_baseline]).astype(np.float32)

    # Only grab unengineered datastreams
    unengineered_streams = pull_unengineered_streams()

    # Grab indices corresponding to unengineered features in unengineered streams (but also with baseline suffix id)
    ue_indices = [
        i for i, feature in enumerate(combined_baseline_names)
        if (
                feature in unengineered_streams
                or any(
            f"{stream}_{suffix}" == feature for stream in unengineered_streams for suffix in baseline_methods_to_use)
        )
    ]

    # Get new x_feature matrix
    x_feature_matrix = x_feature_matrix[:,ue_indices]
    trial_ints = convert_to_unique_ordered_integers(trial_column)

    x_feature_matrix = np.hstack([x_feature_matrix,trial_ints])
    y_gloc_labels = gloc

    all_features = combined_baseline_names
    all_features = [all_features[i] for i in ue_indices]


    ############################################# FEATURE CLEAN AND PREP ##############################################
    """ 
          Optional handling of raw NaN data
    """

    # Remove constant columns (typically no constant columns)
    x_feature_matrix, all_features = remove_constant_columns(x_feature_matrix, all_features)

    # List-wise deletion or clean any residual NaNs
    if impute_type == 2 or impute_type == 1:
        # Remove rows with NaN (temporary solution-should replace with other method eventually)
        y_gloc_labels_noNaN, x_feature_matrix_noNaN, all_features, trials_noNaN = process_NaN(y_gloc_labels, x_feature_matrix,
                                                                                all_features, trial_ints)
    else:
        y_gloc_labels_noNaN, x_feature_matrix_noNaN, trials_noNaN = y_gloc_labels, x_feature_matrix, trial_ints



    ################################################ TRAIN/TEST SPLIT  ################################################
    """ 
          Split data into training/test for optimization loop of sequential optimization framework.
    """

    # Training/Test Split
    x_train, x_test, y_train, y_test,_ , _ = groupedtrial_kfold_split(
        y_gloc_labels_noNaN,x_feature_matrix_noNaN, trials_noNaN, num_splits, kfold_ID)

    # Grab trials as separate
    x_train_trials = x_train[:,-1].reshape(-1, 1)
    x_train = x_train[:,:-1]
    x_test_trials = x_test[:, -1].reshape(-1, 1)
    x_test = x_test[:, :-1]

    # And standardize based on training data
    scaler  = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test  = scaler.transform(x_test)

    # Add indices back as final column
    x_train = np.hstack([x_train,x_train_trials])
    x_test = np.hstack([x_test, x_test_trials])


    ################################################ MACHINE LEARNING ################################################
    save_folder = os.path.join("../ModelSave/CV", runname, str(kfold_ID))
    #performance_metric_summary_single = []
    summaries = []

    # Time Series (Autoregressive Time Aware) Logistic Regression
    if classifier_type == 'LogRegTS' or classifier_type == 'all':
        accuracy, precision, recall, f1, specificity, g_mean = (
            lrts_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                              save_folder=save_folder))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy, precision, recall, f1, specificity, g_mean, ['LSTM'])
        summaries.append(performance_metric_summary_single)

    # Long Short Term Memory RNN
    if classifier_type == 'LSTM' or classifier_type == 'all':
        accuracy, precision, recall, f1, specificity, g_mean = (
            lstm_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                                             save_folder=save_folder))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy, precision, recall, f1, specificity, g_mean,['LSTM'])
        summaries.append(performance_metric_summary_single)

    # Transformer
    if classifier_type == 'Trans' or classifier_type == 'all':
        accuracy, precision, recall, f1, specificity, g_mean = (
            transformer_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                              save_folder=save_folder))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy, precision, recall, f1, specificity, g_mean, ['Trans'])
        summaries.append(performance_metric_summary_single)

    # Temporal Convolutional Network
    if classifier_type == 'TCN' or classifier_type == 'all':
        accuracy, precision, recall, f1, specificity, g_mean = (
            tcn_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                              save_folder=save_folder))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy, precision, recall, f1, specificity, g_mean, ['TCN'])
        summaries.append(performance_metric_summary_single)


    duration = time.time() - start_time
    print(duration)

    if classifier_type == 'all':
        performance_metric_summary = pd.concat(summaries)
        return performance_metric_summary
    else:
        return performance_metric_summary_single

if __name__ == "__main__":

    # Needed for proper debugging of CUDA errors
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    runname = 'ImplicitV0V3_alltest'

    # Test set identifier for 10-fold Model Validation
    num_splits = 10
    kfold_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    kfold_ID = [0, 1]

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

    plot_cross_val_sp(kfold_performance_summary)