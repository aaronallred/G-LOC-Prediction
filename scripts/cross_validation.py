import json
from imputation import *
from baseline_methods import *
from features import *
from feature_selection import *
from GLOC_classifier import *
from GLOC_visualization import *
from imbalance_techniques import *
#from GLOC_data_processing import pull_unengineered_streams
#from GLOC_data_processing import save_metrics_to_csv
#from GLOC_data_processing import process_NaN, process_NaN_raw
import pickle
import time
import pandas as pd
from numpy import number
from openpyxl.styles.builtins import percent
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from LogRegTS_supporting import lrts_binary_class
from NAM_supporting import nam_binary_class
from LSTM_supporting import lstm_binary_class
from Transformer_supporting import transformer_class
from TCN_supporting import tcn_binary_class
from GAM_supporting import gam_binary_class
import json
cfg_path = "config.json"
with open(cfg_path, "r") as f:
    cfg = json.load(f)
datafolder          = cfg["datafolder"]
learning_mode       = cfg["learning_mode"]
engineering_mode    = cfg["engineering_mode"]
window_size         = cfg["window_size"]
stride              = cfg["stride"]
offset              = cfg["offset"]
remove_NaN_trials   = cfg["remove_NaN_trials"]
impute_type         = cfg["impute_type"]
n_neighbors         = cfg["n_neighbors"]
save_impute         = cfg["save_impute"]
load_impute         = cfg["load_impute"]
classifier_type     = cfg["classifier_type"]
train_class         = cfg["train_class"]
class_weight_imb    = cfg["class_weight_imb"]
model_type          = cfg["model_type"]
def main_loop(kfold_ID, num_splits, runname):
    start_time = time.time()
    save_folder = os.path.join("../ModelSave/CV", runname, str(kfold_ID))
    os.makedirs(save_folder, exist_ok=True)
    ################################################### USER INPUTS  ###################################################
    ## Data Folder Location
    # datafolder = '../../'
    #datafolder = '../RunningGLOC/'

    # Random State | 42 - Debug mode
    random_state = 42

    # NOTE: testing | DL = 1, ML =0
    # learning_mode = 1

    # Engineering Mode | 0 = Engineered streams are being used, 1 = Unengineered streams are being used
    # engineering_mode = 0

    ## Deep Learning Classifiers | Pick 'LogRegTS', 'LSTM', 'TCN', 'Trans', or 'all_dl'
    ## Machine Learning Classifiers | Pick 'logreg' 'rf' 'LDA' 'KNN' 'SVM' 'EGB', 'all_ml' or 'all_hpo'
    # classifier_type = 'TCN'

    train_class = True # not yet set up to test and not train (always trains)
    # class_weight_imb= None
    #  class_weight_imb = 'balanced'

    # Data Handling Options
    # remove_NaN_trials = True
    # impute_type = 1
    # n_neighbors = 4

    # save_impute = False   # save post impute?
    # load_impute = False  # skip impute and load from file?

    ## Model Parameters
    # model_type = ['noAFE', 'implicit']
    if 'noAFE' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                     'rawEEG', 'demographics']
        # For processed explicit
        #feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
        #                             'rawEEG', 'processedEEG', 'demographics', 'strain']

    if 'noAFE' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG','BR','temp', 'eyetracking','rawEEG']

    # baseline_methods_to_use = ['v0','v1','v2','v3','v4','v5','v6','v7','v8']
    # baseline_methods_to_use = ['v0','v1','v2','v5','v6','v7','v8']
    baseline_methods_to_use = ['v0','v1','v2','v5','v6','v7','v8']

    # baseline_window = 10  # seconds
    baseline_window = 32.5
    analysis_type = 0
    # Subject & Trial Information (only need to adjust this if doing analysis type 0,1)
    subject_to_analyze = '01'
    trial_to_analyze = '02'

    if learning_mode==1:
        window_size = 10  # seconds
        stride = 1  # seconds
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
    if learning_mode == 0 and impute_type == 1:
        features, indicator_matrix = knn_impute(features, n_neighbors)

    ################################################## REDUCE MEMORY ##################################################

    # Grab columns from gloc_data_reduced and remove gloc_data_reduced variable from memory; Grabs relevant trials
    trial_column = gloc_data_reduced['trial_id']

    trial_ints= convert_to_unique_ordered_integers(trial_column)
    time_column = gloc_data_reduced['Time (s)']
    event_validated_column = gloc_data_reduced['event_validated']
    subject_column = gloc_data_reduced['subject']

    #del gloc_data_reduced

    ################################################## DL-IMPUTE #################################################
    """
        Imputes data using train / test split within imputation to prevent data leakage
    """
    if learning_mode == 1 and impute_type == 1:
        impute_path = os.path.join(save_folder, "imputed_data.pkl")
        _, _, _, _, train_ind, test_ind = groupedtrial_kfold_split(
            gloc, features, trial_ints, num_splits, kfold_ID
        )
        if load_impute and os.path.exists(impute_path):
            with open(impute_path,'rb') as f:
                features = pickle.load(f)
        else:
            features = faster_knn_impute_train_test(
                features, train_ind, test_ind, n_neighbors
            )
            if save_impute:
                with open(impute_path,'wb') as f:
                    pickle.dump(features, f)
        # rebuild sub-blocks
        phys_idx = [i for i,f in enumerate(all_features) if f in all_features_phys]
        ecg_idx  = [i for i,f in enumerate(all_features) if f in all_features_ecg]
        eeg_idx  = [i for i,f in enumerate(all_features) if f in all_features_eeg]
        features_phys = features[:, phys_idx]
        features_ecg  = features[:, ecg_idx]
        features_eeg  = features[:, eeg_idx]


    ################################################## BASELINE DATA ##################################################
    """ 
        Baselines pre-feature data based on 'baseline_methods_to_use'
    """
    combined_baseline, combined_baseline_names, baseline_v0, baseline_names_v0 = (
        baseline_data(
            baseline_methods_to_use,
            trial_column, time_column, event_validated_column, subject_column,
            features, all_features, gloc, baseline_window,
            features_phys, all_features_phys, features_ecg, all_features_ecg, features_eeg, all_features_eeg, baseline_data_filename, list_of_baseline_eeg_processed_files, model_type
        )
    )

    ################################################ GENERATE FEATURES ################################################

    """
        Generates unengineered features from baseline data
    """

    if engineering_mode==1:
        y_gloc_labels, x_feature_matrix, all_features = feature_generation(
            0, offset, stride, window_size,
            combined_baseline, gloc, trial_column, time_column,
            combined_baseline_names, baseline_names_v0, baseline_v0,
            feature_groups_to_analyze
        )
    else:
        drop_cols = {'gloc_label', 'trial_id', 'Time (s)'}
        raw_cols  = [c for c in gloc_data_reduced.columns if c not in drop_cols]
        raw_df = gloc_data_reduced[raw_cols].apply(
            lambda col: pd.to_numeric(col, errors='coerce')
        )

        x_feature_matrix = raw_df.values.astype(np.float32)
        gloc = label_gloc_events(gloc_data_reduced)
        gloc_data_reduced.loc[:, 'gloc_label'] = gloc
        y_gloc_labels    = gloc_data_reduced['gloc_label'].values.astype(np.float32)

        # 4) feature names are the numeric columns we kept
        all_features     = raw_df.columns.tolist()

    ############################################# FEATURE CLEAN AND PREP ##############################################
    """ 
          Optional handling of raw NaN data
    """
    x_feature_matrix, all_features = remove_constant_columns(
        x_feature_matrix, all_features
    )


    if impute_type in (1, 2):
        if engineering_mode == 0:
            # raw → x_feature_matrix and trial_ints align
            y_gloc_labels_noNaN, x_feature_matrix_noNaN, all_features, trials_noNaN = \
                process_NaN(y_gloc_labels, x_feature_matrix, all_features, trial_ints)
        else:
            y_gloc_labels_noNaN, x_feature_matrix_noNaN, all_features = \
                process_NaN(y_gloc_labels, x_feature_matrix, all_features)
            trials_noNaN = None
    else:
        y_gloc_labels_noNaN, x_feature_matrix_noNaN, trials_noNaN = \
            y_gloc_labels, x_feature_matrix, trial_ints

    ################################################ TRAIN/TEST SPLIT  ################################################
    """ 
          Split data into training/test for optimization loop of sequential optimization framework.
    """
    # Keep temporal data if using Dl classifiers
    if learning_mode==1:
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
    else:
        x_train, x_test, y_train, y_test = stratified_kfold_split(y_gloc_labels_noNaN, x_feature_matrix_noNaN,
                                                                  num_splits, kfold_ID)
    ################################################ Feature Selection ################################################

    if learning_mode==0:
        x_train, x_test, selected_features =  feature_selection_lasso(x_train, x_test, y_train, all_features, random_state)

    ################################################ Class Imbalance ################################################

    if learning_mode==0:
        x_train, y_train =  simple_smote(x_train, y_train, random_state)

    ################################################ MACHINE LEARNING ################################################
    #performance_metric_summary_single = []
    summaries = []

    if classifier_type == 'all_hpo' or classifier_type == 'logreg_hpo':
        accuracy_logreg_hpo, precision_logreg_hpo, recall_logreg_hpo, f1_logreg_hpo, specificity_logreg_hpo, g_mean_logreg_hpo = (
            classify_logistic_regression_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                                             save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_logreg_hpo, precision_logreg_hpo, recall_logreg_hpo, f1_logreg_hpo,
            specificity_logreg_hpo, g_mean_logreg_hpo, ['Log Reg'])

        # Logistic Regression | logreg
    if classifier_type == 'all_ml' or classifier_type == 'logreg':
        accuracy_logreg, precision_logreg, recall_logreg, f1_logreg, specificity_logreg, g_mean_logreg = (
            classify_logistic_regression(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                                         save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_logreg, precision_logreg, recall_logreg, f1_logreg,
            specificity_logreg, g_mean_logreg, ['Log Reg'])

        # Random Forest HPO | rf_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'rf_hpo':
        accuracy_rf_hpo, precision_rf_hpo, recall_rf_hpo, f1_rf_hpo, tree_depth_hpo, specificity_rf_hpo, g_mean_rf_hpo = (
            classify_random_forest_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                                       save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_rf_hpo, precision_rf_hpo, recall_rf_hpo, f1_rf_hpo,
            specificity_rf_hpo, g_mean_rf_hpo, ['RF'])

        # Random Forrest | rf
    if classifier_type == 'all_ml' or classifier_type == 'rf':
        accuracy_rf, precision_rf, recall_rf, f1_rf, tree_depth, specificity_rf, g_mean_rf = (
            classify_random_forest(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                                   save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_rf, precision_rf, recall_rf, f1_rf, specificity_rf, g_mean_rf, ['RF'])

        # Linear discriminant analysis HPO | LDA_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'LDA_hpo':
        accuracy_lda_hpo, precision_lda_hpo, recall_lda_hpo, f1_lda_hpo, specificity_lda_hpo, g_mean_lda_hpo = (
            classify_lda_hpo(x_train, x_test, y_train, y_test, random_state,
                             save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_lda_hpo, precision_lda_hpo, recall_lda_hpo, f1_lda_hpo,
            specificity_lda_hpo, g_mean_lda_hpo, ['LDA'])

        # Linear discriminant analysis | LDA
    if classifier_type == 'all_ml' or classifier_type == 'LDA':
        accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda, g_mean_lda = (
            classify_lda(x_train, x_test, y_train, y_test, random_state,
                         save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda, g_mean_lda, ['LDA'])

        # K Nearest Neighbors HPO | KNN_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'KNN_hpo':
        accuracy_knn_hpo, precision_knn_hpo, recall_knn_hpo, f1_knn_hpo, specificity_knn_hpo, g_mean_knn_hpo = (
            classify_knn_hpo(x_train, x_test, y_train, y_test, random_state,
                             save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_knn_hpo, precision_knn_hpo, recall_knn_hpo, f1_knn_hpo,
            specificity_knn_hpo, g_mean_knn_hpo, ['KNN'])

        # K Nearest Neighbors | KNN
    if classifier_type == 'all_ml' or classifier_type == 'KNN':
        accuracy_knn, precision_knn, recall_knn, f1_knn, specificity_knn, g_mean_knn = (
            classify_knn(x_train, x_test, y_train, y_test, random_state,
                         save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_knn, precision_knn, recall_knn, f1_knn, specificity_knn, g_mean_knn, ['KNN'])

        # Support Vector Machine HPO | SVM_hpo
    if classifier_type == 'all_hpo' or classifier_type == 'SVM_hpo':
        accuracy_svm_hpo, precision_svm_hpo, recall_svm_hpo, f1_svm_hpo, specificity_svm_hpo, g_mean_svm_hpo = (
            classify_svm_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                             save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_svm_hpo, precision_svm_hpo, recall_svm_hpo, f1_svm_hpo,
            specificity_svm_hpo, g_mean_svm_hpo, ['SVM'])

        # Support Vector Machine | SVM
    if classifier_type == 'all_ml' or classifier_type == 'SVM':
        accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm, g_mean_svm = (
            classify_svm(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                         save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm, g_mean_svm, ['SVM'])

        # Ensemble with Gradient Boosting HPO | EGB_hpo
    if classifier_type == 'EGB_hpo' or classifier_type == 'all_hpo':
        accuracy_gb_hpo, precision_gb_hpo, recall_gb_hpo, f1_gb_hpo, specificity_gb_hpo, g_mean_gb_hpo = (
            classify_ensemble_with_gradboost_hpo(x_train, x_test, y_train, y_test, random_state,
                                                 save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_gb_hpo, precision_gb_hpo, recall_gb_hpo, f1_gb_hpo,
            specificity_gb_hpo, g_mean_gb_hpo, ['Ensemble w/ GB'])

        # Ensemble with Gradient Boosting | EGB
    if classifier_type == 'EGB' or classifier_type == 'all_ml':
        accuracy_gb, precision_gb, recall_gb, f1_gb, specificity_gb, g_mean_gb = (
            classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test, random_state,
                                             save_folder=save_folder, retrain=train_class))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy_gb, precision_gb, recall_gb, f1_gb, specificity_gb, g_mean_gb, ['Ensemble w/ GB'])

    if classifier_type == 'all_ml':
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

## DEEP LEARNING MODELS

    # Time Series (Autoregressive Time Aware) Logistic Regression
    if classifier_type == 'LogRegTS' or classifier_type == 'all_dl':
        accuracy, precision, recall, f1, specificity, g_mean = (
            lrts_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                              all_features, save_folder=save_folder))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy, precision, recall, f1, specificity, g_mean, ['LogRegTS'])
        save_metrics_to_csv(performance_metric_summary_single, save_folder)
        summaries.append(performance_metric_summary_single)

    # Time Series (Autoregressive Time Aware) Neural Additive Model
    if classifier_type == 'NAM' or classifier_type == 'all_dl':
        accuracy, precision, recall, f1, specificity, g_mean = (
            nam_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                              all_features, save_folder=save_folder))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy, precision, recall, f1, specificity, g_mean, ['NAM'])
        save_metrics_to_csv(performance_metric_summary_single, save_folder)
        summaries.append(performance_metric_summary_single)

    # Long Short Term Memory RNN
    if classifier_type == 'LSTM' or classifier_type == 'all_dl':
        accuracy, precision, recall, f1, specificity, g_mean = (
            lstm_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                              all_features, save_folder=save_folder))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy, precision, recall, f1, specificity, g_mean,['LSTM'])
        save_metrics_to_csv(performance_metric_summary_single, save_folder)
        summaries.append(performance_metric_summary_single)

    # Transformer
    if classifier_type == 'Trans' or classifier_type == 'all_dl':
        accuracy, precision, recall, f1, specificity, g_mean = (
            transformer_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                              all_features, save_folder=save_folder))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy, precision, recall, f1, specificity, g_mean, ['Trans'])
        save_metrics_to_csv(performance_metric_summary_single, save_folder)
        summaries.append(performance_metric_summary_single)

    # Temporal Convolutional Network
    if classifier_type == 'TCN' or classifier_type == 'all_dl':
        accuracy, precision, recall, f1, specificity, g_mean = (
            tcn_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                              all_features, save_folder=save_folder))

        performance_metric_summary_single = single_classifier_performance_summary(
            accuracy, precision, recall, f1, specificity, g_mean, ['TCN'])
        save_metrics_to_csv(performance_metric_summary_single, save_folder)
        summaries.append(performance_metric_summary_single)


    duration = time.time() - start_time
    print(duration)

    if classifier_type == 'all_dl':
        performance_metric_summary = pd.concat(summaries)
        return performance_metric_summary
    else:
        return performance_metric_summary_single
    if classifier_type == 'all':
        return performance_metric_summary
    if classifier_type == 'all_hpo':
        return performance_metric_summary_hpo
    else:
        return performance_metric_summary_single

if __name__ == "__main__":

        # Needed for proper debugging of CUDA errors
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        runname = 'ExplicitV0-8HPO_SMOTE_LASSO_noNAN_all' # ModelSave

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

        # ML vs DL pipelines use slightly different printing function here

        #if learning_mode==1:
        #    plot_cross_val_sp(kfold_performance_summary)
        #else:
        #    plot_cross_val(kfold_performance_summary)