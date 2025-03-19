from GLOC_data_processing import *
from imputation import *
from baseline_methods import *
from features import *
from feature_selection import *
from GLOC_classifier import *
from GLOC_visualization import *
from imbalance_techniques import *

if __name__ == "__main__":

    ################################################### USER INPUTS  ###################################################
    ## Data Folder Location
    datafolder = '../../'
    # datafolder = '../data/'

    ## Classifier | Pick 'logreg' 'rf' 'LDA' 'KNN' 'SVM' 'EGB' or 'all'
    classifier_type = 'all'
    train_class = True

    ## Sequential Optimization Mode | Pick 'none' 'imbalance' 'nan' 'sliding_window' or 'feature_reduction'
    sequential_optimization_mode = 'none'

    ## Imbalance Technique | Pick 'rus' 'ros' 'smote' 'cost_function' 'rus_cf' 'ros_cf' 'smote_cf' 'none' or 'all'
    imbalance_technique = 'all'

    ## Feature Reduction | Pick 'lasso' 'enet' 'ridge' 'mrmr' 'pca' 'target_mean' 'performance' 'shuffle' or 'all'
    feature_reduction_type = 'all'

    # Data Handling Options
    remove_NaN_trials = True
    impute_type = 2

    ## Model Parameters
    model_type = ['noAFE', 'explicit']
    feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                 'rawEEG', 'processedEEG', 'strain', 'demographics']
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

    #### DESCRIPTIONS OF ABOVE PARAMETERS
    ## datafolder: Location of structured data files

    ## classifier_type: Type of Classifier
        # options are 'logreg' 'rf' 'LDA' 'KNN' 'SVM' 'EGB'
        # select 'all' to run all of the above

    ## train_class: Retrain or load pre-existing weights

    ## model_type: Type of Model with Two parameters to specify:
        # either 'AFE' or 'noAFE'
        # either 'explicit' or 'implicit'
            # implicit: does NOT contain direct features for g, strain, demographics
            # explicit: DOES contain direct features for g, strain, demographics

    ## feature_groups_to_analyze: Feature Groups to Pull from GLOC Data File
        # Example with all feature groups:
        # feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G', 'cognitive',
                                     # 'rawEEG', 'processedEEG', 'strain', 'demographics']
        # NOTES:
        # Update from GLOC tagup 01/15/25: Chris said the fNIRS data should not be trusted and should not be used.
            # The light from the eye tracking glasses washed out this data.
        # The cognitive data only exists for some period before and after ROR.
            # removing cognitive features rather than imputing


    ##  baseline_methods_to_use: Baseline Methods to Use on Feature Set
        #  options:
            # V0: no baseline
            # V1: divide by baseline window
            # V2: subtract baseline window
            # V3: pre ROR: divide by baseline window pre GOR, ROR: divide by baseline window pre ROR
            # V4: pre ROR: subtract baseline window pre GOR, ROR: subtract baseline window pre ROR
            # V5: divide by seated resting HR
            # V6: subtract seated resting HR
            # V7: divide by resting EEG
            # V8: subtract resting EEG
            # Example with all feature groups:
            # baseline_methods_to_use = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8']
            # NOTES:
            # ALWAYS use v0- it is needed for several additional features that get computed
            # baseline_methods_to_use = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8']
            # baseline_methods_to_use = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

    ## remove_NaN_trials: Remove Trials that are missing a chosen data stream that has all NaN during the trial

    ## impute_type: Type of Imputation to perform
        # 0: Remove raw NaN rows | 1: KNN impute raw data | 2: remove feature NaN rows | 3: KNN impute features

    ## training_ratio: ML Splits (Training/Test Split, specify proportion of training data 0-1)

    ## baseline_window, window_size, stride,offset, time_start: (all in seconds) Sliding Window Parameters

    ## analysis_type: Flags which data should be analyzed
        # analysis_type = 0: analyze one trial from a subject
            # if analysis_type = 0, then set subject_to_analyze and trial_to_analyze parameters below
        # analysis_type = 1: analyze subject data (all trials for a subject)
            # if analysis_type = 1, then set subject_to_analyze parameter below
        # analysis_type = 2: analyze cohort data (all subjects, all trials)
            # if analysis_type = 2, then no extra parameters need to be set


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

    # Find time window after acceleration before GLOC (to compare our data to LOCINDTI)
    # find_prediction_window(gloc_data_reduced, gloc, time_variable)


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
    if impute_type == 0:
        # Remove rows with NaN (temporary solution-should replace with other method eventually)
        gloc, features, gloc_data_reduced = process_NaN_raw(gloc, features, gloc_data_reduced)
    elif impute_type == 1:
        features, indicator_matrix = knn_impute(features, n_neighbors=5)

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

    ############################################# FEATURE CLEAN AND PREP ##############################################
    """ 
          Optional handling of raw NaN data, depending on 'impute_type' >= 2
    """

    # Remove constant columns
    x_feature_matrix, all_features = remove_constant_columns(x_feature_matrix, all_features)

    if impute_type == 2:
        # Remove rows with NaN (temporary solution-should replace with other method eventually)
        y_gloc_labels_noNaN, x_feature_matrix_noNaN, all_features = process_NaN(y_gloc_labels, x_feature_matrix,
                                                                                all_features)
    elif impute_type == 3:
        y_gloc_labels_noNaN = y_gloc_labels
        x_feature_matrix_noNaN, indicator_matrix = knn_impute(x_feature_matrix, n_neighbors=5)
    else:
        y_gloc_labels_noNaN, x_feature_matrix_noNaN = y_gloc_labels, x_feature_matrix

    ################################################ TRAIN/TEST SPLIT  ################################################
    """ 
          Split data into training/test for optimization loop of sequential optimization framework.
    """

    # Training/Test Split
    x_train, x_test, y_train, y_test = pre_classification_training_test_split(y_gloc_labels_noNaN,
                                                                              x_feature_matrix_noNaN,training_ratio)

    x = 1

    ################################################  CLASS IMBALANCE  ################################################
    """ 
          Explore Class Imbalance Section of Sequential Optimization Framework
    """

    if sequential_optimization_mode == 'imbalance':
        ## Imbalance Technique | Pick 'rus' 'ros' 'smote' 'cost_function' 'rus_cf' 'ros_cf' 'smote_cf' 'none' or 'all'
        if classifier_type == 'all' or classifier_type == 'rus':
            rus_x_train, rus_y_train = resample_rus(x_train, y_train)
            class_weight_imb = 'None'

            performance_metric_summary_rus = (call_all_classifiers(classifier_type, rus_x_train, x_test, rus_y_train,
                                                                   y_test, all_features, train_class, class_weight_imb))

        if classifier_type == 'all' or classifier_type == 'ros':
            ros_x_train, ros_y_train = resample_ros(x_train, y_train)
            class_weight_imb = 'None'

            performance_metric_summary_ros = (call_all_classifiers(classifier_type, ros_x_train, x_test, ros_y_train,
                                                                   y_test, all_features, train_class, class_weight_imb))

        if classifier_type == 'all' or classifier_type == 'smote':
            smote_x_train, smote_y_train = resample_smote(x_train, y_train)
            class_weight_imb = 'None'

            performance_metric_summary_smote = (call_all_classifiers(classifier_type, smote_x_train, x_test, smote_y_train,
                                                                   y_test, all_features, train_class, class_weight_imb))

        if classifier_type == 'all' or classifier_type == 'cost_function':
            class_weight_imb = 'balanced'
            performance_metric_summary_cf = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                                                   y_test, all_features, train_class, class_weight_imb))

        if classifier_type == 'all' or classifier_type == 'rus_cf':
            class_weight_imb = 'balanced'
            if rus_x_train not in globals():
                rus_x_train, rus_y_train = resample_rus(x_train, y_train)

            performance_metric_summary_rus_cf = (call_all_classifiers(classifier_type, rus_x_train, x_test, rus_y_train,
                                                                   y_test, all_features, train_class, class_weight_imb))

        if classifier_type == 'all' or classifier_type == 'ros_cf':
            class_weight_imb = 'balanced'
            if ros_x_train not in globals():
                ros_x_train, ros_y_train = resample_ros(x_train, y_train)

            performance_metric_summary_ros_cf = (call_all_classifiers(classifier_type, ros_x_train, x_test, ros_y_train,
                                                                       y_test, all_features, train_class, class_weight_imb))

        if classifier_type == 'all' or classifier_type == 'smote_cf':
            class_weight_imb = 'balanced'
            if smote_x_train not in globals():
                smote_x_train, smote_y_train = resample_smote(x_train, y_train)

            performance_metric_summary_smote_cf = (call_all_classifiers(classifier_type, smote_x_train, x_test, smote_y_train,
                                                                       y_test, all_features, train_class, class_weight_imb))

        if classifier_type == 'all' or classifier_type == 'none':
            class_weight_imb = 'None'

            performance_metric_summary_none = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                     y_test, all_features, train_class, class_weight_imb))


    ################################################ FEATURE REDUCTION ################################################
    """ 
          Explore Feature Reduction Section of Sequential Optimization Framework
    """

    if sequential_optimization_mode == 'feature_reduction':
        ## Feature Reduction | Pick 'lasso' 'enet' 'ridge' 'mrmr' 'pca' 'target_mean' 'performance' 'shuffle' or 'all'
        if feature_reduction_type == 'all' or feature_reduction_type == 'lasso':
            selected_features_lasso = feature_selection_lasso(x_train, y_train, all_features)

            # Grab relevant feature columns from x_train and x_test
            feature_index = np.where(all_features == selected_features_lasso)
            x_train = x_train[:,feature_index]
            x_test = x_test[:,feature_index]

            # Assess performance for all classifiers
            performance_metric_summary_lasso = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                                                   y_test, selected_features_lasso, train_class, class_weight_imb))

        if feature_reduction_type == 'all' or feature_reduction_type == 'enet':
            selected_features_enet = feature_selection_elastic_net(x_train, y_train, all_features)

            # Grab relevant feature columns from x_train and x_test
            feature_index = np.where(all_features == selected_features_enet)
            x_train = x_train[:,feature_index]
            x_test = x_test[:,feature_index]

            # Assess performance for all classifiers
            performance_metric_summary_lasso = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                                                   y_test, selected_features_enet, train_class, class_weight_imb))

        if feature_reduction_type == 'all' or feature_reduction_type == 'ridge':
            selected_features_ridge = feature_selection_ridge(x_train, y_train, all_features)

            # Grab relevant feature columns from x_train and x_test
            feature_index = np.where(all_features == selected_features_ridge)
            x_train = x_train[:,feature_index]
            x_test = x_test[:,feature_index]

            # Assess performance for all classifiers
            performance_metric_summary_lasso = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                                                   y_test, selected_features_ridge, train_class, class_weight_imb))

        if feature_reduction_type == 'all' or feature_reduction_type == 'mrmr':
            selected_features_mrmr = feature_selection_mrmr(x_train, y_train, all_features)

            # Grab relevant feature columns from x_train and x_test
            feature_index = np.where(all_features == selected_features_mrmr)
            x_train = x_train[:,feature_index]
            x_test = x_test[:,feature_index]

            # Assess performance for all classifiers
            performance_metric_summary_lasso = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                                                   y_test, selected_features_mrmr, train_class, class_weight_imb))

        if feature_reduction_type == 'all' or feature_reduction_type == 'pca':
            selected_features_pca = feature_selection_pca(x_train, y_train, all_features)

            # Grab relevant feature columns from x_train and x_test
            feature_index = np.where(all_features == selected_features_pca)
            x_train = x_train[:,feature_index]
            x_test = x_test[:,feature_index]

            # Assess performance for all classifiers
            performance_metric_summary_lasso = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                                                   y_test, selected_features_pca, train_class, class_weight_imb))

        if feature_reduction_type == 'all' or feature_reduction_type == 'target_mean':
            selected_features_target_mean = feature_selection_target_mean(x_train, y_train, all_features)

            # Grab relevant feature columns from x_train and x_test
            feature_index = np.where(all_features == selected_features_target_mean)
            x_train = x_train[:,feature_index]
            x_test = x_test[:,feature_index]

            # Assess performance for all classifiers
            performance_metric_summary_lasso = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                                                   y_test, selected_features_target_mean, train_class, class_weight_imb))

        if feature_reduction_type == 'all' or feature_reduction_type == 'performance':
            selected_features_performance = feature_selection_performance(x_train, y_train, all_features)

            # Grab relevant feature columns from x_train and x_test
            feature_index = np.where(all_features == selected_features_performance)
            x_train = x_train[:,feature_index]
            x_test = x_test[:,feature_index]

            # Assess performance for all classifiers
            performance_metric_summary_lasso = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                                                   y_test, selected_features_performance, train_class, class_weight_imb))

        if feature_reduction_type == 'all' or feature_reduction_type == 'shuffle':
            selected_features_shuffle = feature_selection_shuffle(x_train, y_train, all_features)

            # Grab relevant feature columns from x_train and x_test
            feature_index = np.where(all_features == selected_features_shuffle)
            x_train = x_train[:,feature_index]
            x_test = x_test[:,feature_index]

            # Assess performance for all classifiers
            performance_metric_summary_lasso = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                                                   y_test, selected_features_shuffle, train_class, class_weight_imb))


    # Feature Selection
    # selected_features_lasso = feature_selection_lasso(x_train, y_train, all_features)
    # selected_features_enet = feature_selection_elastic_net(x_train, y_train, all_features)
    # selected_features_ridge = feature_selection_ridge(x_train, y_train, all_features)
    # selected_features_mrmr = feature_selection_mrmr(x_train, y_train, all_features)
    # selected_features_pca = feature_selection_pca(x_train, y_train, all_features)
    # selected_features_target_mean = feature_selection_target_mean(x_train, y_train, all_features)
    # selected_features_performance = feature_selection_performance(x_train, y_train, all_features)
    # selected_features_shuffle = feature_selection_shuffle(x_train, y_train, all_features)


    ################################################ MACHINE LEARNING ################################################
    if sequential_optimization_mode == 'none':
        # Logistic Regression | logreg
        if classifier_type == 'all' or classifier_type == 'logreg':
            accuracy_logreg, precision_logreg, recall_logreg, f1_logreg, specificity_logreg, g_mean_logreg = (
                classify_logistic_regression(x_train, x_test, y_train, y_test, all_features,retrain=train_class))

        # Random Forrest | rf
        if classifier_type == 'all' or classifier_type == 'rf':
            accuracy_rf, precision_rf, recall_rf, f1_rf, tree_depth, specificity_rf, g_mean_rf = (
                classify_random_forest(x_train, x_test, y_train, y_test, all_features,retrain=train_class))

        # Linear discriminant analysis | LDA
        if classifier_type == 'all' or classifier_type == 'LDA':
            accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda, g_mean_lda = (
                classify_lda(x_train, x_test, y_train, y_test, all_features,retrain=train_class))

        # KNN
        if classifier_type == 'all' or classifier_type == 'KNN':
            accuracy_knn, precision_knn, recall_knn, f1_knn, specificity_knn, g_mean_knn = (
                classify_knn(x_train, x_test, y_train, y_test,retrain=train_class))

        # SVM
        if classifier_type == 'all' or classifier_type == 'SVM':
            accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm, g_mean_svm = (
                classify_svm(x_train, x_test, y_train, y_test,retrain=train_class))

        # Ensemble with Gradient Boosting
        if classifier_type == 'all' or classifier_type == 'EGB':
            accuracy_gb, precision_gb, recall_gb, f1_gb, specificity_gb, g_mean_gb = (
                classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test,retrain=train_class))

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


    # Breakpoint for troubleshooting
    x = 1