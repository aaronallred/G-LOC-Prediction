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

if __name__ == "__main__":

    start_time = time.time()

    ################################################### USER INPUTS  ###################################################
    ## Data Folder Location
    datafolder = '../../'
    # datafolder = '../data/'

    # Random State | 42 - Debug mode
    random_state = 42

    # troubleshoot mode | 0 = No, Proceed with full feature set , 1 = Yes, reduce feature set for testing/troubleshooting
    trouble_shoot_mode = 1

    # Import Feature Matrix | 0 = No, Proceed with Baseline and Feature Extraction , 1 = Yes, Use Existing Pkl
    import_feature_matrix = 1
    feature_matrix_name = 'x_feature_matrix.pkl'
    y_label_name = 'y_gloc_labels.pkl'
    all_features_name = 'all_features.pkl'

    ## Classifier | Pick 'logreg' 'rf' 'LDA' 'KNN' 'SVM' 'EGB' or 'all'
    #                    'logreg_hpo' 'rf_hpo' 'LDA_hpo' 'KNN_hpo' 'SVM_hpo' 'EGB_hpo' or 'all_hpo'
    classifier_type = 'all'
    train_class = True

    ## Sequential Optimization Mode | Pick 'none' 'imbalance' 'nan' 'sliding_window' or 'feature_reduction'
    sequential_optimization_mode = 'none'

    ## Imbalance Technique | Pick 'rus' 'ros' 'smote' 'cost_function' 'rus_cf' 'ros_cf' 'smote_cf' 'none' or 'all'
    # Note: Cost Function techniques ('cost_function' 'rus_cf' 'ros_cf' 'smote_cf') do not work for LDA, KNN, or Ens. Learner
    imbalance_technique = 'none'
    class_weight_imb = None

    ## Feature Reduction | Pick 'lasso' 'enet' 'ridge' 'mrmr' 'pca' 'target_mean' 'performance' 'shuffle' 'none' or 'all'
    # Note: 'shuffle' does not work for KNN or LDA
    feature_reduction_type = 'pca'

    # Data Handling Options
    remove_NaN_trials = True
    impute_type = 1

    ## Model Parameters
    model_type = ['noAFE', 'explicit']
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

    #### DESCRIPTIONS OF ABOVE PARAMETERS
    ## datafolder: Location of structured data files

    ## classifier_type: Type of Classifier
        # options without hyperparameter tuning: 'logreg' 'rf' 'LDA' 'KNN' 'SVM' 'EGB'
            # select 'all' to run all of the above
        # options with hyperparameter tuning: 'logreg_hpo' 'rf_hpo' 'LDA_hpo' 'KNN_hpo' 'SVM_hpo' 'EGB_hpo'
            # select 'all_hpo' to run all of the above

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

    ## remove_NaN_trials: Remove Trials that are missing a chosen data stream that has all NaN during the trial

    ## impute_type: Type of Imputation to perform
        # 1: KNN impute raw data | 2: remove feature NaN rows | 3: KNN impute features

    ## training_ratio: ML Splits (Training/Test Split, specify proportion of training data 0-1)

    ## baseline_window, window_size, stride,offset, time_start: (all in seconds) Sliding Window Parameters

    ## analysis_type: Flags which data should be analyzed
        # analysis_type = 0: analyze one trial from a subject
            # if analysis_type = 0, then set subject_to_analyze and trial_to_analyze parameters below
        # analysis_type = 1: analyze subject data (all trials for a subject)
            # if analysis_type = 1, then set subject_to_analyze parameter below
        # analysis_type = 2: analyze cohort data (all subjects, all trials)
            # if analysis_type = 2, then no extra parameters need to be set

    if import_feature_matrix == 0:
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
            # Remove rows with NaN
            gloc, features, gloc_data_reduced = process_NaN_raw(gloc, features, gloc_data_reduced)
        if impute_type == 1:
            print('Entering impute 1')
            features = faster_knn_impute(features, k=3)

        ################################################## REDUCE MEMORY ##################################################
        print('Exiting impute 1')
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

        # if impute_type == 2 or impute_type == 1:
        #     # Remove rows with NaN (temporary solution-should replace with other method eventually)
        #     y_gloc_labels_noNaN, x_feature_matrix_noNaN, all_features = process_NaN(y_gloc_labels, x_feature_matrix,
        #                                                                             all_features)
        # elif impute_type == 3:
        #     y_gloc_labels_noNaN = y_gloc_labels
        #     x_feature_matrix_noNaN, indicator_matrix = knn_impute(x_feature_matrix, n_neighbors)
        # else:
        #     y_gloc_labels_noNaN, x_feature_matrix_noNaN = y_gloc_labels, x_feature_matrix


        # # Save pkl
        # with open (y_label_name, 'wb') as file:
        #     pickle.dump(y_gloc_labels, file)
        #
        # with open (feature_matrix_name, 'wb') as file:
        #     pickle.dump(x_feature_matrix, file)
        #
        # with open (all_features_name, 'wb') as file:
        #     pickle.dump(all_features, file)


    # Import pkl
    else:
        y_gloc_labels = pd.read_pickle(y_label_name)
        x_feature_matrix = pd.read_pickle(feature_matrix_name)
        all_features = pd.read_pickle(all_features_name)

    ################################################ TRAIN/TEST SPLIT  ################################################
    """ 
          Split data into training/test for optimization loop of sequential optimization framework.
    """

    # Remove all NaN rows from x matrix before train/test split for method 2 & method 1 if there are remaining NaNs
    if impute_type == 2 or impute_type == 1:
        y_gloc_labels, x_feature_matrix, all_features = process_NaN(y_gloc_labels, x_feature_matrix, all_features)


    # Training/Test Split
    x_train_NaN, x_test_NaN, y_train_NaN, y_test_NaN = pre_classification_training_test_split(y_gloc_labels,
                                                                              x_feature_matrix, training_ratio, random_state)

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
        x_train, indicator_matrix_train = knn_impute(x_train_NaN, n_neighbors=3)

        # Impute Test Data
        x_test, indicator_matrix_test = knn_impute(x_test_NaN, n_neighbors=3)

    else:
        # Leave train/test matrix as is
        y_train, x_train = y_train_NaN, x_train_NaN
        y_test, x_test = y_test_NaN, x_test_NaN

    # Reduce feature set to reduce run time if trouble_shoot_mode = 1
    if trouble_shoot_mode == 1:
        x_train = x_train[:, 0:100]
        x_test = x_test[:, 0:100]
        all_features = all_features[0:100]

    ################################################  CLASS IMBALANCE  ################################################
    """ 
          Explore Class Imbalance Section of Sequential Optimization Framework
    """

    if sequential_optimization_mode == 'imbalance':
        ## Imbalance Technique | Pick 'rus' 'ros' 'smote' 'cost_function' 'rus_cf' 'ros_cf' 'smote_cf' 'none' or 'all'
        # Random Under Sampling | rus
        if imbalance_technique == 'all' or imbalance_technique == 'rus':
            # Implement Imbalance Sampling Technique
            rus_x_train, rus_y_train = resample_rus(x_train, y_train, random_state)

            # Summarize Performance
            performance_metric_summary_rus = (call_all_classifiers(classifier_type, rus_x_train, x_test, rus_y_train,
                                                                   y_test, all_features, train_class, class_weight_imb,
                                                                   random_state))

        # Random Over Sampling | ros
        if imbalance_technique == 'all' or imbalance_technique == 'ros':
            # Implement Imbalance Sampling Technique
            ros_x_train, ros_y_train = resample_ros(x_train, y_train, random_state)

            # Summarize Performance
            performance_metric_summary_ros = (call_all_classifiers(classifier_type, ros_x_train, x_test, ros_y_train,
                                                                   y_test, all_features, train_class, class_weight_imb,
                                                                   random_state))

        # Synthetic Minority Over Sampling Technique | smote
        if imbalance_technique == 'all' or imbalance_technique == 'smote':
            # Implement Imbalance Sampling Technique
            smote_x_train, smote_y_train = resample_smote(x_train, y_train, random_state)

            # Summarize Performance
            performance_metric_summary_smote = (call_all_classifiers(classifier_type, smote_x_train, x_test, smote_y_train,
                                                                     y_test, all_features, train_class, class_weight_imb,
                                                                     random_state))

        # Modify Cost Function | cost_function
        if imbalance_technique == 'all' or imbalance_technique == 'cost_function':
            # Implement Imbalance Cost Function Technique
            class_weight_imb = 'balanced'

            # Summarize Performance
            performance_metric_summary_cf = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                                                  y_test, all_features, train_class, class_weight_imb,
                                                                  random_state))

        # Random Under Sampling & Modify Cost Function | rus_cf
        if imbalance_technique == 'all' or imbalance_technique == 'rus_cf':
            # Implement Imbalance Hybrid Technique
            class_weight_imb = 'balanced'
            rus_x_train, rus_y_train = resample_rus(x_train, y_train, random_state)

            # Summarize Performance
            performance_metric_summary_rus_cf = (call_all_classifiers(classifier_type, rus_x_train, x_test, rus_y_train,
                                                                      y_test, all_features, train_class, class_weight_imb,
                                                                      random_state))

        # Random Over Sampling & Modify Cost Function | rus_cf
        if imbalance_technique == 'all' or imbalance_technique == 'ros_cf':
            # Implement Imbalance Hybrid Technique
            class_weight_imb = 'balanced'
            ros_x_train, ros_y_train = resample_ros(x_train, y_train, random_state)

            # Summarize Performance
            performance_metric_summary_ros_cf = (call_all_classifiers(classifier_type, ros_x_train, x_test, ros_y_train,
                                                                      y_test, all_features, train_class, class_weight_imb,
                                                                      random_state))

        # Synthetic Minority Over Sampling Technique & Modify Cost Function | smote_cf
        if imbalance_technique == 'all' or imbalance_technique == 'smote_cf':
            # Implement Imbalance Hybrid Technique
            class_weight_imb = 'balanced'
            smote_x_train, smote_y_train = resample_smote(x_train, y_train, random_state)

            # Summarize Performance
            performance_metric_summary_smote_cf = (call_all_classifiers(classifier_type, smote_x_train, x_test, smote_y_train,
                                                                        y_test, all_features, train_class, class_weight_imb,
                                                                        random_state))

        # No imbalance technique | none
        if imbalance_technique == 'all' or imbalance_technique == 'none':

            # Summarize Performance
            performance_metric_summary_none = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                     y_test, all_features, train_class, class_weight_imb, random_state))


    ################################################ FEATURE REDUCTION ################################################
    """ 
          Explore Feature Reduction Section of Sequential Optimization Framework
    """

    if sequential_optimization_mode == 'feature_reduction':
        ## Feature Reduction | Pick 'lasso' 'enet' 'ridge' 'mrmr' 'pca' 'target_mean' 'performance' 'shuffle' 'none' or 'all'
        # Least Absolute Shrinkage and Selection Operator | lasso
        if feature_reduction_type == 'all' or feature_reduction_type == 'lasso':
            # Implement feature reduction
            x_train_lasso, x_test_lasso, selected_features_lasso = feature_selection_lasso(x_train, x_test, y_train, all_features, random_state)

            # Assess performance for all classifiers
            performance_metric_summary_lasso = (call_all_classifiers(classifier_type, x_train_lasso, x_test_lasso, y_train,
                                                                     y_test, selected_features_lasso, train_class, class_weight_imb,
                                                                     random_state))

        # Elastic Net Regression | enet
        if feature_reduction_type == 'all' or feature_reduction_type == 'enet':
            # Implement feature reduction
            x_train_enet, x_test_enet, selected_features_enet = feature_selection_elastic_net(x_train, x_test, y_train, all_features, random_state)

            # Assess performance for all classifiers
            performance_metric_summary_enet = (call_all_classifiers(classifier_type, x_train_enet, x_test_enet, y_train,
                                                                    y_test, selected_features_enet, train_class, class_weight_imb,
                                                                    random_state))

        # Ridge Regression | ridge
        if feature_reduction_type == 'all' or feature_reduction_type == 'ridge':
            # Implement feature reduction & assess performance of classifiers
            x_train_ridge, x_test_ridge, selected_features_ridge, performance_metric_summary_ridge = (
                ridge_methods(x_train, x_test, y_train, y_test, all_features, classifier_type, train_class, class_weight_imb, random_state))

        # Minimum Redundancy Maximum Relevance | mrmr
        if feature_reduction_type == 'all' or feature_reduction_type == 'mrmr':
            # Implement feature reduction & assess performance of classifiers
            x_train_mrmr, x_test_mrmr, selected_features_mrmr, performance_metric_summary_mrmr = (
                mrmr_methods(x_train, x_test, y_train, y_test, all_features, classifier_type, train_class,
                                   class_weight_imb))

        # Principle Component Analysis | pca
        if feature_reduction_type == 'all' or feature_reduction_type == 'pca':
            # Implement feature reduction
            x_train_pca, x_test_pca, selected_features_pca = dimensionality_reduction_PCA(x_train, x_test, random_state)

            # Assess performance for all classifiers
            performance_metric_summary_pca = (call_all_classifiers(classifier_type, x_train_pca, x_test_pca, y_train,
                                                                   y_test, selected_features_pca, train_class, class_weight_imb,
                                                                   random_state))

        # Select by Target Mean Performance | target_mean
        if feature_reduction_type == 'all' or feature_reduction_type == 'target_mean':
            # Implement feature reduction
            x_train_target_mean, x_test_target_mean, selected_features_target_mean = target_mean_selection(x_train, x_test, y_train, all_features,random_state)

            # Assess performance for all classifiers
            performance_metric_summary_target_mean = (call_all_classifiers(classifier_type, x_train_target_mean, x_test_target_mean, y_train,
                                                                   y_test, selected_features_target_mean, train_class, class_weight_imb, random_state))

        # Select by Single Feature Performance | performance
        if feature_reduction_type == 'all' or feature_reduction_type == 'performance':
            # Implement feature reduction & assess performance of classifiers
            x_train_performance, x_test_performance, selected_features_performance, performance_metric_summary_sfp = (
                sfp_methods(x_train, x_test, y_train, y_test, all_features, train_class, class_weight_imb, random_state))

        # Select by Shuffling | shuffle
        if feature_reduction_type == 'all' or feature_reduction_type == 'shuffle':
            # Implement feature reduction & assess performance of classifiers
            # Note: Select by shuffling does not work for KNN or LDA
            x_train_shuffle, x_test_shuffle, selected_features_shuffle, performance_metric_summary_shuffle = (
                shuffle_methods(x_train, x_test, y_train, y_test, all_features, train_class, class_weight_imb, random_state))

        # No feature reduction methods | none
        if feature_reduction_type == 'all' or feature_reduction_type == 'none':
            # Assess performance for all classifiers
            performance_metric_summary_no_feature_selection = (call_all_classifiers(classifier_type, x_train, x_test, y_train,
                                                               y_test, all_features, train_class, class_weight_imb,
                                                               random_state))

    ################################################ MACHINE LEARNING ################################################
    if sequential_optimization_mode == 'none':

        # Logistic Regression HPO | logreg_hpo
        if classifier_type == 'all_hpo' or classifier_type == 'logreg_hpo':
            accuracy_logreg_hpo, precision_logreg_hpo, recall_logreg_hpo, f1_logreg_hpo, specificity_logreg_hpo, g_mean_logreg_hpo = (
                classify_logistic_regression_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state, retrain=train_class))

        # Logistic Regression | logreg
        if classifier_type == 'all' or classifier_type == 'logreg':
            accuracy_logreg, precision_logreg, recall_logreg, f1_logreg, specificity_logreg, g_mean_logreg = (
                classify_logistic_regression(x_train, x_test, y_train, y_test, class_weight_imb,random_state, retrain=train_class))

        # Random Forest HPO | rf_hpo
        if classifier_type == 'all_hpo' or classifier_type == 'rf_hpo':
            accuracy_rf_hpo, precision_rf_hpo, recall_rf_hpo, f1_rf_hpo, tree_depth_hpo, specificity_rf_hpo, g_mean_rf_hpo  = (
                classify_random_forest_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state, retrain=train_class))

        # Random Forrest | rf
        if classifier_type == 'all' or classifier_type == 'rf':
            accuracy_rf, precision_rf, recall_rf, f1_rf, tree_depth, specificity_rf, g_mean_rf = (
                classify_random_forest(x_train, x_test, y_train, y_test, class_weight_imb, random_state, retrain=train_class))

        # Linear discriminant analysis HPO | LDA_hpo
        if classifier_type == 'all_hpo' or classifier_type == 'LDA_hpo':
            accuracy_lda_hpo, precision_lda_hpo, recall_lda_hpo, f1_lda_hpo, specificity_lda_hpo, g_mean_lda_hpo = (
                classify_lda_hpo(x_train, x_test, y_train, y_test, random_state, retrain=train_class))

        # Linear discriminant analysis | LDA
        if classifier_type == 'all' or classifier_type == 'LDA':
            accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda, g_mean_lda = (
                classify_lda(x_train, x_test, y_train, y_test, random_state, retrain=train_class))

        # K Nearest Neighbors HPO | KNN_hpo
        if classifier_type == 'all_hpo' or classifier_type == 'KNN_hpo':
            accuracy_knn_hpo, precision_knn_hpo, recall_knn_hpo, f1_knn_hpo, specificity_knn_hpo, g_mean_knn_hpo = (
                classify_knn_hpo(x_train, x_test, y_train, y_test, random_state, retrain=train_class))

        # K Nearest Neighbors | KNN
        if classifier_type == 'all' or classifier_type == 'KNN':
            accuracy_knn, precision_knn, recall_knn, f1_knn, specificity_knn, g_mean_knn = (
                classify_knn(x_train, x_test, y_train, y_test, random_state, retrain=train_class))

        # Support Vector Machine HPO | SVM_hpo
        if classifier_type == 'all_hpo' or classifier_type == 'SVM_hpo':
            accuracy_svm_hpo, precision_svm_hpo, recall_svm_hpo, f1_svm_hpo, specificity_svm_hpo, g_mean_svm_hpo = (
                classify_svm_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state, retrain=train_class))

        # Support Vector Machine | SVM
        if classifier_type == 'all' or classifier_type == 'SVM':
            accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm, g_mean_svm = (
                classify_svm(x_train, x_test, y_train, y_test, class_weight_imb, random_state, retrain=train_class))

        # Ensemble with Gradient Boosting HPO | EGB_hpo
        if classifier_type == 'all_hpo' or classifier_type == 'EGB_hpo':
            accuracy_gb_hpo, precision_gb_hpo, recall_gb_hpo, f1_gb_hpo, specificity_gb_hpo, g_mean_gb_hpo = (
                classify_ensemble_with_gradboost_hpo(x_train, x_test, y_train, y_test, random_state, retrain=train_class))

        # Ensemble with Gradient Boosting | EGB
        if classifier_type == 'all' or classifier_type == 'EGB':
            accuracy_gb, precision_gb, recall_gb, f1_gb, specificity_gb, g_mean_gb = (
                classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test, random_state, retrain=train_class))

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
    # Breakpoint for troubleshooting
    x = 1

    duration = time.time() - start_time
    print(duration)