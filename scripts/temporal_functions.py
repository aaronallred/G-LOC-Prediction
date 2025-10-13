import numpy as np
import os
import joblib  # For saving the model
from feature_engine.selection import SelectBySingleFeaturePerformance, SelectByTargetMeanPerformance
from nltk.classify.svm import SvmClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import plot_tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from GLOC_visualization import create_confusion_matrix
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from itertools import islice
from imblearn.metrics import geometric_mean_score
from baseline_methods import baseline_data
from GLOC_data_processing import *
from scripts.GLOC_classifier import stratified_kfold_split
from scripts.feature_selection import feature_selection_lasso
from scripts.features import feature_generation
from scripts.imbalance_techniques import simple_smote, resample_ros
from scripts.imputation import knn_impute, faster_knn_impute
import pickle
from prediction import y_prediction_offset
import matplotlib.pyplot as plt



###### This script will load data and make corrections to the GLOC labels for prediction.
###### A separate script will be used to train and work with the models.
###### Note for BRADY: This will only work if placed in the entire GLOC repo on Alien

def data_with_prediction(backstep,data_rate, classifier_type):

      ################################################### USER INPUTS  ###################################################
        ## Data Folder Location
        # datafolder = '../../'
    datafolder = '../data/'

        # Random State | 42 - Debug mode
    random_state = 42

        ## Classifier | Pick 'logreg' 'rf' 'LDA' 'KNN' 'SVM' 'EGB'
          # classifier_type = 'rf' # example of what would be fed into code
            # Using specified class type, we pull out the txt file that contains hyperparameters and metrics

    if classifier_type == 'logreg':
        # Specifying Methods from Sequential optimization
        baseline_window = 5  # seconds - PULLED FROM NIKKI PAPER
        window_size = 12.5 # seconds - PULLED FROM NIKKI PAPER
        stride = 0.25 # seconds - PULLED FROM NIKKI PAPER
        imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
        feature_reduction_type = 'performance' #- PULLED FROM NIKKI PAPER
        baseline_methods_to_use = ['v0', 'v1', 'v2','v5','v6','v7','v8'] #- PULLED FROM NIKKI PAPER
        impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
        n_neighbors = 5  # - PULLED FROM NIKKI PAPER



    if classifier_type == 'rf':
        # Specifying Methods from Sequential optimization
        baseline_window = 18.75  # seconds - PULLED FROM NIKKI PAPER
        window_size = 7.5  # seconds - PULLED FROM NIKKI PAPER
        stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
        feature_reduction_type = 'none'  # - PULLED FROM NIKKI PAPER
        threshold = 30  # - PULLED FROM NIKKI PAPER
        baseline_methods_to_use = ['v0', 'v1', 'v2','v5','v6','v7','v8']  # - PULLED FROM NIKKI PAPER
        imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
        impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
        n_neighbors = 3  # - PULLED FROM NIKKI PAPER
        # Code for loading txt


    if classifier_type == 'LDA':
        # Specifying Methods from Sequential optimization
        baseline_window = 46.25  # seconds - PULLED FROM NIKKI PAPER
        window_size = 15  # seconds - PULLED FROM NIKKI PAPER
        stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
        feature_reduction_type = 'lasso'  # - PULLED FROM NIKKI PAPER
        baseline_methods_to_use = ['v0', 'v1', 'v2']  # - PULLED FROM NIKKI PAPER
        imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
        impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
        n_neighbors = 3  # - PULLED FROM NIKKI PAPER
        # Code for loading txt


    if classifier_type == 'SVM':
        # Specifying Methods from Sequential optimization
        baseline_window = 32.5  # seconds - PULLED FROM NIKKI PAPER
        window_size = 15  # seconds - PULLED FROM NIKKI PAPER
        stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
        feature_reduction_type = 'ridge'  # - PULLED FROM NIKKI PAPER
        threshold = 10  # - PULLED FROM NIKKI PAPER
        baseline_methods_to_use = ['v0', 'v1', 'v2']  # - PULLED FROM NIKKI PAPER
        impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
        n_neighbors = 3  # - PULLED FROM NIKKI PAPER
        imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER


    if classifier_type == 'EGB':
        # Specifying Methods from Sequential optimization
        baseline_window = 46.25  # seconds - PULLED FROM NIKKI PAPER
        window_size = 12.5  # seconds - PULLED FROM NIKKI PAPER
        stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
        feature_reduction_type = 'ridge'  # - PULLED FROM NIKKI PAPER
        threshold = 20 # - PULLED FROM NIKKI PAPER
        baseline_methods_to_use = ['v0', 'v1', 'v2','v5','v6','v7','v8']  # - PULLED FROM NIKKI PAPER
        imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
        impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
        n_neighbors = 3  # - PULLED FROM NIKKI PAPER
        # Code for loading txt


    if classifier_type == 'KNN':
        # Specifying Methods from Sequential optimization
        baseline_window = 32.5  # seconds - PULLED FROM NIKKI PAPER
        window_size = 15  # seconds - PULLED FROM NIKKI PAPER
        stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
        feature_reduction_type = 'performance'  # - PULLED FROM NIKKI PAPER
        baseline_methods_to_use = ['v0', 'v1', 'v2']  # - PULLED FROM NIKKI PAPER
        imbalance_type = 'ros' # - PULLED FROM NIKKI PAPER
        impute_type = 1 # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
        n_neighbors = 5 # - PULLED FROM NIKKI PAPER
        # Code for loading txt


    train_class = True
    class_weight_imb = None

        # Data Handling Options
    remove_NaN_trials = True # Planning to always remove NaN for offset sequence


        ## Model Parameters
    model_type = ['noAFE', 'explicit']
    if 'noAFE' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                     'rawEEG', 'processedEEG', 'strain', 'demographics']
    if 'noAFE' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking','rawEEG', 'processedEEG']
            # feature_groups_to_analyze = ['ECG']

        # baseline_methods_to_use = ['v0','v1','v2','v3','v4','v5','v6','v7','v8']
    # baseline_methods_to_use = ['v0','v1','v2','v5','v6','v7','v8']

    offset = 0  # seconds
    time_start = 0  # seconds
    training_ratio = 0.8

        # Subject & Trial Information (only need to adjust this if doing analysis type 0,1)
    subject_to_analyze = '01'
    trial_to_analyze = '02'
    analysis_type = 2

    #### Making code more efficient by only running certain sections on first iteration. Data stored in cache
    cache_folder = os.path.join('./cached_data', classifier_type)
    os.makedirs(cache_folder, exist_ok=True)

    # File names to save data .pkl as
    feature_matrix_name = os.path.join(cache_folder, f'x_feature_matrix_method_{classifier_type}.pkl')
    y_label_name = os.path.join(cache_folder, f'y_gloc_labels_method_{classifier_type}.pkl')
    all_features_name = os.path.join(cache_folder, f'all_features_method_{classifier_type}.pkl')

    if backstep == 0:
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


        ############################################### DATA CLEAN AND Some Imputation ###############################################
        """ 
               Optional handling of raw NaN data, depending on 'remove_NaN_trials' 'impute_type' <= 1
            """
          ### Remove full trials with NaN
        if remove_NaN_trials:
          gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc, nan_proportion_df = (
              remove_all_nan_trials(gloc_data_reduced, all_features,
                                    features, features_phys, features_ecg, features_eeg, gloc))

        impute_type = 2 # For debugging, makes it easier to run
        # knn_impute takes way too long

          ### Impute missing row data
        # Impute type == 0 will result in errors unless performed on all variables. not just features.
        if impute_type == 0:
          # Remove rows with NaN
          gloc, features, gloc_data_reduced = process_NaN_raw(gloc, features, gloc_data_reduced)

                ### Impute missing row data
        elif impute_type == 1:
            features, indicator_matrix = faster_knn_impute(features, n_neighbors)

        ################################################## REDUCE MEMORY ##################################################

            # Grab columns from gloc_data_reduced and remove gloc_data_reduced variable from memory
        trial_column = gloc_data_reduced['trial_id']
        time_column = gloc_data_reduced['Time (s)']
        event_validated_column = gloc_data_reduced['event_validated']
        subject_column = gloc_data_reduced['subject']

        del gloc_data_reduced

        save_variables_to_folder(cache_folder, {
            'filename': filename,
            'baseline_data_filename': baseline_data_filename,
            'demographic_data_filename': demographic_data_filename,
            'list_of_eeg_data_files': list_of_eeg_data_files,
            'list_of_baseline_eeg_processed_files': list_of_baseline_eeg_processed_files,
            'features': features,
            'features_phys': features_phys,
            'features_ecg': features_ecg,
            'features_eeg': features_eeg,
            'all_features': all_features,
            'all_features_phys': all_features_phys,
            'all_features_ecg': all_features_ecg,
            'all_features_eeg': all_features_eeg,
            'gloc': gloc,
            'trial_column': trial_column,
            'time_column': time_column,
            'event_validated_column': event_validated_column,
            'subject_column': subject_column,
            'nan_proportion_df': nan_proportion_df if remove_NaN_trials else None,
            'indicator_matrix': indicator_matrix if impute_type == 1 else None
        })

        print('reduce memory complete for', backstep, 'backstep')

    else:
        # Load from cache
        cached_vars = load_variables_from_folder(cache_folder, [
            'filename',
            'baseline_data_filename',
            'demographic_data_filename',
            'list_of_eeg_data_files',
            'list_of_baseline_eeg_processed_files',
            'features',
            'features_phys',
            'features_ecg',
            'features_eeg',
            'all_features',
            'all_features_phys',
            'all_features_ecg',
            'all_features_eeg',
            'gloc',
            'trial_column',
            'time_column',
            'event_validated_column',
            'subject_column',
            'nan_proportion_df',
            'indicator_matrix'
        ])

        # Unpack
        filename = cached_vars['filename']
        baseline_data_filename = cached_vars['baseline_data_filename']
        demographic_data_filename = cached_vars['demographic_data_filename']
        list_of_eeg_data_files = cached_vars['list_of_eeg_data_files']
        list_of_baseline_eeg_processed_files = cached_vars['list_of_baseline_eeg_processed_files']
        features = cached_vars['features']
        features_phys = cached_vars['features_phys']
        features_ecg = cached_vars['features_ecg']
        features_eeg = cached_vars['features_eeg']
        all_features = cached_vars['all_features']
        all_features_phys = cached_vars['all_features_phys']
        all_features_ecg = cached_vars['all_features_ecg']
        all_features_eeg = cached_vars['all_features_eeg']
        gloc = cached_vars['gloc']
        trial_column = cached_vars['trial_column']
        time_column = cached_vars['time_column']
        event_validated_column = cached_vars['event_validated_column']
        subject_column = cached_vars['subject_column']
        nan_proportion_df = cached_vars['nan_proportion_df']
        indicator_matrix = cached_vars['indicator_matrix']
        print('for', backstep, 'backstep... data was recovered from cache')  # debugging

    ###################################################### Prediction Offset ###############################################
    # Flag to see if GLOC has any NaN in it
    if np.isnan(gloc).any():
        print('gloc has nan')
    else:
        print('gloc has no nan')

    # Function inputs
        # Backstep: number of seconds we are trying to predict in advance
        # data_rate: (hz) the data rate at which data is collected.
    gloc = y_prediction_offset(gloc,backstep,data_rate,trial_column) # Call function to shift gloc flags around

    print('gloc offset complete for' ,backstep, '') # debugging

    ################################################ BASELINE ################################################
    """
            Computes baseline data 
        """
      #### Making code more efficient by only running certain sections on first iteration
    if backstep == 0:
        combined_baseline, combined_baseline_names, baseline_v0, baseline_names_v0 = (
          baseline_data(baseline_methods_to_use, trial_column, time_column, event_validated_column, subject_column,
                        features, all_features,
                        gloc, baseline_window, features_phys, all_features_phys, features_ecg, all_features_ecg,
                        features_eeg, all_features_eeg, baseline_data_filename, list_of_baseline_eeg_processed_files,
                        model_type))

        save_variables_to_folder(cache_folder, {
          'combined_baseline': combined_baseline,
          'combined_baseline_names': combined_baseline_names,
          'baseline_v0': baseline_v0,
          'baseline_names_v0': baseline_names_v0
        })
    else:
        cached_vars2 = load_variables_from_folder(cache_folder, [
            # existing variables...
            'combined_baseline',
            'combined_baseline_names',
            'baseline_v0',
            'baseline_names_v0'
        ])
        combined_baseline = cached_vars2['combined_baseline']
        combined_baseline_names = cached_vars2['combined_baseline_names']
        baseline_v0 = cached_vars2['baseline_v0']
        baseline_names_v0 = cached_vars2['baseline_names_v0']


    print('baseline complete for', backstep, 'backstep')  # debugging

    ###################################################### Feature Generation #####################################################
    # Feature generation has to happen for each offset value, cannot make more efficient. Need to do it to window the gloc labels
    if backstep == 0:
        y_gloc_labels, x_feature_matrix, all_features = (
            feature_generation(time_start, offset, stride, window_size, combined_baseline, gloc, trial_column,
                         time_column,
                         combined_baseline_names, baseline_names_v0, baseline_v0, feature_groups_to_analyze))


        # Remove constant columns
        x_feature_matrix, all_features = remove_constant_columns(x_feature_matrix, all_features)


    else:
        #  normal outputs of x and all feats will be deleted later if not calculated at 0 offset
        y_gloc_labels, x_feature_matrix, all_features = (
            feature_generation(time_start, offset, stride, window_size, combined_baseline, gloc, trial_column,
                               time_column,
                               combined_baseline_names, baseline_names_v0, baseline_v0, feature_groups_to_analyze))


    print('generation complete for', backstep, 'backstep')  # debugging

    # Always do NaN evaluation to make sure x and y are the same size
    # Need to remove NAN or IMPUTE:
    impute_type = 2
    if impute_type == 2:
      # Remove rows with NaN (temporary solution-should replace with other method eventually)
      y_gloc_labels, x_feature_matrix, all_features = process_NaN(y_gloc_labels, x_feature_matrix, all_features)

      print('After imputing at a backstep of', backstep, '')  # debugging
      print("IGNORE x_feature_matrix shape:", x_feature_matrix.shape)  # debugging
      print("y_gloc_labels shape:", y_gloc_labels.shape)  # debugging

    ######################################################## FEATURE REDUCTION #######################################################
    # If statement evaluates if we have already done feature reduction of x matrix as is time intensive.
    # We want these x matrices fixed across offsets so if it has been done for backstep = 0, dont do again.
    if backstep == 0:

        print('entering reduction for', backstep, 'backstep')  # debugging
        #########  Reduction ###################
        if feature_reduction_type == 'lasso':
            # Implement feature reduction
            x_feature_matrix, selected_features = feature_selection_lasso_predict(x_feature_matrix, y_gloc_labels, all_features,
                                                                         random_state)
        # Ridge Regression | ridge
        if feature_reduction_type == 'ridge':
            # Implement feature reduction & assess performance of classifiers
            x_feature_matrix, selected_features = (
                feature_selection_ridge_predict(x_feature_matrix, y_gloc_labels, all_features, random_state, threshold))

        # Select by Single Feature Performance | performance
        if feature_reduction_type == 'performance':
            # Implement feature reduction & assess performance of classifiers
            x_feature_matrix, selected_features = (
                feature_selection_performance_predict(x_feature_matrix, y_gloc_labels, all_features, classifier_type,
                                                      random_state))

        # Select by Target mean
        if feature_reduction_type == 'target_mean':
            # Implement feature reduction & assess performance of classifiers
            x_feature_matrix, selected_features = (
                target_mean_selection_predict(x_feature_matrix, y_gloc_labels, all_features,
                                                      random_state))

        # Store generated features and reduced features at 0 seconds offset. FIX TO THIS for every other offset.
        # Need to name these files to specific classifiers
        with open(all_features_name, 'wb') as file:
            pickle.dump(all_features, file)

        print(f"Saved all_features to {all_features_name}, size={os.path.getsize(all_features_name)} bytes")

        with open(feature_matrix_name, 'wb') as file:
            pickle.dump(x_feature_matrix, file)

        print(f"Saved x_features to {feature_matrix_name}, size={os.path.getsize(feature_matrix_name)} bytes")

    else:
        # If we have already reduced features at 0 backstep, just reopen.
        del all_features
        del x_feature_matrix
        if os.path.getsize(all_features_name) == 0:
            raise ValueError(f"File {all_features_name} is empty. Cannot load features.")
        with open(all_features_name, 'rb') as file:
            all_features = pickle.load(file)

        if os.path.getsize(feature_matrix_name) == 0:
            raise ValueError(f"File {feature_matrix_name} is empty. Cannot load feature matrix.")
        with open(feature_matrix_name, 'rb') as file:
            x_feature_matrix = pickle.load(file)
        print('Files accessed for', backstep, 'backstep')  # debugging

    ############################################## CLASS IMBALANCE ####################################################

    # With the way the functions are currently designed need to do ROS on indiviudal train/test splits. Might be able to recode this later

    # Random Over Sampling | ros
    if imbalance_type == 'ros':
        # Implement Imbalance Sampling Technique
        print('Starting sampling methods for', backstep, 'backstep')  # debugging
        x_train, x_test, y_train, y_test = train_test_split(x_feature_matrix, y_gloc_labels, test_size=training_ratio, random_state=random_state)
        ros_x_train, ros_y_train = resample_ros(x_train, y_train, random_state)
        x_test, y_test = resample_ros(x_test, y_test, random_state)
        del x_train, y_train
        x_train = ros_x_train
        y_train = ros_y_train
        print('Sampling methods done for', backstep, 'backstep')  # debugging

    else:
        print('Starting sampling methods for', backstep, 'backstep')  # debugging
        x_train, x_test, y_train, y_test = train_test_split(x_feature_matrix, y_gloc_labels, test_size=training_ratio, random_state=random_state)
        print('Sampling methods done for', backstep, 'backstep')  # debugging

        ################################################ Get Outputs Ready ############################################

    x_feature_return = np.vstack((x_train, x_test))
    y_return = np.vstack((y_train, y_test))

    # Function call end
    return (x_feature_return, y_return)

def cross_validation_split (y_gloc_labels, x_feature_matrix,ID,num_folds):
    ################################################ TRAIN/TEST SPLIT  ################################################
    """
          Split data into training/test with cross validation for performance
    """
    # # Training/Test Split with kfolds
    # inputs are y, x, the number of folds we will use and which fold iteration we are on.
    x_train, x_test, y_train, y_test = stratified_kfold_split(y_gloc_labels, x_feature_matrix,
                                                              num_folds, ID)

    # Simple version of train test split
    # x_train, x_test, y_train, y_test = train_test_split(x_feature_matrix, y_gloc_labels, test_size=0.2, random_state=random_state)

    return (y_train, y_test, x_train, x_test)

def lr_call (y_train, y_test, x_train, x_test):

    # Might be better to call Nikkis code here - GLOC Classifier for each model, get the scores back, and maybe comment
    # out the confusion matrix call. As long as we still retrain it to the new DATA. Unfortunately, we can only load in the old models
    # to get hyperparamters, but we certainly cant just instantly use those models. If we want to still call the classifier function,
    # should have it load in hyperparameters for the retrain section. Can I just load in her best models saved off?
    # model = ModelType
    # clf = load(model) # This should come with a set of hyperparameters
    # hpos = get.params(clf) # Get HP to inspect


   # Fixing these terms for training to see how model works
   model = LogisticRegression(
       penalty='l2',  # L2 regularization
       C=0.5,  # Regularization strength
       solver='saga',  # Scales well for large datasets and supports L1, L2, elasticnet
       max_iter=500,  # Plenty of iterations for convergence
       fit_intercept=True,  # Include bias term
       class_weight='balanced',  # Handle class imbalance
       random_state=42,  # Reproducibility
       tol=1e-4,  # Convergence tolerance
       warm_start=False  # Start fresh each time
   )
   model.fit(x_train, y_train)
   y_predictions = model.predict(x_test)

   # Assess Performance
   accuracy = metrics.accuracy_score(y_test, y_predictions)
   precision = metrics.precision_score(y_test, y_predictions)
   recall = metrics.recall_score(y_test, y_predictions)
   f1 = metrics.f1_score(y_test, y_predictions)
   specificity = metrics.recall_score(y_test, y_predictions, pos_label=0)
   g_mean = geometric_mean_score(y_test, y_predictions)

   return (accuracy,precision,recall,f1,specificity,g_mean)

def plotting_offset_models(offset_ranges,accuracy_model,precision_model,recall_model,f1_model,specificity_model,gmean_model,classifier_name):

    # Convert offset_ranges to a NumPy array for plotting
        offsets = np.array(offset_ranges)

    # Helper function to compute mean, min, and max across folds
        def summarize_range(metric_matrix):
            mean_vals = np.mean(metric_matrix, axis=1)
            min_vals = np.min(metric_matrix, axis=1)
            max_vals = np.max(metric_matrix, axis=1)
            return mean_vals, min_vals, max_vals

            # Prepare figure
        plt.figure(figsize=(12, 8))

        # Assigning metrics colors
        metrics = [
            ('Accuracy', accuracy_model, 'blue'),
            ('Precision', precision_model, 'green'),
            ('Recall', recall_model, 'orange'),
            ('F1 Score', f1_model, 'red'),
            ('Specificity', specificity_model, 'purple'),
            ('G-Mean', gmean_model, 'brown')
        ]

        # Plot each metric individually and loop through them
        for idx, (label, matrix, color) in enumerate(metrics, start=1):
            mean_vals, min_vals, max_vals = summarize_range(matrix)
            plt.subplot(2, 3, idx)

            # Plot mean line
            plt.plot(offsets, mean_vals, color=color, label=label)

            # Plot the mean values as a scatter as well
            plt.scatter(offsets, mean_vals, color=color, edgecolor='black', zorder=5)

            # Plot shaded region for min–max range
            plt.fill_between(offsets, min_vals, max_vals,
                             color='gray', alpha=0.3, label='Range (min–max)')

            plt.title(f'{label} vs Offset')
            plt.xlabel('Offset [s]')
            plt.ylabel(label)
            plt.grid(True)
            plt.legend()
            plt.suptitle(f'Metrics Across Offsets — Classifier: {classifier_name}', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.show()

        ############################ Save each metric matrix as a .pkl file ############################
        results_folder = './prediction_model_results'
        os.makedirs(results_folder, exist_ok=True)

        metric_data = {
            'accuracy': accuracy_model,
            'precision': precision_model,
            'recall': recall_model,
            'f1_score': f1_model,
            'specificity': specificity_model,
            'g_mean': gmean_model
        }

        for metric_name, matrix in metric_data.items():
            filename = f"{metric_name}_results_{classifier_name}.pkl"
            filepath = os.path.join(results_folder, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(matrix, f)
            print(f"Saved {metric_name} to {filepath}")

        return None

def rf_call (y_train, y_test, x_train, x_test):

    # Might be better to call Nikkis code here - GLOC Classifier for each model, get the scores back, and maybe comment
    # out the confusion matrix call. As long as we still retrain it to the new DATA. Unfortunately, we can only load in the old models
    # to get hyperparamters, but we certainly cant just instantly use those models. If we want to still call the classifier function,
    # should have it load in hyperparameters for the retrain section. Can I just load in her best models saved off?
    # model = ModelType
    # clf = load(model) # This should come with a set of hyperparameters
    # hpos = get.params(clf) # Get HP to inspect

   # Fixing these terms for training to see how model works
    model = RandomForestClassifier(
        n_estimators=200,  # More trees for better performance
        max_depth=None,  # Let trees grow fully unless overfitting
        min_samples_split=10,  # Prevent overfitting by requiring more samples to split
        min_samples_leaf=4,  # Minimum samples per leaf for generalization
        max_features='sqrt',  # Good default for classification tasks
        bootstrap=True,  # Use bootstrapped samples
        class_weight='balanced',  # Handle class imbalance
        n_jobs=-1,  # Use all CPU cores
        random_state=42,  # Reproducibility
        verbose=1  # Track training progress
    )

    model.fit(x_train, y_train)
    y_predictions = model.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, y_predictions)
    precision = metrics.precision_score(y_test, y_predictions)
    recall = metrics.recall_score(y_test, y_predictions)
    f1 = metrics.f1_score(y_test, y_predictions)
    specificity = metrics.recall_score(y_test, y_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, y_predictions)

    return (accuracy,precision,recall,f1,specificity,g_mean)

def knn_call(y_train, y_test, x_train, x_test):
    # Might be better to call Nikkis code here - GLOC Classifier for each model, get the scores back, and maybe comment
    # out the confusion matrix call. As long as we still retrain it to the new DATA. Unfortunately, we can only load in the old models
    # to get hyperparamters, but we certainly cant just instantly use those models. If we want to still call the classifier function,
    # should have it load in hyperparameters for the retrain section. Can I just load in her best models saved off?
    # model = ModelType
    # clf = load(model) # This should come with a set of hyperparameters
    # hpos = get.params(clf) # Get HP to inspect

    # Fixing these terms for training to see how model works
    model = KNeighborsClassifier(
        n_neighbors=5,  # Number of neighbors to use
        weights='distance',  # Closer neighbors have more influence
        algorithm='auto',  # Automatically chooses the best algorithm
        leaf_size=30,  # Affects speed/memory trade-off
        p=2,  # Use Euclidean distance (L2 norm)
        n_jobs=-1  # Use all CPU cores for distance computation
    )

    model.fit(x_train, y_train)
    y_predictions = model.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, y_predictions)
    precision = metrics.precision_score(y_test, y_predictions)
    recall = metrics.recall_score(y_test, y_predictions)
    f1 = metrics.f1_score(y_test, y_predictions)
    specificity = metrics.recall_score(y_test, y_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, y_predictions)

    return (accuracy, precision, recall, f1, specificity, g_mean)

def svm_call(y_train, y_test, x_train, x_test):
    # Might be better to call Nikkis code here - GLOC Classifier for each model, get the scores back, and maybe comment
    # out the confusion matrix call. As long as we still retrain it to the new DATA. Unfortunately, we can only load in the old models
    # to get hyperparamters, but we certainly cant just instantly use those models. If we want to still call the classifier function,
    # should have it load in hyperparameters for the retrain section. Can I just load in her best models saved off?
    # model = ModelType
    # clf = load(model) # This should come with a set of hyperparameters
    # hpos = get.params(clf) # Get HP to inspect

    # Fixing these terms for training to see how model works
    model = svm.SVC(
        C=1.0,  # Regularization strength
        kernel='rbf',  # RBF kernel for non-linear decision boundaries
        gamma='scale',  # Automatically adapts to feature variance
        class_weight='balanced',  # Adjusts for class imbalance
        probability=True,  # Enables probability estimates (slower but useful)
        cache_size=500,  # Allocates memory for kernel computations
        tol=1e-3,  # Convergence tolerance
        max_iter=-1,  # No limit on iterations
        random_state=42  # Ensures reproducibility
    )

    model.fit(x_train, y_train)
    y_predictions = model.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, y_predictions)
    precision = metrics.precision_score(y_test, y_predictions)
    recall = metrics.recall_score(y_test, y_predictions)
    f1 = metrics.f1_score(y_test, y_predictions)
    specificity = metrics.recall_score(y_test, y_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, y_predictions)

    return (accuracy, precision, recall, f1, specificity, g_mean)

def lda_call(y_train, y_test, x_train, x_test):
    # Might be better to call Nikkis code here - GLOC Classifier for each model, get the scores back, and maybe comment
    # out the confusion matrix call. As long as we still retrain it to the new DATA. Unfortunately, we can only load in the old models
    # to get hyperparamters, but we certainly cant just instantly use those models. If we want to still call the classifier function,
    # should have it load in hyperparameters for the retrain section. Can I just load in her best models saved off?
    # model = ModelType
    # clf = load(model) # This should come with a set of hyperparameters
    # hpos = get.params(clf) # Get HP to inspect

    # Fixing these terms for training to see how model works
    model = LinearDiscriminantAnalysis(
        solver='lsqr',  # Efficient for large datasets, supports shrinkage
        shrinkage='auto',  # Automatically estimates regularization to improve generalization
        tol=1e-4  # Convergence tolerance
    )

    model.fit(x_train, y_train)
    y_predictions = model.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, y_predictions)
    precision = metrics.precision_score(y_test, y_predictions)
    recall = metrics.recall_score(y_test, y_predictions)
    f1 = metrics.f1_score(y_test, y_predictions)
    specificity = metrics.recall_score(y_test, y_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, y_predictions)

    return (accuracy, precision, recall, f1, specificity, g_mean)

def ensemble_call(y_train, y_test, x_train, x_test):
    # Might be better to call Nikkis code here - GLOC Classifier for each model, get the scores back, and maybe comment
    # out the confusion matrix call. As long as we still retrain it to the new DATA. Unfortunately, we can only load in the old models
    # to get hyperparamters, but we certainly cant just instantly use those models. If we want to still call the classifier function,
    # should have it load in hyperparameters for the retrain section. Can I just load in her best models saved off?
    # model = ModelType
    # clf = load(model) # This should come with a set of hyperparameters
    # hpos = get.params(clf) # Get HP to inspect

    # Fixing these terms for training to see how model works
    model = GradientBoostingClassifier(
        n_estimators=300,  # More trees for better performance
        learning_rate=0.05,  # Lower rate for smoother convergence
        max_depth=5,  # Controls tree complexity
        min_samples_split=10,  # Prevents overfitting
        min_samples_leaf=5,  # Ensures leaves have enough samples
        subsample=0.8,  # Stochastic gradient boosting
        max_features='sqrt',  # Reduces overfitting and speeds up training
        random_state=42,  # Reproducibility
        verbose=1  # Training progress output
    )

    model.fit(x_train, y_train)
    y_predictions = model.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, y_predictions)
    precision = metrics.precision_score(y_test, y_predictions)
    recall = metrics.recall_score(y_test, y_predictions)
    f1 = metrics.f1_score(y_test, y_predictions)
    specificity = metrics.recall_score(y_test, y_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, y_predictions)

    return (accuracy, precision, recall, f1, specificity, g_mean)

def feature_selection_performance_predict(x, y, all_features, classifier_method, random_state):
    # Complete feature selection by single feature performance for each classifier
    if classifier_method == 'logreg':
        sfp = SelectBySingleFeaturePerformance(LogisticRegression(random_state=random_state),cv=3)
    elif classifier_method == 'rf':
        sfp = SelectBySingleFeaturePerformance(RandomForestClassifier(random_state=random_state), cv=3)
    elif classifier_method == 'LDA':
        sfp = SelectBySingleFeaturePerformance(LinearDiscriminantAnalysis(), cv=3)
    elif classifier_method == 'KNN':
        sfp = SelectBySingleFeaturePerformance(KNeighborsClassifier(), cv=3)
    elif classifier_method == 'SVM':
        sfp = SelectBySingleFeaturePerformance(svm.SVC(random_state=random_state), cv=3)
    elif classifier_method == 'EGB':
        sfp = SelectBySingleFeaturePerformance(GradientBoostingClassifier(random_state=random_state), cv=3)

    # fit single feature performance model on data
    sfp.fit(x, y)

    # Reduce x
    x = sfp.transform(x)

    # Use features to drop to determine features to keep
    features_to_drop = sfp.features_to_drop_
    features_to_drop_index = [element[1:] for element in features_to_drop]
    features_to_drop_index = np.array([int(x) for x in features_to_drop_index])
    selected_features = [all_features[index] for index in range(len(all_features)) if index not in features_to_drop_index]

    # Example feature parameters
    # sfp_feature_performance = sfp.feature_performance_
    # sfp_features_to_drop = sfp.features_to_drop_

    # Example Plotting Code to modify
        # r = pd.concat([
        #     pd.Series(sel.feature_performance_),
        #     pd.Series(sel.feature_performance_std_)
        # ], axis=1
        # )
        # r.columns = ['mean', 'std']
        #
        # r['mean'].plot.bar(yerr=[r['std'], r['std']], subplots=True)
        #
        # plt.title("Single feature model Performance")
        # plt.ylabel('R2')
        # plt.xlabel('Features')
        # plt.show()

    return x, selected_features

def feature_selection_ridge_predict(x, y, all_features, random_state, threshold):
    """
    This function finds optimal ridge parameters and fits a ridge model to determine
    most important features. MODIFIED FROM NIKKIS CODE
    """

    # # parameters to be tested on GridSearchCV
    # params = {"alpha": np.arange(0.00001, 10, 500)}
    #
    # # Number of Folds and adding the random state for replication
    # # kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    #
    # # Initializing the Model
    # ridge0 = Ridge()
    #
    # # GridSearchCV with model, params, and stratified 10fold CV
    # ridge_cv = GridSearchCV(ridge0, param_grid=params, cv=10)
    # ridge_cv.fit(x_train, y_train)
    #
    # # Use optimal alpha value from grid search CV
    # alpha_optimal = ridge_cv.best_params_['alpha']
    #
    # # calling the model with the best parameter
    # ridge1 = Ridge(alpha=alpha_optimal)
    # ridge1.fit(x_train, y_train)

    # Define the hyperparameter search space
    search_spaces = {
        'alpha': Real(0.00001, 100),
    }

    # Initializing the Model
    ridge = Ridge()

    # Set up BayesSearchCV
    ridge_cv = BayesSearchCV(
        estimator=ridge,
        search_spaces=search_spaces,
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    # Fit model
    ridge_cv.fit(x, np.ravel(y))

    # Get best model and coefficients
    best_ridge = ridge_cv.best_estimator_
    ridge1_coef = np.abs(best_ridge.coef_)

    # # Use optimal alpha value from CV
    # alpha_optimal = ridge_cv.best_params_['alpha']
    #
    # # calling the model with the best parameter
    # ridge1 = Ridge(alpha=alpha_optimal)
    # ridge1.fit(x_train, y_train)

    # # Using np.abs() to make coefficients positive.
    # ridge1_coef = np.abs(ridge1.coef_)
    ridge1_coef = np.ravel(ridge1_coef)

    # # plotting the Column Names and Importance of Columns.
    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.bar(all_features, ridge_cv)
    # plt.xticks(rotation=90, fontsize=10)
    # plt.grid()
    # plt.title("Feature Selection Based on Ridge")
    # plt.xlabel("Features")
    # plt.ylabel("Importance")
    # plt.ylim(0, 0.7)
    # plt.show()

    percentile_cutoff = 100 - threshold  # e.g., 70th percentile for top 30%
    threshold_value = np.percentile(ridge1_coef, percentile_cutoff)

    # Determine threshold for top n% features
    selected_features = np.array(all_features)[ridge1_coef >= threshold_value]

    if len(selected_features) == 0:
        raise ValueError("No features selected by Ridge. Try lowering the threshold or inspecting coefficients.")

    # Grab relevant feature columns from x_train and x_test
    feature_index = [index for index, element in enumerate(all_features) if element in selected_features]
    x = x[:, feature_index]

    return x, selected_features

def feature_selection_lasso_predict(x, y, all_features, random_state):
    """
    This function finds optimal lasso alpha parameter and fits a lasso model to determine
    most important features. MODIFIED FROM NIKKIS CODE
    """
    # # parameters to be tested on GridSearchCV
    # params = {"alpha": np.arange(0.00001, 10, 500)}
    #
    # # Number of Folds and adding the random state for replication
    # # kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    #
    # # Initializing the Model
    # lasso = Lasso()
    #
    # # GridSearchCV with model, params and 10 stratified folds.
    # lasso_cv = GridSearchCV(lasso, param_grid=params, cv=10)
    # lasso_cv.fit(x_train, y_train)
    #
    # # Use optimal alpha value from grid search CV
    # alpha_optimal = lasso_cv.best_params_['alpha']
    #
    # # calling the model with the best parameter
    # lasso_optimal = Lasso(alpha=alpha_optimal)
    # lasso_optimal.fit(x_train, y_train)

    # Define the hyperparameter search space
    search_spaces = {
        'alpha': Real(1e-5, 100, prior='log-uniform')
    }

    # Initializing the Model
    lasso = Lasso()

    # Set up BayesSearchCV
    lasso_cv = BayesSearchCV(
        estimator=lasso,
        search_spaces=search_spaces,
        cv=3,
        n_iter=50,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    # Fit model
    lasso_cv.fit(x, np.ravel(y))

    # # Use optimal alpha value from CV
    # alpha_optimal = lasso_cv.best_params_['alpha']

    # # calling the model with the best parameter
    # lasso_optimal = Lasso(alpha=alpha_optimal)
    # lasso_optimal.fit(x_train, y_train)
    #
    #
    # # Using np.abs() to make coefficients positive.
    # lasso_optimal_coef = np.abs(lasso_optimal.coef_)

    # Get best model and coefficients
    best_lasso = lasso_cv.best_estimator_
    lasso_optimal_coef = np.abs(best_lasso.coef_)

    # # plotting the Column Names and Importance of Columns.
    # fig,ax = plt.subplots(figsize=(10,10))
    # plt.bar(all_features, lasso_optimal_coef)
    # plt.xticks(rotation=90, fontsize=10)
    # plt.grid()
    # plt.title("Feature Selection Based on Lasso")
    # plt.xlabel("Features")
    # plt.ylabel("Importance")
    # plt.ylim(0, 0.15)
    # plt.show()

    # Subset of the features with nonzero coefficient
    selected_features = np.array(all_features)[lasso_optimal_coef != 0]

    # Grab relevant feature columns from x_train and x_test
    feature_index = [index for index, element in enumerate(all_features) if element in selected_features]
    x = x[:, feature_index]

    return x, selected_features

def save_variables_to_folder(folder_path, variables_dict):
    os.makedirs(folder_path, exist_ok=True)
    for var_name, var_value in variables_dict.items():
        with open(os.path.join(folder_path, f"{var_name}.pkl"), 'wb') as f:
            pickle.dump(var_value, f)

def load_variables_from_folder(folder_path, variable_names):
    loaded_vars = {}
    for var_name in variable_names:
        file_path = os.path.join(folder_path, f"{var_name}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        with open(file_path, 'rb') as f:
            loaded_vars[var_name] = pickle.load(f)
    return loaded_vars

def target_mean_selection_predict(x, y, all_features, random_state):

    # # parameters to be tested on GridSearchCV
    # params = {"threshold": np.arange(0.001, 1, 100)}
    #
    # # Number of Folds and adding the random state for replication
    # # kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    #
    # # Initializing the Model
    # tmp = SelectByTargetMeanPerformance()
    #
    # # GridSearchCV with model, params and 10 stratified folds.
    # tmp_cv = GridSearchCV(tmp, param_grid=params, cv=10)
    # tmp_cv.fit(x_train, y_train)

    # Define the hyperparameter search space
    search_spaces = {
        'threshold': Real(0.001, 1)
    }

    # Initializing the Model
    tmp = SelectByTargetMeanPerformance()

    # Set up BayesSearchCV
    tmp_cv = BayesSearchCV(
        estimator=tmp,
        search_spaces=search_spaces,
        scoring='roc_auc',
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    # Fit model
    tmp_cv.fit(x, y)

    # Get best model and coefficients
    best_tmp = tmp_cv.best_estimator_

    # # Use optimal value for threshold found in GridSearchCV
    # threshold_optimal = tmp_cv.best_params_['threshold']
    #
    # # calling the model with the best parameter
    # tmp1 = SelectByTargetMeanPerformance(scoring="f1", threshold = threshold_optimal, cv=10, regression=False)
    # tmp1.fit(x_train, y_train)

    # tmp = SelectByTargetMeanPerformance(scoring="f1", threshold = 0.01, cv=10, regression=False)
    # tmp.fit_transform(x_train, y_train)

    # find features to keep
    features_to_keep = best_tmp.get_feature_names_out()

    # Convert feature output from selected_features_target_mean to index array and list of features
    selected_features_index = [element[1:] for element in features_to_keep]
    selected_features_index = np.array([int(x) for x in selected_features_index])
    selected_features = [all_features[index] for index in selected_features_index]

    # Grab relevant feature columns from x_train and x_test
    x = best_tmp.transform(x)

    # Example feature code
    # tmp_variables = tmp.variables_
    # tmp_features_to_drop = tmp.features_to_drop_
    # tmp_feature_performance = tmp.feature_performance_
    # tmp_feature_performance_sdtdev = tmp.feature_performance_std_

    # Example Plotting Code to modify
    #     r = pd.concat([
    #         pd.Series(sel.feature_performance_),
    #         pd.Series(sel.feature_performance_std_)
    #     ], axis=1
    #     )
    #     r.columns = ['mean', 'std']
    #
    #     r['mean'].plot.bar(yerr=[r['std'], r['std']], subplots=True)
    #
    #     plt.title("Feature importance")
    #     plt.ylabel('ROC-AUC')
    #     plt.xlabel('Features')
    #     plt.show()

    return x, selected_features

def plot_f1_scores_across_classifiers(offset_ranges, f1_score_dict, window_lengths):
    """
    Plots F1 score curves for multiple classifiers in a shared panel.

    Parameters:
    - offset_ranges: list of offset values
    - f1_score_dict: dict of {classifier_name: f1_score_matrix}
    - window_length: float, x-axis threshold for shaded region
    """
    offsets = np.array(offset_ranges)

    def summarize_range(metric_matrix):
        mean_vals = np.mean(metric_matrix, axis=1)
        min_vals = np.min(metric_matrix, axis=1)
        max_vals = np.max(metric_matrix, axis=1)
        return mean_vals, min_vals, max_vals

    plt.figure(figsize=(14, 10))
    classifier_names = list(f1_score_dict.keys())

    for idx, classifier_name in enumerate(classifier_names, start=1):
        f1_matrix = f1_score_dict[classifier_name]
        mean_vals, min_vals, max_vals = summarize_range(f1_matrix)

        ax = plt.subplot(2, 3, idx)

        # Plot shaded region beyond window_length
        classifier_window = window_lengths.get(classifier_name, None)
        if classifier_window is not None:
            ax.axvspan(classifier_window, offsets.max(), color='lightcoral', alpha=0.2)

        # Plot mean line
        ax.plot(offsets, mean_vals, color='red', label='F1 Score')

        # Plot scatter points
        ax.scatter(offsets, mean_vals, color='red', edgecolor='black', zorder=5)

        # Plot min–max range
        ax.fill_between(offsets, min_vals, max_vals, color='gray', alpha=0.3, label='Range (min–max)')

        ax.set_title(f'F1 Score — {classifier_name}')
        ax.set_xlabel('Offset [s]')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0.15, 1.0)
        ax.grid(True)
        ax.legend()

    plt.suptitle('F1 Score Across Offsets for All Classifiers', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return None