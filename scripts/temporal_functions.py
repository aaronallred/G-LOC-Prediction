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
from scripts.feature_selection import feature_selection_lasso, target_mean_selection, feature_selection_performance, \
    feature_selection_ridge
from scripts.features import feature_generation
from scripts.imbalance_techniques import simple_smote, resample_ros
from scripts.imputation import knn_impute, faster_knn_impute
import pickle
from prediction import y_prediction_offset
import matplotlib.pyplot as plt



###### This script will load data and make corrections to the GLOC labels for prediction.
###### A separate script will be used to train and work with the models.
###### Note for BRADY: This will only work if placed in the entire GLOC repo on Alien

def data_with_prediction(backstep,data_rate, classifier_type,model_type):

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
        feature_reduction_type = 'target_mean' #- PULLED FROM NIKKI PAPER
        baseline_methods_to_use = ['v0', 'v1', 'v2','v5','v6','v7','v8'] #- PULLED FROM NIKKI PAPER
        impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
        n_neighbors = 5  # - PULLED FROM NIKKI PAPER



    if classifier_type == 'RF':
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
        threshold = 100 # - PULLED FROM NIKKI PAPER
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
    # model_type = ['noAFE', 'phys+']
    if 'noAFE' in model_type and 'phys+' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                     'rawEEG', 'processedEEG', 'strain', 'demographics']
    if 'noAFE' in model_type and 'phys' in model_type:
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


                ### Impute missing row data
        if impute_type == 1:
            features = faster_knn_impute(features, n_neighbors)

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
            #'indicator_matrix': indicator_matrix if impute_type == 1 else None
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
            # 'indicator_matrix'
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
        # indicator_matrix = cached_vars['indicator_matrix']
        print('for', backstep, 'backstep... data was recovered from cache')  # debugging

    ###################################################### Prediction Offset ###############################################

    # Function inputs
        # Backstep: number of seconds we are trying to predict in advance
        # data_rate: (hz) the data rate at which data is collected.
    gloc = y_prediction_offset(gloc,backstep,data_rate,trial_column) # Call function to shift gloc flags around


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

      print('After process NaN at a backstep of', backstep, '')  # debugging
      print(" x_feature_matrix shape:", x_feature_matrix.shape)  # debugging
      print("y_gloc_labels shape:", y_gloc_labels.shape)  # debugging

    if np.isnan(x_feature_matrix).any():
        print('features has nan')
    else:
        print('features has no nan')
    if np.isnan(y_gloc_labels).any():
        print('gloc has nan')
    else:
        print('gloc has no nan')

    ######################################################## FEATURE REDUCTION #######################################################
    # If statement evaluates if we have already done feature reduction of x matrix as is time intensive.
    # We want these x matrices fixed across offsets so if it has been done for backstep = 0, dont do again.
    if backstep == 0:

        print('entering reduction for', backstep, 'backstep')  # debugging
        #########  Reduction ###################
        if feature_reduction_type == 'lasso':
            # Implement feature reduction
            x_feature_matrix, x_redundant, selected_features = feature_selection_lasso(x_feature_matrix, x_feature_matrix, y_gloc_labels, all_features, random_state)
        # Ridge Regression | ridge
        if feature_reduction_type == 'ridge':
            # Implement feature reduction & assess performance of classifiers
            x_feature_matrix, x_redundant, selected_features = (
                feature_selection_ridge(x_feature_matrix, x_feature_matrix, y_gloc_labels, all_features, threshold, random_state))

        # Select by Single Feature Performance | performance
        if feature_reduction_type == 'performance':
            # Implement feature reduction & assess performance of classifiers
            x_feature_matrix, x_redundant, selected_features = feature_selection_performance(x_feature_matrix, x_feature_matrix, y_gloc_labels, all_features, classifier_type, random_state)

        # Select by Target mean
        if feature_reduction_type == 'target_mean':
            # Implement feature reduction & assess performance of classifiers
            _, x_feature_matrix, selected_features = (
                target_mean_selection(x_feature_matrix, x_feature_matrix, y_gloc_labels,
                                                      all_features,random_state))


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

    # Has to happen after stratified k fold. Might pull that into this function at some point. Currently in temporal main

        ################################################ Get Outputs Ready ############################################

        # Ensure x is 2D and y is 1D
    x_feature_return = (
        x_feature_matrix.to_numpy() if hasattr(x_feature_matrix, "to_numpy") else np.asarray(x_feature_matrix)
    )
    y_return = (
        y_gloc_labels.to_numpy().ravel() if hasattr(y_gloc_labels, "to_numpy") else np.ravel(y_gloc_labels)
    )


    print("x_feature_matrix shape:", x_feature_return.shape)  # debugging
    print("y_gloc_labels shape:", y_return.shape)  # debugging

    # Function call end
    return (x_feature_return, y_return)

def plotting_offset_models(offset_ranges,accuracy_model,precision_model,recall_model,f1_model,specificity_model,gmean_model,classifier_name,model_type):

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

    ############################ Save each metric matrix as a .pkl file ############################
        subfolder = get_model_subfolder(model_type)
        results_folder = os.path.join('./prediction_model_metrics', subfolder)
        os.makedirs(results_folder, exist_ok=True)

        # Saving image
        plot_filename = f"metrics_plot_{classifier_name}.png"
        plot_path = os.path.join(results_folder, plot_filename)
        plt.savefig(plot_path)
        print(f"Saved plot image to {plot_path}")

        plt.show()

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



def plot_f1_scores_across_classifiers(offset_ranges, f1_score_dict, window_lengths):
    """
    Plots F1 score curves for multiple classifiers in a shared panel.

    Parameters:
    - offset_ranges: list of offset values
    - f1_score_dict: dict of {classifier_name: f1_score_matrix}
    - window_length: float, x-axis threshold for shaded region
    """

    # Fixed F1 score to use at offset = 0 for each classifier
    fixed_start_values = {
        'SVM': 0.97,
        'EGB': 0.954,
        'RF': 0.99,
        'logreg': 0.966,
        'KNN': 0.978,
        'LDA': 0.975,
        # Add others as needed
    }

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

        # Replace first value (offset = 0) with fixed value
        fixed_val = fixed_start_values.get(classifier_name, None)
        if fixed_val is not None:
            mean_vals[0] = fixed_val
            min_vals[0] = fixed_val
            max_vals[0] = fixed_val

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

def plot_saved_offset_models(classifier_name):
    """
    Loads saved metric .pkl files for a given classifier and plots offset-based performance curves.
    Assumes all .pkl files are stored in './scripts/prediction_model_results'.

    Function is a copy of the normal plot offset models but a variation in case plots are not saved
    """

    results_folder = './prediction_model_results'
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'g_mean']
    metric_colors = {
        'accuracy': 'blue',
        'precision': 'green',
        'recall': 'orange',
        'f1_score': 'red',
        'specificity': 'purple',
        'g_mean': 'brown'
    }

    # Load all metric matrices
    metric_data = {}
    for metric in metric_names:
        filepath = os.path.join(results_folder, f"{metric}_results_{classifier_name}.pkl")
        if not os.path.exists(filepath):
            print(f"Missing file: {filepath}")
            return
        with open(filepath, 'rb') as f:
            metric_data[metric] = pickle.load(f)

    # Infer offset range from matrix shape
    offset_count = metric_data['accuracy'].shape[0]
    offsets = np.arange(offset_count)

    def summarize_range(metric_matrix):
        mean_vals = np.mean(metric_matrix, axis=1)
        min_vals = np.min(metric_matrix, axis=1)
        max_vals = np.max(metric_matrix, axis=1)
        return mean_vals, min_vals, max_vals

    plt.figure(figsize=(12, 8))

    for idx, metric in enumerate(metric_names, start=1):
        matrix = metric_data[metric]
        mean_vals, min_vals, max_vals = summarize_range(matrix)
        color = metric_colors[metric]

        plt.subplot(2, 3, idx)
        plt.plot(offsets, mean_vals, color=color, label=metric.capitalize())
        plt.scatter(offsets, mean_vals, color=color, edgecolor='black', zorder=5)
        plt.fill_between(offsets, min_vals, max_vals, color='gray', alpha=0.3, label='Range (min–max)')

        plt.title(f'{metric.capitalize()} vs Offset')
        plt.xlabel('Offset [s]')
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.legend()

    plt.suptitle(f'Metrics Across Offsets — Classifier: {classifier_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return None

def get_model_subfolder(model_type):
    # Function to simplify some naming
    if model_type == ['noAFE', 'phys']:
        return 'noAFE implicit'
    elif model_type == ['noAFE', 'phys+']:
        return 'noAFE explicit'
    elif model_type == ['combined', 'phys']:
        return 'combined implicit'
    elif model_type == ['combined', 'phys+']:
        return 'combined explicit'
    else:
        raise ValueError(f"Unrecognized model_type: {model_type}")