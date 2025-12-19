import json
from collections import OrderedDict

import numpy as np
import os
import joblib  # For saving the model

from baseline_methods import baseline_data
from GLOC_data_processing import *

from scripts.features import feature_generation, sliding_window_max, sliding_window_mean_calc

from scripts.imputation import knn_impute, faster_knn_impute, eeg_condition_impute
import pickle
from prediction import y_prediction_offset, process_NaN_temporal
import matplotlib.pyplot as plt


def data_with_prediction(backstep,data_rate, classifier_type,model_type,select_features):
    # This script will load data and make corrections to the GLOC labels for prediction.
    # Note for BRADY: This will only work if placed in the entire GLOC repo on Alien
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
        feature_reduction_type = 'lasso' #- PULLED FROM NIKKI PAPER
        baseline_methods_to_use = ['v0', 'v1', 'v2','v5','v6','v7','v8'] #- PULLED FROM NIKKI PAPER
        impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
        n_neighbors = 5  # -For imputation PULLED FROM NIKKI PAPER

        ## Investigating different windows per EVAN ANDERSON
        #window_size = 8 # ~ 0.1 hit to f1 score



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
        n_neighbors = 3  # -For imputation PULLED FROM NIKKI PAPER
        # Code for loading txt

        ## Investigating different windows per EVAN ANDERSON
        #window_size = 5 # ~ 0.1 hit to f1 score


    if classifier_type == 'LDA':
        # Specifying Methods from Sequential optimization
        baseline_window = 46.25  # seconds - PULLED FROM NIKKI PAPER
        window_size = 15  # seconds - PULLED FROM NIKKI PAPER
        stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
        feature_reduction_type = 'lasso'  # - PULLED FROM NIKKI PAPER
        baseline_methods_to_use = ['v0', 'v1', 'v2']  # - PULLED FROM NIKKI PAPER
        imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
        impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
        n_neighbors = 3  # -For imputation PULLED FROM NIKKI PAPER
        # Code for loading txt

        ## Investigating different windows per EVAN ANDERSON
        #window_size = 10 # ~ 0.3 hit to f1 score


    if classifier_type == 'SVM':
        # Specifying Methods from Sequential optimization
        baseline_window = 32.5  # seconds - PULLED FROM NIKKI PAPER
        window_size = 15  # seconds - PULLED FROM NIKKI PAPER
        stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
        feature_reduction_type = 'ridge'  # - PULLED FROM NIKKI PAPER
        threshold = 10  # - PULLED FROM NIKKI PAPER
        baseline_methods_to_use = ['v0', 'v1', 'v2']  # - PULLED FROM NIKKI PAPER
        impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
        n_neighbors = 3  # - For imputation PULLED FROM NIKKI PAPER
        imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER

        ## Investigating different windows per EVAN ANDERSON
        #window_size = 8 # ~ 0.2 hit to f1 score


    if classifier_type == 'EGB':
        # Specifying Methods from Sequential optimization
        baseline_window = 46.25  # seconds - PULLED FROM NIKKI PAPER
        window_size = 12.5  # seconds - PULLED FROM NIKKI PAPER
        stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
        feature_reduction_type = 'lasso'  # - PULLED FROM NIKKI PAPER
        baseline_methods_to_use = ['v0', 'v1', 'v2','v5','v6','v7','v8']  # - PULLED FROM NIKKI PAPER
        imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
        impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
        n_neighbors = 3  # -For imputation PULLED FROM NIKKI PAPER
        # Code for loading txt

        ## Investigating different windows per EVAN ANDERSON
        #window_size = 8 # ~ 0.1 hit to f1 score


    if classifier_type == 'KNN':
        # Specifying Methods from Sequential optimization
        baseline_window = 32.5  # seconds - PULLED FROM NIKKI PAPER
        window_size = 15  # seconds - PULLED FROM NIKKI PAPER
        stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
        feature_reduction_type = 'performance'  # - PULLED FROM NIKKI PAPER
        baseline_methods_to_use = ['v0', 'v1', 'v2']  # - PULLED FROM NIKKI PAPER
        imbalance_type = 'ros' # - PULLED FROM NIKKI PAPER
        impute_type = 1 # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
        n_neighbors = 5 # -For imputation PULLED FROM NIKKI PAPER
        # Code for loading txt

        ## Investigating different windows per EVAN ANDERSON
        # window_size = 12 # ~ 0.1 hit to f1 score


    train_class = True
    class_weight_imb = None

        # Data Handling Options
    remove_NaN_trials = True # Planning to always remove NaN for offset sequence


        ## Model Parameters
    # model_type = ['noAFE', 'explicit']
    if 'noAFE' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                     'rawEEG', 'processedEEG', 'strain', 'demographics']
    if 'noAFE' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking','rawEEG', 'processedEEG']
            # feature_groups_to_analyze = ['ECG']
    if 'complete' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                         'rawEEG', 'processedEEG', 'strain', 'demographics']
        baseline_methods_to_use = ['v0', 'v1', 'v2', 'v5', 'v6']
    if 'complete' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'rawEEG', 'processedEEG', 'AFE']
        baseline_methods_to_use = ['v0', 'v1', 'v2', 'v5', 'v6']

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


        if 'complete' in model_type and 'explicit' in model_type:
            # Grab AFE / NonAFE condition indicator column
            condition_idx = all_features.index('condition')
            afe_indicator_column = features[:, condition_idx]

            # Impute raw (using mean) the value of the missing channels for each AFE condition
            gloc_data_reduced, features, features_phys, features_eeg = (
                eeg_condition_impute(gloc_data_reduced,
                                        all_features, all_features_phys, all_features_eeg, afe_indicator_column))

            # Set aside AFE / NonAFE condition indicator for now - to be incorporated back in later
            features = np.delete(features, condition_idx, axis=1)
            all_features = [stream for stream in all_features if stream != 'condition']

            # Add indicator back in for trial and row removal during 'data clean and prep' (will be taken back out)
            gloc_data_reduced[
                "AFE_indicator"] = afe_indicator_column  # Merge afe_indicators back into the predictor set
        if 'noAFE' in model_type:
            # Reduce Dataset based on AFE / nonAFE condition
            gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc = (
                afe_subset(model_type, gloc_data_reduced, all_features,
                           features, features_phys, features_ecg, features_eeg, gloc))

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

        # If complete condition, grab afe_indicator from cleaned dataframe
        if 'complete' in model_type and 'explicit' in model_type:
            afe_indicator_column = gloc_data_reduced["AFE_indicator"].to_numpy(dtype=np.float32).reshape(-1, 1)

        del gloc_data_reduced

        save_variables_to_folder(cache_folder, {
            'gloc': gloc,
            'trial_column': trial_column,
            'time_column': time_column,
            'event_validated_column': event_validated_column,
            'subject_column': subject_column,
            'nan_proportion_df': nan_proportion_df if remove_NaN_trials else None,
            'indicator_afe': afe_indicator_column if 'complete' in model_type and 'explicit' in model_type else None
        })

        print('reduce memory complete for', backstep, 'backstep')

    else:
        # Load from cache
        cached_vars = load_variables_from_folder(cache_folder, [
            'gloc',
            'trial_column',
            'time_column',
            'event_validated_column',
            'subject_column',
            'nan_proportion_df',
            'indicator_afe'
        ])

        # Unpack
        gloc = cached_vars['gloc']
        trial_column = cached_vars['trial_column']
        time_column = cached_vars['time_column']
        event_validated_column = cached_vars['event_validated_column']
        subject_column = cached_vars['subject_column']
        nan_proportion_df = cached_vars['nan_proportion_df']
        afe_indicator_column = cached_vars['indicator_afe']
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

        ################################# FEATURE GENERATION ########################################
    # Feature generation must run for each offset to window G-LOC labels
    if backstep == 0:
        # Generate features and windowed G-LOC labels
        y_gloc_labels, x_feature_matrix, all_features = feature_generation(
            time_start, offset, stride, window_size,
            combined_baseline, gloc, trial_column, time_column,
            combined_baseline_names, baseline_names_v0, baseline_v0,
            feature_groups_to_analyze
        )

        ################################################ Feature Reduction ################################################

        # Add windowed AFE indicator if required by model type
        if 'complete' in model_type and 'explicit' in model_type:
            afe_indicator_column_windowed, gloc_compare, _ = sliding_window_max(
                afe_indicator_column, trial_column, time_column, gloc,
                offset, stride, window_size, time_start
            )
            x_feature_matrix = np.hstack([x_feature_matrix, afe_indicator_column_windowed])
            all_features.append('AFE_indicator_windowed')

        # Convert feature matrix to DataFrame for column selection
        x_feature_matrix = pd.DataFrame(x_feature_matrix, columns=all_features)
        x_feature_matrix = x_feature_matrix[select_features]
        x_feature_matrix = x_feature_matrix.to_numpy()

        # Remove constant columns
        x_feature_matrix, select_features = remove_constant_columns(x_feature_matrix, select_features)

        ################################################ NaN Processing ################################################

        # Remove rows with NaNs and track removed indices
        y_gloc_labels, x_feature_matrix, all_features, removed_ind = process_NaN_temporal(
            y_gloc_labels, x_feature_matrix, select_features)

        save_variables_to_folder(cache_folder, {
            'removed_ind': removed_ind
        })

    else:
        y_gloc_labels, _, _, _, _, _ = sliding_window_mean_calc(
            time_start, offset, stride, window_size,
            combined_baseline, gloc, trial_column, time_column,
            combined_baseline_names
        )

        y_gloc_labels = np.vstack(list(y_gloc_labels.values())).astype(np.float32)
        # Trim G-LOC labels using previously removed row indices
        cached_vars2 = load_variables_from_folder(cache_folder, [
            # existing variables...
            'removed_ind'
        ])
        removed_ind = cached_vars2['removed_ind']
        mask = np.ones(len(y_gloc_labels), dtype=bool)
        mask[removed_ind] = False
        y_gloc_labels = y_gloc_labels[mask]


    print('generation complete for', backstep, 'backstep')  # debugging


    ######################################################## FEATURE Saving #######################################################
    # Fix X Matrix to the 0 offset case
    if backstep == 0:
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
        if os.path.getsize(all_features_name) == 0:
            raise ValueError(f"File {all_features_name} is empty. Cannot load features.")
        with open(all_features_name, 'rb') as file:
            all_features = pickle.load(file)

        if os.path.getsize(feature_matrix_name) == 0:
            raise ValueError(f"File {feature_matrix_name} is empty. Cannot load feature matrix.")
        with open(feature_matrix_name, 'rb') as file:
            x_feature_matrix = pickle.load(file)
        print('Files accessed for', backstep, 'backstep')  # debugging

    ################################################ Get Outputs Ready ############################################
        # Ensure x is 2D and y is 1D
    x_feature_return = (
        x_feature_matrix.to_numpy() if hasattr(x_feature_matrix, "to_numpy") else np.asarray(x_feature_matrix)
    )
    y_return = (
        y_gloc_labels.to_numpy().ravel() if hasattr(y_gloc_labels, "to_numpy") else np.ravel(y_gloc_labels)
    )


    # Function call end
      #return (x_feature_return, y_return)
    return (x_feature_return, y_return)

def plotting_offset_models(offset_ranges,accuracy_model,precision_model,recall_model,f1_model,specificity_model,gmean_model,classifier_name,model_type,subfolder2=None):

    if classifier_name == 'logreg':
        window_size = 12.5 # seconds - PULLED FROM NIKKI PAPER

        if subfolder2 is not None:
            ## Investigating different windows per EVAN ANDERSON
            window_size = 8 # ~ 0.1 hit to f1 score
    if classifier_name == 'RF':
        window_size = 7.5  # seconds - PULLED FROM NIKKI PAPER

        if subfolder2 is not None:
            ## Investigating different windows per EVAN ANDERSON
            window_size = 5 # ~ 0.1 hit to f1 score
    if classifier_name == 'LDA':
        window_size = 15  # seconds - PULLED FROM NIKKI PAPER

        if subfolder2 is not None:
            ## Investigating different windows per EVAN ANDERSON
            window_size = 10 # ~ 0.3 hit to f1 score
    if classifier_name == 'SVM':
        window_size = 15  # seconds - PULLED FROM NIKKI PAPER

        if subfolder2 is not None:
            ## Investigating different windows per EVAN ANDERSON
            window_size = 8 # ~ 0.2 hit to f1 score
    if classifier_name == 'EGB':
        window_size = 12.5  # seconds - PULLED FROM NIKKI PAPER

        if subfolder2 is not None:
            ## Investigating different windows per EVAN ANDERSON
            window_size = 8 # ~ 0.1 hit to f1 score
    if classifier_name == 'KNN':
        window_size = 15  # seconds - PULLED FROM NIKKI PAPER

        if subfolder2 is not None:
            ## Investigating different windows per EVAN ANDERSON
            window_size = 12 # ~ 0.1 hit to f1 score

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

        # Add vertical dashed line at window_size
        plt.axvline(x=window_size, color='black', linestyle='--', linewidth=1.5, label='Window Size')

        # fix the y axis
        # plt.ylim(0, 1)

        plt.title(f'{label} vs Offset')
        plt.xlabel('Offset [s]')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()
        plt.suptitle(f'Metrics Across Offsets — Classifier: {classifier_name} {get_model_subfolder(model_type)}', fontsize=16, fontweight='bold')

    plt.tight_layout()

    ############################ Save each metric matrix as a .pkl file ############################
    subfolder = get_model_subfolder(model_type)
    if subfolder2 is not None:
        results_folder = os.path.join('./prediction_model_metrics', subfolder,subfolder2)
        os.makedirs(results_folder, exist_ok=True)
    else:
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

def plot_metrics_from_cache(classifier_name, model_type):
    """
    Loads stored metric matrices and offset ranges from storage and plots them.
    Only requires classifier and model type.
    """

    # Define window size per classifier
    window_sizes = {
        'logreg': 12.5,
        'RF': 7.5,
        'LDA': 15,
        'SVM': 15,
        'EGB': 12.5,
        'KNN': 15
    }
    window_size = window_sizes.get(classifier_name, 12.5)

    # Locate results folder
    subfolder = get_model_subfolder(model_type)
    results_folder = os.path.join('./prediction_model_metrics', subfolder)

    # Load all metric matrices
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'g_mean']
    metric_colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    metric_data = {}

    for name in metric_names:
        filepath = os.path.join(results_folder, f"{name}_results_{classifier_name}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing metric file: {filepath}")
        with open(filepath, 'rb') as f:
            metric_data[name] = pickle.load(f)

    # Infer offset range from matrix shape
    num_offsets = metric_data['accuracy'].shape[0]
    offsets = np.arange(num_offsets)

    # Plotting
    plt.figure(figsize=(12, 8))

    def summarize_range(matrix):
        return np.mean(matrix, axis=1), np.min(matrix, axis=1), np.max(matrix, axis=1)

    for idx, (name, color) in enumerate(zip(metric_names, metric_colors), start=1):
        mean_vals, min_vals, max_vals = summarize_range(metric_data[name])
        plt.subplot(2, 3, idx)
        plt.plot(offsets, mean_vals, color=color, label=name.capitalize())
        plt.scatter(offsets, mean_vals, color=color, edgecolor='black', zorder=5)
        plt.fill_between(offsets, min_vals, max_vals, color='gray', alpha=0.3, label='Range (min–max)')
        plt.axvline(x=window_size, color='black', linestyle='--', linewidth=1.5, label='Window Size')

        plt.title(f'{name.capitalize()} vs Offset')
        plt.xlabel('Offset [s]')
        plt.ylabel(name.capitalize())
        plt.grid(True)
        plt.legend()

    if model_type == ['noAFE', 'phys']:
        model_name = 'noAFE phys'
    elif model_type == ['noAFE', 'phys+']:
        model_name = 'noAFE phys+'
    elif model_type == ['combined', 'phys']:
        model_name = 'Complete phys'
    elif model_type == ['combined', 'phys+']:
        model_name = 'Complete phys+'
    elif model_type == ['noAFE', 'explicit']:
        model_name = 'noAFE phys+'
    elif model_type == ['combined', 'implicit']:
        model_name = 'Complete phys'
    elif model_type == ['combined', 'explicit']:
        model_name = 'Complete phys+'
    elif model_type == ['noAFE', 'implicit']:
        model_name = 'noAFE phys'
    elif model_type == ['complete', 'implicit']:
        model_name = 'Complete phys'
    elif model_type == ['complete', 'explicit']:
        model_name = 'Complete phys+'


    plt.suptitle(f'Metrics Across Offsets — Classifier: {classifier_name} {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save plot
    plot_filename = f"metrics_plot_{classifier_name}_replot.png"
    plot_path = os.path.join(results_folder, plot_filename)
    plt.savefig(plot_path)
    print(f"Saved plot image to {plot_path}")

    plt.show()
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

def plot_f1_scores_across_classifiers(f1_score_dict, window_lengths, model_type_name, shared_plot=False):
    """
    Plots F1 score curves for multiple classifiers.

    Parameters:
    - f1_score_dict: dict mapping classifier → F1 score matrix (offsets × folds)
    - window_lengths: dict mapping classifier → window size (for vertical reference lines)
    - shared_plot: if True, plot all classifiers in one panel; otherwise use subplots
    """

    # Helper: summarize each offset by mean/min/max across folds
    def summarize_range(metric_matrix):
        return (
            np.mean(metric_matrix, axis=1),
            np.min(metric_matrix, axis=1),
            np.max(metric_matrix, axis=1)
        )

    # Color palette for consistent classifier coloring
    classifier_colors = {
        'logreg': '#440154',  # purple
        'RF': '#3b528b',      # blue
        'LDA': '#21918c',     # teal
        'KNN': '#5ec962',     # green
        'SVM': '#35b779',     # light green
        'EGB': '#fde725'      # yellow
    }

    # List of classifiers to plot
    classifier_names = list(f1_score_dict.keys())

    # ----------------------------------------------------------------------
    # SHARED PLOT MODE: all classifiers in one panel
    # ----------------------------------------------------------------------
    if shared_plot:
        plt.figure(figsize=(14, 10))
        ax = plt.gca()
        handles, labels = [], []

        for classifier_name in classifier_names:
            # Extract F1 matrix and compute summary curves
            f1_matrix = f1_score_dict[classifier_name]
            mean_vals, min_vals, max_vals = summarize_range(f1_matrix)
            offsets = np.arange(f1_matrix.shape[0])

            # Choose classifier color
            color = classifier_colors.get(classifier_name, 'gray')

            # Plot mean curve + scatter points
            h_mean, = ax.plot(offsets, mean_vals, color=color, label=f'{classifier_name} F1')
            ax.scatter(offsets, mean_vals, color=color, edgecolor='black', zorder=5)

            # (Optional range shading commented out)
            # ax.fill_between(offsets, min_vals, max_vals, color='gray', alpha=0.3)

            handles.extend([h_mean])
            labels.extend([f'{classifier_name} F1', f'{classifier_name} Range'])

        # Axis formatting
        ax.set_xlabel('Offset [s]')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0.5, 1.0)
        ax.set_xlim(0, 20)
        ax.grid(True)
        ax.legend(loc='lower right')

        plt.title(f'F1 Score Across Offsets for All Classifiers — {model_type_name}',
                  fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------------
    # SUBPLOT MODE: one panel per classifier, with window-length reference line
    # ----------------------------------------------------------------------
    else:
        plt.figure(figsize=(14, 10))

        for idx, classifier_name in enumerate(classifier_names, start=1):
            f1_matrix = f1_score_dict[classifier_name]
            mean_vals, min_vals, max_vals = summarize_range(f1_matrix)

            # Offsets inferred from number of rows
            offsets = np.arange(f1_matrix.shape[0])

            ax = plt.subplot(2, 3, idx)

            # Optional vertical line marking the classifier's window size
            classifier_window = window_lengths.get(classifier_name, None)
            if classifier_window is not None:
                ax.axvline(x=classifier_window, color='black', linestyle='--',
                           linewidth=1.5, label='Window Size')

            # Plot mean curve + scatter + min–max shading
            color = classifier_colors.get(classifier_name, 'gray')
            ax.plot(offsets, mean_vals, color=color, label='F1 Score')
            ax.scatter(offsets, mean_vals, color=color, edgecolor='black', zorder=5)
            ax.fill_between(offsets, min_vals, max_vals, color='gray', alpha=0.3,
                            label='Range (min–max)')

            # Axis formatting
            ax.set_title(f'F1 Score — {classifier_name}')
            ax.set_xlabel('Offset [s]')
            ax.set_ylabel('F1 Score')
            ax.set_ylim(0.5, 1.0)
            ax.grid(True)
            ax.legend()

        plt.suptitle(f'F1 Score Across Offsets for All Classifiers — {model_type_name}',
                     fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    return None

def get_model_subfolder(model_type):
    # Function to simplify some naming
    if model_type == ['noAFE', 'phys']:
        return 'Implicit noAFE'
    elif model_type == ['noAFE', 'phys+']:
        return 'Explicit noAFE'
    elif model_type == ['combined', 'phys']:
        return 'Implicit complete'
    elif model_type == ['combined', 'phys+']:
        return 'Explicit complete'
    elif model_type == ['noAFE', 'explicit']:
        return 'Explicit noAFE'
    elif model_type == ['combined', 'implicit']:
        return 'Implicit complete'
    elif model_type == ['combined', 'explicit']:
        return 'Explicit complete'
    elif model_type == ['noAFE', 'implicit']:
        return 'Implicit noAFE'
    elif model_type == ['complete', 'implicit']:
        return 'Implicit complete'
    elif model_type == ['complete', 'explicit']:
        return 'Explicit complete'
    else:
        raise ValueError(f"Unrecognized model_type: {model_type}")

def get_hyperparameters_from_json(classifier: str, model_type: str):
    # Function to load in median hyperparameters from a simple JSON
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    json_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type, f'median_hyperparameters_{classifier}.json')

    with open(json_path, 'r') as f:
        data = json.load(f)

    best_params = OrderedDict(data['best_params'])
    selected_features = data['selected_features']
    score = data['f1_score']
    foldID = data['fold_id']
    foldID = int(foldID)


    return best_params, selected_features, foldID, score

def sanitize_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    else:
        return obj


def get_median_hyperparameters(classifier: str, model_type: str):
    # Function to find the median hyperparameters given the 10 folds of CV are completed
    # NOTE: Files should be stored as defined in this function

    # Resolve project root directory
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Will store (fold_id, f1_score) for all 10 folds
    fold_scores = []

    # ------------------------------------------------------------
    # Step 1: Load F1 scores from each fold's HPO summary
    # ------------------------------------------------------------
    for fold_id in range(10):
        fold_str = str(fold_id)

        # Build path to the fold summary depending on classifier type
        if classifier == 'RF':
            perf_path = os.path.join(BASE_DIR, 'PerformanceSave', 'CrossValidation', 'rf_hpo', model_type, fold_str,
                                     'FoldSummary.pkl')
        elif classifier == 'KNN':
            perf_path = os.path.join(BASE_DIR, 'PerformanceSave', 'CrossValidation', 'KNN_hpo', model_type, fold_str,
                                     'FoldSummary.pkl')
        elif classifier == 'EGB':
            perf_path = os.path.join(BASE_DIR, 'PerformanceSave', 'CrossValidation', 'EGB_hpo', model_type, fold_str,
                                     'FoldSummary.pkl')
        elif classifier == 'logreg':
            perf_path = os.path.join(BASE_DIR, 'PerformanceSave', 'CrossValidation', 'logreg_hpo', model_type, fold_str,
                                     'FoldSummary.pkl')
        elif classifier == 'SVM':
            perf_path = os.path.join(BASE_DIR, 'PerformanceSave', 'CrossValidation', 'SVM_hpo', model_type, fold_str,
                                     'FoldSummary.pkl')
        elif classifier == 'LDA':
            perf_path = os.path.join(BASE_DIR, 'PerformanceSave', 'CrossValidation', 'LDA_hpo', model_type, fold_str,
                                     'FoldSummary.pkl')
        else:
            raise ValueError(f"Unsupported classifier: {classifier}")

        # Load fold summary and extract F1 score
        perf_dict = joblib.load(perf_path)
        perf_df = perf_dict[fold_str]
        f1_score = perf_df['f1-score'].iloc[0]

        # Store fold result
        fold_scores.append({'fold_id': fold_str, 'f1_score': f1_score})

    # ------------------------------------------------------------
    # Step 2: Identify the median-performing fold
    # ------------------------------------------------------------
    fold_scores.sort(key=lambda x: x['f1_score'])  # sort by F1
    median_fold = fold_scores[len(fold_scores) // 2]
    median_fold_id = median_fold['fold_id']
    median_f1 = median_fold['f1_score']

    # ------------------------------------------------------------
    # Step 3: Load model + selected features for the median fold
    # ------------------------------------------------------------
    if classifier == 'RF':
        model_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type, median_fold_id, 'random_forest_model.pkl')
    elif classifier == 'KNN':
        model_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type, median_fold_id, 'KNN_model.pkl')
    elif classifier == 'EGB':
        model_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type, median_fold_id, 'ensemble_model.pkl')
    elif classifier == 'logreg':
        model_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type, median_fold_id, 'logistic_regression_model.pkl')
    elif classifier == 'SVM':
        model_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type, median_fold_id, 'SVM_model.pkl')
    elif classifier == 'LDA':
        model_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type, median_fold_id, 'LDA_model.pkl')

    # Default feature path
    features_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type, median_fold_id,
                                 f'SelectedFeatures{classifier}.pkl')

    # Logistic regression uses a different filename convention
    if classifier == 'logreg':
        features_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type, median_fold_id,
                                     'SelectedFeaturesLR.pkl')

    # Load model and selected features
    model = joblib.load(model_path)
    selected_features = joblib.load(features_path)

    # ------------------------------------------------------------
    # Step 4: Package and save median fold results
    # ------------------------------------------------------------
    median_result = {
        'fold_id': median_fold_id,
        'f1_score': median_f1,
        'best_params': model.best_params_,
        'selected_features': selected_features
    }

    output_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type, f'median_hyperparameters_{classifier}.json')
    with open(output_path, 'w') as f:
        json.dump(sanitize_for_json(median_result), f, indent=4)

    print(f"Saved median fold data to {output_path}")
    return None




