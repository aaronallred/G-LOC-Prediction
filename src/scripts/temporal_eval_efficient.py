from imputation import *
from baseline_methods import *
from GLOC_classifier import *
from GLOC_visualization import *
from imbalance_techniques import *
import pickle
import time
import os
from sklearn.preprocessing import StandardScaler

from LogRegTS_supporting import lrts_binary_class_load
from NAM_supporting import nam_binary_class_load
from LSTM_supporting import lstm_binary_class_load
from TCN_supporting import tcn_binary_class_load
from Transformer_supporting import transformer_class_load


def main_loop(kfold_ID, num_splits, param_path, impute_path, horizons, save_folder, classifier_type, model_type):
    start_time = time.time()

    ################################################### USER INPUTS  ###################################################
    ## Data Folder Location
    datafolder = '../data/'

    # Random State | 42 - Debug mode
    random_state = 42

    train_class = True  # not yet set up to test and not train (always trains)
    class_weight_imb = 'balanced'

    # Data Handling Options
    remove_NaN_trials = True
    impute_type = 1
    n_neighbors = 4

    save_impute = True   # save post impute?
    load_impute = True   # skip impute and load from file?

    ## Model Parameters
    if 'noAFE' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                     'rawEEG', 'processedEEG', 'demographics', 'strain']

    if 'noAFE' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'rawEEG']

    if 'complete' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'rawEEG']

    if 'complete' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                     'rawEEG', 'processedEEG', 'demographics', 'strain']

    # Set baseline characteristics. Depends on model type
    baseline_window = 32.5  # seconds
    if 'noAFE' in model_type:
        baseline_methods_to_use = ['v0', 'v1', 'v2', 'v5', 'v6', 'v7', 'v8']
    else:
        baseline_methods_to_use = ['v0', 'v1', 'v2', 'v5', 'v6']

    # Analysis Type 2 loads in all participants and all trials
    analysis_type = 2
    # Subject & Trial Information (only need to adjust this if doing analysis type 0 or 1)
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
    if 'complete' not in model_type:
        gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc = (
            afe_subset(model_type, gloc_data_reduced, all_features,
                       features, features_phys, features_ecg, features_eeg, gloc))

    ############################################# EEG Specific Imputation #############################################
    if 'complete' in model_type:
        # Compute AFE / NonAFE condition indicator column
        condition_idx = all_features.index('condition')
        afe_indicator_column = features[:, condition_idx]

        # Impute (using mean) the value of the missing channels for each AFE condition
        gloc_data_reduced, features, features_phys, features_eeg = (
            eeg_condition_impute(gloc_data_reduced,
                                 all_features, all_features_phys, all_features_eeg, afe_indicator_column))

        # Set aside AFE / NonAFE condition indicator for now - to be incorporated back in later
        features = np.delete(features, condition_idx, axis=1)
        all_features = [stream for stream in all_features if stream != 'condition']

        gloc_data_reduced["AFE_indicator"] = afe_indicator_column  # Merge afe_indicators back into the predictor set

    ############################################### DATA CLEAN AND PREP ###############################################
    """
       Optional handling of raw NaN data, depending on 'remove_NaN_trials' 'impute_type' <= 1
    """

    ### Remove full trials with NaN
    if remove_NaN_trials:
        gloc_data_reduced, features, features_phys, features_ecg, features_eeg, gloc, nan_proportion_df = (
            remove_all_nan_trials(gloc_data_reduced, all_features,
                                  features, features_phys, features_ecg, features_eeg, gloc, verbose=True))

    ################################################## REDUCE MEMORY ##################################################

    # Grab columns from gloc_data_reduced and remove gloc_data_reduced variable from memory
    trial_column = gloc_data_reduced['trial_id']
    trial_ints = convert_to_unique_ordered_integers(trial_column)
    time_column = gloc_data_reduced['Time (s)']
    event_validated_column = gloc_data_reduced['event_validated']
    subject_column = gloc_data_reduced['subject']
    if "complete" in model_type:
        afe_indicator_column = gloc_data_reduced["AFE_indicator"].to_numpy(dtype=np.float32).reshape(-1, 1)

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

        # Load or compute imputed features
        # NOTE: impute_path is PROVIDED by caller; do not overwrite it.
        if load_impute and os.path.exists(impute_path):
            with open(impute_path, 'rb') as f:
                features = pickle.load(f)
            print(f"Loaded imputed data from {impute_path}")
        else:
            features = faster_knn_impute_train_test(features, train_ind, test_ind, n_neighbors)

            if save_impute:
                os.makedirs(os.path.dirname(impute_path), exist_ok=True)
                with open(impute_path, 'wb') as f:
                    pickle.dump(features, f)
                print(f"Saved imputed data to {impute_path}")

        # Calculate new sub-feature arrays
        phys_indices = [i for i, feature in enumerate(all_features) if (feature in all_features_phys)]
        features_phys = features[:, phys_indices]

        ecg_indices = [i for i, feature in enumerate(all_features) if (feature in all_features_ecg)]
        features_ecg = features[:, ecg_indices]

        eeg_indices = [i for i, feature in enumerate(all_features) if (feature in all_features_eeg)]
        features_eeg = features[:, eeg_indices]

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
    x_feature_matrix = x_feature_matrix[:, ue_indices]
    trial_ints = convert_to_unique_ordered_integers(trial_column)

    x_feature_matrix = np.hstack([x_feature_matrix, trial_ints])
    y_gloc_labels = gloc

    all_features = combined_baseline_names
    all_features = [all_features[i] for i in ue_indices]

    ############################################# FEATURE CLEAN AND PREP ##############################################
    """
          Optional handling of raw NaN data
    """

    # Remove constant columns (typically no constant columns)
    x_feature_matrix, all_features = remove_constant_columns(x_feature_matrix, all_features)

    # Add back in as 2nd to last column (needs to be 2nd to last for advanced - could be last for traditional)
    if "complete" in model_type:
        x_feature_matrix = np.hstack([
            x_feature_matrix[:, :-1],
            afe_indicator_column,
            x_feature_matrix[:, -1:]
        ])

    # List-wise deletion or clean any residual NaNs
    if impute_type == 2 or impute_type == 1:
        # Remove rows with NaN (temporary solution-should replace with other method eventually)
        y_gloc_labels_noNaN, x_feature_matrix_noNaN, all_features, trials_noNaN = process_NaN(y_gloc_labels,
                                                                                              x_feature_matrix,
                                                                                              all_features, trial_ints)
    else:
        y_gloc_labels_noNaN, x_feature_matrix_noNaN, trials_noNaN = y_gloc_labels, x_feature_matrix, trial_ints

    ################################################ TRAIN/TEST SPLIT  ################################################
    """
          Split data into training/test for optimization loop of sequential optimization framework.
    """

    # Training/Test Split
    x_train, x_test, y_train, y_test, _, _ = groupedtrial_kfold_split(
        y_gloc_labels_noNaN, x_feature_matrix_noNaN, trials_noNaN, num_splits, kfold_ID)

    # Grab trials as separate
    x_train_trials = x_train[:, -1].reshape(-1, 1)
    x_train = x_train[:, :-1]
    x_test_trials = x_test[:, -1].reshape(-1, 1)
    x_test = x_test[:, :-1]

    # And standardize based on training data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Add indices back as final column
    x_train = np.hstack([x_train, x_train_trials])
    x_test = np.hstack([x_test, x_test_trials])

    ################################################ MACHINE LEARNING ################################################

    horizon_performance_summary = dict()

    # Transformer
    if classifier_type == 'Trans' or classifier_type == 'all':
        for horizon in horizons:
            accuracy, precision, recall, f1, specificity, g_mean = (
                transformer_class_load(
                    x_train, x_test, y_train, y_test, horizon, class_weight_imb, random_state,
                    all_features, param_path=param_path, save_folder=save_folder
                )
            )

            single_run = single_classifier_performance_summary(
                accuracy, precision, recall, f1, specificity, g_mean, ['Trans']
            )

            # Preserve plotting contract
            single_run['horizon'] = horizon
            single_run['fold'] = kfold_ID

            method_key = f"fold{str(kfold_ID)}_h{str(horizon)}"
            horizon_performance_summary[method_key] = single_run

    # Time Series (Lagged Features) Logistic Regression
    if classifier_type == 'LogRegTS' or classifier_type == 'all':
        for horizon in horizons:
            accuracy, precision, recall, f1, specificity, g_mean = (
                lrts_binary_class_load(
                    x_train, x_test, y_train, y_test, horizon, class_weight_imb, random_state,
                    all_features, param_path=param_path, save_folder=save_folder
                )
            )

            single_run = single_classifier_performance_summary(
                accuracy, precision, recall, f1, specificity, g_mean, ['LogRegTS']
            )

            single_run['horizon'] = horizon
            single_run['fold'] = kfold_ID

            method_key = f"fold{str(kfold_ID)}_h{str(horizon)}"
            horizon_performance_summary[method_key] = single_run

    # Time Series (Lagged Features) Neural Additive Model
    if classifier_type == 'NAM' or classifier_type == 'all':
        for horizon in horizons:
            accuracy, precision, recall, f1, specificity, g_mean = (
                nam_binary_class_load(
                    x_train, x_test, y_train, y_test, horizon, class_weight_imb, random_state,
                    all_features, param_path=param_path, save_folder=save_folder
                )
            )

            single_run = single_classifier_performance_summary(
                accuracy, precision, recall, f1, specificity, g_mean, ['NAM']
            )

            single_run['horizon'] = horizon
            single_run['fold'] = kfold_ID

            method_key = f"fold{str(kfold_ID)}_h{str(horizon)}"
            horizon_performance_summary[method_key] = single_run

    # Long Short Term Memory RNN
    if classifier_type == 'LSTM' or classifier_type == 'all':
        for horizon in horizons:
            accuracy, precision, recall, f1, specificity, g_mean = (
                lstm_binary_class_load(
                    x_train, x_test, y_train, y_test, horizon, class_weight_imb, random_state,
                    all_features, param_path=param_path, save_folder=save_folder
                )
            )

            single_run = single_classifier_performance_summary(
                accuracy, precision, recall, f1, specificity, g_mean, ['LSTM']
            )

            single_run['horizon'] = horizon
            single_run['fold'] = kfold_ID

            method_key = f"fold{str(kfold_ID)}_h{str(horizon)}"
            horizon_performance_summary[method_key] = single_run

    # Transformer
    if classifier_type == 'Trans' or classifier_type == 'all':
        for horizon in horizons:
            accuracy, precision, recall, f1, specificity, g_mean = (
                transformer_class_load(
                    x_train, x_test, y_train, y_test, horizon, class_weight_imb, random_state,
                    all_features, param_path=param_path, save_folder=save_folder
                )
            )

            single_run = single_classifier_performance_summary(
                accuracy, precision, recall, f1, specificity, g_mean, ['Trans']
            )

            single_run['horizon'] = horizon
            single_run['fold'] = kfold_ID

            method_key = f"fold{str(kfold_ID)}_h{str(horizon)}"
            horizon_performance_summary[method_key] = single_run

    # Temporal Convolutional Network
    if classifier_type == 'TCN' or classifier_type == 'all':
        for horizon in horizons:
            accuracy, precision, recall, f1, specificity, g_mean = (
                tcn_binary_class_load(
                    x_train, x_test, y_train, y_test, horizon, class_weight_imb, random_state,
                    all_features, param_path=param_path, save_folder=save_folder
                )
            )

            single_run = single_classifier_performance_summary(
                accuracy, precision, recall, f1, specificity, g_mean, ['TCN']
            )

            single_run['horizon'] = horizon
            single_run['fold'] = kfold_ID

            method_key = f"fold{str(kfold_ID)}_h{str(horizon)}"
            horizon_performance_summary[method_key] = single_run

    duration = time.time() - start_time
    print(duration)

    return horizon_performance_summary


if __name__ == "__main__":

    ## Classifier | Pick 'LogRegTS', 'LSTM', 'TCN', 'Trans', or 'all'
    classifier_type = 'NAM'

    # Model type (determines data subset) | Pick 'noAFE/complete' or 'implicit/explicit'. Temporal is just 'explicit'
    model_type = ['noAFE', 'explicit']

    # Naming run and save location for summary  files
    run_name = 'NAMAllFolds'
    # Folder name where models and performance metrics will be saved
    subFolder = "TemporalPrediction_ExplicitnonAFE"

    # Root directory for loading hyperparams & post-imputation data
    root_load_path = "../ModelSave/CV/Explicit_nonAFE_final"

    # Needed for proper debugging of CUDA errors, normally commented out
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # For loading model parameters from previously saved CV run (specify kfold_ID_Load for median performing fold)
    if classifier_type == 'Trans':
        if 'complete' in model_type:
            kfold_ID_Load = 4  # Median performer for Transformer Explicit Complete
        else:
            kfold_ID_Load = 7
    elif classifier_type == 'TCN':
        if 'complete' in model_type:
            kfold_ID_Load = 4  # Median performer for TCN Explicit Complete
        else:
            kfold_ID_Load = 5
    elif classifier_type == 'LSTM':
        if 'complete' in model_type:
            kfold_ID_Load = 4  # Median performer for LSTM Explicit Complete
        else:
            kfold_ID_Load = 3
    elif classifier_type == 'LogRegTS':
        if 'complete' in model_type:
            kfold_ID_Load = 3  # Median performer for LogRegTS Explicit Complete
        else:
            kfold_ID_Load = 4
    elif classifier_type == 'NAM':
        if 'complete' in model_type:
            kfold_ID_Load = 4  # Median performer for LogRegTS Explicit Complete
        else:
            kfold_ID_Load = 2
    else:
        raise ValueError(
            f"Unsupported classifier_type '{classifier_type}'." "Need to specify median fold to load hyperparameters.")
    param_path = os.path.join(root_load_path, str(kfold_ID_Load))

    # Define horizon range set
    horizons = list(range(0, 501, 25))

    # Define CV range set
    kfold_IDs = list(range(10))


    # Test set splits for 10-fold Model Validation (never changes)
    num_splits = 10

    # Make Performance Save Folder
    summary_loc = os.path.join("../PerformanceSave",subFolder)
    os.makedirs(summary_loc, exist_ok=True)

    # Pre-Allocate Performance Summary Dictionary (same structure as before)
    horizon_performance_summary = dict()

    # Loop through folds (main_loop handles horizons internally)
    for kfold_ID in kfold_IDs:

        # Model Save Folder
        model_save_folder = os.path.join("../ModelSave/", subFolder, run_name, str(kfold_ID))
        os.makedirs(model_save_folder, exist_ok=True)

        # For loading imputation (if saved) - caller provides full file path
        impute_path = os.path.join(root_load_path, str(kfold_ID), "imputed_data.pkl")

        # Run main loop (returns dict keyed by fold[X]_h[Y])
        fold_results = main_loop(kfold_ID, num_splits, param_path, impute_path, horizons, model_save_folder,
                                 classifier_type, model_type)

        # Merge into master dict and preserve per-horizon saving behavior
        for method_key, single_run in fold_results.items():
            horizon_performance_summary[method_key] = single_run

            save_folder = os.path.join(summary_loc, run_name)
            save_file = f'FoldSummary_{method_key}.pkl'
            save_path = os.path.join(save_folder, save_file)

            # Ensure the save folder exists
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            with open(save_path, 'wb') as file:
                pickle.dump(horizon_performance_summary, file)

    # Save pkl summary (same as before)
    save_folder = os.path.join(summary_loc, run_name)
    save_file = 'AllHorizons.pkl'
    save_path = os.path.join(save_folder, save_file)

    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(save_path, 'wb') as file:
        pickle.dump(horizon_performance_summary, file)

    plot_metrics_over_offsets(horizon_performance_summary)