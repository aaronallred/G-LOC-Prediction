import pickle
import os
from sklearn.preprocessing import StandardScaler

from .imputation import *
from .baseline_methods import *
from .GLOC_classifier import *
from .imbalance_techniques import *

def load_and_prepare_data_advanced(
        model_type,
        num_splits,
        kfold_ID,
        impute_path,
        impute_type=1,
        n_neighbors=4,
        baseline_window = 32.5,
        datafolder = "../data/",
        analysis_type = 2,
        remove_NaN_trials=True,
        save_impute=True,
        load_impute=True,

    ):
    """
    Function Loads Raw data and Prepares the Predictor / Target Sets for Advanced Classifiers

    Args:
        model_type --> A list of model type characteristics i.e. 'Complete/nonAFE' and 'explicit/implicit'
        num_splits --> The number of splits of the training and test data set for K-fold CV. Nominally set to 10
        kfold_ID   --> The id of the train / test split. If num split is 10, kfold is [0, 9]
        impute_type--> Sets the imputation method. Since Sequential, use impute type equal to 1 (input raw data)
        n_neighbors--> Sets the KNN imputation # of neighbors. Since Sequential, use 4 neighbors
        baseline_window --> Sets the baseline window duration. Since Sequential, use 32.5 s
        datafolder --> location of AFRL provided data from the experiment: raw data that is processed
        analysis_type --> Determines what data to use. 2: all data. 1: one participant (set in function), 0: one trial
        remove_NaN_trials=True --> removes trials that have an all NaN sensor instead of imputing an all NaN array
        save_impute --> dumps data post-impute into a pickle file (this is convenient as imputation has large compute)
        load_impute --> checks if there is a saved impute pickle and loads it if available

    Returns:
        x_train, x_test --> Aray of predictors. rows are over time, columns are over predictors
            Adtl Note: The last column is the trial ID. Needed for the advanced classifier slicing
        y_train, y_test --> An array of binary labels corresponding to GLOC or no GLOC (1 or 0, respectively)
            Adtl Note: Not shifted by horizon for advanced classifiers (happens in the data loader inside)
        all_features  --> List of features in x_train and x_test
    """


    ################################################### USER INPUTS  ###################################################
    # Subject & Trial Information (only need to adjust this if doing analysis type 0 or 1)
    subject_to_analyze = '01'
    trial_to_analyze = '02'

    ## Model Parameters
    if 'noAFE' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                     'rawEEG', 'processedEEG', 'demographics', 'strain']

    if 'noAFE' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'rawEEG']

    if 'complete' in model_type and 'implicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'rawEEG', 'AFE'] # AFE is removed downstream

    if 'complete' in model_type and 'explicit' in model_type:
        feature_groups_to_analyze = ['ECG', 'BR', 'temp', 'eyetracking', 'AFE', 'G',
                                     'rawEEG', 'processedEEG', 'demographics', 'strain']

    # NOTE:
    # AFE indicator is required for EEG imputation in complete models,
    # but is only included as a predictive feature for explicit models.

    # Set baseline characteristics. Depends on model type
    if 'noAFE' in model_type:
        baseline_methods_to_use = ['v0', 'v1', 'v2', 'v5', 'v6', 'v7', 'v8']
    else:
        baseline_methods_to_use = ['v0', 'v1', 'v2', 'v5', 'v6']


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
    ####
    #  Note: This runs for 'complete' models, but because we are only using shared/overlapping EEG features for the
    #      'complete' case, this block doesn't do anything. Imputation occurs only for non-shared EEG features are used.
    #       This block requires 'AFE' to be an
    ####
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

    # Add back in as 2nd to last column for explicit only
    # (needs to be 2nd to last for advanced pipeline - could be last for traditional)
    if "complete" in model_type and "explicit" in model_type:
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


    return x_train, x_test, y_train, y_test, all_features