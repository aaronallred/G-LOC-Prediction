import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

class DataManager:
    IMPLICIT_FEATURE_GROUPS = {"ECG", "BR", "temp", "eyetracking", "rawEEG"} # All physiological signals
    EXPLICIT_FEATURE_GROUPS = IMPLICIT_FEATURE_GROUPS.union({"AFE", "G", "processedEEG", "demographics", "strain"}) # All physiological and participant info
    COMPLETE_FEATURE_GROUPS = {"AFE"} # Includes nonAFE and AFE trials

    def __init__(self, file_path):
        self.file_path = file_path

    def get_data(
            model_type,
            num_splits,
            kfold_ID,
            impute_path,
            subject_to_analyze = None,
            trial_to_analyze = None,
            impute_type = 1,
            n_neighbors = 4,
            baseline_window = 32.5,
            datafolder = "../data/",
            analysis_type = 2,
            remove_NaN_trials = True,
            save_impute = True,
            load_impute = True
        ):
        """
            Function Loads Raw data and Prepares the Predictor / Target Sets for Advanced Classifiers

            Parameters:
                model_type (List[str])   --> A list of model type characteristics i.e. 'Complete/nonAFE' and 'explicit/implicit'
                num_splits (int)         --> The number of splits of the training and test data set for K-fold CV. Nominally set to 10
                kfold_ID (int)           --> The id of the train / test split. If num split is 10, kfold is [0, 9]
                impute_path (str)        --> The path to save/load the imputed data pickle file
                subject_to_analyze (str) --> If analysis type is 1, the participant number to analyze
                trial_to_analyze (str)   --> If analysis type is 0, the trial number to analyze
                impute_type (int)        --> Sets the imputation method. Since Sequential, use impute type equal to 1 (input raw data)
                n_neighbors (int)        --> Sets the KNN imputation # of neighbors. Since Sequential, use 4 neighbors
                baseline_window (float)  --> Sets the baseline window duration. Since Sequential, use 32.5 s
                datafolder (str)         --> location of AFRL provided data from the experiment: raw data that is processed
                analysis_type (int)      --> Determines what data to use. 2: all data. 1: one participant (set in function), 0: one trial
                remove_NaN_trials (bool) --> removes trials that have an all NaN sensor instead of imputing an all NaN array
                save_impute (bool)       --> dumps data post-impute into a pickle file (this is convenient as imputation has large compute)
                load_impute (bool)       --> checks if there is a saved impute pickle and loads it if available

            Returns:
                x_train, x_test --> Array of predictors. rows are over time, columns are over predictors
                    *Additional Note: The last column is the trial ID. Needed for the advanced classifier slicing
                y_train, y_test --> An array of binary labels corresponding to GLOC or no GLOC (1 or 0, respectively)
                    *Additional Note: Not shifted by horizon for advanced classifiers (happens in the data loader inside)
                all_features    --> List of all feature names in x_train and x_test
        """
        # Determining features to use for the dataset
        feature_groups_to_analyze = set()
        if model_type[0] == "Complete":
            feature_groups_to_analyze = DataManager.COMPLETE_FEATURE_GROUPS
        # noAFE restriction is imposed later when filtering out rows

        if model_type[1] == "Implicit":
            feature_groups_to_analyze = feature_groups_to_analyze.union(DataManager.IMPLICIT_FEATURE_GROUPS)
        elif model_type[1] == "Explicit":
            feature_groups_to_analyze = feature_groups_to_analyze.union(DataManager.EXPLICIT_FEATURE_GROUPS)

        # NOTE:
        # AFE indicator is required for EEG imputation in complete models,
        # but is only included as a predictive feature for explicit models.

        # Set baseline characteristics. Depends on model type
        if model_type[0] == "noAFE":
            baseline_methods_to_use = ["v0", "v1", "v2", "v5", "v6", "v7", "v8"]
        else:
            baseline_methods_to_use = ["v0", "v1", "v2", "v5", "v6"]

    def get_ml_splitter(self, n_splits=5):
        return StratifiedKFold(n_splits=n_splits)

    def get_dl_splitter(self, n_splits=5):
        # Maybe DL uses simple KFold or specific time-series split
        return KFold(n_splits=n_splits)

    def save_split_indices(self, train_idx, val_idx, path):
        # Save indices to recreate the split later
        pass