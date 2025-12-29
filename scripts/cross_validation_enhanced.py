import pickle
import time
import os
from matplotlib import pyplot as plt
import pandas as pd

from GLOC_data_pipeline import load_and_prepare_data_advanced
from GLOC_data_processing import save_metrics_to_csv
from GLOC_classifier import single_classifier_performance_summary
from GLOC_visualization import plot_cross_val_sp

from LogRegTS_supporting import lrts_binary_class
from NAM_supporting import nam_binary_class
from LSTM_supporting import lstm_binary_class
from Transformer_supporting import transformer_class
from TCN_supporting import tcn_binary_class



# Module-level variables
CLASSIFIER_LOADERS = {
        "LogRegTS": lrts_binary_class,
        "NAM": nam_binary_class,
        "LSTM": lstm_binary_class,
        "TCN": tcn_binary_class,
        "Trans": transformer_class,
    }
DATA_FOLDER = "../data/"
DEFAULT_RANDOM_STATE = 42
DEFAULT_BASELINE_WINDOW = 32.5
DEFAULT_IMPUTE_TYPE = 1
DEFAULT_IMPUTE_NEIGHBORS = 4
DEFAULT_ANALYSIS_TYPE = 2
DEFAULT_CLASS_BALANCE = 'balanced'
DEFAULT_REMOVE_NAN_TRIALS = True


def main_loop(model_type,
              kfold_ID,
              num_splits,
              impute_path,
              save_folder,
              classifier_type,
              random_state=DEFAULT_RANDOM_STATE,
              class_weight_imb=DEFAULT_CLASS_BALANCE):
    """
    Function loops Through and Evaluations advanced classifiers

    Args:
        model_type --> A list of model type characteristics i.e. 'Complete/nonAFE' and 'explicit/implicit'
        kfold_ID   --> The id of the train / test split. If num split is 10, kfold is [0, 9]
        num_splits --> The number of splits of the training and test data set for K-fold CV. Nominally set to 10
        impute_path --> The path to the saved post-impute data (if saved)
        save_folder --> The path to where model results are saved
        classifier_type --> The type of classifier 'LogRegTS', 'NAM', 'LSTM', 'Trans', 'TCN', of 'all' to run all

        random_state --> For stochastic processes, set to 42
        class_weight_imb --> Set to 'balanced' for oversampling minor class at a ratio of occurrence. Alt is None

    Mediating Args (these are set inside the function for data processing and do not change)
        impute_type--> Sets the imputation method. Since Sequential, use impute type equal to 1 (input raw data)
        n_neighbors--> Sets the KNN imputation # of neighbors. Since Sequential, use 4 neighbors
        baseline_window --> Sets the baseline window duration. Since Sequential, use 32.5 s
        datafolder --> location of AFRL provided data from the experiment: raw data that is processed
        remove_NaN_trials=True --> removes trials that have an all NaN sensor instead of imputing an all NaN array
        save_impute --> dumps data post-impute into a pickle file (this is convenient as imputation has large compute)
        load_impute --> checks if there is a saved impute pickle and loads it if available

    Returns:
        performance_summary --> dataframe of performance metrics, for classifier, fold, and horizon
    """


    ############################################# LOAD AND PREPARE DATA ##############################################
    start_time = time.time()

    x_train, x_test, y_train, y_test, all_features = load_and_prepare_data_advanced(
            model_type=model_type,
            num_splits=num_splits,
            kfold_ID=kfold_ID,
            impute_path=impute_path,
            impute_type=DEFAULT_IMPUTE_TYPE,
            n_neighbors=DEFAULT_IMPUTE_NEIGHBORS,
            baseline_window=DEFAULT_BASELINE_WINDOW,
            datafolder=DATA_FOLDER,
            analysis_type=DEFAULT_ANALYSIS_TYPE,
            remove_NaN_trials=DEFAULT_REMOVE_NAN_TRIALS,
            save_impute=True,
            load_impute=True,
    )

    duration = time.time() - start_time
    print(f"Data Preparation Duration: {duration}")

    ################################################ MACHINE LEARNING  ################################################

    # List to hold fold results
    summaries = []

    # Determine which classifiers to loop through
    classifiers_to_run = (
        CLASSIFIER_LOADERS.keys()
        if classifier_type == "all"
        else [classifier_type]
    )

    # Loop through all classifiers
    start_time = time.time()

    for clf in classifiers_to_run:
        loader = CLASSIFIER_LOADERS[clf]

        kwargs = dict(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            class_weight_imb=class_weight_imb,
            random_state=random_state,
            all_features=all_features,
            save_folder=save_folder,
        )

        # Run classifier with above arguments
        accuracy, precision, recall, f1, specificity, g_mean = loader(**kwargs)

        # Get dataframe of results
        single_run = single_classifier_performance_summary(
            accuracy, precision, recall, f1, specificity, g_mean, [clf]
        )

        # Augment dataframe with additional parameters for later analysis
        single_run["fold"] = kfold_ID
        single_run["classifier"] = clf

        # Save results from this fold and append the list
        save_metrics_to_csv(single_run, save_folder)
        summaries.append(single_run)

    duration = time.time() - start_time
    print(f"Model Evaluation Duration: {duration}")

    performance_summary = pd.concat(summaries)

    return performance_summary


if __name__ == "__main__":

    """
        This script runs through folds and train/test splits. Each fold performs an Optuna hyperparameter search
    """

    ## Classifier | Pick 'LogRegTS', 'LSTM', 'TCN', 'Trans', or 'all'
    classifier_type = 'all'

    # Model type (determines data subset) | Pick 'noAFE/complete' or 'implicit/explicit'. Temporal is just 'explicit'
    model_type = ['complete', 'implicit']

    # Folder name where models and performance metrics will be saved or loaded
    subFolder = "CrossValidation"

    # Naming run and save location for summary  files
    run_name = "Implicit_Complete_final"


    # Needed for proper debugging of CUDA errors, normally commented out
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Define folds to loop through
    kfold_IDs = list(range(0, 10))

    # Test set splits for 10-fold Model Validation (doesn't typically change)
    num_splits = 10

    # Root directory for saving hyperparams & post-imputation data
    root_load_path = os.path.join("../ModelSave/CV/",run_name)

    # Make Performance Save Folder
    summary_loc = os.path.join("../PerformanceSave",subFolder)
    os.makedirs(summary_loc, exist_ok=True)

    # Pre-Allocate Performance Summary Dictionary (same structure as before)
    kfold_performance_summary = dict()

    # Loop through folds (main_loop handles horizons internally)
    for kfold_ID in kfold_IDs:

        # Model Save Folder
        model_save_folder = os.path.join(root_load_path, str(kfold_ID))
        os.makedirs(model_save_folder, exist_ok=True)

        # For loading imputation (if saved) - caller provides full file path
        impute_path = os.path.join(model_save_folder, "imputed_data.pkl")

        # Run main loop (returns dictionary of results)
        method_key = str(kfold_ID)
        kfold_performance_summary[method_key] = main_loop(kfold_ID=kfold_ID,
                                 num_splits=num_splits,
                                 impute_path=impute_path,
                                 save_folder=model_save_folder,
                                 classifier_type=classifier_type,
                                 model_type=model_type)


        # Save as the code loops through folds to ensure progress is saved if the code faults
        save_folder = os.path.join(summary_loc, run_name)
        save_file = f'FoldSummary_{kfold_ID}.pkl'
        save_path = os.path.join(save_folder, save_file)

        # Ensure the save folder exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        with open(save_path, 'wb') as file:
            pickle.dump(kfold_performance_summary, file)

    # Save pkl summary
    save_folder = os.path.join(summary_loc, run_name)
    save_file = 'CrossValidation.pkl'
    save_path = os.path.join(save_folder, save_file)

    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(save_path, 'wb') as file:
        pickle.dump(kfold_performance_summary, file)

    plot_cross_val_sp(kfold_performance_summary)
    plt.show()