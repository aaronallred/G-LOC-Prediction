import numpy as np
import os
import joblib  # For saving the model
from numpy import ravel
import time

from sklearn.ensemble import RandomForestClassifier

from GLOC_data_processing import *
from scripts.GLOC_classifier import stratified_kfold_split, classify_logistic_regression, classify_random_forest, \
    classify_lda, classify_svm, classify_knn, classify_ensemble_with_gradboost
from scripts.imbalance_techniques import resample_ros
from scripts.temporal_functions import plotting_offset_models, data_with_prediction, \
    plot_f1_scores_across_classifiers, plot_saved_offset_models, get_model_subfolder, get_hyperparameters
import pickle

from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split

from feature_selection import feature_selection_lasso
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

######## This is the new main file that will be used for the prediction side of things to loop and through each
######## classifier and get performance. Will only have 2 other scripts that it calls from

# The general structure will be a loop that calls in data_testing but will only do certain sections more than once.
# 0.04 is the smallest step size we can have (this is 25hz step)
# Does not work to have a .5 second step size.

offset_ranges = np.arange(0,2,1) #FOR FULL RUNS: (0,21,1)
data_rate = 25 # (hz)
preference = 3 # Which section of the code do we want to run
random_state = 42
class_weight_imb = None


if preference == 3:
    # This preference section will load in pkl files and train/validate models according to offset data
    # NOTE: This code sequence will throw an error as it starts about number of cores using to process

    # Can adjust this as needed to specify what classifiers we want to test
    # options are: SVM , EGB, KNN, logreg, RF , LDA
    classifiers_to_test = ['RF']

    for m in range(len(classifiers_to_test)):
        # Initialize the arrays and class type
        start_time = time.time()
        classifier = classifiers_to_test[m]
        model_type = ['complete','explicit'] # specify model type to run

        num_kfold = 3  # Number of kfolds we will use for validation, FOR FULL RUNS 10
        accuracy_model = np.zeros((len(offset_ranges), num_kfold))
        precision_model = np.zeros((len(offset_ranges), num_kfold))
        recall_model = np.zeros((len(offset_ranges), num_kfold))
        f1_model = np.zeros((len(offset_ranges), num_kfold))
        specificity_model = np.zeros((len(offset_ranges), num_kfold))
        g_mean_model = np.zeros((len(offset_ranges), num_kfold))

        # Preallocate some folders
        subfolder_name = get_model_subfolder(model_type)
        save_folder = os.path.join("../prediction_models", subfolder_name)
        os.makedirs(save_folder, exist_ok=True)

        # Grab Optimized (median) hyperparameters for classifier
        # hyperparameters = get_hyperparameters(classifier, get_model_subfolder(model_type))

        print('Starting loop for ', classifier)  # debugging

        for i in range(len(offset_ranges)):

            # Call prediction function to obtain x and y. All methods are implemented in this function
            (x,y) = data_with_prediction(offset_ranges[i], data_rate, classifier,model_type)

            # Start nested loop for each fold to be evaluated
            for k in range(num_kfold):
                # Call models and collect performance data
                print('Splitting for Kfold k of', k)  # debugging
                x_train, x_test, y_train, y_test = stratified_kfold_split(ravel(y),x,num_kfold, k,random_state)

                print('Classifier call for k of', k)  # debugging
                if classifier == 'RF':
                    (accuracy, precision, recall, f1, tree_depth, specificity, g_mean) = classify_random_forest(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                           save_folder,model_name="random_forest_model_temporal.pkl",retrain=True)
                if classifier == 'LDA':
                    (accuracy, precision, recall, f1, specificity, g_mean) = classify_lda(x_train, x_test, y_train, y_test, random_state,
                                                                                          save_folder,
                                                                                          model_name="LDA_model_temporal.pkl",
                                                                                          retrain=True)
                if classifier == 'logreg':
                    (accuracy, precision, recall, f1, specificity, g_mean) = classify_logistic_regression(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                           save_folder,model_name="logistic_regression_model_temporal.pkl",retrain=True)
                if classifier == 'SVM':
                    (accuracy, precision, recall, f1, specificity, g_mean) = classify_svm(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                           save_folder,model_name="svm_model_temporal.pkl",retrain=True)
                if classifier == 'KNN':
                    # Implement Imbalance Sampling Technique ONLY FOR KNN. Need to think of better code implementation for this
                    ros_x_train, ros_y_train = resample_ros(x_train, y_train, random_state)
                    (accuracy, precision, recall, f1, specificity, g_mean) = classify_knn(ros_x_train, x_test, ros_y_train, y_test, random_state,
                           save_folder,model_name="knn_model_temporal.pkl",retrain=True)
                if classifier == 'EGB':
                    (accuracy, precision, recall, f1, specificity, g_mean) = classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test, random_state,
                                     save_folder,model_name="ensemble_model_temporal.pkl", retrain = True)

                # Storing each value in arrays
                accuracy_model[i, k] = accuracy
                precision_model[i, k] = precision
                recall_model[i, k] = recall
                f1_model[i, k] = f1
                specificity_model[i, k] = specificity
                g_mean_model[i, k] = g_mean

            print('Success for offset of', offset_ranges[i], 'using classifier:', classifier)

            # Clear variables at end of loop before reassignment on next iteration of loop
            del y_train, y_test, x_train, x_test

        # Plotting the results, function also saves the data to a folder for one particular model, outside the loop
        plotting_offset_models(offset_ranges, accuracy_model, precision_model, recall_model, f1_model, specificity_model, g_mean_model,classifier,model_type)

            # # Print performance metrics
            # print(f"\nLogistic Regression Performance Metrics for offset of {offset_ranges[i]}:")
            # print("Accuracy: ", accuracy)
            # print("Precision: ", precision)
            # print("Recall: ", recall)
            # print("F1 Score: ", f1)
            # print("Specificity: ", specificity)
            # print("G-Mean: ", g_mean)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Total time for classifier '{classifier}': {elapsed:.2f} seconds")


if preference == 4:
    # This code preference will work with already derived/stored data as needed.

    # Folder where results are stored
    results_folder = './prediction_model_results'

    # List of classifier names
    # just classifiers_to_test
    classifiers_to_test = ['rf', 'LDA', 'SVM', 'EGB', 'KNN', 'logreg']

    # Load each F1 score file into a dictionary
    f1_score_dict = {}

    for clf in classifiers_to_test:
        filename = f"f1_score_results_{clf}.pkl"
        filepath = os.path.join(results_folder, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                f1_score_dict[clf] = pickle.load(f) # store f1 scores into a dictionary
            print(f"Loaded F1 scores for {clf}")
        else:
            print(f"File not found: {filepath}")

    # Now do window_lengths for each classifier type
    window_lengths = {
        'logreg': 12.5, # FROM NIKKI PAPER
        'rf': 7.5, # FROM NIKKI PAPER
        'LDA': 15, # FROM NIKKI PAPER
        'SVM': 15, # FROM NIKKI PAPER
        'EGB': 12.5, # FROM NIKKI PAPER
        'KNN': 15, # FROM NIKKI PAPER
    }

    plot_f1_scores_across_classifiers(offset_ranges, f1_score_dict, window_lengths)


if preference == 5:
    # Use this preference section to plot all metrics of specific saved models.
    plot_saved_offset_models('RF')







