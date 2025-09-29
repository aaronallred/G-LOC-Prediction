import numpy as np
import os
import joblib  # For saving the model
from numpy import ravel
import time
from sklearn.linear_model import LogisticRegression
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
from scripts.temporal_functions import plotting_offset_models, cross_validation_split, data_with_prediction, \
    plot_f1_scores_across_classifiers
import pickle

from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split

from temporal_functions import rf_call
from temporal_functions import lda_call
from temporal_functions import ensemble_call
from temporal_functions import knn_call
from temporal_functions import lr_call
from temporal_functions import svm_call
from feature_selection import feature_selection_lasso
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

start_time = time.time()
######## This is the new main file that will be used for the prediction side of things to loop and through each
######## classifier and get performance. Will only have 2 other scripts that it calls from

# The general structure will be a loop that calls in data_testing but will only do certain sections more than once.
# 0.04 is the smallest step size we can have (this is 25hz step)
# Does not work to have a .5 second step size.
########## offset_ranges = np.arange(0,20,0.04)

offset_ranges = np.arange(0,20,1)
data_rate = 25 # (hz)
preference = 3 # Which section of the code do we want to run
random_state = 42

if preference == 1:
    # This preference section will load in all the data and clean it for various offset ranges.
    # Code is time INTENSIVE
    for i in range(len(offset_ranges)):
        (y,x,all_feats) = data_with_prediction(offset_ranges[i],data_rate) # Call function
        filename = f"y_offset_{offset_ranges[i]}.pkl" # Create unique filename for y matrix of this offset
        with open(filename, 'wb') as file:
            pickle.dump(y, file)

    with open("all_features.pkl", 'wb') as file:
        pickle.dump(all_feats, file)

    with open("x_features.pkl", 'wb') as file:
        pickle.dump(x, file)


if preference == 2:
    # This preference section will do feature selection on the non offset y data to scale down the features.
    # This preference only needs to be run once. Takes about an hour to do the lasso
    # The reduction results in a feature matrix only about 53 columns wide vs the some 30 thousand it was before?

    # Load in the data
    with open('all_features.pkl', 'rb') as file:
        all_features = pickle.load(file)
    with open('x_features.pkl', 'rb') as file:
        x_feature_matrix = pickle.load(file)

    # Call function to get y if there was zero offset, to do lasso on
    (yy,x,allall_feats) = data_with_prediction(0,data_rate)

    # Call lasso to get reduced x matrix and reassign to file.
    (x_reduced,x_meaningless, selected_features) = feature_selection_lasso(x,x,yy,allall_feats, random_state)

    # Save as new file to be used in new function calls
    with open("x_features_lasso.pkl", 'wb') as file:
        pickle.dump(x_reduced, file)

if preference == 3:
    # This preference section will load in pkl files and train/validate models according to offset data
    # NOTE: This code sequence will throw an error as it starts about number of cores using to process

    # Can adjust this as needed to specify what classifiers we want to test
    # options are: SVM , EGB, KNN, logreg, RF , LDA
    classifiers_to_test = ['SVM', 'EGB', 'KNN','logreg','rf', 'LDA']

    for m in range(len(classifiers_to_test)):
        # Initialize the arrays and class type
        classifier = classifiers_to_test[m]

        num_kfold = 5  # Number of kfolds we will use for validation
        accuracy_model = np.zeros((len(offset_ranges), num_kfold))
        precision_model = np.zeros((len(offset_ranges), num_kfold))
        recall_model = np.zeros((len(offset_ranges), num_kfold))
        f1_model = np.zeros((len(offset_ranges), num_kfold))
        specificity_model = np.zeros((len(offset_ranges), num_kfold))
        g_mean_model = np.zeros((len(offset_ranges), num_kfold))

        print('Starting loop for ', classifier)  # debugging

        for i in range(len(offset_ranges)):

            # Call prediction function to obtain x and y. All methods are implemented in this function
            (x,y) = data_with_prediction(offset_ranges[i], data_rate, classifier)
            # Start nested loop for each fold to be evaluated
            for k in range(num_kfold):
                # Call models and collect performance data
                print('Cross validating for k of', k)  # debugging
                y_train, y_test, x_train, x_test = cross_validation_split(ravel(y),x,k, num_kfold)

                print('Classifier call for k of', k)  # debugging
                if classifier == 'rf':
                    (accuracy, precision, recall, f1, specificity, g_mean) = rf_call(y_train, y_test, x_train, x_test)
                if classifier == 'LDA':
                    (accuracy, precision, recall, f1, specificity, g_mean) = lda_call(y_train, y_test, x_train, x_test)
                if classifier == 'logreg':
                    (accuracy, precision, recall, f1, specificity, g_mean) = lr_call(y_train, y_test, x_train, x_test)
                if classifier == 'SVM':
                    (accuracy, precision, recall, f1, specificity, g_mean) = svm_call(y_train, y_test, x_train, x_test)
                if classifier == 'KNN':
                    (accuracy, precision, recall, f1, specificity, g_mean) = knn_call(y_train, y_test, x_train, x_test)
                if classifier == 'EGB':
                    (accuracy, precision, recall, f1, specificity, g_mean) = ensemble_call(y_train, y_test, x_train, x_test)

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
        plotting_offset_models(offset_ranges, accuracy_model, precision_model, recall_model, f1_model, specificity_model, g_mean_model,classifier)

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









