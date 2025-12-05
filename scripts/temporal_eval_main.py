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
    plot_f1_scores_across_classifiers, get_model_subfolder, \
    data_with_prediction_verification, get_median_hyperparameters, get_hyperparameters_from_json, \
    plot_metrics_from_cache
import pickle

from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split

from feature_selection import feature_selection_lasso
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

######## This is the new main file that will be used for the prediction side of things to loop and through each
######## classifier and get performance

# The general structure will be a loop that calls in data_testing but will only do certain sections more than once.
# 0.04 is the smallest step size we can have (this is 25hz step)
# Does not work to have a .5 second step size.

offset_ranges = np.arange(0,21,1) #FOR FULL RUNS: (0,21,1)
data_rate = 25 # (hz)
preference = 7 # Which section of the code do we want to run
random_state = 42
class_weight_imb = None


if preference == 3:
    # This preference section will load in pkl files and train/validate models according to offset data

    # Can adjust this as needed to specify what classifiers we want to test
    # options are: SVM , EGB, KNN, logreg, RF , LDA
    classifiers_to_test = ['KNN']

    for m in range(len(classifiers_to_test)):
        # Initialize the arrays and class type
        start_time = time.time()
        classifier = classifiers_to_test[m]
        model_type = ['noAFE','explicit'] # specify model type to run


        num_kfold = 10  # Number of kfolds we will use for validation, FOR FULL RUNS 10
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
        get_median_hyperparameters(classifier, get_model_subfolder(model_type))
        hyperparameters, select_features, foldID_check, score_check = get_hyperparameters_from_json(classifier,
                                                                                                    get_model_subfolder(
                                                                                                        model_type))

        print('Score check at 0 ', score_check)  # debugging
        print('Starting loop for ', classifier)  # debugging

        for i in range(len(offset_ranges)):

            # Call prediction function to obtain x and y. All methods are implemented in this function
            (x,y) = data_with_prediction(offset_ranges[i], data_rate, classifier,model_type, select_features)

            # Start nested loop for each fold to be evaluated
            for k in range(num_kfold):
                # Call models and collect performance data
                print('Splitting for Kfold k of', k)  # debugging
                x_train, x_test, y_train, y_test = stratified_kfold_split(ravel(y),x,num_kfold, k,random_state)

                if classifier == 'RF':
                    (accuracy, precision, recall, f1, tree_depth, specificity, g_mean) = classify_random_forest(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                           save_folder,model_name="random_forest_model_temporal.pkl",retrain=False,temporal=True,best_params=hyperparameters)
                if classifier == 'LDA':
                    (accuracy, precision, recall, f1, specificity, g_mean) = classify_lda(x_train, x_test, y_train, y_test, random_state,
                                                                                          save_folder,
                                                                                          model_name="LDA_model_temporal.pkl",
                                                                                          retrain=False,temporal=True,best_params=hyperparameters)
                if classifier == 'logreg':
                    (accuracy, precision, recall, f1, specificity, g_mean) = classify_logistic_regression(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                           save_folder,model_name="logistic_regression_model_temporal.pkl",retrain=False,temporal=True,best_params=hyperparameters)
                if classifier == 'SVM':
                    (accuracy, precision, recall, f1, specificity, g_mean) = classify_svm(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                           save_folder,model_name="svm_model_temporal.pkl",retrain=False,temporal=True,best_params=hyperparameters)
                if classifier == 'KNN':
                    # Implement Imbalance Sampling Technique ONLY FOR KNN. Need to think of better code implementation for this
                    ros_x_train, ros_y_train = resample_ros(x_train, y_train, random_state)
                    (accuracy, precision, recall, f1, specificity, g_mean) = classify_knn(x_train, x_test, y_train, y_test, random_state,
                           save_folder,model_name="knn_model_temporal.pkl",retrain=False,temporal=True,best_params=hyperparameters)
                if classifier == 'EGB':
                    (accuracy, precision, recall, f1, specificity, g_mean) = classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test, random_state,
                                     save_folder,model_name="ensemble_model_temporal.pkl", retrain=False,temporal=True,best_params=hyperparameters)

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

    # List of classifiers to evaluate
    classifiers_to_test = ['RF', 'LDA', 'SVM', 'KNN', 'logreg']
    model_type = ['complete', 'explicit']  # specify model type to run

    # Load each F1 score file into a dictionary
    f1_score_dict = {}
    model_subfolder = get_model_subfolder(model_type)
    results_folder = os.path.join('./prediction_model_metrics', model_subfolder)

    for clf in classifiers_to_test:
        filename = f"f1_score_results_{clf}.pkl"
        filepath = os.path.join(results_folder, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                f1_score_dict[clf] = pickle.load(f)
            print(f"Loaded F1 scores for {clf}")
        else:
            print(f"File not found: {filepath}")

    # Define window lengths per classifier
    window_lengths = {
        'logreg': 12.5,
        'RF': 7.5,
        'LDA': 15,
        'SVM': 15,
        'EGB': 12.5,
        'KNN': 15,
    }

    plot_f1_scores_across_classifiers(f1_score_dict, window_lengths, get_model_subfolder(model_type))

if preference == 5:
    # Use this preference section to plot all metrics of specific saved models.
    model_type = ['complete', 'explicit']  # specify model type to run
    plot_metrics_from_cache('SVM', model_type)

if preference == 6:
    # This part of the code will be used to verify pipeline with given F1 scores and hyperparameters
    classifiers_to_test = ['LDA']
    classifier = classifiers_to_test[0]
    model_type = ['noAFE', 'explicit']  # specify model type to run

    num_kfold = 10  # For validation RUNS = 10
    kfold_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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
    hyperparameters, select_features, foldID_check,score_check = get_hyperparameters_from_json(classifier, get_model_subfolder(model_type))

    offset_ranges = (0,1,1)
    print('Starting loop for ', classifier)  # debugging

    for i in range(len(offset_ranges)):

        # Call models and collect performance data

        # Call prediction function to obtain x and y. All methods are implemented in this function
        x_train, x_test, y_train, y_test = data_with_prediction_verification(offset_ranges[i], data_rate, classifier, model_type, int(foldID_check), select_features)


        print('Classifier call for k of', int(foldID_check))  # debugging
        if classifier == 'RF':
            (accuracy, precision, recall, f1, tree_depth, specificity, g_mean) = classify_random_forest(x_train,
                                                                                                        x_test,
                                                                                                        y_train,
                                                                                                        y_test,
                                                                                                        class_weight_imb,
                                                                                                        random_state,
                                                                                                        save_folder,
                                                                                                        model_name="random_forest_model_temporal.pkl",
                                                                                                        retrain=False,temporal=True,best_params=hyperparameters)
        if classifier == 'LDA':
            (accuracy, precision, recall, f1, specificity, g_mean) = classify_lda(x_train, x_test, y_train, y_test,
                                                                                  random_state,
                                                                                  save_folder,
                                                                                  model_name="LDA_model_temporal.pkl",
                                                                                  retrain=False,temporal=True,best_params=hyperparameters)
        if classifier == 'logreg':
            (accuracy, precision, recall, f1, specificity, g_mean) = classify_logistic_regression(x_train, x_test,
                                                                                                  y_train, y_test,
                                                                                                  class_weight_imb,
                                                                                                  random_state,
                                                                                                  save_folder,
                                                                                                  model_name="logistic_regression_model_temporal.pkl",
                                                                                                  retrain=False,temporal=True,best_params=hyperparameters)
        if classifier == 'SVM':
            (accuracy, precision, recall, f1, specificity, g_mean) = classify_svm(x_train, x_test, y_train, y_test,
                                                                                  class_weight_imb, random_state,
                                                                                  save_folder,
                                                                                  model_name="svm_model_temporal.pkl",
                                                                                  retrain=False,temporal=True,best_params=hyperparameters)
        if classifier == 'KNN':
            # Implement Imbalance Sampling Technique ONLY FOR KNN. Need to think of better code implementation for this
            ros_x_train, ros_y_train = resample_ros(x_train, y_train, random_state)
            (accuracy, precision, recall, f1, specificity, g_mean) = classify_knn(ros_x_train, x_test, ros_y_train,
                                                                                  y_test, random_state,
                                                                                  save_folder,
                                                                                  model_name="knn_model_temporal.pkl",
                                                                                  retrain=False,temporal=True,best_params=hyperparameters)
        if classifier == 'EGB':
            (accuracy, precision, recall, f1, specificity, g_mean) = classify_ensemble_with_gradboost(x_train,
                                                                                                      x_test,
                                                                                                      y_train,
                                                                                                      y_test,
                                                                                                      random_state,
                                                                                                      save_folder,
                                                                                                      model_name="ensemble_model_temporal.pkl",
                                                                                                      retrain=False,temporal=True,best_params=hyperparameters)

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

if preference == 7:
    # Comparing feature space of different models.
    # Ensure that classifiers have hyperparameters already saved into JSON
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2

    offset_ranges = np.arange(0, 41, 1)  # FOR FULL RUNS: (0,21,1)

    # Step 1: Load selected features from JSON
    model_type = ['noAFE', 'explicit']

    # LDA
    classifier = 'LDA'
    _, select_featuresLDA, _, _ = get_hyperparameters_from_json(classifier, get_model_subfolder(model_type))

    # KNN
    classifier = 'KNN'
    _, select_featuresKNN, _, _ = get_hyperparameters_from_json(classifier, get_model_subfolder(model_type))

    # Step 2: Compute shared and unique features
    lda_set = set(select_featuresLDA)
    knn_set = set(select_featuresKNN)
    shared_features = lda_set & knn_set
    lda_unique = lda_set - knn_set
    knn_unique = knn_set - lda_set

    # Step 3: Visualize overlap
    plt.figure(figsize=(6, 5))
    venn2([knn_set, lda_set], set_labels=('KNN', 'LDA'))
    plt.title('Feature Overlap: KNN vs LDA')
    plt.show()

    # Step 4: Print feature breakdown
    print(f"\n Shared features ({len(shared_features)}):")
    for feat in sorted(shared_features):
        print(f"  - {feat}")

    print(f"\n Features selected only by LDA ({len(lda_unique)}):")
    for feat in sorted(lda_unique):
        print(f"  - {feat}")

    print(f"\n Features selected only by KNN ({len(knn_unique)}):")
    for feat in sorted(knn_unique):
        print(f"  - {feat}")

    # Step 5: Filter KNN and LDA features to shared only
    select_featuresKNN = [f for f in select_featuresKNN if f in shared_features]
    select_featuresLDA = [f for f in select_featuresLDA if f in shared_features]

    # Step 6: Forecasting evaluation using shared features
    classifiers_to_test = ['KNN']  # or ['KNN', 'LDA'] to test both

    for classifier in classifiers_to_test:
        start_time = time.time()
        model_type = ['noAFE', 'explicit']
        num_kfold = 10

        accuracy_model = np.zeros((len(offset_ranges), num_kfold))
        precision_model = np.zeros((len(offset_ranges), num_kfold))
        recall_model = np.zeros((len(offset_ranges), num_kfold))
        f1_model = np.zeros((len(offset_ranges), num_kfold))
        specificity_model = np.zeros((len(offset_ranges), num_kfold))
        g_mean_model = np.zeros((len(offset_ranges), num_kfold))

        subfolder_name = get_model_subfolder(model_type)
        save_folder = os.path.join("../prediction_models_featureSHARE", subfolder_name)
        os.makedirs(save_folder, exist_ok=True)

        get_median_hyperparameters(classifier, subfolder_name)
        hyperparameters, select_features, foldID_check, score_check = get_hyperparameters_from_json(classifier,
                                                                                                    subfolder_name)

        # Override features with shared ones
        if classifier == 'KNN':
            select_features = select_featuresKNN
        elif classifier == 'LDA':
            select_features = select_featuresLDA

        print('Score check at 0 ', score_check)
        print('Starting loop for ', classifier)

        for i in range(len(offset_ranges)):
            x, y = data_with_prediction(offset_ranges[i], data_rate, classifier, model_type, select_features)

            for k in range(num_kfold):
                print('Splitting for Kfold k of', k)
                x_train, x_test, y_train, y_test = stratified_kfold_split(ravel(y), x, num_kfold, k, random_state)

                if classifier == 'RF':
                    accuracy, precision, recall, f1, tree_depth, specificity, g_mean = classify_random_forest(
                        x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                        save_folder, model_name="random_forest_model_temporal.pkl", retrain=False,
                        temporal=True, best_params=hyperparameters)
                elif classifier == 'LDA':
                    accuracy, precision, recall, f1, specificity, g_mean = classify_lda(
                        x_train, x_test, y_train, y_test, random_state,
                        save_folder, model_name="LDA_model_temporal.pkl", retrain=False,
                        temporal=True, best_params=hyperparameters)
                elif classifier == 'KNN':
                    ros_x_train, ros_y_train = resample_ros(x_train, y_train, random_state)
                    accuracy, precision, recall, f1, specificity, g_mean = classify_knn(
                        x_train, x_test, y_train, y_test, random_state,
                        save_folder, model_name="knn_model_temporal.pkl", retrain=False,
                        temporal=True, best_params=hyperparameters)
                elif classifier == 'logreg':
                    accuracy, precision, recall, f1, specificity, g_mean = classify_logistic_regression(
                        x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                        save_folder, model_name="logistic_regression_model_temporal.pkl", retrain=False,
                        temporal=True, best_params=hyperparameters)
                elif classifier == 'SVM':
                    accuracy, precision, recall, f1, specificity, g_mean = classify_svm(
                        x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                        save_folder, model_name="svm_model_temporal.pkl", retrain=False,
                        temporal=True, best_params=hyperparameters)
                elif classifier == 'EGB':
                    accuracy, precision, recall, f1, specificity, g_mean = classify_ensemble_with_gradboost(
                        x_train, x_test, y_train, y_test, random_state,
                        save_folder, model_name="ensemble_model_temporal.pkl", retrain=False,
                        temporal=True, best_params=hyperparameters)

                accuracy_model[i, k] = accuracy
                precision_model[i, k] = precision
                recall_model[i, k] = recall
                f1_model[i, k] = f1
                specificity_model[i, k] = specificity
                g_mean_model[i, k] = g_mean

            print('Success for offset of', offset_ranges[i], 'using classifier:', classifier)
            del y_train, y_test, x_train, x_test

        subfolder2 = 'FeatureShare' # Adding folder level for looking at performance when limiting features
        plotting_offset_models(offset_ranges, accuracy_model, precision_model, recall_model,
                               f1_model, specificity_model, g_mean_model, classifier, model_type,subfolder2)

        end_time = time.time()
        print(f"Total time for classifier '{classifier}': {end_time - start_time:.2f} seconds")






