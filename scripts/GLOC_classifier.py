import numpy as np
import os
import joblib  # For saving the model
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
from GLOC_data_processing import *


import optuna
from LSTM_supporting import *
from GLOC_visualization import create_confusion_matrix, roc_curve_plot, prediction_time_plot
from sklearn.utils.class_weight import compute_class_weight

def check_event_columns(gloc_data):
    """
    This function was created to understand if ALOC data was available. It will check which
    unique values exist in the event and event_validated columns.
    """

    vals_event = gloc_data['event'].unique()
    vals_event_validated = gloc_data['event_validated'].unique()

    return vals_event, vals_event_validated

# Training Test Split
# USING RANDOM STATE = 42
def pre_classification_training_test_split(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio, random_state):
    """
    This function splits the X and y matrix into training and test matrix.
    """

    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(x_feature_matrix_noNaN, y_gloc_labels_noNaN,
                                                                    test_size=(1 - training_ratio), random_state=random_state, stratify = y_gloc_labels_noNaN)

    return x_train, x_test, y_train, y_test

# Training Test Split Using Stratified K-Fold
# USING RANDOM STATE = 42
def stratified_kfold_split(Y, X, num_splits, kfold_ID, random_state=42):
    """
    This function splits the X and y matrix into training and test matrix.
    """

    # Stratified K-Fold setup
    # Use random state to ensure repeatability across runs and classifiers
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)

    # Safety check to ensure that kfold_ID is within the fold indices
    n_folds = skf.get_n_splits()
    if kfold_ID < 0 or kfold_ID >= n_folds:
        raise ValueError(f"Fold index {kfold_ID} out of range (must be between 0 and {n_folds - 1})")

    # Grab train and test indices given the skf generator format for a specific kfold_ID
    train_index, test_index = next(islice(skf.split(X, Y), kfold_ID, kfold_ID + 1))

    # Extract the corresponding data for the given kfold_ID
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]


    return x_train, x_test, y_train, y_test


# Logistic Regression Classifier
def classify_logistic_regression(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                                 save_folder="../ModelSave",model_name="logistic_regression_model.pkl",retrain=True):
    """
    This function fits and assesses performance of a logistic regression ML classifier for the data
    specified. Within this function, a separate confusion matrix function is called. Additional
    plotting capabilities include logistic regression visualization.
    """
    # feature_subset = feature_selection_lasso(x_train, y_train, all_features)

    if retrain:
        # Use Default Parameters & Fit Model
        #weights = {0: 1.0, 1: 10.0}
        logreg = LogisticRegression(class_weight = class_weight_imb, random_state=random_state, max_iter=1000).fit(x_train, np.ravel(y_train))
    else:
        model_path = os.path.join(save_folder, model_name)
        logreg = joblib.load(model_path)

    # Predict
    label_predictions = logreg.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nLogistic Regression Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Create Confusion Matrix
    create_confusion_matrix(np.ravel(y_test), label_predictions, 'Log. Reg.')

    # Plot 0/1 Classification
    # for i in range(np.size(sliding_window_mean,1)):
    #     x_test_squeeze = x_testing[:,i].squeeze()
    #     y_test_squeeze = y_testing.squeeze()
    #     sns.scatterplot(x=x_test_squeeze, y=label_predictions, hue=y_test_squeeze)
    #     plt.title('GLOC Classification- Logistic Regression')
    #     plt.xlabel(all_features[i])
    #     plt.ylabel('Predicted')
    #     plt.legend()
    #     plt.show()
    #
    #     # Plot Logistic Regression
    #     fig, ax = plt.subplots()
    #     y_prob = logreg.predict_proba(x_testing)
    #     sns.scatterplot(x=x_test_squeeze, y=y_prob[:, 1], hue=y_test_squeeze)
    #     plt.title('GLOC Classification- Logistic Regression')
    #     plt.xlabel(all_features[i])
    #     plt.ylabel('Predicted')
    #     plt.show()

    # Save model
    if retrain:
        save_model_weights(logreg, save_folder, model_name)

    return accuracy, precision, recall, f1, specificity, g_mean

# Random Forest Classifier
def classify_random_forest(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                           save_folder="../ModelSave",model_name="random_forest_model.pkl",retrain=True):
    """
    This function fits and assesses performance of a random forest ML classifier for the data
    specified. Within this function, a separate confusion matrix function is called. Additional
    visualization capabilities include random forest visualization.
    """

    if retrain:
        # Use Default Parameters & Fit Model
        rf = RandomForestClassifier(class_weight= class_weight_imb, random_state=random_state).fit(x_train, np.ravel(y_train))
    else:
        model_path = os.path.join(save_folder, model_name)
        rf = joblib.load(model_path)

    # Predict
    label_predictions = rf.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nRandom Forest Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Find Tree Depth
    tree_depth = [estimator.get_depth() for estimator in rf.estimators_]

    # Visualize Decision Tree
    # fn = all_features
    # cn = ['No GLOC', 'GLOC']
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    # plot_tree(rf.estimators_[0],
    #               feature_names=fn,
    #               class_names=cn,
    #               filled=True)
    # plt.show()

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Random Forest')

    # Save model
    if retrain:
        save_model_weights(rf, save_folder, model_name)

    return accuracy, precision, recall, f1, tree_depth, specificity, g_mean

# Linear Discriminant Analysis
def classify_lda(x_train, x_test, y_train, y_test, random_state,
                 save_folder="../ModelSave",model_name="LDA_model.pkl",retrain=True):
    """
    This function fits and assesses performance of a linear discriminant analysis ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    if retrain:
        # Use Default Parameters & Fit Model
        lda = LinearDiscriminantAnalysis().fit(x_train, np.ravel(y_train))
    else:
        model_path = os.path.join(save_folder, model_name)
        lda = joblib.load(model_path)

    # Predict
    label_predictions = lda.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nLinear Discriminant Analysis Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Linear Discriminant Analysis')

    # Save model
    if retrain:
        save_model_weights(lda, save_folder, model_name)

    return accuracy, precision, recall, f1, specificity, g_mean

# k Nearest Neighbors
def classify_knn(x_train, x_test, y_train, y_test, random_state,
                 save_folder="../ModelSave",model_name="KNN_model.pkl",retrain=True):
    """
    This function fits and assesses performance of a K Nearest Neighbors ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    if retrain:
        # Use Default Parameters & Fit Model
        neigh = KNeighborsClassifier().fit(x_train, np.ravel(y_train))
    else:
        model_path = os.path.join(save_folder, model_name)
        neigh = joblib.load(model_path)

    # Predict
    label_predictions = neigh.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nKNN Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'kNN')

    # Save model
    if retrain:
        save_model_weights(neigh, save_folder, model_name)

    return accuracy, precision, recall, f1, specificity, g_mean

# Support Vector Machine
def classify_svm(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                 save_folder="../ModelSave",model_name="svm_model.pkl", retrain = True):
    """
    This function fits and assesses performance of a Support Vector Machine ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    if retrain:
        # Use Default Parameters & Fit Model
        svm_class = svm.SVC(class_weight=class_weight_imb).fit(x_train, np.ravel(y_train))
    else:
        model_path = os.path.join(save_folder, model_name)
        svm_class = joblib.load(model_path)

    # Predict
    label_predictions = svm_class.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nSVM Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Support Vector Machine')

    # Save model
    if retrain:
        save_model_weights(svm_class, save_folder, model_name)

    return accuracy, precision, recall, f1, specificity, g_mean

# Ensemble Learner with Gradient Boost
def classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test, random_state,
                                     save_folder="../ModelSave",model_name="ensemble_model.pkl", retrain = True):
    """
    This function fits and assesses performance of an Ensemble Learner w/ Grad Boost ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    # Use Default Parameters & Fit Model
    if retrain:
        gb = GradientBoostingClassifier(random_state=random_state).fit(x_train, np.ravel(y_train))
    else:
        model_path = os.path.join(save_folder, model_name)
        gb = joblib.load(model_path)

    # Predict
    label_predictions = gb.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nEnsemble Learner with Gradient Boosting Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Gradient Boosting')

    # Save model
    if retrain:
        save_model_weights(gb, save_folder, model_name)

    return accuracy, precision, recall, f1, specificity, g_mean

# Ensemble Learner with Gradient Boost
def faster_classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test, random_state,
                                     save_folder="../ModelSave",model_name="ensemble_model.pkl", retrain = True):
    """
    This function fits and assesses performance of an Ensemble Learner w/ Grad Boost ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    # Use Default Parameters & Fit Model
    if retrain:
        gb = GradientBoostingClassifier(random_state=random_state, n_estimators=50,max_features='sqrt').fit(x_train, np.ravel(y_train))
    else:
        model_path = os.path.join(save_folder, model_name)
        gb = joblib.load(model_path)

    # Predict
    label_predictions = gb.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nEnsemble Learner with Gradient Boosting Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Gradient Boosting')

    # Save model
    if retrain:
        save_model_weights(gb, save_folder, model_name)

    return accuracy, precision, recall, f1, specificity, g_mean

def call_all_classifiers(classifier_type, x_train, x_test, y_train, y_test, all_features,train_class, class_weight_imb,
                         random_state):
    # Logistic Regression | logreg
    if classifier_type == 'all' or classifier_type == 'logreg':
        accuracy_logreg, precision_logreg, recall_logreg, f1_logreg, specificity_logreg, g_mean_logreg = (
            classify_logistic_regression(x_train, x_test, y_train, y_test, class_weight_imb,random_state,
                                         retrain=train_class))

    # Random Forrest | rf
    if classifier_type == 'all' or classifier_type == 'rf':
        accuracy_rf, precision_rf, recall_rf, f1_rf, tree_depth, specificity_rf, g_mean_rf = (
            classify_random_forest(x_train, x_test, y_train, y_test, class_weight_imb,random_state,
                                   retrain=train_class))

    # Linear discriminant analysis | LDA
    if classifier_type == 'all' or classifier_type == 'LDA':
        accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda, g_mean_lda = (
            classify_lda(x_train, x_test, y_train, y_test,random_state,
                         retrain=train_class))

    # KNN
    if classifier_type == 'all' or classifier_type == 'KNN':
        accuracy_knn, precision_knn, recall_knn, f1_knn, specificity_knn, g_mean_knn = (
            classify_knn(x_train, x_test, y_train, y_test, random_state, retrain=train_class))

    # SVM
    if classifier_type == 'all' or classifier_type == 'SVM':
        accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm, g_mean_svm = (
            classify_svm(x_train, x_test, y_train, y_test, class_weight_imb,random_state, retrain=train_class))

    # Ensemble with Gradient Boosting
    if classifier_type == 'all' or classifier_type == 'EGB':
        accuracy_gb, precision_gb, recall_gb, f1_gb, specificity_gb, g_mean_gb = (
            classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test,random_state,
                                             retrain=train_class))

    performance_metric_summary = (summarize_performance_metrics(accuracy_logreg, accuracy_rf, accuracy_lda,
                                                                    accuracy_knn, accuracy_svm, accuracy_gb,
                                                                    precision_logreg, precision_rf, precision_lda,
                                                                    precision_knn, precision_svm, precision_gb,
                                                                    recall_logreg, recall_rf, recall_lda, recall_knn,
                                                                    recall_svm, recall_gb, f1_logreg, f1_rf, f1_lda,
                                                                    f1_knn, f1_svm, f1_gb, specificity_logreg,
                                                                    specificity_rf, specificity_lda, specificity_knn,
                                                                    specificity_svm, specificity_gb, g_mean_logreg,
                                                                    g_mean_rf, g_mean_lda, g_mean_knn,
                                                                    g_mean_svm, g_mean_gb))

    return performance_metric_summary



def save_model_weights(model,save_folder,model_name):
    """
    Saves Model Weights to save folder
    """

    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save the trained model to the specified folder
    model_path = os.path.join(save_folder, model_name)
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")


# Logistic Regression Classifier
def classify_logistic_regression_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                                     save_folder="../ModelSave",model_name="logistic_regression_model.pkl",retrain=True):
    """
    This function fits and assesses performance of a logistic regression ML classifier for the data
    specified after finding the optimal hyperparameters. Within this function, a separate confusion matrix
    function is called. Additional plotting capabilities include logistic regression visualization.
    """

    if retrain:
        # # Determine optimal hyperparameters of the model
        # param_grid = {'penalty': ['l1', 'l2', 'elasticnet', None],
        #               'C': [0.01, 0.1, 0.5, 1, 5, 10, 100],
        #               'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        #               }
        # logreg = LogisticRegression(max_iter = 1000, class_weight = class_weight_imb, random_state = random_state)
        #
        # clf = GridSearchCV(logreg, param_grid = param_grid, cv = 10, scoring='f1')
        #
        # clf.fit(x_train, np.ravel(y_train))

        # Define the hyperparameter search space
        search_spaces = {
            'penalty': Categorical(['elasticnet']),
            'C': Real(0.01, 100, prior='log-uniform'),
            'solver': Categorical(['saga']),
            'l1_ratio': Real(0.0, 1.0)  # Add l1_ratio for 'elasticnet'
        }

        logreg = LogisticRegression(max_iter=1000, class_weight=class_weight_imb, random_state=random_state)

        # Set up BayesSearchCV
        clf = BayesSearchCV(
            estimator=logreg,
            search_spaces=search_spaces,
            n_iter=30,  # You can adjust this based on how long you're willing to search
            scoring='f1',
            cv=3,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
            error_score=np.nan  # 'nan' to silently skip failing configs
        )

        clf.fit(x_train, np.ravel(y_train))

    else:
        model_path = os.path.join(save_folder, model_name)
        clf = joblib.load(model_path)

    # Predict
    label_predictions = clf.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nLogistic Regression HPO Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Log. Reg.')

    # Save model
    if retrain:
        save_model_weights(clf, save_folder, model_name)

    return accuracy, precision, recall, f1, specificity, g_mean

# Random Forest Classifier
def classify_random_forest_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                           save_folder="../ModelSave",model_name="random_forest_model.pkl",retrain=True):
    """
    This function fits and assesses performance of a random forest ML classifier for the data
    specified. Within this function, a separate confusion matrix function is called. Additional
    visualization capabilities include random forest visualization.
    """

    if retrain:
        # Determine optimal hyperparameters of the model
        # param_grid = {'n_estimators': [10, 50, 100, 300,  500, 1000],
        #               'criterion': ['gini', 'entropy', 'log_loss'],
        #               'max_depth': [3, 5, 10, 30, 50, 70, 100, None],
        #               'max_features': ['sqrt', 'log2'],
        #               'min_samples_leaf': [1, 2, 4],
        #               'min_samples_split': [1, 2, 4],
        #               'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.5]
        #               }
        #
        # rf = RandomForestClassifier(class_weight = class_weight_imb, random_state = random_state)
        #
        # clf = GridSearchCV(rf, param_grid = param_grid, cv = 10, scoring='f1')
        #
        # clf.fit(x_train, np.ravel(y_train))

        search_space = {
            'n_estimators': Integer(10, 1000),
            'criterion': Categorical(['gini', 'entropy', 'log_loss']),
            'max_depth': Integer(3, 100),
            'max_features': Categorical(['sqrt', 'log2']),
            'min_samples_leaf': Integer(1, 4),
            'min_samples_split': Integer(2, 10),  # start from 2, not 1
            'min_weight_fraction_leaf': Real(0.0, 0.5)
        }

        rf = RandomForestClassifier(class_weight=class_weight_imb, random_state=random_state)

        # Use BayesSearchCV instead of GridSearchCV
        clf = BayesSearchCV(
            estimator=rf,
            search_spaces=search_space,
            n_iter=30,  # Number of parameter settings that are sampled
            cv=3,
            scoring='f1',
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
            error_score=np.nan  # 'nan' to silently skip failing configs
        )

        clf.fit(x_train, np.ravel(y_train))

    else:
        model_path = os.path.join(save_folder, model_name)
        clf = joblib.load(model_path)

    # Predict
    label_predictions = clf.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nRandom Forest HPO Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Get the best RandomForest model from the search
    best_rf = clf.best_estimator_

    # Find Tree Depth
    tree_depth = [estimator.get_depth() for estimator in best_rf.estimators_]

    # Visualize Decision Tree
    # fn = all_features
    # cn = ['No GLOC', 'GLOC']
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    # plot_tree(rf.estimators_[0],
    #               feature_names=fn,
    #               class_names=cn,
    #               filled=True)
    # plt.show()

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Random Forest')

    # Save model
    if retrain:
        save_model_weights(clf, save_folder, model_name)

    return accuracy, precision, recall, f1, tree_depth, specificity, g_mean

# Linear Discriminant Analysis
def classify_lda_hpo(x_train, x_test, y_train, y_test, random_state,
                 save_folder="../ModelSave",model_name="LDA_model.pkl",retrain=True):
    """
    This function fits and assesses performance of a linear discriminant analysis ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    if retrain:
        # Determine optimal hyperparameters of the model
        # param_grid = {'solver': ['svd', 'lsqr', 'eigen'],
        #               'shrinkage': [None, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 'auto'],
        #               'tol': [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
        #               }
        # lda = LinearDiscriminantAnalysis()
        #
        # clf = GridSearchCV(lda, param_grid = param_grid, cv = 10, scoring='f1')
        #
        # clf.fit(x_train, np.ravel(y_train))

        search_spaces = [
            # Case 1: solver = 'svd' (shrinkage not allowed)
            {
                'solver': Categorical(['svd']),
                'tol': Real(1e-10, 1e-2, prior='log-uniform')
            },
            # Case 2: solver = 'lsqr' or 'eigen' (shrinkage allowed)
            {
                'solver': Categorical(['lsqr', 'eigen']),
                'shrinkage': Categorical([0.1, 0.2, 0.4, 0.6, 0.8, 1, 'auto']),
                'tol': Real(1e-10, 1e-2, prior='log-uniform')
            }
        ]

        # Define the model
        lda = LinearDiscriminantAnalysis()

        # Set up BayesSearchCV
        clf = BayesSearchCV(
            estimator=lda,
            search_spaces=search_spaces,
            n_iter=30,  # You can increase this for a more thorough search
            cv=3,
            scoring='f1',
            random_state=0,
            n_jobs=-1,
            verbose=1,
            error_score=np.nan  # 'nan' to silently skip failing configs
        )

        clf.fit(x_train, np.ravel(y_train))

    else:
        model_path = os.path.join(save_folder, model_name)
        clf = joblib.load(model_path)

    # Predict
    label_predictions = clf.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nLinear Discriminant Analysis HPO Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Linear Discriminant Analysis')

    # Save model
    if retrain:
        save_model_weights(clf, save_folder, model_name)

    return accuracy, precision, recall, f1, specificity, g_mean

# k Nearest Neighbors
def classify_knn_hpo(x_train, x_test, y_train, y_test, random_state,
                 save_folder="../ModelSave",model_name="KNN_model.pkl",retrain=True):
    """
    This function fits and assesses performance of a K Nearest Neighbors ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    if retrain:

        # Determine optimal hyperparameters of the model
        # param_grid = {'n_neighbors' : [3,5,7,9,11,13,15,20,25,30],
        #               'weights' : ['uniform','distance'],
        #               'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
        #               'metric' : ['minkowski','euclidean','manhattan']}
        #
        # neigh = KNeighborsClassifier()
        #
        # clf = GridSearchCV(neigh, param_grid = param_grid, cv = 10, scoring='f1')
        #
        # clf.fit(x_train, np.ravel(y_train))

        search_spaces = {
            'n_neighbors': Integer(3, 30),
            'weights': Categorical(['uniform', 'distance']),
            'algorithm': Categorical(['auto', 'brute']),  # avoids kd_tree/ball_tree issues
            'metric': Categorical(['minkowski']),
            'p': Integer(1, 2)  # p=1 (manhattan), p=2 (euclidean)
        }

        # Define the model
        knn = KNeighborsClassifier()

        # Set up BayesSearchCV
        clf = BayesSearchCV(
            estimator=knn,
            search_spaces=search_spaces,
            n_iter=30,  # Customize as needed
            cv=3,
            scoring='f1',
            random_state=0,
            n_jobs=-1,
            verbose=1,
            error_score=np.nan  # 'nan' to silently skip failing configs
        )

        # Fit the model
        clf.fit(x_train, np.ravel(y_train))

    else:
        model_path = os.path.join(save_folder, model_name)
        clf = joblib.load(model_path)

    # Predict
    label_predictions = clf.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nKNN HPO Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'kNN')

    # Save model
    if retrain:
        save_model_weights(clf, save_folder, model_name)

    return accuracy, precision, recall, f1, specificity, g_mean


# Support Vector Machine
def classify_svm_hpo(x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                 save_folder="../ModelSave",model_name="svm_model.pkl", retrain = True):
    """
    This function fits and assesses performance of a Support Vector Machine ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    if retrain:
        # Determine optimal hyperparameters of the model
        # param_grid = {'C': [0.1, 1, 10, 100, 1000],
        #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto'],
        #               'kernel': ['rbf','linear', 'poly', 'sigmoid'],
        #               'degree': [3, 4, 5],
        #               'tol': [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
        #               }
        #
        # svm_class = svm.SVC(class_weight=class_weight_imb)
        #
        # clf = GridSearchCV(svm_class, param_grid = param_grid, cv = 10, scoring='f1')
        #
        # clf.fit(x_train, np.ravel(y_train))

        search_spaces = [
            {
                'kernel': Categorical(['linear', 'rbf', 'sigmoid']),
                'C': Real(0.1, 1000, prior='log-uniform'),
                'gamma': Categorical(['scale', 0.1, 0.01, 0.001, 0.0001]),
                'tol': Real(1e-6, 1e-2, prior='log-uniform')
            },
            {
                'kernel': Categorical(['poly']),
                'C': Real(0.1, 1000, prior='log-uniform'),
                'gamma': Categorical(['scale', 0.1, 0.01, 0.001]),
                'degree': Integer(2, 5),
                'tol': Real(1e-6, 1e-2, prior='log-uniform')
            }
        ]

        # Define the model
        svm_class = svm.SVC(class_weight=class_weight_imb)

        # Set up BayesSearchCV
        clf = BayesSearchCV(
            estimator=svm_class,
            search_spaces=search_spaces,
            n_iter=30,  # You can tune this
            cv=3,
            scoring='f1',
            random_state=42,
            n_jobs=-1,
            verbose=1,
            error_score=np.nan  # 'nan' to silently skip failing configs
        )

        clf.fit(x_train, np.ravel(y_train))

    else:
        model_path = os.path.join(save_folder, model_name)
        clf = joblib.load(model_path)

    # Predict
    label_predictions = clf.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nSVM HPO Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Support Vector Machine')

    # Save model
    if retrain:
        save_model_weights(clf, save_folder, model_name)

    return accuracy, precision, recall, f1, specificity, g_mean


# Ensemble Learner with Gradient Boost
def classify_ensemble_with_gradboost_hpo(x_train, x_test, y_train, y_test, random_state,
                                     save_folder="../ModelSave",model_name="ensemble_model.pkl", retrain = True):
    """
    This function fits and assesses performance of an Ensemble Learner w/ Grad Boost ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    if retrain:
        # Determine optimal hyperparameters of the model
        # param_grid = {'n_estimators': [10, 50, 100, 300,  500, 1000],
        #               'learning_rate': [0.01, 0.1, 0.2, 1, 5],
        #               'max_depth': [3, 5, 10, 30, 50, 70, 100, None],
        #               'max_features': ['sqrt', 'log2', None],
        #               'min_samples_leaf': [1, 2, 4],
        #               'min_samples_split': [1, 2, 4],
        #               'loss': ['log_loss', 'exponential'],
        #               'criterion': ['friedman_mse', 'squared_error'],
        #               'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.5]
        #               }
        #
        # gb = GradientBoostingClassifier(random_state = random_state)
        #
        # clf = GridSearchCV(gb, param_grid = param_grid, cv = 10, scoring='f1')
        #
        # clf.fit(x_train, np.ravel(y_train))

        search_spaces = {
            'n_estimators': Integer(50, 1000),
            'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
            'max_depth': Integer(3, 20),
            'max_features': Categorical(['sqrt', 'log2', None]),
            'min_samples_leaf': Integer(1, 4),
            'min_samples_split': Integer(2, 4),
            'loss': Categorical(['log_loss']),
            'min_weight_fraction_leaf': Real(0.0, 0.5)
        }

        # Define the model
        gb = GradientBoostingClassifier(random_state=random_state)

        # Set up BayesSearchCV
        clf = BayesSearchCV(
            estimator=gb,
            search_spaces=search_spaces,
            n_iter=30,  # Number of iterations can be adjusted
            cv=3,
            scoring='f1',
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
            error_score=np.nan  # 'nan' to silently skip failing configs
        )

        # Fit the model
        clf.fit(x_train, np.ravel(y_train))

    else:
        model_path = os.path.join(save_folder, model_name)
        clf = joblib.load(model_path)

    # Predict
    label_predictions = clf.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)
    g_mean = geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nEnsemble Learner with Gradient Boosting HPO Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Gradient Boosting')

    # Save model
    if retrain:
        save_model_weights(clf, save_folder, model_name)

    return accuracy, precision, recall, f1, specificity, g_mean

def lstm_binary_class_old(X,Y,training_ratio,stride):
    """ Long Short Term Memory Classifier for Binary Classification"""

    # Define window size and step size
    time_window = 20 # seconds of window to feed at a time
    window_size = round(time_window/stride)  # Length of each subsequence window
    step_size = round(window_size/2)  # Step size for moving the window set at 50%

    # Train Test Split
    train_dataset, test_dataset, train_windows_tensor, train_labels_tensor, test_windows_tensor, test_labels_tensor = (
        train_test_split_trials(X, Y, window_size, step_size, training_ratio))

    # Define model parameters to be tuned with validation dataset
    num_epochs = 25
    hidden_dim = 64*2
    output_dim = 1  # Binary classification output
    num_layers = 2
    dropout = 0.3
    batch_size = 32
    learning_rate = 0.001

    # Create DataLoaders with smaller batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_dim = train_windows_tensor.shape[2]
    model = LSTM_supporting.LSTM_Classifier(input_dim, hidden_dim, output_dim, num_layers, dropout)
    criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    predictors_over_time = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            # Get model predictions
            outputs = model(x_batch)
            preds = (outputs >= 0.5).float()  # Convert probabilities to binary predictions

            # Append predictions and true labels for metrics calculation
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

            # Append the input data for plotting predictors
            predictors_over_time.extend(x_batch.numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    predictors_over_time = np.array(predictors_over_time)

    # Calculate metrics
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds)
    precision = metrics.precision_score(all_labels, all_preds)
    recall = metrics.recall_score(all_labels, all_preds)

    # Display in command line results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

    # Plot Results
    create_confusion_matrix(all_labels, all_preds, 'LSTM')
    roc_curve_plot(all_labels, all_preds)
    prediction_time_plot(all_labels, all_preds,predictors_over_time)


def lstm_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state, save_folder):

    # Compute class weights to address imbalance
    class_weights = compute_class_weight(class_weight_imb, classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Perform Hyperparameter Tuning with Optuna using only Training data
    objective = make_objective(x_train, y_train, class_weights)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    # Grab the Hyperparameters from the best set
    best_params = study.best_trial.params
    print("Best Trial:", study.best_trial)

    hidden_dim = best_params["hidden_dim"]
    num_layers = best_params["num_layers"]
    dropout = best_params["dropout"]
    learning_rate = best_params["lr"]
    batch_size = best_params["batch_size"]
    sequence_length = best_params["sequence_length"]
    stride = best_params["stride"]
    step_size = round(sequence_length * stride)

    # Train with all training data
    train_dataset, _, train_windows_tensor, _, _, train_labels_tensor = (
        train_test_split_trials(x_train, y_train, sequence_length, step_size, training_ratio=1.0)
    )

    # Create the test dataset format
    _, test_dataset, _, _, _, _ = (
        train_test_split_trials(x_test, y_test, sequence_length, step_size, training_ratio=1.0)
    )

    input_dim = train_windows_tensor.shape[2]
    model = LSTMClassifier(input_dim, hidden_dim, 1, num_layers, dropout)

    # Move device to the model after it is created
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model.to(device)

    # Create sample weights using the windowed labels
    sample_weights = class_weights[train_labels_tensor]  # Use the windowed labels here

    # Weighted Random Sampler for the windowed data
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    # Prepare the DataLoader with the sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    criterion = nn.BCELoss(weight=class_weights)  # Weighted BCE loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    patience = 5  # How many epochs to wait before stopping if no improvement
    best_val_loss = float('inf')  # Initialize best validation loss as infinity
    epochs_without_improvement = 0  # Counter for epochs without improvement

    # Train model with X epochs
    num_epochs = 50
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 100 == 0:  # Print loss every 100 batches for verbosity
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Early stopping check: Monitor validation loss
        val_loss = 0
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(test_loader)  # Calculate average validation loss
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print("Validation loss improved, saving model...")
            # Optionally, save the best model here
            torch.save(model.state_dict(), "best_lstm_model.pth")  # Save model
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    # Save Model
    torch.save(model.state_dict(), save_folder)

    # Evaluate final model
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds, all_labels, predictors_over_time = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            # Append the input data for plotting predictors
            predictors_over_time.extend(x_batch.numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    predictors_over_time = np.array(predictors_over_time)

    # Assess Performance
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    precision = metrics.precision_score(all_labels, all_preds)
    recall = metrics.recall_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds)
    specificity = metrics.recall_score(all_labels, all_preds, pos_label=0)  # Specificity
    g_mean = geometric_mean_score(all_labels, all_preds)

    # Print performance metrics
    print("\nPerformance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    return accuracy, precision, recall, f1, specificity, g_mean