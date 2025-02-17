import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import plot_tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from GLOC_visualization import create_confusion_matrix
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from mrmr import mrmr_classif

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
def pre_classification_training_test_split(y_gloc_labels_noNaN, x_feature_matrix_noNaN, training_ratio):
    """
    This function splits the X and y matrix into training and test matrix.
    """

    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(x_feature_matrix_noNaN, y_gloc_labels_noNaN,
                                                                    test_size=(1 - training_ratio), random_state=42)

    return x_train, x_test, y_train, y_test


# Logistic Regression Classifier
def classify_logistic_regression(x_train, x_test, y_train, y_test, all_features):
    """
    This function fits and assesses performance of a logistic regression ML classifier for the data
    specified. Within this function, a separate confusion matrix function is called. Additional
    plotting capabilities include logistic regression visualization.
    """
    # feature_subset = feature_selection_lasso(x_train, y_train, all_features)

    # Use Default Parameters & Fit Model
    logreg = LogisticRegression(class_weight = "balanced", random_state=42, max_iter=1000).fit(x_train, np.ravel(y_train))

    # Predict
    label_predictions = logreg.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)

    # Print performance metrics
    print("\nLogistic Regression Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Log. Reg.')

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

    return accuracy, precision, recall, f1, specificity

# Random Forest Classifier
def classify_random_forest(x_train, x_test, y_train, y_test, all_features):
    """
    This function fits and assesses performance of a random forest ML classifier for the data
    specified. Within this function, a separate confusion matrix function is called. Additional
    visualization capabilities include random forest visualization.
    """

    # Use Default Parameters & Fit Model
    rf = RandomForestClassifier(class_weight="balanced", random_state=42).fit(x_train, np.ravel(y_train))

    # Predict
    label_predictions = rf.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)

    # Print performance metrics
    print("\nRandom Forest Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)

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
    # create_confusion_matrix(y_testing, label_predictions, 'Random Forest')

    return accuracy, precision, recall, f1, tree_depth, specificity

# Linear Discriminant Analysis
def classify_lda(x_train, x_test, y_train, y_test, all_features):
    """
    This function fits and assesses performance of a linear discriminant analysis ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    # Use Default Parameters & Fit Model
    lda = LinearDiscriminantAnalysis().fit(x_train, np.ravel(y_train))

    # Predict
    label_predictions = lda.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)

    # Print performance metrics
    print("\nLinear Discriminant Analysis Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Linear Discriminant Analysis')

    return accuracy, precision, recall, f1, specificity

# k Nearest Neighbors
def classify_knn(x_train, x_test, y_train, y_test):
    """
    This function fits and assesses performance of a K Nearest Neighbors ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    # Use Default Parameters & Fit Model
    neigh = KNeighborsClassifier().fit(x_train, np.ravel(y_train))

    # Predict
    label_predictions = neigh.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)

    # Print performance metrics
    print("\nKNN Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'kNN')

    return accuracy, precision, recall, f1, specificity

# Support Vector Machine
def classify_svm(x_train, x_test, y_train, y_test):
    """
    This function fits and assesses performance of a Support Vector Machine ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    # Use Default Parameters & Fit Model
    svm_class = svm.SVC(kernel="linear", class_weight="balanced").fit(x_train, np.ravel(y_train))

    # Predict
    label_predictions = svm_class.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)

    # Print performance metrics
    print("\nSVM Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Support Vector Machine')

    return accuracy, precision, recall, f1, specificity

# Ensemble Learner with Gradient Boost
def classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test):
    """
    This function fits and assesses performance of an Ensemble Learner w/ Grad Boost ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    # Use Default Parameters & Fit Model
    gb = GradientBoostingClassifier(random_state=42).fit(x_train, np.ravel(y_train))

    # Predict
    label_predictions = gb.predict(x_test)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, label_predictions)
    precision = metrics.precision_score(y_test, label_predictions)
    recall = metrics.recall_score(y_test, label_predictions)
    f1 = metrics.f1_score(y_test, label_predictions)
    specificity = metrics.recall_score(y_test, label_predictions, pos_label=0)

    # Print performance metrics
    print("\nEnsemble Learner with Gradient Boosting Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)

    # Create Confusion Matrix
    create_confusion_matrix(y_test, label_predictions, 'Gradient Boosting')

    return accuracy, precision, recall, f1, specificity