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

def categorize_gloc(gloc_data_reduced):
    # Create GLOC Classifier Vector
    event_validated = gloc_data_reduced['event_validated'].to_numpy()
    gloc_classifier = np.zeros(event_validated.shape)

    gloc_indices = np.argwhere(event_validated == 'GLOC')
    rtc_indices = np.argwhere(event_validated == 'return to consciousness')

    for i in range(gloc_indices.shape[0]):
        gloc_classifier[gloc_indices[i, 0]:rtc_indices[i, 0]] = 1

    return gloc_classifier

def check_for_aloc(gloc_data):
    aloc_search_event = gloc_data['event'].to_numpy()
    aloc_search_event_validated = gloc_data['event_validated'].to_numpy()
    aloc_indices_event = np.argwhere((aloc_search_event != 'GLOC') & (aloc_search_event != 'NO VALUE'))
    aloc_indices_event_validated = np.argwhere((aloc_search_event_validated != 'GLOC') & (aloc_search_event_validated != 'NO VALUE'))

    other_vals_event = aloc_search_event[aloc_indices_event]
    other_vals_event_validated = aloc_search_event_validated[aloc_indices_event_validated]
    return other_vals_event, other_vals_event_validated

# Logistic Regression Classifier
# USING RANDOM STATE = 42
def classify_logistic_regression(gloc_window, sliding_window_mean, training_ratio, all_features):

    # Train/Test Split
    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window, test_size=(1-training_ratio), random_state=42)

    # Use Default Parameters & Fit Model
    logreg = LogisticRegression(class_weight = "balanced", random_state=42).fit(x_training, np.ravel(y_training))

    # Predict
    label_predictions = logreg.predict(x_testing)

    # Assess Performance
    print("\nLogistic Regression Performance Metrics:")
    print("Accuracy: ", metrics.accuracy_score(y_testing, label_predictions))
    print("Precision: ", metrics.precision_score(y_testing, label_predictions))
    print("Recall: ", metrics.recall_score(y_testing, label_predictions))
    print("F1 Score: ", metrics.f1_score(y_testing, label_predictions))

    # Create Confusion Matrix
    create_confusion_matrix(y_testing, label_predictions, 'Log. Reg.')

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

# Random Forest Classifier
# USING RANDOM STATE = 42
def classify_random_forest(gloc_window, sliding_window_mean, training_ratio, all_features):
    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window,
                                                                    test_size=(1 - training_ratio), random_state=42)

    # Use Default Parameters & Fit Model
    rf = RandomForestClassifier(class_weight="balanced", random_state=42).fit(x_training, np.ravel(y_training))

    # Predict
    label_predictions = rf.predict(x_testing)

    # Assess Performance
    print("\nRandom Forest Performance Metrics:")
    print("Accuracy: ", metrics.accuracy_score(y_testing, label_predictions))
    print("Precision: ", metrics.precision_score(y_testing, label_predictions))
    print("Recall: ", metrics.recall_score(y_testing, label_predictions))
    print("F1 Score: ", metrics.f1_score(y_testing, label_predictions))

    # Visualize Decision Tree
    fn = all_features
    cn = ['No GLOC', 'GLOC']
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    plot_tree(rf.estimators_[0],
                   feature_names=fn,
                   class_names=cn,
                   filled=True)
    plt.show()

    # Create Confusion Matrix
    create_confusion_matrix(y_testing, label_predictions, 'Random Forest')

# Linear Discriminant Analysis
# USING RANDOM STATE = 42
def classify_lda(gloc_window, sliding_window_mean, training_ratio, all_features):

    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window,
                                                                    test_size=(1 - training_ratio), random_state=42)

    # Use Default Parameters & Fit Model

    lda = LinearDiscriminantAnalysis().fit(x_training, np.ravel(y_training))

    # Predict
    label_predictions = lda.predict(x_testing)

    # Assess Performance
    print("\nLinear Discriminant Analysis Performance Metrics:")
    print("Accuracy: ", metrics.accuracy_score(y_testing, label_predictions))
    print("Precision: ", metrics.precision_score(y_testing, label_predictions))
    print("Recall: ", metrics.recall_score(y_testing, label_predictions))
    print("F1 Score: ", metrics.f1_score(y_testing, label_predictions))

    # Create Confusion Matrix
    create_confusion_matrix(y_testing, label_predictions, 'Linear Discriminant Analysis')

# k Nearest Neighbors
# USING RANDOM STATE = 42
def classify_knn(gloc_window, sliding_window_mean, training_ratio):
    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window,
                                                                    test_size=(1 - training_ratio), random_state=42)

    # Use Default Parameters & Fit Model
    neigh = KNeighborsClassifier().fit(x_training, np.ravel(y_training))

    # Predict
    label_predictions = neigh.predict(x_testing)

    # Assess Performance
    print("\nKNN Performance Metrics:")
    print("Accuracy: ", metrics.accuracy_score(y_testing, label_predictions))
    print("Precision: ", metrics.precision_score(y_testing, label_predictions))
    print("Recall: ", metrics.recall_score(y_testing, label_predictions))
    print("F1 Score: ", metrics.f1_score(y_testing, label_predictions))

    # Visualize KNN

    # Create Confusion Matrix
    create_confusion_matrix(y_testing, label_predictions, 'kNN')

# Support Vector Machine
# USING RANDOM STATE = 42
def classify_svm(gloc_window, sliding_window_mean, training_ratio):
    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window,
                                                                    test_size=(1 - training_ratio), random_state=42)

    # Use Default Parameters & Fit Model
    svm_class = svm.SVC().fit(x_training, np.ravel(y_training))

    # Predict
    label_predictions = svm_class.predict(x_testing)

    # Assess Performance
    print("\nSVM Performance Metrics:")
    print("Accuracy: ", metrics.accuracy_score(y_testing, label_predictions))
    print("Precision: ", metrics.precision_score(y_testing, label_predictions))
    print("Recall: ", metrics.recall_score(y_testing, label_predictions))
    print("F1 Score: ", metrics.f1_score(y_testing, label_predictions))

    # Visualize SVM

    # Create Confusion Matrix
    create_confusion_matrix(y_testing, label_predictions, 'Support Vector Machine')

# Ensemble Learner with Gradient Boost
# USING RANDOM STATE = 42
def classify_ensemble_with_gradboost(gloc_window, sliding_window_mean, training_ratio):
    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window,
                                                                    test_size=(1 - training_ratio), random_state=42)

    # Use Default Parameters & Fit Model
    gb = GradientBoostingClassifier(random_state=42).fit(x_training, np.ravel(y_training))

    # Predict
    label_predictions = gb.predict(x_testing)

    # Assess Performance
    print("\nEnsemble Learner with Gradient Boosting Performance Metrics:")
    print("Accuracy: ", metrics.accuracy_score(y_testing, label_predictions))
    print("Precision: ", metrics.precision_score(y_testing, label_predictions))
    print("Recall: ", metrics.recall_score(y_testing, label_predictions))
    print("F1 Score: ", metrics.f1_score(y_testing, label_predictions))

    # Create Confusion Matrix
    create_confusion_matrix(y_testing, label_predictions, 'Gradient Boosting')