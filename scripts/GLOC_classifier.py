from pygam import GAM, s, f

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

def label_gloc_events(gloc_data_reduced):
    """
    This function creates a g-loc label for the data based on the event_validated column. The event
    is labeled as 1 between GLOC and Return to Consciousness.
    """

    # Create GLOC Classifier Vector
    event_validated = gloc_data_reduced['event_validated'].to_numpy()
    gloc_classifier = np.zeros(event_validated.shape)

    gloc_indices = np.argwhere(event_validated == 'GLOC')
    rtc_indices = np.argwhere(event_validated == 'return to consciousness')

    for i in range(gloc_indices.shape[0]):
        gloc_classifier[gloc_indices[i, 0]:rtc_indices[i, 0]] = 1

    return gloc_classifier

def check_event_columns(gloc_data):
    """
    This function was created to understand if ALOC data was available. It will check which
    unique values exist in the event and event_validated columns.
    """

    vals_event = gloc_data['event'].unique()
    vals_event_validated = gloc_data['event_validated'].unique()

    return vals_event, vals_event_validated

# Logistic Regression Classifier
# USING RANDOM STATE = 42
def classify_logistic_regression(gloc_window, sliding_window_mean, training_ratio, all_features):
    """
    This function fits and assesses performance of a logistic regression ML classifier for the data
    specified. Within this function, a separate confusion matrix function is called. Additional
    plotting capabilities include logistic regression visualization.
    """

    # Train/Test Split
    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window, test_size=(1-training_ratio), random_state=42)

    # Use Default Parameters & Fit Model
    logreg = LogisticRegression(class_weight = "balanced", random_state=42, max_iter=1000).fit(x_training, np.ravel(y_training))

    # Predict
    label_predictions = logreg.predict(x_testing)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_testing, label_predictions)
    precision = metrics.precision_score(y_testing, label_predictions)
    recall = metrics.recall_score(y_testing, label_predictions)
    f1 = metrics.f1_score(y_testing, label_predictions)

    print("\nLogistic Regression Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)

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

    return accuracy, precision, recall, f1

# Random Forest Classifier
# USING RANDOM STATE = 42
def classify_random_forest(gloc_window, sliding_window_mean, training_ratio, all_features):
    """
    This function fits and assesses performance of a random forest ML classifier for the data
    specified. Within this function, a separate confusion matrix function is called. Additional
    visualization capabilities include random forest visualization.
    """

    # Train/Test Split
    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window,
                                                                    test_size=(1 - training_ratio), random_state=42)

    # Use Default Parameters & Fit Model
    rf = RandomForestClassifier(class_weight="balanced", random_state=42).fit(x_training, np.ravel(y_training))

    # Predict
    label_predictions = rf.predict(x_testing)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_testing, label_predictions)
    precision = metrics.precision_score(y_testing, label_predictions)
    recall = metrics.recall_score(y_testing, label_predictions)
    f1 = metrics.f1_score(y_testing, label_predictions)

    print("\nRandom Forest Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)

    # Find Tree Depth
    tree_depth = [estimator.get_depth() for estimator in rf.estimators_]

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

    return accuracy, precision, recall, f1

# Linear Discriminant Analysis
# USING RANDOM STATE = 42
def classify_lda(gloc_window, sliding_window_mean, training_ratio, all_features):
    """
    This function fits and assesses performance of a linear discriminant analysis ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    # Train/Test Split
    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window,
                                                                    test_size=(1 - training_ratio), random_state=42)

    # Use Default Parameters & Fit Model
    lda = LinearDiscriminantAnalysis().fit(x_training, np.ravel(y_training))

    # Predict
    label_predictions = lda.predict(x_testing)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_testing, label_predictions)
    precision = metrics.precision_score(y_testing, label_predictions)
    recall = metrics.recall_score(y_testing, label_predictions)
    f1 = metrics.f1_score(y_testing, label_predictions)

    print("\nLinear Discriminant Analysis Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)

    # Create Confusion Matrix
    create_confusion_matrix(y_testing, label_predictions, 'Linear Discriminant Analysis')

    return accuracy, precision, recall, f1

# k Nearest Neighbors
# USING RANDOM STATE = 42
def classify_knn(gloc_window, sliding_window_mean, training_ratio):
    """
    This function fits and assesses performance of a K Nearest Neighbors ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    # Train/Test Split
    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window,
                                                                    test_size=(1 - training_ratio), random_state=42)

    # Use Default Parameters & Fit Model
    neigh = KNeighborsClassifier().fit(x_training, np.ravel(y_training))

    # Predict
    label_predictions = neigh.predict(x_testing)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_testing, label_predictions)
    precision = metrics.precision_score(y_testing, label_predictions)
    recall = metrics.recall_score(y_testing, label_predictions)
    f1 = metrics.f1_score(y_testing, label_predictions)

    print("\nKNN Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)

    # Create Confusion Matrix
    create_confusion_matrix(y_testing, label_predictions, 'kNN')

    return accuracy, precision, recall, f1

# Support Vector Machine
# USING RANDOM STATE = 42
def classify_svm(gloc_window, sliding_window_mean, training_ratio):
    """
    This function fits and assesses performance of a Support Vector Machine ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    # Train/Test Split
    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window,
                                                                    test_size=(1 - training_ratio), random_state=42)

    # Use Default Parameters & Fit Model
    svm_class = svm.SVC(kernel="linear", class_weight="balanced").fit(x_training, np.ravel(y_training))

    # Predict
    label_predictions = svm_class.predict(x_testing)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_testing, label_predictions)
    precision = metrics.precision_score(y_testing, label_predictions)
    recall = metrics.recall_score(y_testing, label_predictions)
    f1 = metrics.f1_score(y_testing, label_predictions)

    print("\nSVM Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)

    # Create Confusion Matrix
    create_confusion_matrix(y_testing, label_predictions, 'Support Vector Machine')

    return accuracy, precision, recall, f1

# Ensemble Learner with Gradient Boost
# USING RANDOM STATE = 42
def classify_ensemble_with_gradboost(gloc_window, sliding_window_mean, training_ratio):
    """
    This function fits and assesses performance of an Ensemble Learner w/ Grad Boost ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    # Train/Test Split
    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window,
                                                                    test_size=(1 - training_ratio), random_state=42)

    # Use Default Parameters & Fit Model
    gb = GradientBoostingClassifier(random_state=42).fit(x_training, np.ravel(y_training))

    # Predict
    label_predictions = gb.predict(x_testing)

    # Assess Performance
    accuracy = metrics.accuracy_score(y_testing, label_predictions)
    precision = metrics.precision_score(y_testing, label_predictions)
    recall = metrics.recall_score(y_testing, label_predictions)
    f1 = metrics.f1_score(y_testing, label_predictions)

    print("\nEnsemble Learner with Gradient Boosting Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)

    # Create Confusion Matrix
    create_confusion_matrix(y_testing, label_predictions, 'Gradient Boosting')

    return accuracy, precision, recall, f1

def gam_classifier(X,y):

    ## model
    gam = GAM(s(0, n_splines=5), distribution='binomial', link='logit')
    gam.fit(X, y)

    ## plotting
    plt.figure();
    fig, axs = plt.subplots(1, 1);

    titles = ['HR 95% Conf Bound']

    ax = axs
    i=0
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    plt.scatter(X, y, facecolor='gray', edgecolors='none')

    # continuing last example with the mcycle dataset
    for response in gam.sample(X, y, quantity='y', n_draws=50, sample_at_X=XX):
        plt.scatter(XX, response, alpha=.03, color='k')

    ax.set_title(titles[i]);

    plt.show()

    return gam