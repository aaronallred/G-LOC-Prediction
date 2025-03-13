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
from GLOC_visualization import create_confusion_matrix
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from mrmr import mrmr_classif
from imblearn.metrics import geometric_mean_score

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
def classify_logistic_regression(x_train, x_test, y_train, y_test, all_features,
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
        logreg = LogisticRegression(class_weight = 'balanced', random_state=42, max_iter=1000).fit(x_train, np.ravel(y_train))
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
    g_mean = imblearn.metrics.geometric_mean_score(y_test, label_predictions)

    # Print performance metrics
    print("\nLogistic Regression Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

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

    # Save model
    if retrain:
        save_model_weights(logreg, save_folder, model_name)

    return accuracy, precision, recall, f1, specificity, g_mean

# Random Forest Classifier
def classify_random_forest(x_train, x_test, y_train, y_test, all_features,
                           save_folder="../ModelSave",model_name="random_forest_model.pkl",retrain=True):
    """
    This function fits and assesses performance of a random forest ML classifier for the data
    specified. Within this function, a separate confusion matrix function is called. Additional
    visualization capabilities include random forest visualization.
    """

    if retrain:
        # Use Default Parameters & Fit Model
        rf = RandomForestClassifier(class_weight="balanced", random_state=42).fit(x_train, np.ravel(y_train))
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
    g_mean = imblearn.metrics.geometric_mean_score(y_test, label_predictions)

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
    # create_confusion_matrix(y_testing, label_predictions, 'Random Forest')

    # Save model
    if retrain:
        save_model_weights(rf, save_folder, model_name)

    return accuracy, precision, recall, f1, tree_depth, specificity, g_mean

# Linear Discriminant Analysis
def classify_lda(x_train, x_test, y_train, y_test, all_features,
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
    g_mean = imblearn.metrics.geometric_mean_score(y_test, label_predictions)

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
def classify_knn(x_train, x_test, y_train, y_test,
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
    g_mean = imblearn.metrics.geometric_mean_score(y_test, label_predictions)

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
def classify_svm(x_train, x_test, y_train, y_test,
                 save_folder="../ModelSave",model_name="svm_model.pkl", retrain = True):
    """
    This function fits and assesses performance of a Support Vector Machine ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    if retrain:
        # Use Default Parameters & Fit Model
        svm_class = svm.SVC(kernel="linear", class_weight="balanced").fit(x_train, np.ravel(y_train))
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
    g_mean = imblearn.metrics.geometric_mean_score(y_test, label_predictions)

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
def classify_ensemble_with_gradboost(x_train, x_test, y_train, y_test,
                                     save_folder="../ModelSave",model_name="ensemble_model.pkl", retrain = True):
    """
    This function fits and assesses performance of an Ensemble Learner w/ Grad Boost ML classifier for
    the data specified. Within this function, a separate confusion matrix function is called.
    """

    # Use Default Parameters & Fit Model
    if retrain:
        gb = GradientBoostingClassifier(random_state=42).fit(x_train, np.ravel(y_train))
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
    g_mean = imblearn.metrics.geometric_mean_score(y_test, label_predictions)

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