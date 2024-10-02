import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def categorize_gloc(gloc_data):
    # Create GLOC Classifier Vector
    event = gloc_data['event'].to_numpy()
    event_validated = gloc_data['event_validated'].to_numpy()
    gloc_classifier = np.zeros(event.shape)

    gloc_indices = np.argwhere(event == 'GLOC')
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

#def kNN_classifier(gloc_data)

# Logistic Regression Classifier
def classify_logistic_regression(gloc_window, sliding_window_mean, training_ratio, all_features):

    # Train/Test Split
    x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window, test_size=(1-training_ratio), random_state=42)

    # Use Default Parameters & Fit Model
    logreg = LogisticRegression(class_weight = "balanced", random_state=42).fit(x_training, y_training)

    # Predict
    label_predictions = logreg.predict(x_testing)

    # Assess Performance
    print("Accuracy: ", metrics.accuracy_score(y_testing, label_predictions))
    print("Precision: ", metrics.precision_score(y_testing, label_predictions))
    print("Recall: ", metrics.recall_score(y_testing, label_predictions))
    print("F1 Score: ", metrics.f1_score(y_testing, label_predictions))

    # Create Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(y_testing, label_predictions)

    # Plot Confusion Matrix
    prediction_names = ['No GLOC', 'GLOC']
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(prediction_names))
    plt.xticks(tick_marks, prediction_names)
    plt.yticks(tick_marks, prediction_names)
    sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    # Plot 0/1 Classification
    for i in range(np.size(sliding_window_mean,1)):
        x_test_squeeze = x_testing[:,i].squeeze()
        y_test_squeeze = y_testing.squeeze()
        sns.scatterplot(x=x_test_squeeze, y=label_predictions, hue=y_test_squeeze)
        plt.title('GLOC Classification- Logistic Regression')
        plt.xlabel(all_features[i])
        plt.ylabel('Predicted')
        plt.legend()
        plt.show()

        # Plot Logistic Regression
        fig, ax = plt.subplots()
        y_prob = logreg.predict_proba(x_testing)
        sns.scatterplot(x=x_test_squeeze, y=y_prob[:, 1], hue=y_test_squeeze)
        plt.title('GLOC Classification- Logistic Regression')
        plt.xlabel(all_features[i])
        plt.ylabel('Predicted')
        plt.show()

# def classify_random_forest(gloc_window, sliding_window_mean, training_ratio):
#     x_training, x_testing, y_training, y_testing = train_test_split(sliding_window_mean, gloc_window,
#                                                                     test_size=(1 - training_ratio), random_state=42)
#
#     # Use Default Parameters & Fit Model
#     logreg = (class_weight="balanced", random_state=42).fit(x_training, y_training)
#
#     # Predict
#     label_predictions = logreg.predict(x_testing)
#
#     # Assess Performance
#     print("Accuracy: ", metrics.accuracy_score(y_testing, label_predictions))
#     print("Precision: ", metrics.precision_score(y_testing, label_predictions))
#     print("Recall: ", metrics.recall_score(y_testing, label_predictions))
#     print("F1 Score: ", metrics.f1_score(y_testing, label_predictions))
#     X, y = make_classification(n_samples=1000, n_features=1,
#                                n_informative=2, n_redundant=0,
#                                random_state=0, shuffle=False)
#     clf = RandomForestClassifier(max_depth=2, random_state=0)
#
#     clf.fit(X, y)
#     RandomForestClassifier(...)
#
#     print(clf.predict([[0, 0, 0, 0]]))