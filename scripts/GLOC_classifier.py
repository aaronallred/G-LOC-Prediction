from pygam import GAM, s, f

import numpy as np
from scipy.constants import precision
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

def gam_classifier_sub(X,y,training_ratio,all_features):

    # just grab means for now
    # X = X[:,list(range(18)) + [-1]]

    # also test stddevs
    # X = X[:,list(range(18, 36)) + [-1]]

    # also test just range
    X = X[:, list(range(54, 72)) + [-1]]

    # Split into train and test where last two
    last_column = X[:, -1]
    unique_values = np.unique(last_column)
    # Calculate training split of the unique values
    num_unique = len(unique_values)
    num_to_select = int(num_unique * training_ratio)

    # get train indices
    random_indices = np.random.choice(unique_values, size=num_to_select, replace=False)
    selected_indices = np.where(np.isin(last_column, random_indices))[0]
    X_train = X[selected_indices]
    y_train = y[selected_indices]

    # get test indices
    all_indices = np.arange(X.shape[0])
    complement_indices = np.setdiff1d(all_indices, selected_indices)
    X_test = X[complement_indices]
    y_test = y[complement_indices]


    rows,cols = np.shape(X)
    ## model
    # gam = GAM(s(0)+s(1)+s(2)+s(3)+s(4)+s(5)+s(6)+f(cols-1), distribution='binomial', link='logit')
    gam = GAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) +
              s(10) + s(11) + s(12) + s(13) + s(14) + s(15) + s(16) + s(17) + f(cols-1), distribution='binomial', link='logit')
    # gam = GAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) +
    #             s(10) + s(11) + s(12) + s(13) + s(14) + s(15) + s(16) + s(17) + s(18) +
    #             s(19) + s(20) + s(21) + s(22) + s(23) + s(24) + s(25) + s(26) + s(27) +
    #             s(28) + s(29) + s(30) + s(31) + s(32) + s(33) + s(34) + s(35) + s(36) +
    #             s(37) + s(38) + s(39) + s(40) + s(41) + s(42) + s(43) + s(44) + s(45) +
    #             s(46) + s(47) + s(48) + s(49) + s(50) + s(51) + s(52) + s(53) + s(54) +
    #             s(55) + s(56) + s(57) + s(58) + s(59) + s(60) + s(61) + s(62) + s(63) +
    #             s(64) + s(65) + s(66) + s(67) + s(68) + s(69) + s(70) + s(71) + f(cols-1), distribution='binomial', link='logit')

    gam.fit(X_train, y_train)

    predictions = []
    X_test_len,X_test_width = np.shape(X_test)

    for i in np.unique(X_train[:,-1]):
        series_indicator_test = i*np.ones((X_test_len,1)) # Use the indicator for each series
        X_test_temp = np.column_stack([X_test[:,0:-1], series_indicator_test])
        y_pred = gam.predict(X_test_temp)
        predictions.append(y_pred)

    y_pred_averaged = np.mean(predictions, axis=0)

    y_testing = y_test
    label_predictions = np.round(y_pred_averaged)
    accuracy = metrics.accuracy_score(y_testing, label_predictions)
    precision = metrics.precision_score(y_testing, label_predictions)
    recall = metrics.recall_score(y_testing, label_predictions)
    f1 = metrics.f1_score(y_testing, label_predictions)

    # Create Confusion Matrix
    create_confusion_matrix(y_testing, label_predictions, 'GAM')

    plots = 2
    if plots==1:
        XX = gam.generate_X_grid(term=1, n=500)
        XX[:,-1] = 1

        plt.plot(XX, gam.predict(XX), 'r--')
        # plt.plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--')

        plt.scatter(X_train[:,0], y_train, facecolor='gray', edgecolors='none')
        plt.title('95% prediction interval')
        plt.show()

    elif plots == 2:
        fig, axs = plt.subplots(1, 8)

        single = np.unique(X_train[:,-1])
        titles = all_features
        for i, ax in enumerate(axs):
            XX = gam.generate_X_grid(term=i)
            XX[:, -1] = single[0]
            pdep, confi = gam.partial_dependence(term=i, width=.95, X=XX)

            ax.plot(XX[:, i], pdep)
            ax.plot(XX[:, i], confi, c='r', ls='--')
            # ax.set_title(titles[i]);
        plt.show()

    return accuracy, precision, recall, f1, gam

def gam_classifier_cat(X,y,training_ratio,all_features):
    # concatenated version of GAM

    # just grab means for now
    X = X[:,list(range(18)) + [-1]]

    # also test stddevs
    # X = X[:,list(range(18, 36)) + [-1]]

    # also test just range
    #X = X[:, list(range(54, 72)) + [-1]]

    # Split into train and test where last two
    last_column = X[:, -1]
    unique_values = np.unique(last_column)
    # Calculate training split of the unique values
    num_unique = len(unique_values)
    num_to_select = int(num_unique * training_ratio)

    # get train indices
    random_indices = np.random.choice(unique_values, size=num_to_select, replace=False)
    selected_indices = np.where(np.isin(last_column, random_indices))[0]
    X_train = X[selected_indices]
    y_train = y[selected_indices]

    # get test indices
    all_indices = np.arange(X.shape[0])
    complement_indices = np.setdiff1d(all_indices, selected_indices)
    X_test = X[complement_indices]
    y_test = y[complement_indices]


    rows,cols = np.shape(X)
    ## model
    # gam = GAM(s(0)+s(1)+s(2)+s(3)+s(4)+s(5)+s(6)+f(cols-1), distribution='binomial', link='logit')
    gam = GAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) +
              s(10) + s(11) + s(12) + s(13) + s(14) + s(15) + s(16) + s(17), distribution='binomial', link='logit')
    # gam = GAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) +
    #             s(10) + s(11) + s(12) + s(13) + s(14) + s(15) + s(16) + s(17) + s(18) +
    #             s(19) + s(20) + s(21) + s(22) + s(23) + s(24) + s(25) + s(26) + s(27) +
    #             s(28) + s(29) + s(30) + s(31) + s(32) + s(33) + s(34) + s(35) + s(36) +
    #             s(37) + s(38) + s(39) + s(40) + s(41) + s(42) + s(43) + s(44) + s(45) +
    #             s(46) + s(47) + s(48) + s(49) + s(50) + s(51) + s(52) + s(53) + s(54) +
    #             s(55) + s(56) + s(57) + s(58) + s(59) + s(60) + s(61) + s(62) + s(63) +
    #             s(64) + s(65) + s(66) + s(67) + s(68) + s(69) + s(70) + s(71) + f(cols-1), distribution='binomial', link='logit')

    # train
    gam.fit(X_train, y_train)

    # test
    y_pred = gam.predict(X_test)
    y_testing = y_test
    label_predictions = np.round(y_pred)
    accuracy = metrics.accuracy_score(y_testing, label_predictions)
    precision = metrics.precision_score(y_testing, label_predictions)
    recall = metrics.recall_score(y_testing, label_predictions)
    f1 = metrics.f1_score(y_testing, label_predictions)

    # Create Confusion Matrix
    create_confusion_matrix(y_testing, label_predictions, 'GAM')

    plots = 2
    if plots==1:
        XX = gam.generate_X_grid(term=1, n=500)
        XX[:,-1] = 1

        plt.plot(XX, gam.predict(XX), 'r--')
        # plt.plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--')

        plt.scatter(X_train[:,0], y_train, facecolor='gray', edgecolors='none')
        plt.title('95% prediction interval')
        plt.show()

    elif plots == 2:
        fig, axs = plt.subplots(1, 8)

        single = np.unique(X_train[:,-1])
        titles = all_features
        for i, ax in enumerate(axs):
            XX = gam.generate_X_grid(term=i)
            XX[:, -1] = single[0]
            pdep, confi = gam.partial_dependence(term=i, width=.95, X=XX)

            ax.plot(XX[:, i], pdep)
            ax.plot(XX[:, i], confi, c='r', ls='--')
            # ax.set_title(titles[i]);
        plt.show()

    return accuracy, precision, recall, f1, gam




def lstm_binary_class(X,y,training_ratio,all_features):

    class LSTMClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.3):
            super(LSTMClassifier, self).__init__()

            # LSTM layer
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

            # Fully connected layer
            self.fc = nn.Linear(hidden_dim, output_dim)

            # Sigmoid activation for binary classification
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # LSTM layer
            lstm_out, _ = self.lstm(x)

            # Get the last output of the sequence
            lstm_out = lstm_out[:, -1, :]

            # Fully connected layer
            out = self.fc(lstm_out)

            # Apply sigmoid for binary classification
            out = self.sigmoid(out)

            return out

    # Train Test Split
    Y = y
    # Split data based on trials first so no overlap of time series
    unique_trials = np.unique(X[:, -1])  # Get unique trial identifiers
    train_trials, test_trials = train_test_split(unique_trials, test_size=1-training_ratio, random_state=42)

    # Collect sequences for training and testing based on trial groups
    train_sequences = [X[X[:, -1] == trial, :-1] for trial in train_trials]
    test_sequences = [X[X[:, -1] == trial, :-1] for trial in test_trials]

    train_labels = [Y[X[:, -1] == trial] for trial in train_trials]
    test_labels = [Y[X[:, -1] == trial] for trial in test_trials]

    # Convert sequences and labels to tensors and create DataLoaders
    train_sequences_tensor = [torch.tensor(seq, dtype=torch.float32) for seq in train_sequences]
    train_labels_tensor = [torch.tensor(lbl, dtype=torch.float32) for lbl in train_labels]
    test_sequences_tensor = [torch.tensor(seq, dtype=torch.float32) for seq in test_sequences]
    test_labels_tensor = [torch.tensor(lbl, dtype=torch.float32) for lbl in test_labels]

    # Pack into TensorDataset and DataLoader for batch processing
    train_dataset = TensorDataset(torch.nn.utils.rnn.pad_sequence(train_sequences_tensor, batch_first=True),
                                  torch.nn.utils.rnn.pad_sequence(train_labels_tensor, batch_first=True))
    test_dataset = TensorDataset(torch.nn.utils.rnn.pad_sequence(test_sequences_tensor, batch_first=True),
                                 torch.nn.utils.rnn.pad_sequence(test_labels_tensor, batch_first=True))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Define model parameters
    input_dim = train_sequences_tensor[0].shape[1]
    hidden_dim = 64
    output_dim = 1  # Binary classification output
    num_layers = 2
    dropout = 0.3

    # Initialize model, loss function, and optimizer
    model = LSTMClassifier(input_dim, hidden_dim, output_dim, num_layers, dropout)
    criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(x_batch)
            y_batch = y_batch[:, -1].view(-1, 1)  # Take the label for the last time step
            loss = criterion(outputs, y_batch)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            # Get model predictions
            outputs = model(x_batch)
            preds = (outputs >= 0.5).float()  # Convert probabilities to binary predictions

            # Append predictions and true labels for metrics calculation
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch[:, -1].view(-1, 1).numpy())  # Use last label in each sequence for evaluation

    # Calculate metrics
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds)
    precision = metrics.precision_score(all_labels, all_preds)
    recall = metrics.recall_score(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")


    return accuracy, precision, recall, f1