from DL_supporting import *
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import TensorDataset
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from imblearn.metrics import geometric_mean_score
from sklearn.utils.class_weight import compute_class_weight
import optuna

from GLOC_visualization import prediction_time_plot

# Build LSTM architecture
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.3, bidirectional=False):

        super(LSTMClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :] # If we were predicting a single output label e.g., traditional learners
        out = self.fc(last_out)  # (batch, seq_len, 1)

        return out


def make_objective(x_train, y_train, class_weights, random_state, save_folder, use_sampler, objective_var, all_features):
    """
    LSTM Objective Function for Optuna.

    Function objective (below) has access to all global arguments passed to this function.
    Returns Stopping Metric (here F1 score) as the objective (set to maximize)
        The Stopping Metric (stopping_metric) is what cuts off training during hyperparameter tuning
    Saves hyperparameters from the best validation performance for building final model

    """
    def objective(trial):
        # Hyperparameters
        baseline_method = trial.suggest_categorical("baseline_method", [0, 1, 2, 3, 4, 5])
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5) if num_layers > 1 else 0
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        sequence_length = trial.suggest_int("sequence_length", 25, 250)
        step_size = trial.suggest_int("step_size", 25, 75)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        bidirectional = trial.suggest_categorical("bidirectional", [False])
        threshold = trial.suggest_float('threshold', 0.1, 0.9)

        x_train_ds, _ = baseline_down_select(x_train, all_features, baseline_method)

        # Create training and validation sets from x_train/y_train
        train_dataset, val_dataset, train_windows_tensor, train_labels_tensor, val_windows_tensor, val_labels_tensor = (
            train_test_split_trials(
                x_train_ds,y_train,sequence_length,step_size,test_ratio=0.2, random_state = random_state, end_label=True)
        )

        # Prepare the data for training and evaluation
        sampler = build_sampler(train_labels_tensor, class_weights) if use_sampler else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Build model
        input_dim = train_windows_tensor.shape[2]
        model = LSTMClassifier(input_dim, hidden_dim, 1, num_layers, dropout, bidirectional)
        device = get_device()
        model.to(device)
        criterion, optimizer = build_training_components(model, class_weights, learning_rate, weight_decay, device)

        # Train and Evaluate with Early Stopping
        best_model_state, best_epoch, best_stopping_metric = train_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer, device, threshold, objective_var)

        # Restore best weights
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Store best epoch in trial attributes
        trial.set_user_attr("best_epoch", best_epoch)

        # Save model to file
        os.makedirs(save_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_folder, f"LSTM_best_model_trial_{trial.number}.pt"))

        return best_stopping_metric

    return objective

def lstm_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state, all_features, save_folder):
    """
        Main Temporal Convolutional Network Script
        Input: train and test split of data
        Output: model performance of trained model evaluated on test data
    """
    # User Options
    use_sampler = True # Optionally use sampler to sample the minority class (ROS)
    final_early_stop = False # Optionally use early stopping for final train (always uses early stop in tuning)
    objective_var = 'F1' # F1 or else use 1-Loss. (param used by Optuna and Early Stop during hyperparameter tuning)

    # Compute class weights to address imbalance (depending on class_weight_imb pass)
    class_weights = compute_class_weight(class_weight_imb, classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Perform Hyperparameter Tuning with Optuna using only Training data where Objective is F1 Score
    objective = make_objective(x_train, y_train, class_weights, random_state, save_folder, use_sampler,
                               objective_var, all_features)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=4)

    # Print out the optimal hyperparameters
    best_params = study.best_trial.params
    print("Best Trial:", study.best_trial)

    # Grab the hyperparameters from the best set
    hidden_dim = best_params["hidden_dim"]
    num_layers = best_params["num_layers"]
    dropout = best_params["dropout"] if num_layers > 1 else 0
    learning_rate = best_params["lr"]
    batch_size = best_params["batch_size"]
    sequence_length = best_params["sequence_length"]
    step_size = best_params["step_size"]
    weight_decay = best_params["weight_decay"]
    bidirectional = best_params["bidirectional"]

    threshold = best_params['threshold']
    num_epochs = max(study.best_trial.user_attrs.get("best_epoch", 15),15) # enforce min of 10 epochs

    baseline_method = best_params["baseline_method"]
    x_train_ds, _ = baseline_down_select(x_train, all_features, baseline_method)
    x_test_ds, _ = baseline_down_select(x_test, all_features, baseline_method)

    # Build training (potential validation) datasets for final train
    if final_early_stop:
        # Train with most training data but set aside a validation dataset for early stopping
        train_dataset, val_dataset, train_windows_tensor, train_labels_tensor, _, _ = (
            train_test_split_trials(x_train_ds, y_train, sequence_length, step_size, test_ratio=0.2
                                    ,random_state=random_state, end_label=True)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        # Train with all training data and train to a finite set of epochs (from the best hyperparameter run)
        train_dataset, _, train_windows_tensor, train_labels_tensor, _, _ = (
            train_test_split_trials(x_train_ds, y_train, sequence_length, step_size, test_ratio=None,
                                    random_state=random_state, end_label=True)
        )

    # Create the test dataset, formatted into sequences
    test_dataset, _, _, _, _, _ = (
        train_test_split_trials(x_test_ds, y_test, sequence_length, step_size=10, test_ratio=None,
                                random_state=random_state, end_label=True)
    )

    # Build model
    input_dim = train_windows_tensor.shape[2]
    model = LSTMClassifier(input_dim, hidden_dim, 1, num_layers, dropout, bidirectional)
    device = get_device()
    model.to(device)
    criterion, optimizer = build_training_components(model, class_weights, learning_rate, weight_decay, device)

    # Prepare the data for training
    sampler = build_sampler(train_labels_tensor, class_weights) if use_sampler else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    # Train model
    if final_early_stop:
        best_model_state, _, _ = train_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer, device, threshold, objective_var)
        if best_model_state:
            model.load_state_dict(best_model_state)

    else:
        for epoch in range(num_epochs):
            train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)

    # Save trained model
    os.makedirs(save_folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_folder, f"LSTM_trained_model.pt"))

    # Evaluate final model
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels, predictors_over_time,_ = evaluate(model, test_loader,  threshold, device, criterion)

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    predictors_over_time = np.array(predictors_over_time)
    prediction_time_plot(all_labels, all_preds, predictors_over_time)

    # Assess Performance
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    precision = metrics.precision_score(all_labels, all_preds)
    recall = metrics.recall_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds)
    specificity = metrics.recall_score(all_labels, all_preds, pos_label=0)  # Specificity
    g_mean = geometric_mean_score(all_labels, all_preds)

    # Print performance metrics
    print("\nPerformance Metrics for LSTM:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    return accuracy, precision, recall, f1, specificity, g_mean