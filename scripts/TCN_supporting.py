from DL_supporting import *
import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np
import os
from torch.utils.data import DataLoader
from imblearn.metrics import geometric_mean_score
from sklearn.utils.class_weight import compute_class_weight
import optuna
import gc
import joblib
import json

from GLOC_visualization import prediction_time_plot

# Build TCN architecture
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        for layer in [self.conv1, self.conv2, self.downsample] if self.downsample else [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        # Match sequence lengths before addition
        if res.size(-1) != out.size(-1):
            min_len = min(res.size(-1), out.size(-1))
            res = res[:, :, -min_len:]
            out = out[:, :, -min_len:]

        return out + res

class TCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, kernel_size=3, dropout=0.3):
        super(TCNClassifier, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else hidden_dim
            layers.append(
                TemporalBlock(in_channels, hidden_dim, kernel_size, stride=1,
                              dilation=dilation_size, padding=(kernel_size-1)*dilation_size, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input shape: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        out = self.network(x)
        # Take last time step (assumes full sequence processed)
        out = out.permute(0, 2, 1)  # (batch, features, seq_len) → (batch, seq_len, features)
        out = out[:,-1,:]
        return self.fc(out)  # (batch, seq_len, 1)


def make_objective(x_train, y_train, class_weights, random_state, save_folder, use_sampler, objective_var, all_features):
    """
    TCN Objective Function for Optuna.

    Function objective (below) has access to all global arguments passed to this function.
    Returns Stopping Metric (here F1 score) as the objective (set to maximize)
        The Stopping Metric (stopping_metric) is what cuts off training during hyperparameter tuning
    Saves hyperparameters from the best validation performance for building final model

    """
    def objective(trial):
        # Hyperparameters
        baseline_method = trial.suggest_categorical("baseline_method", [0, 1, 2, 3, 4, 5])
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        optimizer_type = trial.suggest_categorical("optimizer_type", ['AdamW'])
        momentum = 0.9 if optimizer_type == "SGD" else None
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5) if num_layers > 1 else 0
        sequence_length = trial.suggest_int("sequence_length", 25, 250)
        step_size = trial.suggest_int("step_size", 25, 75)
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
        threshold = trial.suggest_float('threshold', 0.1, 0.9)

        x_train_ds, _ = baseline_down_select(x_train, all_features, baseline_method)

        # step_size = round(sequence_length * stride)
        min_length = (kernel_size - 1) * (2 ** (num_layers - 1)) + 1
        if sequence_length < min_length:
            raise optuna.exceptions.TrialPruned()

        # Create training and validation sets from x_train/y_train
        train_dataset, val_dataset, train_windows_tensor, train_labels_tensor, val_windows_tensor, val_labels_tensor = (
            train_test_split_trials(
                x_train_ds,y_train,sequence_length,step_size,test_ratio=0.2,random_state = random_state, end_label=True)
        )

        # Prepare the data for training and evaluation
        workers = get_optimal_workers()
        sampler = build_sampler(train_labels_tensor, class_weights) if use_sampler else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

        # Build model
        input_dim = train_windows_tensor.shape[2]
        model = TCNClassifier(input_dim, hidden_dim, 1, num_layers, kernel_size=kernel_size, dropout=dropout)
        device = get_device()
        model.to(device)
        criterion, optimizer = build_training_components(model, class_weights, learning_rate, weight_decay, momentum,
                                                         device, loss='BCE', optimizer_type=optimizer_type)

        # Train and Evaluate with Early Stopping
        best_model_state, best_epoch, best_stopping_metric = train_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer, device, threshold, objective_var)

        # Restore best weights
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Store best epoch in trial attributes
        trial.set_user_attr("best_epoch", best_epoch)

        # Save model to file
        try:
            best_value = trial.study.best_value
        except (ValueError, AttributeError):
            best_value = float("-inf")
        if best_stopping_metric >= best_value:
            os.makedirs(save_folder, exist_ok=True)

            # Save model weights
            torch.save(model.state_dict(), os.path.join(save_folder, "TCN_best_model.pt"))

            # Save hyperparameters
            with open(os.path.join(save_folder, "TCN_best_params.json"), "w") as f:
                json.dump(trial.params, f, indent=4)

            # Save additional metadata
            metadata = {
                "trial_number": trial.number,
                "objective_value": best_stopping_metric,
                "best_epoch": trial.user_attrs.get("best_epoch", None)
            }
            with open(os.path.join(save_folder, "TCN_best_trial_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)

        del model
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        return best_stopping_metric

    return objective

def tcn_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state, all_features, save_folder):
    """
        Main Temporal Convolutional Network Script
        Input: train and test split of data
        Output: model performance of trained model evaluated on test data
    """
    # User Options
    use_sampler = True # Optionally use sampler to sample the minority class (ROS)
    final_early_stop = False # Optionally use early stopping for final train (always uses early stop in tuning)
    objective_var = 'F1' # 'F1' 'Acc' or else uses 1-Loss. (used by Optuna and Early Stop during hyperparameter tuning)
    trials = 100 # The number of trials in for the Bayesian Search

    # Compute class weights to address imbalance (depending on class_weight_imb pass)
    class_weights = compute_class_weight(class_weight_imb, classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Perform Hyperparameter Tuning with Optuna using only Training data where Objective is F1 Score
    objective = make_objective(x_train, y_train, class_weights, random_state, save_folder, use_sampler,
                               objective_var, all_features)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, catch=(RuntimeError, ValueError))

    # Print out the optimal hyperparameters
    study_path = os.path.join(save_folder, "TCN_optuna_study.pkl")
    joblib.dump(study, study_path)
    best_params = study.best_trial.params
    print("Best Trial:", study.best_trial)

    # Grab the hyperparameters from the best set
    batch_size = best_params["batch_size"]
    optimizer_type = best_params["optimizer_type"]
    momentum = 0.9 if optimizer_type == "SGD" else None
    learning_rate = best_params["lr"]
    weight_decay = best_params["weight_decay"]
    hidden_dim = best_params["hidden_dim"]
    num_layers = best_params["num_layers"]
    dropout = best_params["dropout"] if num_layers > 1 else 0
    kernel_size = best_params['kernel_size']
    sequence_length = best_params["sequence_length"]
    step_size = best_params["step_size"]
    threshold = best_params['threshold']
    num_epochs = max(study.best_trial.user_attrs.get("best_epoch", 15),15) # enforce min of 10 epochs

    baseline_method = best_params["baseline_method"]
    x_train_ds, _ = baseline_down_select(x_train, all_features, baseline_method)
    x_test_ds, _ = baseline_down_select(x_test, all_features, baseline_method)

    # Build training (potential validation) datasets for final train
    workers = get_optimal_workers()
    if final_early_stop:
        # Train with most training data but set aside a validation dataset for early stopping
        train_dataset, val_dataset, train_windows_tensor, train_labels_tensor, _, _ = (
            train_test_split_trials(x_train_ds, y_train, sequence_length, step_size, test_ratio=0.2,
                                    random_state = random_state, end_label=True)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    else:
        # Train with all training data and train to a finite set of epochs (from the best hyperparameter run)
        train_dataset, _, train_windows_tensor, train_labels_tensor, _, _ = (
            train_test_split_trials(x_train_ds, y_train, sequence_length, step_size, test_ratio=None,
                                    random_state = random_state, end_label=True)
        )

    # Create the test dataset, formatted into sequences
    test_dataset, _, _, _, _, _ = (
        train_test_split_trials(x_test_ds, y_test, sequence_length, step_size=10, test_ratio=None,
                                random_state = random_state, end_label=True)
    )

    # Build model
    input_dim = train_windows_tensor.shape[2]
    model = TCNClassifier(input_dim, hidden_dim, 1, num_layers, kernel_size=kernel_size, dropout=dropout)
    device = get_device()
    model.to(device)
    criterion, optimizer = build_training_components(model, class_weights, learning_rate, weight_decay, momentum,
                                                     device, loss='BCE', optimizer_type=optimizer_type)

    # Prepare the data for training
    sampler = build_sampler(train_labels_tensor, class_weights) if use_sampler else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=workers)

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
    torch.save(model.state_dict(), os.path.join(save_folder, f"TCN_trained_model.pt"))

    # Evaluate final model
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
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
    print("\nPerformance Metrics for TCN:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    return accuracy, precision, recall, f1, specificity, g_mean