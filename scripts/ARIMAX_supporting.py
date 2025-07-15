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
from statsmodels.tsa.statespace.sarimax import SARIMAX

from GLOC_visualization import prediction_time_plot


def make_objective(x_train, y_train, random_state, save_folder, objective_var):
    """
    ARIMAX Objective for Optuna hyperparameter tuning.
    Fits SARIMAX with exogenous variables (x_train).
    Evaluates on validation set using binary classification metrics after thresholding predictions.
    """

    def objective(trial):
        # Suggest ARIMA order params (p,d,q)
        p = trial.suggest_int('p', 0, 3)
        d = trial.suggest_int('d', 0, 1)
        q = trial.suggest_int('q', 0, 3)

        # Seasonal order (optional)
        P = trial.suggest_int('P', 0, 1)
        D = trial.suggest_int('D', 0, 1)
        Q = trial.suggest_int('Q', 0, 1)
        s = trial.suggest_categorical('s', [0, 12])  # seasonality period, 0 means no seasonality

        threshold = trial.suggest_float('threshold', 0.1, 0.9)

        # Train/validation split (simple split assuming time series order)
        split_idx = int(len(y_train)*0.8)
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
        x_tr, x_val = x_train[:split_idx], x_train[split_idx:]

        try:
            model = SARIMAX(
                y_tr,
                exog=x_tr,
                order=(p,d,q),
                seasonal_order=(P,D,Q,s) if s > 0 else (0,0,0,0),
                enforce_stationarity=False,
                enforce_invertibility=False,
                initialization='approximate_diffuse'
            )
            model_fit = model.fit(disp=False)

            # Predict on validation data
            pred = model_fit.predict(start=split_idx, end=len(y_train)-1, exog=x_val)
            pred_class = (pred > threshold).astype(int)

            # Compute metric (use F1 or other objective)
            if objective_var == 'F1':
                score = f1_score(y_val, pred_class)
            elif objective_var == 'Accuracy':
                score = accuracy_score(y_val, pred_class)
            else:
                score = f1_score(y_val, pred_class)  # fallback

        except Exception as e:
            print(f"ARIMAX training failed: {e}")
            return 0.0  # Fail gracefully in tuning

        return score

    return objective

def arimax_class(x_train, x_test, y_train, y_test, random_state, save_folder, objective_var='F1'):
    """
    ARIMAX model training, hyperparameter tuning with Optuna, and evaluation.
    """

    # Run Optuna tuning
    objective = make_objective(x_train, y_train, random_state, save_folder, objective_var)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    best_params = study.best_trial.params
    print("Best ARIMAX params:", best_params)

    # Fit best model on full training data
    try:
        if best_params['s'] > 0:
            seasonal_order = (best_params['P'], best_params['D'], best_params['Q'], best_params['s'])
        else:
            seasonal_order = (0,0,0,0)

        model = SARIMAX(
            y_train,
            exog=x_train,
            order=(best_params['p'], best_params['d'], best_params['q']),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            initialization='approximate_diffuse'
        )
        model_fit = model.fit(disp=False)

        # Predict on test data
        pred = model_fit.predict(start=len(y_train), end=len(y_train)+len(y_test)-1, exog=x_test)
        all_preds = (pred > best_params['threshold']).astype(int)

    except Exception as e:
        print(f"ARIMAX final training failed: {e}")
        return None

    all_labels = y_test

    # Assess Performance
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    precision = metrics.precision_score(all_labels, all_preds)
    recall = metrics.recall_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds)
    specificity = metrics.recall_score(all_labels, all_preds, pos_label=0)  # Specificity
    g_mean = geometric_mean_score(all_labels, all_preds)

    print("\nARIMAX Performance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision:", precision)
    print("Recall:   ", recall)
    print("F1 Score: ", f1)
    print("Specificity:", specificity)
    print("G-Mean:  ", g_mean)

    return accuracy, precision, recall, f1, specificity, g_mean









def make_objective(x_train, y_train, class_weights, random_state, save_folder, use_sampler, objective_var):
    """
    LSTM Objective Function for Optuna.

    Function objective (below) has access to all global arguments passed to this function.
    Returns Stopping Metric (here F1 score) as the objective (set to maximize)
        The Stopping Metric (stopping_metric) is what cuts off training during hyperparameter tuning
    Saves hyperparameters from the best validation performance for building final model

    """
    def objective(trial):
        # Hyperparameters
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5) if num_layers > 1 else 0
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        sequence_length = trial.suggest_int("sequence_length", 25, 250)
        stride = trial.suggest_float("stride", 0.25, 1.0)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        bidirectional = trial.suggest_categorical("bidirectional", [False])
        threshold = trial.suggest_float('threshold', 0.1, 0.9)
        step_size = round(sequence_length * stride)

        # Create training and validation sets from x_train/y_train
        train_dataset, val_dataset, train_windows_tensor, train_labels_tensor, val_windows_tensor, val_labels_tensor = (
            train_test_split_trials(
                x_train,y_train,sequence_length,step_size,test_ratio=0.2,random_state = random_state)
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
        torch.save(model.state_dict(), os.path.join(save_folder, f"TCN_best_model_trial_{trial.number}.pt"))

        return best_stopping_metric

    return objective

def lstm_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state, save_folder):
    """
        Main Temporal Convolutional Network Script
        Input: train and test split of data
        Output: model performance of trained model evaluated on test data
    """
    # User Options
    use_sampler = False # Optionally use sampler to sample the minority class (ROS)
    final_early_stop = False # Optionally use early stopping for final train (always uses early stop in tuning)
    objective_var = 'F1' # F1 or else use 1-Loss. (param used by Optuna and Early Stop during hyperparameter tuning)

    # Compute class weights to address imbalance (depending on class_weight_imb pass)
    class_weights = compute_class_weight(class_weight_imb, classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Perform Hyperparameter Tuning with Optuna using only Training data where Objective is F1 Score
    objective = make_objective(x_train, y_train, class_weights, random_state, save_folder, use_sampler,objective_var)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

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
    stride = best_params["stride"]
    weight_decay = best_params["weight_decay"]
    bidirectional = best_params["bidirectional"]
    step_size = round(sequence_length * stride)

    threshold = best_params['threshold']
    num_epochs = max(study.best_trial.user_attrs.get("best_epoch", 15),15) # enforce min of 10 epochs

    if final_early_stop:
        # Train with most training data but set aside a validation dataset for early stopping
        train_dataset, val_dataset, train_windows_tensor, _, _, train_labels_tensor = (
            train_test_split_trials(x_train, y_train, sequence_length, step_size, test_ratio=0.2)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        # Train with all training data and train to a finite set of epochs (from the best hyperparameter run)
        train_dataset, _, train_windows_tensor, _, _, train_labels_tensor = (
            train_test_split_trials(x_train, y_train, sequence_length, step_size, test_ratio=None)
        )

    # Create the test dataset, formatted into sequences
    test_dataset, _, _, _, _, _ = (
        train_test_split_trials(x_test, y_test, sequence_length, step_size, test_ratio=None)
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
    torch.save(model.state_dict(), os.path.join(save_folder, f"trained_model.pt"))

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
    print("\nPerformance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    return accuracy, precision, recall, f1, specificity, g_mean