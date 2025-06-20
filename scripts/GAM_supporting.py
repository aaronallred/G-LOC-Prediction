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
import pickle
from pygam import LogisticGAM, s, f

from GLOC_visualization import prediction_time_plot

def extract_feature_summary(dataset):
    features, labels = [], []
    for x, y in dataset:
        # x shape: [seq_len, n_features]
        x_summary = x.mean(dim=0).numpy()  # now shape: [n_features]
        features.append(x_summary)
        labels.append(y.item())
    return np.array(features), np.array(labels)

def make_objective(x_train, y_train, class_weights, random_state, save_folder, use_sampler, objective_var):
    def objective(trial):
        # Hyperparameters for pyGAM
        lam = trial.suggest_float("lam", 0.1, 10.0, log=True)
        n_splines = trial.suggest_int("n_splines", 5, 30)
        sequence_length = trial.suggest_int("sequence_length", 1, 25)
        stride = trial.suggest_float("stride", 0.25, 1.0)
        step_size = round(sequence_length * stride)

        # Split into windows
        train_dataset, val_dataset, *_ = train_test_split_trials(
            x_train, y_train, sequence_length, step_size, test_ratio=0.2, random_state = random_state, end_label=True)

        # Extract fixed-length features
        x_train_flat, y_train_flat = extract_feature_summary(train_dataset)
        x_val_flat, y_val_flat = extract_feature_summary(val_dataset)

        # Build and train pyGAM model
        n_features = x_train_flat.shape[1]
        terms = sum([s(i,n_splines=n_splines) for i in range(n_features)], start=s(0))
        gam = LogisticGAM(terms, lam=lam)
        gam.fit(x_train_flat, y_train_flat)

        preds = gam.predict(x_val_flat)
        preds = np.round(preds)

        if objective_var == "F1":
            score = metrics.f1_score(y_val_flat, preds)
        elif objective_var == "Acc":
            score = metrics.accuracy_score(y_val_flat, preds)
        else:
            score = -metrics.log_loss(y_val_flat, preds)

        return score

    return objective

def gam_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state, all_features, save_folder):
    """
        Main Transformer Script
        Input: train and test split of data
        Output: model performance of trained model evaluated on test data
    """
    # User Options
    use_sampler = False # Optionally use sampler to sample the minority class (ROS)
    objective_var = 'F1' # 'F1' 'Acc' or else uses 1-Loss. (used by Optuna)

    # Compute class weights to address imbalance (depending on class_weight_imb pass)
    class_weights = compute_class_weight(class_weight_imb, classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Perform Hyperparameter Tuning with Optuna using only Training data
    objective = make_objective(x_train, y_train, class_weights, random_state, save_folder, use_sampler, objective_var)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    # Get best trial hyperparameters
    best_params = study.best_trial.params
    sequence_length = best_params["sequence_length"]
    stride = best_params["stride"]
    step_size = round(sequence_length * stride)

    # Extract final training and testing sets
    train_dataset, _, *_ = train_test_split_trials(x_train, y_train, sequence_length, step_size, test_ratio=None,
                                                   random_state = random_state, end_label=True)
    test_dataset, _, *_ = train_test_split_trials(x_test, y_test, sequence_length, step_size, test_ratio=None,
                                                  random_state = random_state, end_label=True)

    # Extract fixed-length features
    x_train_flat, y_train_flat = extract_feature_summary(train_dataset)
    x_test_flat, y_test_flat = extract_feature_summary(test_dataset)

    # Train final pyGAM model
    n_features = x_train_flat.shape[1]
    terms = sum([s(i, n_splines=best_params["n_splines"]) for i in range(n_features)], start=s(0))
    for i in range(1, n_features):
        terms += s(i, n_splines=best_params["n_splines"])
    gam = LogisticGAM(terms, lam=best_params["lam"])
    gam.fit(x_train_flat, y_train_flat)

    # Save
    os.makedirs(save_folder, exist_ok=True)
    filename = "logistic_gam_model.pkl"
    filepath = os.path.join(save_folder, filename)
    with open(filepath, "wb") as f:
        pickle.dump(gam, f)
    print(f"Model saved to: {filepath}")

    # Evaluate
    preds = np.round(gam.predict(x_test_flat))
    prediction_time_plot(y_test_flat, preds, preds)  # predictors_over_time dummy = preds

    accuracy = metrics.accuracy_score(y_test_flat, preds)
    precision = metrics.precision_score(y_test_flat, preds)
    recall = metrics.recall_score(y_test_flat, preds)
    f1 = metrics.f1_score(y_test_flat, preds)
    specificity = metrics.recall_score(y_test_flat, preds, pos_label=0)
    g_mean = geometric_mean_score(y_test_flat, preds)

    print("\nPerformance Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Specificity: ", specificity)
    print("G-Mean: ", g_mean)

    return accuracy, precision, recall, f1, specificity, g_mean