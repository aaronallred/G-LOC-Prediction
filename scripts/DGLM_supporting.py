from DL_supporting import *
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
from sklearn.utils.class_weight import compute_class_weight

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import optuna
from GLOC_visualization import prediction_time_plot


class DGLM(PyroModule):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.coeff_scale = 1.0

    def model(self, x, y=None):
        batch_size, seq_len, input_dim = x.shape
        probs = []
        for t in range(seq_len):
            beta_t = pyro.sample(
                f"beta_{t}",
                dist.Normal(torch.zeros(input_dim), self.coeff_scale * torch.ones(input_dim)).to_event(1)
            )
            logit = (x[:, t, :] * beta_t).sum(-1)
            p = torch.sigmoid(logit)
            probs.append(p)
            if y is not None:
                pyro.sample(f"obs_{t}", dist.Bernoulli(p).to_event(0), obs=y[:, t])
        return probs

    def guide(self, x, y=None):
        seq_len, input_dim = x.shape[1], x.shape[2]
        for t in range(seq_len):
            mu_q = pyro.param(f"mu_q_{t}", torch.randn(input_dim))
            sigma_q = pyro.param(f"sigma_q_{t}", torch.ones(input_dim), constraint=dist.constraints.positive)
            pyro.sample(f"beta_{t}", dist.Normal(mu_q, sigma_q).to_event(1))

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            probs = self.model(x)
        return torch.stack(probs, dim=1)


def train_dglm(model, train_loader, num_epochs=30, lr=1e-3):
    pyro.clear_param_store()
    svi = SVI(model.model, model.guide, Adam({"lr": lr}), loss=Trace_ELBO())

    for epoch in range(num_epochs):
        total_loss = 0.
        for xb, yb in train_loader:
            xb, yb = xb.float(), yb.float()
            total_loss += svi.step(xb, yb)
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")


def evaluate_dglm(model, test_loader, threshold=0.5):
    all_preds, all_labels, probs_over_time = [], [], []

    for xb, yb in test_loader:
        probs = model.predict(xb.float())
        preds = (probs > threshold).int()
        all_preds.append(preds)
        all_labels.append(yb.int())
        probs_over_time.append(probs)

    all_preds = torch.cat(all_preds, dim=0).view(-1).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).view(-1).cpu().numpy()
    probs_over_time = torch.cat(probs_over_time, dim=0).cpu().numpy()
    return all_preds, all_labels, probs_over_time


def make_objective(x_train, y_train, random_state, save_folder):
    def objective(trial):
        # Hyperparameters
        sequence_length = trial.suggest_int("sequence_length", 50, 250)
        stride = trial.suggest_float("stride", 0.25, 1.0)
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        num_epochs = trial.suggest_int("num_epochs", 10, 40)
        threshold = trial.suggest_float("threshold", 0.1, 0.9)
        step_size = round(sequence_length * stride)

        # Train/Val Split
        train_dataset, val_dataset, _, _, _, _ = train_test_split_trials(
            x_train, y_train, sequence_length, step_size, test_ratio=0.2, random_state=random_state)

        workers = get_optimal_workers()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

        # Model
        input_dim = x_train.shape[2]
        model = DGLM(input_dim)

        # Train
        train_dglm(model, train_loader, num_epochs=num_epochs, lr=lr)

        # Evaluate
        preds, labels, _ = evaluate_dglm(model, val_loader, threshold=threshold)
        f1 = metrics.f1_score(labels, preds)
        trial.set_user_attr("best_epoch", num_epochs)

        # Save model param store
        os.makedirs(save_folder, exist_ok=True)
        pyro.get_param_store().save(os.path.join(save_folder, f"dglm_param_store_trial_{trial.number}.pt"))
        return f1

    return objective


def dglm_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state, save_folder):
    objective = make_objective(x_train, y_train, random_state, save_folder)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    best_params = study.best_trial.params
    print("Best Trial:", study.best_trial)

    # Final training setup
    sequence_length = best_params["sequence_length"]
    stride = best_params["stride"]
    batch_size = best_params["batch_size"]
    step_size = round(sequence_length * stride)
    lr = best_params["lr"]
    num_epochs = max(study.best_trial.user_attrs.get("best_epoch", 20), 10)
    threshold = best_params["threshold"]

    # Full Train
    train_dataset, _, _, _, _, _ = train_test_split_trials(
        x_train, y_train, sequence_length, step_size, test_ratio=None)
    test_dataset, _, _, _, _, _ = train_test_split_trials(
        x_test, y_test, sequence_length, step_size, test_ratio=None)

    workers = get_optimal_workers()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    input_dim = x_train.shape[2]
    model = DGLM(input_dim)
    train_dglm(model, train_loader, num_epochs=num_epochs, lr=lr)

    # Save final param store
    os.makedirs(save_folder, exist_ok=True)
    pyro.get_param_store().save(os.path.join(save_folder, f"trained_dglm_param_store.pt"))

    # Evaluate
    preds, labels, probs_over_time = evaluate_dglm(model, test_loader, threshold=threshold)
    prediction_time_plot(labels, preds, probs_over_time)

    accuracy = metrics.accuracy_score(labels, preds)
    precision = metrics.precision_score(labels, preds)
    recall = metrics.recall_score(labels, preds)
    f1 = metrics.f1_score(labels, preds)
    specificity = metrics.recall_score(labels, preds, pos_label=0)
    g_mean = geometric_mean_score(labels, preds)

    print("\nPerformance Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Specificity:", specificity)
    print("G-Mean:", g_mean)

    return accuracy, precision, recall, f1, specificity, g_mean