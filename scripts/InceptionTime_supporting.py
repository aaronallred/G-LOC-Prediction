import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import optuna
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# InceptionTime blocks and model
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[10, 20, 40], bottleneck_channels=32):
        super(InceptionBlock, self).__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False) if in_channels > 1 else nn.Identity()

        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_channels if in_channels > 1 else 1, out_channels, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batchnorm = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x_bottleneck = self.bottleneck(x)
        out = [conv(x_bottleneck) for conv in self.convs]
        out.append(self.conv_pool(self.maxpool(x)))
        out = torch.cat(out, dim=1)
        return self.relu(self.batchnorm(out))

class InceptionTime(nn.Module):
    def __init__(self, input_dim, n_classes=1, num_blocks=3, in_channels=1, out_channels=32):
        super(InceptionTime, self).__init__()
        self.blocks = nn.Sequential(*[
            InceptionBlock(in_channels if i == 0 else out_channels * 4, out_channels)
            for i in range(num_blocks)
        ])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels * 4, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, time)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

# Trial-wise split
def train_test_split_trials(X,Y,window_size,step_size,test_ratio):
    # Creates train test split based on trials
    # Split data by trials
    unique_trials = np.unique(X[:, -1])  # Get unique trial identifiers
    train_trials, test_trials = train_test_split(unique_trials, test_size = test_ratio, random_state=42)

    # Prepare train and test windows
    train_windows, train_labels = [], []
    test_windows, test_labels = [], []

    for trial in train_trials:
        trial_sequence = X[X[:, -1] == trial, :-1]  # Exclude trial identifier
        trial_labels = Y[X[:, -1] == trial]
        windows, labels = create_windows(trial_sequence, trial_labels, window_size, step_size)
        train_windows.extend(windows)
        train_labels.extend(labels)

    for trial in test_trials:
        trial_sequence = X[X[:, -1] == trial, :-1]
        trial_labels = Y[X[:, -1] == trial]
        windows, labels = create_windows(trial_sequence, trial_labels, window_size, step_size)
        test_windows.extend(windows)
        test_labels.extend(labels)

    # Convert data to torch tensors and create DataLoaders
    train_windows_tensor = torch.tensor(np.array(train_windows), dtype=torch.float32)
    train_labels_tensor = torch.tensor(np.array(train_labels), dtype=torch.float32).view(-1, 1)
    test_windows_tensor = torch.tensor(np.array(test_windows), dtype=torch.float32)
    test_labels_tensor = torch.tensor(np.array(test_labels), dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(train_windows_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_windows_tensor, test_labels_tensor)

    return (train_dataset, test_dataset,
            train_windows_tensor, train_labels_tensor, test_windows_tensor, test_labels_tensor)

# Objective function
def make_objective(x_train, y_train, class_weights, save_folder):
    def objective(trial):
        out_channels = trial.suggest_categorical("out_channels", [16, 32, 64])
        num_blocks = trial.suggest_int("num_blocks", 2, 6)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        sequence_length = trial.suggest_int("sequence_length", 500, 1000)
        stride = trial.suggest_float("stride", 0.25, 1.0)
        step_size = round(sequence_length * stride)

        # Create training and validation sets from x_train/y_train
        train_dataset, val_dataset, train_windows_tensor, train_labels_tensor, val_windows_tensor, val_labels_tensor =(
            train_test_split_trials(x_train, y_train, sequence_length, step_size, test_ratio=0.2))

        input_dim = train_windows_tensor.shape[2]
        model = InceptionTime(input_dim, num_blocks=num_blocks, out_channels=out_channels)

        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        model.to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Early stopping parameters for cutting off runs if validation performance does not improve
        best_f1 = 0.0
        patience = 5
        wait = 0
        num_epochs = 50
        best_model_state = None

        for epoch in range(num_epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    preds = (outputs >= 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())

            val_f1 = metrics.f1_score(all_labels, all_preds)
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation F1 Score: {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                wait = 0
                best_model_state = model.state_dict()
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch + 1} with best F1 score: {best_f1:.4f}")
                    break

        # Restore best weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save model to file
        os.makedirs(save_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_folder, f"LSTM_best_model_trial_{trial.number}.pt"))

        return best_f1

    return objective

# Final training and evaluation
def inceptiontime_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state, save_folder):

    # Compute class weights to address imbalance
    class_weights = compute_class_weight(class_weight_imb, classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Perform Hyperparameter Tuning with Optuna using only Training data
    objective = make_objective(x_train, y_train, class_weights, save_folder)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    best_epoch = study.best_trial.user_attrs["best_epoch"]

    # Train with most training data
    train_dataset, val_dataset, train_windows_tensor, _, _, train_labels_tensor = (
        train_test_split_trials(x_train, y_train, sequence_length, step_size, test_ratio=0.2)
    )

    # Create the test dataset format
    test_dataset, _, _, _, _, _ = (
        train_test_split_trials(x_test, y_test, sequence_length, step_size, test_ratio=None)
    )

    model = InceptionTime(input_dim=x_train.shape[2], num_blocks=best_params["num_blocks"],
                          out_channels=best_params["out_channels"])

    # Move device to the model after it is created
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model.to(device)

    # Create sample weights using the windowed labels
    sample_weights = class_weights[train_labels_tensor.long()].squeeze()  # Use the windowed labels here

    # Weighted Random Sampler for the windowed data
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    # Prepare the DataLoader with the sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Early stopping parameters
    patience = 5  # How many epochs to wait before stopping if no improvement
    best_val_f1 = 0
    epochs_without_improvement = 0  # Counter for epochs without improvement
    num_epochs = 50

    # Train model with early stopping
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 100 == 0:  # Print loss every 100 batches for verbosity
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Early stopping check: Monitor validation loss
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                preds = (outputs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_f1 = metrics.f1_score(all_labels, all_preds)
        print(f"Validation F1 Score: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            # Save Model
            print("Validation F1 improved, saving model...")
            os.makedirs(save_folder, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_folder, f"trained_model.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    # Evaluate final model
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds, all_labels, predictors_over_time = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            # Append the input data for plotting predictors
            predictors_over_time.extend(x_batch.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # predictors_over_time = np.array(predictors_over_time)
    # prediction_time_plot(all_labels, all_preds, predictors_over_time)

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
