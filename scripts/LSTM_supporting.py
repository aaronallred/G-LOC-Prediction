import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import TensorDataset
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

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

def create_windows(sequence, labels, window_size, step_size):
    """Creates windows of a given size with a certain step from to structure sequences as inputs to paired labels."""
    windows = []
    window_labels = []
    for start in range(0, len(sequence) - window_size + 1, step_size):
        end = start + window_size
        windows.append(sequence[start:end])
        window_labels.append(max(labels[start:end]))  # Take the maximum label within the window
    return windows, window_labels

def train_test_split_trials(X,Y,window_size,step_size,training_ratio):
    # Creates train test split based on trials
    # Split data by trials
    unique_trials = np.unique(X[:, -1])  # Get unique trial identifiers
    train_trials, test_trials = train_test_split(unique_trials, test_size=1 - training_ratio, random_state=42)

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
    train_windows_tensor = torch.tensor(train_windows, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).view(-1, 1)
    test_windows_tensor = torch.tensor(test_windows, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(train_windows_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_windows_tensor, test_labels_tensor)

    return (train_dataset, test_dataset,
            train_windows_tensor, train_labels_tensor, test_windows_tensor, test_labels_tensor)


# Objective function for Optuna
def make_objective(x_train, y_train, class_weights):
    def objective(trial):
        # Hyperparameters
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        sequence_length = trial.suggest_int("sequence_length", 20, 100, step=10)
        stride = trial.suggest_float("stride", 0.1, 2.0, step=0.1)
        step_size = round(sequence_length * stride)

        # Create training and validation sets from x_train/y_train
        train_dataset, val_dataset, train_windows_tensor, train_labels_tensor, val_windows_tensor, val_labels_tensor = (
            train_test_split_trials(x_train, y_train, sequence_length, step_size, training_ratio=0.8)
        )

        # Ensure class_weights are a tensor
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

        # Create sampler for class imbalance
        sample_weights = class_weights_tensor[train_labels_tensor]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        input_dim = train_windows_tensor.shape[2]
        model = LSTMClassifier(input_dim, hidden_dim, 1, num_layers, dropout)

        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        model.to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.BCELoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Early stopping params
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
                    print(f"Early stopping at epoch {epoch+1} with best F1 score: {best_f1:.4f}")
                    break

        # Restore best weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save model to file
        torch.save(model.state_dict(), f"best_model_trial_{trial.number}.pt")

        return best_f1

    return objective