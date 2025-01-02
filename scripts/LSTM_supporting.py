import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import numpy as np

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
    train_trials, test_trials = train_test_split(unique_trials, test_size=1 - training_ratio, random_state=60)


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