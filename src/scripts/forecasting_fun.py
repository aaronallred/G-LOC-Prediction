import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import numpy as np

# Splits dataset based on trial identifier (located in the last column of predictors)
def train_test_split_trials_forecast(X, Y, window_size, step_size, test_ratio, random_state=42,
                            end_label=False, horizon=50):
    """
    Creates a train-test split based on trials
    Split data by trials
    Split trials into sequences (with horizon)
    """
    unique_trials = np.unique(X[:, -1])
    train_trials, test_trials = train_test_split(unique_trials, test_size=test_ratio,
                                                 random_state=random_state)

    # Prepare train and test windows
    train_windows, train_labels = [], []
    test_windows, test_labels = [], []

    for trial in train_trials:
        trial_sequence = X[X[:, -1] == trial, :-1]  # Exclude trial identifier
        trial_labels = Y[X[:, -1] == trial]
        windows, labels = create_windows_forecast(trial_sequence, trial_labels,
                                         window_size, step_size, end_label, horizon)
        train_windows.extend(windows)
        train_labels.extend(labels)

    for trial in test_trials:
        trial_sequence = X[X[:, -1] == trial, :-1]
        trial_labels = Y[X[:, -1] == trial]
        windows, labels = create_windows_forecast(trial_sequence, trial_labels,
                                         window_size, step_size, end_label, horizon)
        test_windows.extend(windows)
        test_labels.extend(labels)

    # Convert data to torch tensors and create datasets
    train_windows_tensor = torch.tensor(np.array(train_windows), dtype=torch.float32)
    train_labels_tensor = torch.tensor(np.array(train_labels), dtype=torch.float32)
    test_windows_tensor = torch.tensor(np.array(test_windows), dtype=torch.float32)
    test_labels_tensor = torch.tensor(np.array(test_labels), dtype=torch.float32)

    train_dataset = TensorDataset(train_windows_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_windows_tensor, test_labels_tensor)

    return (train_dataset, test_dataset,
            train_windows_tensor, train_labels_tensor,
            test_windows_tensor, test_labels_tensor)


# Creates windowed sequences
def create_windows_forecast(sequence, labels, window_size, step_size, end_label=True, horizon=25):
    """
    Creates windows with paired labels, considering the forecasting horizon
    """

    windows = []
    window_labels = []
    for start in range(0, len(sequence) - window_size - horizon + 1, step_size):
        end = start + window_size
        windows.append(sequence[start:end])

        if end_label:
            window_labels.append(labels[end - 1 + horizon])  # label at horizon
        else:
            window_labels.append(labels[start + horizon : end + horizon])  # extended labels

    return windows, window_labels