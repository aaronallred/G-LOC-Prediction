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
        return self.fc(out)  # (batch, seq_len, 1)

def build_sampler(train_labels_tensor, class_weights):
    # Determine if window has GLOC event
    mean_labels = train_labels_tensor.float().mean(dim=1)  # shape: (num_windows,)
    binary_labels = (mean_labels > 0).long()

    # Apply class weights
    sample_weights = class_weights[binary_labels]  # shape: (num_windows,)
    # Create sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    return sampler

# Define a differentiable F1 Loss function
class SoftF1Loss(nn.Module):
    def __init__(self, pos_weight=1.0, epsilon=1e-7):
        super().__init__()
        self.pos_weight = pos_weight
        self.epsilon = epsilon

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Apply weights: positive samples get pos_weight, negative get 1
        weights = torch.ones_like(targets)
        weights[targets == 1] = self.pos_weight

        # Weighted true positives, false positives, false negatives
        tp = (probs * targets * weights).sum()
        fp = (probs * (1 - targets)).sum()  # FP not weighted
        fn = ((1 - probs) * targets * weights).sum()

        soft_f1 = 2 * tp / (2 * tp + fp + fn + self.epsilon)
        return 1 - soft_f1

# Define Training and Evaluation Functions
def train(model, loader, criterion, optimizer, device, epoch, num_epochs, verbose = 1):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        loss = criterion(outputs.view(-1), y_batch.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if verbose == 2:  # Print loss every 100 batches for verbosity
            print(f"Epoch [{epoch + 1}/{num_epochs}] Batch Loss: {loss.item():.16f}")

    if verbose > 0:
        avg_epoch_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Epoch Loss: {avg_epoch_loss:.16f}")

# Define Evaluation Loop
def evaluate(model, loader, threshold, device):
    model.eval()
    all_preds, all_labels, predictors_over_time = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            preds = (outputs >= threshold).float()
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(y_batch.view(-1).cpu().numpy())

            # Append the input data for plotting predictors
            predictors_over_time.extend(x_batch.cpu().numpy())

    return all_preds, all_labels, predictors_over_time

# Define Evaluation with early stopping
def evaluate_with_early_stopping(model, val_loader,  threshold, device, best_model_state,
                                 best_f1=0,patience_counter=0, patience=5, epoch = None, num_epochs= None):

    # Run Evaluation
    all_preds, all_labels, _ = evaluate(model, val_loader, threshold, device)

    # Calculate F1 metric and use for determining stopping
    f1 = metrics.f1_score(all_labels, all_preds)
    print(f"Epoch [{epoch + 1}/{num_epochs}]: Evaluation F1 Score = {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
    stop_early = patience_counter >= patience

    return f1, best_f1, patience_counter, stop_early, best_model_state

# Define Data Handling Functions
def create_windows(sequence, labels, window_size, step_size):
    # Creates windows with paired labels.
    windows = []
    window_labels = []
    for start in range(0, len(sequence) - window_size + 1, step_size):
        end = start + window_size
        windows.append(sequence[start:end])
        window_labels.append(labels[start:end])  # Take the maximum label within the window
    return windows, window_labels

def train_test_split_trials(X,Y,window_size,step_size,test_ratio, random_state = 42):
    # Creates train test split based on trials
    # Split data by trials
    unique_trials = np.unique(X[:, -1])  # Get unique trial identifiers
    train_trials, test_trials = (
        train_test_split(unique_trials, test_size = test_ratio, random_state=random_state))

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
    train_labels_tensor = torch.tensor(np.array(train_labels), dtype=torch.float32)
    test_windows_tensor = torch.tensor(np.array(test_windows), dtype=torch.float32)
    test_labels_tensor = torch.tensor(np.array(test_labels), dtype=torch.float32)

    train_dataset = TensorDataset(train_windows_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_windows_tensor, test_labels_tensor)

    return (train_dataset, test_dataset,
            train_windows_tensor, train_labels_tensor, test_windows_tensor, test_labels_tensor)


# Objective function for Optuna
def make_objective(x_train, y_train, class_weights, random_state, save_folder, use_sampler):
    def objective(trial):
        # Hyperparameters
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5) if num_layers > 1 else 0
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [ 32, 64])
        sequence_length = trial.suggest_int("sequence_length", 50, 100)
        stride = trial.suggest_float("stride", 0.25, 1.0)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        threshold = trial.suggest_float('threshold', 0.1, 0.9)

        step_size = round(sequence_length * stride)

        # Create training and validation sets from x_train/y_train
        train_dataset, val_dataset, train_windows_tensor, train_labels_tensor, val_windows_tensor, val_labels_tensor = (
            train_test_split_trials(
                x_train,
                y_train,
                sequence_length,
                step_size,
                test_ratio=0.2,
                random_state = random_state)
        )

        if use_sampler:
            sampler = build_sampler(train_labels_tensor, class_weights)
        else:
            sampler = None

        input_dim = train_windows_tensor.shape[2]
        model = TCNClassifier(input_dim, hidden_dim, 1, num_layers, kernel_size=kernel_size, dropout=dropout)

        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        model.to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
        criterion = SoftF1Loss(pos_weight=class_weights[1].to(device))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Early stopping parameters for cutting off runs if validation performance does not improve
        best_f1, patience_counter = 0.0, 0
        patience = 5
        num_epochs = 100
        best_model_state = None

        for epoch in range(num_epochs):
            train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)

            # Validation with Early Stopping
            f1, best_f1, patience_counter, stop_early, best_model_state = evaluate_with_early_stopping(
                model, val_loader, threshold, device,
                best_model_state, best_f1, patience_counter, patience, epoch, num_epochs)

            if stop_early:
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

def tcn_binary_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state, save_folder):

    # Optionally use sampler to sample the minority class (ROS)
    use_sampler = False

    # Compute class weights to address imbalance
    class_weights = compute_class_weight(class_weight_imb, classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Perform Hyperparameter Tuning with Optuna using only Training data where Objective is F1 Score
    objective = make_objective(x_train, y_train, class_weights, random_state, save_folder, use_sampler)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    # Print out the optimal hyperparameters
    best_params = study.best_trial.params
    print("Best Trial:", study.best_trial)

    # Grab the Hyperparameters from the best set
    hidden_dim = best_params["hidden_dim"]
    num_layers = best_params["num_layers"]
    dropout = best_params["dropout"] if num_layers > 1 else 0
    learning_rate = best_params["lr"]
    batch_size = best_params["batch_size"]
    sequence_length = best_params["sequence_length"]
    stride = best_params["stride"]
    weight_decay = best_params["weight_decay"]
    kernel_size = best_params['kernel_size']
    step_size = round(sequence_length * stride)
    threshold = best_params['threshold']

    # Train with most training data but set aside a validation dataset for early stopping
    train_dataset, val_dataset, train_windows_tensor, _, _, train_labels_tensor = (
        train_test_split_trials(x_train, y_train, sequence_length, step_size, test_ratio=0.2)
    )

    # Create the test dataset, formatted into sequences
    test_dataset, _, _, _, _, _ = (
        train_test_split_trials(x_test, y_test, sequence_length, step_size, test_ratio=None)
    )

    input_dim = train_windows_tensor.shape[2]
    model = TCNClassifier(input_dim, hidden_dim, 1, num_layers, kernel_size=kernel_size, dropout=dropout)

    # Move device to the model after it is created
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model.to(device)

    # Build or skip using a class imbalance sampler
    if use_sampler:
        sampler = build_sampler(train_labels_tensor, class_weights)
    else:
        sampler = None

    # Prepare the DataLoader with the sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer for iteratively solving the minimum of the loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train model with early stopping
    best_f1, patience_counter = 0.0, 0
    patience = 5
    num_epochs = 100
    best_model_state = None

    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)

        # Validation with Early Stopping
        f1, best_f1, patience_counter, stop_early, best_model_state = evaluate_with_early_stopping(
            model, val_loader, threshold, device,
            best_model_state, best_f1, patience_counter, patience, epoch, num_epochs)

        if stop_early:
            print(f"Early stopping at epoch {epoch + 1} with best F1 score: {best_f1:.4f}")
            break

        os.makedirs(save_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_folder, f"trained_model.pt"))

    # Evaluate final model
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels, predictors_over_time = evaluate(model, test_loader,  threshold, device)

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