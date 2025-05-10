import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
import optuna
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit
import torch.optim as optim

# Convert Dataset to Pytorch tensors of Dtype Float 32
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, labels):
        # Convert each sequence and label to tensors
        self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        self.labels = [torch.tensor(lbl, dtype=torch.float32) for lbl in labels]

        # Pad sequences and labels
        self.padded_sequences = pad_sequence(self.sequences, batch_first=True, padding_value=0.0)  # [B, T, F]
        self.padded_labels = pad_sequence(self.labels, batch_first=True, padding_value=0.0)  # [B, T]

        # Create attention masks: 1 for real tokens, 0 for padding
        self.attn_masks = torch.tensor([
            [1] * len(seq) + [0] * (self.padded_sequences.shape[1] - len(seq))
            for seq in self.sequences
        ], dtype=torch.bool)

    def __len__(self):
        return len(self.padded_sequences)

    def __getitem__(self, idx):
        return self.padded_sequences[idx], self.padded_labels[idx], self.attn_masks[idx]

# Create Simple Transformer Model using pytorch with hyperparamater variables
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)  # [B, T, d_model]
        # Transformer expects attn_mask to be [B, T], bool mask where True = keep, False = pad
        out = self.transformer(x)
        logits = self.classifier(out).squeeze(-1)  # [B, T]

        return logits

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
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, predictors_over_time = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(y_batch.view(-1).cpu().numpy())

            # Append the input data for plotting predictors
            predictors_over_time.extend(x_batch.cpu().numpy())

    return all_preds, all_labels, predictors_over_time

# Define Evaluation with early stopping
def evaluate_with_early_stopping(model, val_loader, device, best_model_state, best_f1=0,
                                 patience_counter=0, patience=5, epoch = None):

    # Run Evaluation
    all_preds, all_labels, _ = evaluate(model, val_loader, device)

    # Calculate F1 metric and use for determining stopping
    f1 = metrics.f1_score(all_labels, all_preds)
    print(f"Epoch {epoch}: F1 Score = {f1:.4f}")

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

# Define Optuna Objective Function to be Optimized
def make_objective(x_train, y_train, class_weights, random_state, save_folder, use_sampler):
    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dim_feedforward = trial.suggest_int("dim_feedforward", 64, 512)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        sequence_length = trial.suggest_int("sequence_length", 100, 1000)
        stride = trial.suggest_float("stride", 0.25, 1.0)

        step_size = round(sequence_length * stride)

        # Create training and validation sets from x_train/y_train
        train_dataset, val_dataset, train_windows_tensor, train_labels_tensor, val_windows_tensor, val_labels_tensor = (
            train_test_split_trials(
                x_train,
                y_train,
                sequence_length,
                step_size,
                test_ratio=0.2,
                random_state=random_state)
        )

        if use_sampler:
            sampler = build_sampler(train_labels_tensor, class_weights)
        else:
            sampler = None

        input_dim = train_windows_tensor.shape[2]
        model = TransformerClassifier(input_dim, d_model, nhead, num_layers, dim_feedforward, dropout)

        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        model.to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Early stopping parameters for cutting off runs if validation performance does not improve
        best_f1, patience_counter = 0.0, 0
        patience = 5
        num_epochs = 100
        best_model_state = None

        for epoch in range(num_epochs):
            train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)

            # Validation with Early Stopping
            f1, best_f1, patience_counter, stop_early, best_model_state = evaluate_with_early_stopping(
                model, val_loader, device, best_model_state, best_f1, patience_counter, patience, epoch)

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

def trans_win_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state, save_folder):

    # Optionally use sampler to sample the minority class (ROS)
    use_sampler = False

    # Compute class weights to address imbalance
    class_weights = compute_class_weight(class_weight_imb, classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(class_weights * 100, dtype=torch.float)

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

    # Train with most training data but set aside a validation dataset for early stopping
    train_dataset, val_dataset, train_windows_tensor, _, _, train_labels_tensor = (
        train_test_split_trials(x_train, y_train, sequence_length, step_size, test_ratio=0.2)
    )

    # Create the test dataset, formatted into sequences
    test_dataset, _, _, _, _, _ = (
        train_test_split_trials(x_test, y_test, sequence_length, step_size, test_ratio=None)
    )

    input_dim = train_windows_tensor.shape[2]
    model = TransformerClassifier(input_dim, d_model, nhead, num_layers, dim_feedforward, dropout)

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
            model, val_loader, device, best_model_state, best_f1, patience_counter, patience, epoch)

        if stop_early:
            print(f"Early stopping at epoch {epoch + 1} with best F1 score: {best_f1:.4f}")
            break

        os.makedirs(save_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_folder, f"trained_model.pt"))

    # Evaluate final model
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels, predictors_over_time = evaluate(model, test_loader, device)

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