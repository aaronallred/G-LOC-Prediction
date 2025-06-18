import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import TensorDataset
import numpy as np
import os
import torch.optim as optim
from torch.amp import autocast, GradScaler

### Define Data Handling Functions
# Baseline down selection
def baseline_down_select(x,all_features,method):
    if method == 0: # No baseline columns
        include =  {name for name in all_features if name.endswith('_v0')}

    elif method == 1: # v1 baseline
        v0_base = {name[:-3] for name in all_features if name.endswith('_v0')}
        v1_base = {name[:-3] for name in all_features if name.endswith('_v1')}
        unique_v0_names = [f"{base}_v0" for base in v0_base if base not in v1_base]
        unique_v1_names = [f"{base}_v1" for base in v1_base]

        include = unique_v0_names + unique_v1_names

    elif method == 2: # v2 baseline
        v0_base = {name[:-3] for name in all_features if name.endswith('_v0')}
        v2_base = {name[:-3] for name in all_features if name.endswith('_v2')}
        unique_v0_names = [f"{base}_v0" for base in v0_base if base not in v2_base]
        unique_v2_names = [f"{base}_v2" for base in v2_base]

        include = unique_v0_names + unique_v2_names

    elif method == 3:
        v0_base = {name[:-3] for name in all_features if name.endswith('_v0')}
        v1_base = {name[:-3] for name in all_features if name.endswith('_v1')}
        v5_base = {name[:-3] for name in all_features if name.endswith('_v5')}
        unique_v0_names = [f"{base}_v0" for base in v0_base if base not in v1_base and base not in v5_base]
        unique_v1_names = [f"{base}_v1" for base in v1_base if base not in v5_base]
        unique_v5_names = [f"{base}_v5" for base in v5_base]

        include = unique_v0_names + unique_v1_names + unique_v5_names

    elif method == 4:
        v0_base = {name[:-3] for name in all_features if name.endswith('_v0')}
        v2_base = {name[:-3] for name in all_features if name.endswith('_v2')}
        v6_base = {name[:-3] for name in all_features if name.endswith('_v6')}
        unique_v0_names = [f"{base}_v0" for base in v0_base if base not in v2_base and base not in v6_base]
        unique_v2_names = [f"{base}_v2" for base in v2_base if base not in v6_base]
        unique_v6_names = [f"{base}_v6" for base in v6_base]

        include = unique_v0_names + unique_v2_names + unique_v6_names

    else:
        include = all_features

    # Grab indices of included features
    i_indices = [i for i, feature in enumerate(all_features) if feature in include]

    # Add last column index
    i_indices.append(x.shape[1] - 1)

    x = x[:,i_indices]

    return x, include


# Creates windowed sequences
def create_windows(sequence, labels, window_size, step_size, end_label):
    # Creates windows with paired labels.
    windows = []
    window_labels = []
    for start in range(0, len(sequence) - window_size + 1, step_size):
        end = start + window_size
        windows.append(sequence[start:end])
        if end_label:
            window_labels.append(labels[end-1])  # Take the last label only
        else:
            window_labels.append(labels[start:end])  # Take full window of labels
    return windows, window_labels

# Splits dataset based on trial identifier (located in last column of predictors)
def train_test_split_trials(X,Y,window_size,step_size,test_ratio, random_state = 42, end_label=False):
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
        windows, labels = create_windows(trial_sequence, trial_labels, window_size, step_size, end_label)
        train_windows.extend(windows)
        train_labels.extend(labels)

    for trial in test_trials:
        trial_sequence = X[X[:, -1] == trial, :-1]
        trial_labels = Y[X[:, -1] == trial]
        windows, labels = create_windows(trial_sequence, trial_labels, window_size, step_size, end_label)
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

### Builds model components
# Checks device hardware and uses the best available hardware (cuda for NVIDIA, mps for Mac metal, and cpu otherwise)
def get_device():
    return (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

# Check how many cores are available and return best use for DataLoader
def get_optimal_workers():
    num_cpus = os.cpu_count()

    if num_cpus > 4:
        workers = 0 # use 4 is there are main threads to spare
    else:
        workers = 0 # Default amount

    return workers

# Builds model loss criterion and optimizer (default is BCE)
def build_training_components(model, class_weights, lr, weight_decay, device, loss='BCE'):
    if loss == 'BCE':
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
    elif loss == 'F1':
        criterion = SoftF1Loss(pos_weight=class_weights[1].to(device))
    else:
        # Default is just Binary cross entropy
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return criterion, optimizer

# builds a sampler for random oversampling the minority class (not used)
def build_sampler(train_labels_tensor, class_weights):
    # Determine if window has GLOC event
    # Ensure 2D shape for mean computation
    if train_labels_tensor.dim() == 1:
        binary_labels = train_labels_tensor.long()
    else:
        mean_labels = train_labels_tensor.float().mean(dim=1)  # shape: (num_windows,)
        binary_labels = (mean_labels > 0).long()

    # Apply class weights
    sample_weights = class_weights[binary_labels]  # shape: (num_windows,)
    # Create sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    return sampler

# Define a differentiable F1 Loss function (not currently used)
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

### Evaluation Routines
# Define training routines
def train(model, loader, criterion, optimizer, device, epoch, num_epochs, verbose = 1):
    model.train()

    use_amp = device.type in ['cuda']
    scaler = GradScaler(enabled=use_amp)

    epoch_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type=device.type):
                outputs = model(x_batch)
                loss = criterion(outputs.reshape(-1), y_batch.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(x_batch)
            loss = criterion(outputs.reshape(-1), y_batch.reshape(-1))
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        if verbose == 2:  # Print loss every 100 batches for verbosity
            print(f"Epoch [{epoch + 1}/{num_epochs}] Batch Loss: {loss.item():.16f}")

    if verbose > 0:
        avg_epoch_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Epoch Loss: {avg_epoch_loss:.16f}")

# If using early stopping, use this function (used during hyperparameter tuning | optional during final train)
def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device,
                               threshold, objective_var, num_epochs=100, patience=10, min_epochs=15):
    best_stopping_metric, patience_counter = 0.0, 0
    best_epoch = num_epochs
    best_model_state = None

    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)

        stopping_metric, best_stopping_metric, patience_counter, stop_early, best_model_state = (
            evaluate_with_early_stopping(
            model, val_loader, threshold, device, criterion, objective_var,
            best_model_state, best_stopping_metric, patience_counter, patience, epoch, num_epochs, min_epochs))

        if stopping_metric == best_stopping_metric:
            best_epoch = epoch + 1

        if stop_early:
            print(f"Early stopping at epoch {epoch + 1} with best stopping metric: {best_stopping_metric:.4f}")
            break

    return best_model_state, best_epoch, best_stopping_metric

### Evaluation Routines
# Define Evaluation Loop
def evaluate(model, loader, threshold, device, criterion):
    model.eval()
    all_preds, all_labels, predictors_over_time = [], [], []

    eval_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs.reshape(-1), y_batch.reshape(-1))
            eval_loss += loss.item()

            preds = (outputs >= threshold).float()
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(y_batch.view(-1).cpu().numpy())

            # Append the input data for plotting predictors
            predictors_over_time.extend(x_batch[-1].cpu().numpy())

    avg_eval_loss = eval_loss / len(loader)

    return all_preds, all_labels, predictors_over_time, avg_eval_loss

# Define Evaluation with early stopping
def evaluate_with_early_stopping(model, val_loader,  threshold, device, criterion, objective_var, best_model_state,
                                 best_stopping_metric=0,patience_counter=0, patience=10,
                                 epoch = None, num_epochs= None, min_epochs = 15):

    # Run Evaluation
    all_preds, all_labels, _, avg_eval_loss = evaluate(model, val_loader, threshold, device, criterion)

    # Calculate stopping metric (we are using F1 but could be '1-loss' on validation set instead)
    stopping_metric = compute_stopping_metric(epoch,num_epochs,avg_eval_loss,all_labels,all_preds,
                                              metric_type = objective_var)

    if epoch + 1 >=min_epochs:
        if stopping_metric >  best_stopping_metric:
            best_stopping_metric = stopping_metric
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        stop_early = patience_counter >= patience
    else:
        stop_early = False

    return stopping_metric, best_stopping_metric, patience_counter, stop_early, best_model_state

# Adjustable stopping metric for early stopping
def compute_stopping_metric(epoch,num_epochs, avg_eval_loss, all_labels, all_preds, metric_type = 'F1'):
    if metric_type == 'F1':
        stopping_metric = metrics.f1_score(all_labels, all_preds)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Evaluation F1 Score: {stopping_metric:.4f}")

    elif metric_type == 'Acc':
        stopping_metric = metrics.accuracy_score(all_labels, all_preds)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Evaluation Accuracy Score: {stopping_metric:.4f}")

    else:
        stopping_metric = 1-avg_eval_loss # Use 1-loss since Optuna and Early Stop are set to maximize and not minimize
        print(f"Epoch [{epoch + 1}/{num_epochs}] Evaluation 1-Loss: {stopping_metric:.4f}")

    return stopping_metric