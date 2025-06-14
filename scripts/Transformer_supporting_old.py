import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import optuna
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit
from collections import defaultdict

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

    def forward(self, x, attn_mask):
        x = self.input_proj(x)  # [B, T, d_model]
        # Transformer expects attn_mask to be [B, T], bool mask where True = keep, False = pad
        out = self.transformer(x, src_key_padding_mask=attn_mask.bool())
        logits = self.classifier(out).squeeze(-1)  # [B, T]

        return logits

# Define Pad Collate Function for DataLoader
def pad_collate_fn(batch):
    # Unpack each item in the batch
    sequences, labels, masks = zip(*batch)  # Now we expect 3 elements per batch item

    # Pad sequences and attention masks
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)  # [B x T_max x F]
    attention_masks = torch.tensor([
        [1] * len(seq) + [0] * (padded_seqs.shape[1] - len(seq)) for seq in sequences
    ], dtype=torch.bool)

    # Stack labels
    labels = torch.stack(labels).float()  # Ensure labels are correctly stacked

    return padded_seqs, attention_masks, labels

# Define Training Loop
def train(model, loader, criterion, optimizer, device):
    model.train()
    for x_batch, y_batch, mask in loader:
        x_batch, y_batch, mask = x_batch.to(device), y_batch.to(device), mask.to(device)
        optimizer.zero_grad()
        preds = model(x_batch, mask)

        loss = criterion(preds[mask], y_batch[mask])
        loss.backward()
        optimizer.step()

# Define Evaluation Loop
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch, mask in loader:
            x_batch, y_batch, mask = x_batch.to(device), y_batch.to(device), mask.to(device)
            logits = model(x_batch, mask)
            probs = torch.sigmoid(logits) # Convert logits to probabilities
            preds = (probs >= 0.5).float()  # Threshold at 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return all_preds, all_labels

# Define Evaluation with early stopping
def evaluate_with_early_stopping(model, val_loader, device, best_f1=0, patience_counter=0, patience=5, epoch = None):
    all_preds, all_labels = evaluate(model, val_loader, device)

    f1 = metrics.f1_score(all_labels, all_preds)
    print(f"Epoch {epoch}: F1 Score = {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
    stop_early = patience_counter >= patience

    return f1, best_f1, patience_counter, stop_early

# Group Based Split (keeping train and test trials together)
def group_based_split(x, y, trial_ids, test_size=0.2, random_state=42):

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(x, y, groups=trial_ids))

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    # Convert data to torch tensors and create DataLoaders
    x_train_tensor = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(np.array(x_val), dtype=torch.float32)
    y_val_tensor = torch.tensor(np.array(y_val), dtype=torch.float32).view(-1, 1)

    return x_train_tensor, x_val_tensor, y_train_tensor, y_val_tensor

def split_trials(X, Y, trial_ids):
    from collections import defaultdict
    trial_dict = defaultdict(list)

    for x, y, tid in zip(X, Y, trial_ids):
        trial_dict[tid].append((x, y))

    sequences, labels, groups = [], [], []
    for tid, data in trial_dict.items():
        trial_x = np.stack([x for x, _ in data])  # Shape: [T x F]
        trial_y = np.array([y for _, y in data])  # Shape: [T]
        sequences.append(trial_x)
        labels.append(trial_y)
        groups.append(tid)

    return sequences, labels, groups

# Define Optuna Objective Function to be Optimized
def make_objective(train_set,val_set, class_weights, save_folder):
    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dim_feedforward = trial.suggest_int("dim_feedforward", 64, 512)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [1])

        input_dim = train_set[0][0].shape[1]
        model = TransformerClassifier(input_dim, d_model, nhead, num_layers, dim_feedforward, dropout)

        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))

        # Train and Evaluate with Early Stopping
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
        val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=pad_collate_fn)

        patience = 5
        best_f1 = 0
        patience_counter = 0
        num_epochs = 50

        for epoch in range(num_epochs):
            # Train on training set
            train(model, train_loader, criterion, optimizer, device)

            # Evaluate on validation set
            f1, best_f1, patience_counter, stop_early = evaluate_with_early_stopping(
                model, val_loader, device, best_f1, patience_counter, patience, epoch)

            if stop_early:
                print(f"Early stopping at epoch {epoch+1} with best F1 score: {best_f1:.4f}")
                break

        # Save model to file
        torch.save(model.state_dict(), os.path.join(save_folder, f"trans_best_model_trial_{trial.number}.pt"))

        return best_f1

    return objective

def transformer_class(x_train, x_test, y_train, y_test, class_weight_imb, random_state, save_folder):

    # Make save folder if needed
    os.makedirs(save_folder, exist_ok=True)

    # Split data into train, validation, and test
    trial_ids_train = x_train[:, -1]
    trial_ids_test = x_test[:, -1]
    x_train = x_train[:,:-1]  # Discard last column (trial ids)
    x_test = x_test[:,:-1]    # Discard last column (trial ids)

    # Put into torch tensor
    train_seqs, train_labels, train_groups = split_trials(x_train, y_train, trial_ids_train)
    test_seqs, test_labels, _ = split_trials(x_test, y_test, trial_ids_test) # used after hyperparameter

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, val_idx = next(gss.split(train_seqs, train_labels, groups=train_groups))

    # Create Datasets
    train_set = TimeSeriesDataset([train_seqs[i] for i in train_idx], [train_labels[i] for i in train_idx])
    val_set = TimeSeriesDataset([train_seqs[i] for i in val_idx], [train_labels[i] for i in val_idx])
    test_set = TimeSeriesDataset(test_seqs, test_labels)

    # Compute the proportionality of the class weights used for training
    class_weights = compute_class_weight(class_weight_imb, classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Perform Hyperparameter Tuning with Optuna using only Training data
    objective = make_objective(train_set,val_set, class_weights, save_folder)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    # Grab the Hyperparameters from the best set
    best_params = study.best_trial.params
    print("Best Trial:", study.best_trial)

    # Final model training with best hyperparameters
    model = TransformerClassifier(
        input_dim=train_set[0][0].shape[1],
        d_model=best_params["d_model"],
        nhead=best_params["nhead"],
        num_layers=best_params["num_layers"],
        dim_feedforward=best_params["dim_feedforward"],
        dropout=best_params["dropout"]
    )

    # Grab device to train on depending on hardware
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model.to(device)

    # Define optimization method, loss criterion, and train / test dataloaders in pytorch
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))

    final_train_set = torch.utils.data.ConcatDataset([train_set, val_set])
    train_loader = DataLoader(final_train_set, batch_size=best_params["batch_size"], shuffle=True,
                              collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_set, batch_size=best_params["batch_size"], collate_fn=pad_collate_fn)

    # Train using full training dataset
    num_epochs = 20
    for epoch in range(num_epochs):  # Train longer after tuning
        train(model, train_loader, criterion, optimizer, device)

    # Final evaluation using test dataset
    all_preds, all_labels = evaluate(model, test_loader, device)

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