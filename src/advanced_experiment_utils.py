from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def baseline_down_select(x: np.ndarray, all_features: List[str], method: int) -> tuple:
    if method == 0:
        include = {f for f in all_features if f.endswith('_v0')}
    elif method == 1:
        v0, v1 = {f[:-3] for f in all_features if f.endswith('_v0')}, {f[:-3] for f in all_features if
                                                                       f.endswith('_v1')}
        include = [f"{b}_v0" for b in v0 if b not in v1] + [f"{b}_v1" for b in v1]
    elif method == 2:
        v0, v2 = {f[:-3] for f in all_features if f.endswith('_v0')}, {f[:-3] for f in all_features if
                                                                       f.endswith('_v2')}
        include = [f"{b}_v0" for b in v0 if b not in v2] + [f"{b}_v2" for b in v2]
    elif method == 3:
        v0, v1, v5, v7 = [{f[:-3] for f in all_features if f.endswith(f'_{v}')} for v in ['v0', 'v1', 'v5', 'v7']]
        include = [f"{b}_v0" for b in v0 if b not in v1 and b not in v5 and b not in v7] + \
                  [f"{b}_v1" for b in v1 if b not in v5 and b not in v7] + \
                  [f"{b}_v5" for b in v5 if b not in v7] + [f"{b}_v7" for b in v7]
    elif method == 4:
        v0, v2, v6, v8 = [{f[:-3] for f in all_features if f.endswith(f'_{v}')} for v in ['v0', 'v2', 'v6', 'v8']]
        include = [f"{b}_v0" for b in v0 if b not in v2 and b not in v6 and b not in v8] + \
                  [f"{b}_v2" for b in v2 if b not in v6 and b not in v8] + \
                  [f"{b}_v6" for b in v6 if b not in v8] + [f"{b}_v8" for b in v8]
    else:
        include = all_features

    indices = [i for i, f in enumerate(all_features) if f in include] + [x.shape[1] - 1]
    return x[:, indices], include


def train_test_split_trials(X, Y, window_size, step_size, test_ratio, random_state=42, end_label=False):
    unique_trials = np.unique(X[:, -1])
    if test_ratio is not None:
        train_trials, test_trials = train_test_split(unique_trials, test_size=test_ratio, random_state=random_state)
    else:
        train_trials, test_trials = unique_trials, []

    def make_set(trials, step):
        w_out, l_out = [], []
        for t in trials:
            seq = X[X[:, -1] == t, :-1]
            lbl = Y[X[:, -1] == t]
            for start in range(0, len(seq) - window_size + 1, step):
                end = start + window_size
                w_out.append(seq[start:end])
                l_out.append(lbl[end - 1] if end_label else lbl[start:end])
        return torch.tensor(np.array(w_out), dtype=torch.float32), torch.tensor(np.array(l_out), dtype=torch.float32)

    tr_w, tr_l = make_set(train_trials, step_size)
    te_w, te_l = make_set(test_trials, 10 if test_ratio is not None else step_size)
    return TensorDataset(tr_w, tr_l), TensorDataset(te_w, te_l), tr_w, tr_l, te_w, te_l


def build_training_components(model, weights, params, device):
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1].to(device))
    if params["optimizer_type"] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"],
                                    momentum=0.9)
    return criterion, optimizer


def build_sampler(labels, weights):
    binary_labels = labels.long() if labels.dim() == 1 else (labels.float().mean(dim=1) > 0).long()
    sample_weights = weights[binary_labels]
    return torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def run_train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb).reshape(-1), yb.reshape(-1))
        loss.backward()
        optimizer.step()


def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, threshold, objective_var,
                              patience=10, min_epochs=15):
    best_metric, best_state, best_epoch, patience_counter = -1.0, None, 0, 0
    for epoch in range(100):
        run_train_epoch(model, train_loader, criterion, optimizer, device)
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                outputs = model(xb.to(device))
                preds.extend((outputs.reshape(-1) >= threshold).float().cpu().numpy())
                targets.extend(yb.reshape(-1).cpu().numpy())

        actual, predicted = np.array(targets), np.array(preds)
        if objective_var == 'F1':
            score = metrics.f1_score(actual, predicted, zero_division=0)
        elif objective_var == 'Acc':
            score = metrics.accuracy_score(actual, predicted)
        else:
            score = 1.0  # fallback matching logic bounds safely

        if epoch + 1 >= min_epochs:
            if score > best_metric:
                best_metric, best_epoch, patience_counter = score, epoch + 1, 0
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= patience: break
    return best_state, best_epoch, best_metric


def get_advanced_predictions_and_targets(
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int,
        step_size: int,
        batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    dataset, _, _, target_tensor, _, _ = train_test_split_trials(
        X, y, sequence_length, step_size=step_size, test_ratio=None, end_label=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.model.eval()
    predictions = []
    with torch.no_grad():
        for x_batch, _ in loader:
            outputs = model.model(x_batch.to(model.device))
            preds = (outputs.reshape(-1) >= model.best_params["threshold"]).float()
            predictions.extend(preds.cpu().numpy())

    return target_tensor.numpy(), np.array(predictions)
