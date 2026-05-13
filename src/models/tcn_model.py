import os
import joblib
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel


class _SimpleTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride = stride,
            padding = padding,
            dilation = dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size, 
            stride = stride,
            padding = padding, 
            dilation = dilation
        ) 
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


class _SimpleTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim = 1, num_layers = 2, kernel_size = 3, dropout = 0.3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else hidden_dim
            layers.append(
                _SimpleTCNBlock(
                    in_channels, 
                    hidden_dim, 
                    kernel_size, 
                    stride = 1,
                    dilation = dilation_size, 
                    padding = (kernel_size - 1) * dilation_size, 
                    dropout = dropout
                )
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input shape: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        out = self.network(x)
        # Take last time step (assumes full sequence processed)
        out = out.permute(0, 2, 1)  # (batch, features, seq_len) → (batch, seq_len, features)
        out = out[:,-1,:]
        return self.fc(out)  # (batch, seq_len, 1)


class TCNModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.is_traditional = False
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def _ensure_3d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            return X.reshape(X.shape[0], 1, X.shape[1])
        return X

    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] | None = None) -> None:
        if params is None:
            params = {}
        X_proc = self._ensure_3d(X)
        input_dim = params.get("input_dim") or X_proc.shape[2]
        num_filters = int(params.get("num_filters", 32))
        lr = float(params.get("lr", 1e-3))
        epochs = int(params.get("epochs", 5))
        batch_size = int(params.get("batch_size", 32))
        random_seed = int(params.get("random_seed", 42))

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.model = _SimpleTCN(input_dim, num_filters).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        X_tensor = torch.from_numpy(X_proc.astype(np.float32)).to(self.device)
        # Ensure targets shape (N,1) to match logits (N,1)
        y_tensor = torch.from_numpy(y.reshape(-1, 1).astype(np.float32)).to(self.device)

        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        self.best_params = {"input_dim": input_dim, "num_filters": num_filters, "lr": lr, "epochs": epochs, "batch_size": batch_size}

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            return {}
        X_proc = self._ensure_3d(X)
        self.model.eval()
        X_tensor = torch.from_numpy(X_proc.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int).ravel()

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
        }
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(path, "tcn_state.pt"))
        metadata = {"best_params": self.best_params, "split_info": self.split_info, "config": self.config}
        joblib.dump(metadata, os.path.join(path, "tcn_metadata.pkl"))

    def load(self, path: str) -> None:
        metadata_path = os.path.join(path, "tcn_metadata.pkl")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.best_params = metadata.get("best_params", {})
            self.split_info = metadata.get("split_info", {})
        state_path = os.path.join(path, "tcn_state.pt")
        if os.path.exists(state_path):
            if self.model is None:
                input_dim = self.best_params.get("input_dim")
                num_filters = self.best_params.get("num_filters", 32)
                self.model = _SimpleTCN(input_dim, num_filters).to(self.device)
            self.model.load_state_dict(torch.load(state_path, map_location=self.device))
            self.model.eval()

    def hpo_defaults(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "n_trials": 10,
            "timeout": None,
            "metric": "f1",
            "train_fraction": 0.8,
            "sampler_seed": None,
            "pruner_startup_trials": 3,
            "pruner_warmup_steps": 0,
        }

    def build_hpo_search_space(self, trial, X_train: np.ndarray, random_seed: int) -> Dict[str, Any]:
        input_dim = int(X_train.shape[2] if X_train.ndim == 3 else (X_train.shape[1] if X_train.ndim == 2 else 1))
        return {
            "input_dim": input_dim,
            "num_filters": int(trial.suggest_categorical("num_filters", [32, 64, 128, 256])),
            "lr": float(trial.suggest_float("lr", 1e-4, 1e-1, log=True)),
            "epochs": int(trial.suggest_int("epochs", 3, 20)),
            "batch_size": int(trial.suggest_categorical("batch_size", [16, 32, 64, 128])),
            "random_seed": int(random_seed),
        }

    def get_name(self) -> str:
        return "TCN"

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([])
        X_proc = self._ensure_3d(X)
        self.model.eval()
        X_tensor = torch.from_numpy(X_proc.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (probs >= 0.5).astype(int).ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([])
        X_proc = self._ensure_3d(X)
        self.model.eval()
        X_tensor = torch.from_numpy(X_proc.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs_flat = probs.ravel()
        return np.vstack([1 - probs_flat, probs_flat]).T
