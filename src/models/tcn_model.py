import os
import joblib
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel


class _SimpleTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, channels, seq_len)
        return self.relu(self.conv(x))


class _SimpleTCN(nn.Module):
    def __init__(self, input_dim: int, num_filters: int = 32, kernel_size: int = 3):
        super().__init__()
        # We'll treat features as channels and seq_len=1 by default
        # So we create a conv1d that operates across a fake seq dimension
        self.net = nn.Sequential(
            _SimpleTCNBlock(input_dim, num_filters, kernel_size=kernel_size),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_filters, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expect x shape: (batch, seq_len, features)
        if x.ndim == 3:
            # convert to (batch, channels, seq_len)
            x = x.permute(0, 2, 1)
        return self.net(x).squeeze(-1)


class TCNModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.is_traditional = False
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tune(self, X, y, groups=None) -> None:
        return None

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
        y_tensor = torch.from_numpy(y.reshape(-1,).astype(np.float32)).to(self.device)

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
        preds = (probs >= 0.5).astype(int)

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
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([])
        X_proc = self._ensure_3d(X)
        self.model.eval()
        X_tensor = torch.from_numpy(X_proc.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1 - probs, probs]).T
