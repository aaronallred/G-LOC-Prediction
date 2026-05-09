import os
import joblib
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel


class _SimpleLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # take last timestep
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class LSTMModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.is_traditional = False
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tune(self, X, y, groups=None) -> None:
        return None

    def _ensure_3d(self, X: np.ndarray) -> np.ndarray:
        # If 2D (N, D), treat as seq_len=1
        if X.ndim == 2:
            return X.reshape(X.shape[0], 1, X.shape[1])
        return X

    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] | None = None) -> None:
        if params is None:
            params = {}
        X_proc = self._ensure_3d(X)
        input_dim = params.get("input_dim") or X_proc.shape[2]
        hidden_dim = int(params.get("hidden_dim", 32))
        num_layers = int(params.get("num_layers", 1))
        lr = float(params.get("lr", 1e-3))
        epochs = int(params.get("epochs", 5))
        batch_size = int(params.get("batch_size", 32))
        random_seed = int(params.get("random_seed", 42))

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.model = _SimpleLSTM(input_dim, hidden_dim, num_layers).to(self.device)
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

        self.best_params = {"input_dim": input_dim, "hidden_dim": hidden_dim, "num_layers": num_layers, "lr": lr, "epochs": epochs, "batch_size": batch_size}

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
            torch.save(self.model.state_dict(), os.path.join(path, "lstm_state.pt"))
        metadata = {"best_params": self.best_params, "split_info": self.split_info, "config": self.config}
        joblib.dump(metadata, os.path.join(path, "lstm_metadata.pkl"))

    def load(self, path: str) -> None:
        metadata_path = os.path.join(path, "lstm_metadata.pkl")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.best_params = metadata.get("best_params", {})
            self.split_info = metadata.get("split_info", {})
        state_path = os.path.join(path, "lstm_state.pt")
        if os.path.exists(state_path):
            if self.model is None:
                input_dim = self.best_params.get("input_dim")
                hidden_dim = self.best_params.get("hidden_dim", 32)
                num_layers = self.best_params.get("num_layers", 1)
                self.model = _SimpleLSTM(input_dim, hidden_dim, num_layers).to(self.device)
            self.model.load_state_dict(torch.load(state_path, map_location=self.device))
            self.model.eval()

    def get_name(self) -> str:
        return "LSTM"

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
