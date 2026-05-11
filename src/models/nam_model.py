import os
import joblib
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from src.models.base import BaseModel


class _NAM(nn.Module):
    """
    Vectorized NAM:
      - still learns a separate MLP per feature (separate weights per feature)
      - computes all features in parallel
    """
    def __init__(self, num_features, window_length, hidden_dim=4, num_layers=1, dropout=0.2):
        super().__init__()
        self.num_features = num_features
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Per-feature weights for each layer:
        # Layer 0: 1 -> hidden
        self.W0 = nn.Parameter(torch.empty(num_features, hidden_dim, 1))
        self.b0 = nn.Parameter(torch.empty(num_features, hidden_dim))

        # Hidden layers: hidden -> hidden
        self.Wh = nn.ParameterList()
        self.bh = nn.ParameterList()
        for _ in range(num_layers - 1):
            self.Wh.append(nn.Parameter(torch.empty(num_features, hidden_dim, hidden_dim)))
            self.bh.append(nn.Parameter(torch.empty(num_features, hidden_dim)))

        # Final: hidden -> 1
        self.Wf = nn.Parameter(torch.empty(num_features, 1, hidden_dim))
        self.bf = nn.Parameter(torch.empty(num_features, 1))

        self.reset_parameters()
        self.final_activation = nn.Sigmoid()

    def reset_parameters(self):
        # Kaiming init per-feature
        nn.init.kaiming_uniform_(self.W0, a=5**0.5)
        nn.init.zeros_(self.b0)
        for W, b in zip(self.Wh, self.bh):
            nn.init.kaiming_uniform_(W, a=5**0.5)
            nn.init.zeros_(b)
        nn.init.kaiming_uniform_(self.Wf, a=5**0.5)
        nn.init.zeros_(self.bf)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F_ = x.shape
        assert F_ == self.num_features, f"Expected {self.num_features} features, got {F_}"

        # Flatten time+batch, keep feature axis
        # x_flat: (BT, F, 1)
        x_flat = x.reshape(B * T, F_, 1)

        # Layer 0: (BT, F, H) = einsum over per-feature weights
        # W0: (F, H, 1)
        h = torch.einsum("bfi,fhi->bfh", x_flat, self.W0) + self.b0.unsqueeze(0)
        h = F.relu(h)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Hidden layers
        for W, b in zip(self.Wh, self.bh):
            # W: (F, H, H)
            h = torch.einsum("bfh,foh->bfo", h, W) + b.unsqueeze(0)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Final to 1: (BT, F, 1)
        out = torch.einsum("bfh,fkh->bfk", h, self.Wf) + self.bf.unsqueeze(0)  # k=1

        # Reshape back to (B, T, F)
        out = out.squeeze(-1).reshape(B, T, F_)

        # NAM combine: sum over features, mean over time
        combined = out.sum(dim=2).mean(dim=1, keepdim=True)  # (B, 1)

        return self.final_activation(combined)

class NAMModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.is_traditional = False
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tune(self, X, y, groups=None) -> None:
        return None

    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] | None = None) -> None:
        if params is None:
            params = {}
        # Input may be 2D (N, D) or 3D (N, T, D). Normalize to (N, T, D).
        if X.ndim == 2:
            # treat as sequence length 1
            X_proc = X.reshape(X.shape[0], 1, X.shape[1])
        else:
            X_proc = X

        input_dim = params.get("input_dim") or X_proc.shape[2]
        window_length = int(params.get("window_length", X_proc.shape[1]))
        hidden_dim = int(params.get("hidden_dim", 16))
        lr = float(params.get("lr", 1e-3))
        epochs = int(params.get("epochs", 5))
        batch_size = int(params.get("batch_size", 32))
        random_seed = int(params.get("random_seed", 42))

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # _NAM expects (num_features, window_length, ...)
        self.model = _NAM(num_features=input_dim, window_length=window_length, hidden_dim=hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        X_tensor = torch.from_numpy(X_proc.astype(np.float32)).to(self.device)
        # Ensure targets match logits shape (N, 1)
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

        self.best_params = {"input_dim": input_dim, "hidden_dim": hidden_dim, "lr": lr, "epochs": epochs, "batch_size": batch_size}

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            return {}
        self.model.eval()
        X_tensor = torch.from_numpy(X.reshape(X.shape[0], 1, X.shape[1]).astype(np.float32)) if X.ndim == 2 else torch.from_numpy(X.astype(np.float32))
        X_tensor = X_tensor.to(self.device)
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
            torch.save(self.model.state_dict(), os.path.join(path, "nam_state.pt"))
        metadata = {"best_params": self.best_params, "split_info": self.split_info, "config": self.config}
        joblib.dump(metadata, os.path.join(path, "nam_metadata.pkl"))

    def load(self, path: str) -> None:
        metadata_path = os.path.join(path, "nam_metadata.pkl")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.best_params = metadata.get("best_params", {})
            self.split_info = metadata.get("split_info", {})
        state_path = os.path.join(path, "nam_state.pt")
        if os.path.exists(state_path):
            if self.model is None:
                input_dim = self.best_params.get("input_dim")
                hidden_dim = self.best_params.get("hidden_dim", 16)
                self.model = _NAM(input_dim, hidden_dim).to(self.device)
            self.model.load_state_dict(torch.load(state_path, map_location=self.device))
            self.model.eval()

    def get_name(self) -> str:
        return "NAM"

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([])
        self.model.eval()
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([])
        self.model.eval()
        X_tensor = torch.from_numpy(X.reshape(X.shape[0], 1, X.shape[1]).astype(np.float32)) if X.ndim == 2 else torch.from_numpy(X.astype(np.float32))
        X_tensor = X_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs_flat = probs.ravel()
        return np.vstack([1 - probs_flat, probs_flat]).T
