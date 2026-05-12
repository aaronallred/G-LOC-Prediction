import os
import joblib
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel


class _SimpleLogistic(nn.Module):
    def __init__(self, input_dim, output_dim = 1):
        super(_SimpleLogistic, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Flatten the time series window for logistic regression
        batch_size, feature_dim = x.shape
        x_flat = x.view(batch_size, -1)  # Shape: (batch_size, seq_len * feature_dim)
        return self.linear(x_flat)


class LogRegTS(BaseModel):
    """PyTorch-based logistic regression for advanced pipeline (LogRegTS).

    Lightweight, CPU-friendly implementation intended for testing and CV.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.is_traditional = False
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tune(self, X, y, groups=None) -> None:
        # Placeholder for HPO (Optuna) in future
        return None

    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] | None = None) -> None:
        # X: (N, D) numpy, y: (N,) numpy
        if params is None:
            params = {}
        input_dim = params.get("input_dim") or (X.shape[1] if X is not None else None)
        if input_dim is None:
            raise ValueError("LogRegTS.train requires input_dim in params or X with known shape")

        lr = float(params.get("lr", 1e-2))
        epochs = int(params.get("epochs", 5))
        batch_size = int(params.get("batch_size", 32))
        random_seed = int(params.get("random_seed", 42))

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.model = _SimpleLogistic(input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        # Ensure targets have shape (N, 1) to match logits shape (N, 1)
        y_tensor = torch.from_numpy(y.reshape(-1, 1).astype(np.float32)).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # Save best_params summary for cross-validation median extraction
        self.best_params = {"lr": lr, "epochs": epochs, "batch_size": batch_size, "input_dim": input_dim}

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            return {}
        self.model.eval()
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int).ravel()

        # Compute metrics using sklearn-like behavior without importing heavy libs repeatedly
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
        }
        return metrics

    def classify_traditional(
        self,
        x_train,
        x_test,
        y_train,
        y_test,
        class_weight_imb,
        random_state,
        save_folder,
        model_name,
        strategy=None,
        best_params=None,
    ):
        """Legacy-compatible wrapper that trains/evaluates this PyTorch model and returns the legacy metric tuple."""
        # Use provided best_params if available, else fallback to stored best_params
        params = best_params if best_params is not None else getattr(self, "best_params", None)
        try:
            # Train using our advanced train() method
            self.train(x_train, y_train, params=params)
        except Exception:
            # Fallback: attempt a minimal train with defaults
            self.train(x_train, y_train, params={})

        preds = self.predict(x_test)
        # Ensure 1d array
        preds_flat = preds.ravel() if hasattr(preds, 'ravel') else preds

        # Use BaseModel helper to compute legacy tuple
        try:
            legacy_metrics = self._legacy_binary_metrics(y_test, preds_flat)
        except Exception:
            # Compute basic metrics if helper fails
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(y_test, preds_flat)
            precision = precision_score(y_test, preds_flat)
            recall = recall_score(y_test, preds_flat)
            f1 = f1_score(y_test, preds_flat)
            specificity = 0.0
            g_mean = 0.0
            legacy_metrics = (accuracy, precision, recall, f1, specificity, g_mean)

        # Print legacy metrics for compatibility
        self._print_legacy_metrics(f"{self.get_name()} Performance Metrics:", legacy_metrics)
        return legacy_metrics

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(path, "logregts_state.pt"))
        metadata = {
            "best_params": self.best_params,
            "split_info": self.split_info,
            "config": self.config,
        }
        joblib.dump(metadata, os.path.join(path, "logregts_metadata.pkl"))

    def load(self, path: str) -> None:
        metadata_path = os.path.join(path, "logregts_metadata.pkl")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.best_params = metadata.get("best_params", {})
            self.split_info = metadata.get("split_info", {})
        state_path = os.path.join(path, "logregts_state.pt")
        if os.path.exists(state_path):
            if self.model is None:
                # Need input_dim to rebuild model from metadata
                input_dim = self.best_params.get("input_dim")
                if input_dim is None:
                    raise ValueError("Cannot reconstruct model: input_dim missing in metadata")
                self.model = _SimpleLogistic(input_dim).to(self.device)
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
        # Match legacy ranges used in src/scripts
        input_dim = int(X_train.shape[2] if X_train.ndim == 3 else (X_train.shape[1] if X_train.ndim == 2 else 1))
        return {
            "input_dim": input_dim,
            "lr": float(trial.suggest_float("lr", 1e-4, 1e-1, log=True)),
            "epochs": int(trial.suggest_int("epochs", 3, 20)),
            "batch_size": int(trial.suggest_categorical("batch_size", [16, 32, 64, 128])),
            "random_seed": int(random_seed),
        }

    def get_name(self) -> str:
        return "LogRegTS"

    # Optional helpers for compatibility
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([])
        self.model.eval()
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (probs >= 0.5).astype(int).ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([])
        self.model.eval()
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        # return two-column proba like sklearn (ensure shape (N,2))
        probs_flat = probs.ravel()
        return np.vstack([1 - probs_flat, probs_flat]).T
