import joblib
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class TransformerNetwork(nn.Module):
    def __init__(self, input_dim, d_model = 64, nhead = 4, num_layers = 2, dim_feedforward = 128, dropout = 0.3, output_dim = 1):
        super(TransformerNetwork, self).__init__()
        # Input linear projection to d_model (embedding dimension)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Transformer encoder layer and stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True # batch_first=True matches LSTM's batch, seq, feature
        )  
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        # Final linear layer to output dimension (binary classification)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)  # -> (batch_size, seq_len, d_model)

        # Transformer Encoder expects input shape (batch_size, seq_len, d_model) with batch_first=True
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        x = x[:,-1,:]

        # You can do per-timestep output (sequence output) to match your LSTM output shape (batch, seq_len, 1)
        out = self.fc(x)  # (batch_size, seq_len, output_dim)

        return out


class TransformerModel(BaseModel):
    """
    Transformer classifier skeleton that inherits from BaseModel.

    This class defines the same lifecycle as the other models:
    tune -> train -> evaluate -> save/load.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_traditional = False


    def train(self, X, y, params: Dict = None) -> None:
        """
        Train the final Transformer model.

        Args:
            X: Input features. Expected shape (N, D) or (N, seq_len, D) — flatten to seq_len=1 when needed.
            y: Labels.
            params: Optional hyperparameters; falls back to best/default params.
        """
        if params is None:
            params = self.best_params if self.best_params else self._get_default_params()

        logger.info("Training TransformerModel with params: %s", params)

        input_dim = params.get("input_dim")
        if input_dim is None:
            # try to infer from X
            if hasattr(X, 'shape'):
                if X.ndim == 2:
                    input_dim = X.shape[1]
                elif X.ndim == 3:
                    input_dim = X.shape[2]
        if input_dim is None:
            raise ValueError("TransformerModel.train requires 'input_dim' in params or X with known shape.")

        d_model = params.get("d_model", 64)
        nhead = params.get("nhead", 4)
        num_layers = params.get("num_layers", 2)
        dim_feedforward = params.get("dim_feedforward", 128)
        dropout = params.get("dropout", 0.1)
        lr = float(params.get("learning_rate", 1e-3))
        epochs = int(params.get("num_epochs", 5))
        batch_size = int(params.get("batch_size", 32))
        random_seed = int(params.get("random_seed", 42))

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.model = TransformerNetwork(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_dim=params.get("output_dim", 1),
        ).to(self.device)

        # Prepare data
        if X.ndim == 2:
            X_proc = X.reshape(X.shape[0], 1, X.shape[1])
        else:
            X_proc = X

        X_tensor = torch.from_numpy(X_proc.astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(y.reshape(-1, 1).astype(np.float32)).to(self.device)

        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        self.best_params = {"input_dim": input_dim, "d_model": d_model, "nhead": nhead, "num_layers": num_layers, "learning_rate": lr, "num_epochs": epochs, "batch_size": batch_size}

    def evaluate(self, X, y) -> Dict[str, float]:
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return {}

        # Prepare data similar to train
        if X.ndim == 2:
            X_proc = X.reshape(X.shape[0], 1, X.shape[1])
        else:
            X_proc = X

        self.model.eval()
        X_tensor = torch.from_numpy(X_proc.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int).reshape(-1)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
        }
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, preds)
        if cm.shape == (2, 2):
            tn = cm[0, 0]
            fp = cm[0, 1]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0
        metrics["specificity"] = float(specificity)
        metrics["g_mean"] = float(np.sqrt(max(0.0, metrics["recall"] * metrics["specificity"])))
        return metrics

    def save(self, path: str) -> None:
        """
        Save model state and metadata.

        Args:
            path: Directory path for artifact persistence.
        """
        logger.info("Saving TransformerModel to %s", path)

        metadata = {
            "best_params": self.best_params,
            "split_info": self.split_info,
            "config": self.config,
        }

        joblib.dump(metadata, f"{path}/transformer_metadata.pkl")

        if self.model is not None:
            torch.save(self.model.state_dict(), f"{path}/transformer_model.pt")

    def load(self, path: str) -> None:
        """
        Load model state and metadata.

        Args:
            path: Directory path where artifacts were saved.
        """
        logger.info("Loading TransformerModel from %s", path)

        metadata = joblib.load(f"{path}/transformer_metadata.pkl")
        self.best_params = metadata.get("best_params", {})
        self.split_info = metadata.get("split_info", {})

        params = self.best_params if self.best_params else self._get_default_params()
        input_dim = params.get("input_dim")
        if input_dim is None:
            raise ValueError("Loaded metadata/config must include 'input_dim' to rebuild model.")

        self.model = TransformerNetwork(
            input_dim=input_dim,
            d_model=params.get("d_model", 64),
            nhead=params.get("nhead", 4),
            num_layers=params.get("num_layers", 2),
            dim_feedforward=params.get("dim_feedforward", 128),
            dropout=params.get("dropout", 0.1),
            output_dim=params.get("output_dim", 1),
        ).to(self.device)
        self.model.load_state_dict(torch.load(f"{path}/transformer_model.pt", map_location=self.device))
        self.model.eval()

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels for input features.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return np.array([])

        # TODO: Replace placeholder with proper preprocessing + batched inference.
        logger.info("TransformerModel.predict is a skeleton implementation")
        return np.array([])

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for input features.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return np.array([])

        # TODO: Replace placeholder with sigmoid probability output.
        logger.info("TransformerModel.predict_proba is a skeleton implementation")
        return np.array([])

    def _get_default_params(self) -> Dict[str, Any]:
        """
        Return default Transformer hyperparameters.
        """
        return {
            "input_dim": self.config.get("input_dim", None),
            "output_dim": self.config.get("output_dim", 1),
            "d_model": self.config.get("d_model", 64),
            "nhead": self.config.get("nhead", 4),
            "num_layers": self.config.get("num_layers", 2),
            "dim_feedforward": self.config.get("dim_feedforward", 128),
            "dropout": self.config.get("dropout", 0.1),
            "learning_rate": self.config.get("learning_rate", 1e-3),
            "weight_decay": self.config.get("weight_decay", 0.0),
            "batch_size": self.config.get("batch_size", 64),
            "num_epochs": self.config.get("num_epochs", 20),
        }

    def hpo_defaults(self) -> Dict[str, Any]:
        """Return default HPO configuration for Optuna."""
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
        """Define the hyperparameter search space for Optuna tuning."""

        input_dim = int(X_train.shape[2] if X_train.ndim == 3 else (X_train.shape[1] if X_train.ndim == 2 else 1))
        d_model = int(trial.suggest_categorical("d_model", [32, 64, 128]))
        nhead = int(trial.suggest_categorical("nhead", [2, 4, 8]))
        if d_model % nhead != 0:
            nhead = 2
        return {
            "input_dim": input_dim,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": int(trial.suggest_int("num_layers", 1, 3)),
            "dim_feedforward": int(trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512])),
            "dropout": float(trial.suggest_float("dropout", 0.1, 0.5)),
            "learning_rate": float(trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)),
            "num_epochs": int(trial.suggest_int("num_epochs", 3, 20)),
            "batch_size": int(trial.suggest_categorical("batch_size", [16, 32, 64, 128])),
            "random_seed": int(random_seed),
        }

    def get_name(self) -> str:
        return "Trans"