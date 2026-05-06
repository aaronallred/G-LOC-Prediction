import joblib
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class TransformerNetwork(nn.Module):
    """
    Minimal Transformer backbone used by TransformerModel.

    This class is intentionally lightweight and serves as a starting point
    for later task-specific architecture updates.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.head(x)


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

    def tune(self, X, y, groups=None) -> None:
        """
        Run hyperparameter tuning (Optuna/Bayesian search placeholder).

        Args:
            X: Input features.
            y: Labels.
            groups: Optional group labels for grouped CV.
        """
        logger.info("Starting hyperparameter tuning for TransformerModel")
        # TODO: Add Optuna objective and search space for transformer hyperparameters.
        # Suggested search dimensions:
        # - d_model, nhead, num_layers, dim_feedforward, dropout
        # - learning_rate, weight_decay, batch_size
        # - sequence_length, step_size, threshold
        pass

    def train(self, X, y, params: Dict = None) -> None:
        """
        Train the final Transformer model.

        Args:
            X: Input features.
            y: Labels.
            params: Optional hyperparameters; falls back to best/default params.
        """
        if params is None:
            params = self.best_params if self.best_params else self._get_default_params()

        logger.info("Training TransformerModel with params: %s", params)

        # TODO: Add data windowing, DataLoader creation, loss, optimizer, and training loop.
        # Build a network skeleton so downstream code can inspect model structure.
        input_dim = params.get("input_dim")
        if input_dim is None:
            raise ValueError("TransformerModel.train requires 'input_dim' in params or config.")

        self.model = TransformerNetwork(
            input_dim=input_dim,
            d_model=params.get("d_model", 64),
            nhead=params.get("nhead", 4),
            num_layers=params.get("num_layers", 2),
            dim_feedforward=params.get("dim_feedforward", 128),
            dropout=params.get("dropout", 0.1),
            output_dim=params.get("output_dim", 1),
        ).to(self.device)

    def evaluate(self, X, y) -> Dict[str, float]:
        """
        Evaluate the trained model.

        Args:
            X: Input features.
            y: Labels.

        Returns:
            Dict of evaluation metrics.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return {}

        # TODO: Add batched inference and full metrics set.
        logger.info("TransformerModel.evaluate is a skeleton implementation")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

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

    def get_name(self) -> str:
        return "Trans"