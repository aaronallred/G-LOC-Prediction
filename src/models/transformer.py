# src/models/transformer.py
from typing import Any, Dict, Optional

import numpy as np
import optuna
import torch
import torch.nn as nn

from src.advanced_experiment_utils import baseline_down_select
from src.models.base import AdvancedModel


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.3, output_dim: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])


class TransformerModel(AdvancedModel):
    """Wrapper managing hardware memory guardrails and execution loops for Transformer."""

    @property
    def name(self) -> str:
        return "Trans"

    def get_hpo_search_space(self):
        model = self

        def _builder(trial: Any, X_train: np.ndarray, random_seed: int) -> Dict[str, Any]:
            baseline_method = trial.suggest_categorical("baseline_method", [0, 1, 2, 3, 4, 5])
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            sequence_length = trial.suggest_int("sequence_length", 25, 200)
            d_model = trial.suggest_categorical("d_model", [64, 128, 256])
            num_layers = trial.suggest_int("num_layers", 1, 3)

            if sequence_length > 125:
                batch_size = min(batch_size, 64)

            _, include_features = baseline_down_select(X_train, model.all_features, baseline_method)
            feature_count = len(include_features) + 1

            estimated_bytes = feature_count * batch_size * sequence_length * d_model * 4
            if estimated_bytes > 8 * 1024 ** 3:
                raise optuna.exceptions.TrialPruned()

            return {
                "baseline_method": baseline_method,
                "batch_size": batch_size,
                "optimizer_type": trial.suggest_categorical("optimizer_type", ['AdamW']),
                "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "d_model": d_model,
                "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
                "num_layers": num_layers,
                "dim_feedforward": trial.suggest_categorical("dim_feedforward", [128, 256, 512]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5) if num_layers > 1 else 0.0,
                "sequence_length": sequence_length,
                "step_size": trial.suggest_int("step_size", 25, 75),
                "threshold": trial.suggest_float('threshold', 0.1, 0.9)
            }

        return _builder

    def _build_model(self, model_hyperparameters: Optional[Dict[str, Any]] = None, *, input_dim: Optional[int] = None,
                     **kwargs) -> Any:
        if model_hyperparameters is None or input_dim is None:
            return None
        return TransformerClassifier(
            input_dim=input_dim,
            d_model=model_hyperparameters["d_model"],
            nhead=model_hyperparameters["nhead"],
            num_layers=model_hyperparameters["num_layers"],
            dim_feedforward=model_hyperparameters["dim_feedforward"],
            dropout=model_hyperparameters.get("dropout", 0.0)
        )
