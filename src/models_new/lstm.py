# src/models_new/lstm.py
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
from src.models_new.base import AdvancedModel

class LSTM(nn.Module):
    """Pure PyTorch LSTM sequence classification module."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, 
                 num_layers: int = 2, dropout: float = 0.3, bidirectional: bool = False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0, 
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class LSTMModel(AdvancedModel):
    """Refactored wrapper managing hyperparameter search space and lifetime loops for LSTM."""

    @property
    def name(self) -> str:
        return "LSTM"



    def build_hpo_search_space(self, trial: Any, X_train: np.ndarray, random_seed: int) -> Dict[str, Any]:
        return {
            "baseline_method": trial.suggest_categorical("baseline_method", [0, 1, 2, 3, 4, 5]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "optimizer_type": trial.suggest_categorical("optimizer_type", ['AdamW']),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "sequence_length": trial.suggest_int("sequence_length", 25, 250),
            "step_size": trial.suggest_int("step_size", 25, 75),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "bidirectional": trial.suggest_categorical("bidirectional", [False]),
            "threshold": trial.suggest_float('threshold', 0.1, 0.9)
        }

    def _instantiate_architecture(self, input_dim: int, params: Dict[str, Any]) -> nn.Module:
        return LSTM(
            input_dim=input_dim,
            hidden_dim=params["hidden_dim"],
            output_dim=1,
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            bidirectional=params["bidirectional"]
        )

    def _build_model(self, model_hyperparameters: Dict[str, Any]) -> Any:
        return None