# src/models_new/logreg_ts.py
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
from src.models_new.base import AdvancedModel

class LogisticRegressionTS(nn.Module):
    """Time Series Autoregressive Logistic Regression mapping a flattened sequence window to an output."""
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.linear(x_flat)


class LogRegTSModel(AdvancedModel):
    """Wrapper managing hyperparameter spaces and lifecycle tracking for LogRegTS."""

    @property
    def name(self) -> str:
        return "LogRegTS"



    def build_hpo_search_space(self, trial: Any, X_train: np.ndarray, random_seed: int) -> Dict[str, Any]:
        return {
            "baseline_method": trial.suggest_categorical("baseline_method", [0, 1, 2, 3, 4, 5]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "optimizer_type": trial.suggest_categorical("optimizer_type", ['AdamW']),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "sequence_length": trial.suggest_int("sequence_length", 25, 250),
            "threshold": trial.suggest_float('threshold', 0.1, 0.9),
            "step_size": trial.suggest_int("step_size", 25, 75)
        }

    def _instantiate_architecture(self, input_dim: int, params: Dict[str, Any]) -> nn.Module:
        # sequence_length * feature_dimension calculates total flattened input size
        flattened_dim = params["sequence_length"] * input_dim
        return LogisticRegressionTS(input_dim=flattened_dim, output_dim=1)