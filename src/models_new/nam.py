# src/models_new/nam.py
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AdvancedModel


class FastNAM(nn.Module):
    """Vectorized Neural Additive Model running feature networks in parallel via einsum."""

    def __init__(self, num_features: int, window_length: int, hidden_dim: int = 4,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.num_features = num_features
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.W0 = nn.Parameter(torch.empty(num_features, hidden_dim, 1))
        self.b0 = nn.Parameter(torch.empty(num_features, hidden_dim))
        self.Wh = nn.ParameterList(
            [nn.Parameter(torch.empty(num_features, hidden_dim, hidden_dim)) for _ in range(num_layers - 1)])
        self.bh = nn.ParameterList([nn.Parameter(torch.empty(num_features, hidden_dim)) for _ in range(num_layers - 1)])
        self.Wf = nn.Parameter(torch.empty(num_features, 1, hidden_dim))
        self.bf = nn.Parameter(torch.empty(num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W0, a=5 ** 0.5)
        nn.init.zeros_(self.b0)
        for W, b in zip(self.Wh, self.bh):
            nn.init.kaiming_uniform_(W, a=5 ** 0.5)
            nn.init.zeros_(b)
        nn.init.kaiming_uniform_(self.Wf, a=5 ** 0.5)
        nn.init.zeros_(self.bf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_ = x.shape
        x_flat = x.reshape(B * T, F_, 1)

        h = F.relu(torch.einsum("bfi,fhi->bfh", x_flat, self.W0) + self.b0.unsqueeze(0))
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)

        for W, b in zip(self.Wh, self.bh):
            h = F.relu(torch.einsum("bfh,foh->bfo", h, W) + b.unsqueeze(0))
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        out = torch.einsum("bfh,fkh->bfk", h, self.Wf) + self.bf.unsqueeze(0)
        out = out.squeeze(-1).reshape(B, T, F_)
        return out.sum(dim=2).mean(dim=1, keepdim=True)


class NAMModel(AdvancedModel):
    """Refactored wrapper managing hyperparameter search space and lifetime loops for NAM."""

    @property
    def name(self) -> str:
        return "NAM"



    def build_hpo_search_space(self, trial: Any, X_train: np.ndarray, random_seed: int) -> Dict[str, Any]:
        return {
            "baseline_method": trial.suggest_categorical("baseline_method", [0, 1, 2, 3, 4, 5]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "optimizer_type": trial.suggest_categorical("optimizer_type", ['AdamW']),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [4, 8, 16]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "sequence_length": trial.suggest_int("sequence_length", 25, 250),
            "threshold": trial.suggest_float('threshold', 0.1, 0.9),
            "step_size": trial.suggest_int("step_size", 25, 75)
        }

    def _instantiate_architecture(self, input_dim: int, params: Dict[str, Any]) -> nn.Module:
        return FastNAM(
            num_features=input_dim,
            window_length=params["sequence_length"],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        )