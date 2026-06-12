# src/models_new/tcn.py
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import optuna
from src.models_new.base import AdvancedModel


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, dilation: int, padding: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                               dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding,
                               dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        layers = [self.conv1, self.conv2]
        if self.downsample:
            layers.append(self.downsample)
        for layer in layers:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)

        if res.size(-1) != out.size(-1):
            min_len = min(res.size(-1), out.size(-1))
            res = res[:, :, -min_len:]
            out = out[:, :, -min_len:]
        return out + res


class TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1,
                 num_layers: int = 2, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else hidden_dim
            layers.append(
                TemporalBlock(
                    in_channels, hidden_dim, kernel_size, stride=1,
                    dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape mutation matching legacy execution: (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        out = self.network(x).permute(0, 2, 1)
        return self.fc(out[:, -1, :])


class TCNModel(AdvancedModel):
    """Wrapper managing hyperparameter optimization validation boundaries and execution for TCN."""

    @property
    def name(self) -> str:
        return "TCN"



    def build_hpo_search_space(self, trial: Any, X_train: np.ndarray, random_seed: int) -> Dict[str, Any]:
        num_layers = trial.suggest_int("num_layers", 1, 3)
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
        sequence_length = trial.suggest_int("sequence_length", 25, 250)

        # Enforce sequence constraint logic strictly matching legacy validation checks
        min_length = (kernel_size - 1) * (2 ** (num_layers - 1)) + 1
        if sequence_length < min_length:
            raise optuna.exceptions.TrialPruned()

        return {
            "baseline_method": trial.suggest_categorical("baseline_method", [0, 1, 2, 3, 4, 5]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "optimizer_type": trial.suggest_categorical("optimizer_type", ['AdamW']),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
            "num_layers": num_layers,
            "dropout": trial.suggest_float("dropout", 0.1, 0.5) if num_layers > 1 else 0.0,
            "sequence_length": sequence_length,
            "step_size": trial.suggest_int("step_size", 25, 75),
            "kernel_size": kernel_size,
            "threshold": trial.suggest_float('threshold', 0.1, 0.9)
        }

    def _instantiate_architecture(self, input_dim: int, params: Dict[str, Any]) -> nn.Module:
        return TCNClassifier(
            input_dim=input_dim,
            hidden_dim=params["hidden_dim"],
            output_dim=1,
            num_layers=params["num_layers"],
            kernel_size=params["kernel_size"],
            dropout=params.get("dropout", 0.0)
        )