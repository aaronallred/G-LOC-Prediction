import os
import joblib
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models_new.base import BaseModel

class _SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim = 1, num_layers = 2, dropout = 0.3, bidirectional = False):
        super(_SimpleLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first = True, 
            dropout = dropout,
            bidirectional = bidirectional
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :] # If we were predicting a single output label e.g., traditional learners
        out = self.fc(last_out)  # (batch, seq_len, 1)

        return out



class LSTMModel(BaseModel):
    @property
    def name(self) -> str:
        return "LSTM_Classifier"

    @property
    def data_pipeline_hyperparameters(self) -> Dict[str, Any]:
        hparams = self.get_model_parameters()
        return {
            "sequence_length": hparams.get("sequence_length", 50),
            "step_size": hparams.get("step_size", 25)
        }

    @property
    def hpo_search_space(self) -> Dict[str, Any]:
        """Defines the bounds using a standardized schema for the base class to parse."""
        return {
            "hidden_dim": {"type": "categorical", "choices": [64, 128, 256, 512]},
            "num_layers": {"type": "int", "low": 1, "high": 3},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "batch_size": {"type": "categorical", "choices": [64, 128]}
        }

    def _build_model(self) -> nn.Module:
        hparams = self.get_model_parameters()
        return _SimpleLSTM(
            input_dim = hparams.get("input_dim", 10), # Should be inferred from data dynamically
            hidden_dim = hparams.get("hidden_dim", 128),
            num_layers = hparams.get("num_layers", 2),
            dropout = hparams.get("dropout", 0.3)
        )