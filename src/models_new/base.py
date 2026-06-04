from abc import ABC, abstractmethod
from enum import Enum
import joblib
import numpy as np
from sklearn.base import clone
from typing import Any, Dict

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

class ModelInitStrategy(Enum):
    """Enum for specifying how to initialize a model during traditional classification.
    
    Replaces the confusing (retrain, temporal) boolean pair with a single semantic strategy.
    """
    RETRAIN_WITH_DEFAULTS = "retrain_with_defaults"
    """Train model using default hyperparameters (no best_params needed)."""
    
    RETRAIN_WITH_CONFIG_PARAMS = "retrain_with_config_params"
    """Train model using hyperparameters from best_params dict (e.g., from JSON config)."""
    
    LOAD_SAVED_MODEL = "load_saved_model"
    """Load a pre-trained model from disk (save_folder and model_name required)."""

class BaseModel(ABC):
    def __init__(self, model_hyperparameters: Dict[str, Any] = None):
        self.model = self._build_model(model_hyperparameters)

    @property
    @abstractmethod
    def is_traditional_model(self) -> bool:
        """Indicates whether this model should use the traditional ML pipeline."""
        pass

    @property
    @abstractmethod
    def data_pipeline_hyperparameters(self) -> Dict[str, Any]:
        """Extracts hyperparameters relevant to the data pipeline."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the model name."""
        pass

    @property
    @abstractmethod
    def model_object(self) -> Any:
        """Returns the underlying model object (e.g. sklearn estimator or PyTorch nn.Module)."""
        pass

    @property
    @abstractmethod
    def hpo_search_space(self) -> Dict[str, Any]:
        """Returns the hyperparameter search space for this model."""
        pass



    @abstractmethod
    def _build_model(self) -> Any:
        """Initializes the underlying PyTorch or sklearn model."""
        pass

    @abstractmethod
    def train(self, X: Any, y: Any, random_state: int = 42):
        """Hides the PyTorch training loop or calls sklearn.fit()."""
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        pass

    @abstractmethod
    def save_model(self, path: str):
        """Saves the underlying model artifacts."""
        pass

    @abstractmethod
    def load_model(self, path: str):
        """Loads the underlying model artifacts into self.model."""
        pass

    @abstractmethod
    def get_model_parameters(self) -> Dict[str, Any]:
        """Returns the model hyperparameters."""
        pass

    @abstractmethod
    def set_model_parameters(self, model_hyperparameters: Dict[str, Any]):
        """Updates the underlying model's hyperparameters after initialization."""
        pass



class TraditionalModel(BaseModel):
    @property
    def is_traditional_model(self) -> bool:
        return True

    @property
    def model_object(self) -> Any:
        """Returns the underlying model object (e.g. sklearn estimator or PyTorch nn.Module)."""
        return clone(self.model)



    def train(self, X: np.ndarray, y: np.ndarray):
        """Fit the model using sklearn's fit method."""
        if "random_state" in self.model.get_params():
            self.model.fit(X, y, random_state = self.model.get_params()["random_state"])
        else:
            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using sklearn's predict method."""
        return self.model.predict(X)

    def save_model(self, path: str) -> None:
        """Saves the sklearn model using joblib."""
        joblib.dump(self.model, path)

    def load_model(self, path: str) -> None:
        """Loads the sklearn model using joblib."""
        self.model = joblib.load(path)

    def get_model_parameters(self) -> Dict[str, Any]:
        """Returns the model hyperparameters."""
        return self.model.get_params()

    def set_model_parameters(self, model_hyperparameters: Dict[str, Any]) -> None:
        """Updates the sklearn model's hyperparameters after initialization."""
        self.model.set_params(**model_hyperparameters)
    


class DeepLearningModel(BaseModel):
    def __init__(self, model_hyperparameters: Dict[str, Any] = None):
        super().__init__(model_hyperparameters or {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_traditional_model(self) -> bool:
        return False

    @property
    def model_object(self) -> Any:
        return copy.deepcopy(self.model)

    def _prepare_loader(self, X: np.ndarray, y: np.ndarray = None, shuffle: bool = False) -> DataLoader:
        """Converts raw arrays to PyTorch DataLoaders."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
            
        batch_size = self.get_model_parameters().get("batch_size", 64)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Standardized PyTorch training loop with optional early stopping."""
        train_loader = self._prepare_loader(X_train, y_train, shuffle=True)
        hparams = self.get_model_parameters()
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=hparams.get("lr", 1e-3), 
            weight_decay=hparams.get("weight_decay", 1e-4)
        )

        epochs = hparams.get("num_epochs", 15)
        self.model.train()
        self.model.to(self.device)

        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = nn.BCEWithLogitsLoss()
                loss.backward()
                optimizer.step()
                
            # Optional: Add validation logic here if X_val is provided

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Standardized inference loop."""
        test_loader = self._prepare_loader(X, shuffle=False)
        self.model.eval()
        self.model.to(self.device)
        
        predictions = []
        with torch.no_grad():
            for batch_x, in test_loader:
                outputs = torch.sigmoid(self.model(batch_x.to(self.device)))
                predictions.extend(outputs.cpu().numpy())
                
        return np.array(predictions).flatten()

    def tune(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
        """Encapsulates Optuna HPO strictly within the model adapter."""
        def objective(trial: optuna.Trial):
            # 1. Parse search space and suggest values
            trial_hparams = {}
            for param, config in self.hpo_search_space.items():
                if config["type"] == "categorical":
                    trial_hparams[param] = trial.suggest_categorical(param, config["choices"])
                elif config["type"] == "float":
                    trial_hparams[param] = trial.suggest_float(param, config["low"], config["high"], log=config.get("log", False))
                elif config["type"] == "int":
                    trial_hparams[param] = trial.suggest_int(param, config["low"], config["high"])

            # 2. Update state and rebuild architecture
            self.set_model_parameters(trial_hparams)
            self.model = self._build_model()
            
            # 3. Train and evaluate (implement a validation split internally for tuning)
            # For brevity, assuming train() computes and returns a validation metric
            val_metric = self.train(X_train, y_train) 
            return val_metric

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params