from abc import ABC, abstractmethod
from enum import Enum
import joblib
import numpy as np
from sklearn.base import clone
from typing import Any, Dict, Optional

import optuna
from skopt import BayesSearchCV
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

    # @property
    # @abstractmethod
    # def model_object(self) -> Any:
    #     """Returns the underlying model object (e.g. sklearn estimator or PyTorch nn.Module)."""
    #     pass

    @property
    @abstractmethod
    def hpo_search_space(self) -> Dict[str, Any]:
        """Returns the hyperparameter search space for this model."""
        pass



    @abstractmethod
    def _build_model(self, model_hyperparameters: Dict[str, Any] | None) -> Any:
        """Initializes the underlying PyTorch or sklearn model."""
        pass

    @abstractmethod
    def tune(self, X: Any, y: Any) -> Any:
        """Hides the hyperparameter optimization for each model."""
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



    def tune(self, X: np.ndarray, y: np.ndarray, random_seed: int, class_weight: Optional[str] = None) -> (Dict[str, Any], BayesSearchCV):
        """Hides the hyperparameter optimization for each model."""
        valid_params = self.model.get_params()
        estimator_params = {k: v for k, v in {"random_state": random_seed, "class_weight": class_weight}.items() if
                            k in valid_params} # Add `random_state` and `class_weight` parameters if model supports it
        self.model.set_params(**estimator_params)
        search = BayesSearchCV(
            estimator = self.model,
            search_spaces = self.hpo_search_space,
            n_iter = 3,
            cv = 3,
            scoring = "f1",
            random_state = random_seed,
            verbose = 1,
            error_score = np.nan,
        )
        search.fit(X, np.ravel(y))

        best_params = dict(search.best_params_)
        summary = {
            "best_params": best_params,
            "best_score": float(search.best_score_),
            "best_index": int(search.best_index_),
            "n_iter": 30,
            "cv": 3,
            "scoring": "f1",
        }

        return {"best_params": best_params, "summary": summary}, search

    def train(self, X: np.ndarray, y: np.ndarray):
        """Fit the model using sklearn's fit method."""
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
    def __init__(self, model_hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Injects the neural network class and the Optuna search space generator.
        """
        self.model_class = model_class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_params = None

        # Super init calls _build_model
        super().__init__(model_hyperparameters)

    @property
    def is_traditional_model(self) -> bool:
        return False

    @property
    def model_object(self) -> Any:
        return copy.deepcopy(self.model)

    def _build_model(self, hparams: Dict[str, Any]) -> nn.Module:
        """Initializes the injected PyTorch model using provided hyperparameters."""
        if not hparams:
            return None
        # Extract architecture-specific params and ignore training params (like lr, batch_size)
        arch_params = {k: v for k, v in hparams.items() if k not in ['lr', 'batch_size', 'weight_decay', 'epochs']}
        return self.model_class(**arch_params).to(self.device)

    def tune(self, X_train: torch.Tensor, y_train: torch.Tensor, n_trials: int = 50) -> Dict[str, Any]:
        """Runs the Optuna HPO using the injected search space."""

        # Pre-split an internal validation set to avoid data leakage during HPO
        X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

        def objective(trial):
            hparams = self.search_space_fn(trial)
            model = self._build_model(hparams)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=hparams.get('lr', 1e-3),
                weight_decay=hparams.get('weight_decay', 1e-4)
            )
            criterion = nn.BCEWithLogitsLoss()

            train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=hparams.get('batch_size', 64), shuffle=True)

            # Minimal training loop for the trial
            model.train()
            for epoch in range(hparams.get('epochs', 15)):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    out = model(batch_X)
                    loss = criterion(out, batch_y.unsqueeze(1).float())
                    loss.backward()
                    optimizer.step()

            # Evaluate trial on internal validation set
            model.eval()
            with torch.no_grad():
                preds = torch.sigmoid(model(X_v.to(self.device))).round().cpu()
                score = f1_score(y_v.cpu(), preds)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_trial.params
        self.model = self._build_model(self.best_params)  # Lock in the best architecture
        return self.best_params

    def train(self, X: torch.Tensor, y: torch.Tensor):
        """Final training pass using the best model configuration."""
        if not self.model:
            raise ValueError("Model is not initialized. Run tune() or provide valid hparams first.")

        hparams = self.best_params or self.get_model_parameters()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=hparams.get('lr', 1e-3))
        criterion = nn.BCEWithLogitsLoss()

        loader = DataLoader(TensorDataset(X, y), batch_size=hparams.get('batch_size', 64), shuffle=True)

        self.model.train()
        for epoch in range(hparams.get('epochs', 15)):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_X)
                loss = criterion(out, batch_y.unsqueeze(1).float())
                loss.backward()
                optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            out = self.model(X.to(self.device))
            return torch.sigmoid(out).round().cpu()