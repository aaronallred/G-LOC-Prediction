from abc import ABC, abstractmethod
from enum import Enum
import joblib
import numpy as np
from typing import Any, Dict, Optional, List

import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from src.advanced_experiment_utils import (
    baseline_down_select,
    train_test_split_trials,
    build_training_components,
    build_sampler,
    train_with_early_stopping,
    run_train_epoch,
)
from imblearn.metrics import geometric_mean_score
from sklearn import metrics as sklearn_metrics


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
    """Unified base class for all machine learning models.

    Both traditional (sklearn) and advanced (PyTorch) models share the same
    interface so that the cross-validation and other orchestration code can
    treat them polymorphically.
    """

    def __init__(self, model_hyperparameters: Dict[str, Any] = None):
        self.model = self._build_model(model_hyperparameters)
        self.best_params: Dict[str, Any] = model_hyperparameters or {}
        self.all_features: List[str] = []
        self.searcher: Optional[Any] = None

    @property
    @abstractmethod
    def is_traditional_model(self) -> bool:
        pass

    @property
    @abstractmethod
    def data_pipeline_hyperparameters(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_hpo_search_space(self) -> Any:
        pass

    @abstractmethod
    def _build_model(self, model_hyperparameters: Dict[str, Any] | None) -> Any:
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, params: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        preds = self.predict(X)
        return {
            "accuracy": sklearn_metrics.accuracy_score(y, preds),
            "precision": sklearn_metrics.precision_score(y, preds),
            "recall": sklearn_metrics.recall_score(y, preds),
            "f1": sklearn_metrics.f1_score(y, preds),
            "specificity": sklearn_metrics.recall_score(y, preds, pos_label=0),
            "g_mean": geometric_mean_score(y, preds),
        }

    @abstractmethod
    def save_model(self, path: str) -> None:
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        pass

    @abstractmethod
    def get_model_parameters(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def set_model_parameters(self, model_hyperparameters: Dict[str, Any]) -> None:
        pass


class TraditionalModel(BaseModel):
    @property
    def is_traditional_model(self) -> bool:
        return True

    @property
    def hpo_search_space(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_hpo_search_space(self) -> Any:
        return self.hpo_search_space

    def train(self, X: np.ndarray, y: np.ndarray, params: Optional[Dict[str, Any]] = None) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save_model(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load_model(self, path: str) -> None:
        self.model = joblib.load(path)

    def get_model_parameters(self) -> Dict[str, Any]:
        return self.model.get_params()

    def set_model_parameters(self, model_hyperparameters: Dict[str, Any]) -> None:
        self.model.set_params(**model_hyperparameters)


class AdvancedModel(BaseModel):
    def __init__(self, model_hyperparameters: Optional[Dict[str, Any]] = None):
        super().__init__(model_hyperparameters)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hpo_config: Dict[str, Any] = {}

    @property
    def is_traditional_model(self) -> bool:
        return False

    @property
    def data_pipeline_hyperparameters(self) -> Dict[str, Any]:
        return {}

    def _build_model(self, model_hyperparameters: Optional[Dict[str, Any]] = None, *, input_dim: Optional[int] = None, **kwargs) -> Any:
        if input_dim is None:
            return None
        raise NotImplementedError("Advanced model subclasses must implement _build_model with input_dim to instantiate architecture.")

    def train(self, X: np.ndarray, y: np.ndarray, params: Optional[Dict[str, Any]] = None) -> None:
        target_params = params if params is not None else self.best_params
        self.best_params = target_params

        X_ds, _ = baseline_down_select(X, self.all_features, target_params["baseline_method"])
        final_early_stop = self.hpo_config.get("final_early_stop", False)

        if final_early_stop:
            train_ds, val_ds, train_w, _, _, _ = train_test_split_trials(
                X_ds, y, target_params["sequence_length"], target_params["step_size"], test_ratio=0.2, end_label=True
            )
            val_loader = DataLoader(val_ds, batch_size=target_params["batch_size"], shuffle=False)
        else:
            train_ds, _, train_w, _, _, _ = train_test_split_trials(
                X_ds, y, target_params["sequence_length"], target_params["step_size"], test_ratio=None, end_label=True
            )
            val_loader = None

        self.model = self._build_model(target_params, input_dim=train_w.shape[2]).to(self.device)
        class_weights = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1]), y=y), dtype=torch.float)
        criterion, optimizer = build_training_components(self.model, class_weights, target_params, self.device)

        train_loader = DataLoader(
            train_ds, batch_size=target_params["batch_size"],
            sampler=build_sampler(train_ds.tensors[1], class_weights) if self.hpo_config.get("use_sampler",
                                                                                             True) else None
        )

        if final_early_stop and val_loader is not None:
            best_state, _, _ = train_with_early_stopping(
                self.model, train_loader, val_loader, criterion, optimizer, self.device,
                target_params["threshold"], self.hpo_config.get("metric", "f1")
            )
            if best_state:
                self.model.load_state_dict(best_state)
        else:
            num_epochs = max(target_params.get("best_epoch", 15), 15)
            for epoch in range(num_epochs):
                run_train_epoch(self.model, train_loader, criterion, optimizer, self.device)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_ds, _ = baseline_down_select(X, self.all_features, self.best_params["baseline_method"])
        dummy_y = np.zeros(len(X_ds))
        dataset, _, _, _, _, _ = train_test_split_trials(
            X=X_ds, Y=dummy_y, window_size=self.best_params["sequence_length"], step_size=10, test_ratio=None,
            end_label=True
        )
        loader = DataLoader(dataset, batch_size=self.best_params["batch_size"], shuffle=False)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for x_batch, _ in loader:
                outputs = self.model(x_batch.to(self.device))
                preds = (outputs.reshape(-1) >= self.best_params["threshold"]).float()
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))

    def get_model_parameters(self) -> Dict[str, Any]:
        return self.best_params

    def set_model_parameters(self, model_hyperparameters: Dict[str, Any]) -> None:
        self.best_params = model_hyperparameters
