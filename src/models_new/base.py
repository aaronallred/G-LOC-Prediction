from abc import ABC, abstractmethod
from enum import Enum
import joblib
import numpy as np
from sklearn.base import clone
from typing import Any, Dict, Optional, List

import optuna
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from skopt import BayesSearchCV
from torch.utils.data import DataLoader

from src.advanced_experiment_utils import *

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
    def train(self, X: Any, y: Any):
        """Hides the PyTorch training loop or calls sklearn.fit()."""
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        pass

    # @abstractmethod
    # def save_model(self, path: str):
    #     """Saves the underlying model artifacts."""
    #     pass
    #
    # @abstractmethod
    # def load_model(self, path: str):
    #     """Loads the underlying model artifacts into self.model."""
    #     pass
    #
    # @abstractmethod
    # def get_model_parameters(self) -> Dict[str, Any]:
    #     """Returns the model hyperparameters."""
    #     pass
    #
    # @abstractmethod
    # def set_model_parameters(self, model_hyperparameters: Dict[str, Any]):
    #     """Updates the underlying model's hyperparameters after initialization."""
    #     pass



class TraditionalModel(BaseModel):
    @property
    def is_traditional_model(self) -> bool:
        return True



    def tune(self, X: np.ndarray, y: np.ndarray, random_seed: int, class_weight: Optional[str] = None) -> (Dict[str, Any], BayesSearchCV):
        """Hides the hyperparameter optimization for each model."""
        valid_params = self.model.get_params()
        estimator_params = {k: v for k, v in {"random_state": random_seed, "class_weight": class_weight}.items() if
                            k in valid_params} # Add `random_state` and `class_weight` parameters if model supports it
        self.model.set_params(**estimator_params)
        search = BayesSearchCV(
            estimator = self.model,
            search_spaces = self.hpo_search_space,
            n_iter = 30,
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


class AdvancedModel(BaseModel):
    def __init__(self, model_hyperparameters: Optional[Dict[str, Any]] = None):
        super().__init__(model_hyperparameters)
        self.best_params = model_hyperparameters or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.all_features: List[str] = []
        self.hpo_config: Dict[str, Any] = {}

    @property
    def is_traditional_model(self) -> bool:
        return False

    @property
    def hpo_search_space(self) -> Dict[str, Any]:
        return {}

    @property
    def data_pipeline_hyperparameters(self) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def build_hpo_search_space(self, trial: Any, X_train: np.ndarray, random_seed: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _instantiate_architecture(self, input_dim: int, params: Dict[str, Any]) -> torch.nn.Module:
        pass

    def tune(self, X: np.ndarray, y: np.ndarray, hpo_config: Dict[str, Any], random_seed: int) -> Dict[str, Any]:
        """Tunes hyperparameters using external metric calculation for optimization feedback."""
        self.hpo_config = hpo_config
        metric_name = hpo_config.get("metric", "f1")
        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            params = self.build_hpo_search_space(trial, X, random_seed)
            params["objective_var"] = hpo_config.get("metric")

            X_ds, _ = baseline_down_select(X, self.all_features, params["baseline_method"])
            train_dataset, val_dataset, train_windows_tensor, _, _, val_windows_labels = train_test_split_trials(
                X = X_ds,
                Y = y,
                window_size = params["sequence_length"],
                step_size = params["step_size"],
                test_ratio = 0.2,
                random_state = random_seed,
                end_label = True
            )

            trial_net = self._instantiate_architecture(train_windows_tensor.shape[2], params).to(self.device)
            class_weights = torch.tensor(compute_class_weight('balanced', classes=np.array([0, 1]), y=y),
                                         dtype=torch.float)
            criterion, optimizer = build_training_components(trial_net, class_weights, params, self.device)

            train_loader = DataLoader(
                train_dataset, batch_size=params["batch_size"],
                sampler=build_sampler(train_dataset.tensors[1], class_weights) if hpo_config["use_sampler"] else None
            )
            val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

            best_state, _, _ = train_with_early_stopping(
                model = trial_net,
                train_loader = train_loader,
                val_loader = val_loader,
                criterion = criterion,
                optimizer = optimizer,
                device = self.device,
                threshold = params["threshold"],
                objective_var = params["objective_var"]
            )
            if best_state:
                trial_net.load_state_dict(best_state)

            # Temporary internal evaluation execution for Optuna loop
            trial_net.eval()
            trial_preds = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    out = trial_net(xb.to(self.device))
                    trial_preds.extend((out.reshape(-1) >= params["threshold"]).float().cpu().numpy())

            return float(f1_score(val_windows_labels.numpy(), np.array(trial_preds)))

        study.optimize(objective, n_trials=int(hpo_config["n_trials"]), catch=(RuntimeError, ValueError))
        self.best_params = dict(study.best_params)
        return {
            "best_params": self.best_params,
            "summary": {
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "metric": metric_name
            }
        }

    def train(self, X: np.ndarray, y: np.ndarray, params: Optional[Dict[str, Any]] = None) -> None:
        """Fits network weights to the configuration parameters."""
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

        self.model = self._instantiate_architecture(train_w.shape[2], target_params).to(self.device)
        class_weights = torch.tensor(compute_class_weight('balanced', classes=np.array([0, 1]), y=y), dtype=torch.float)
        criterion, optimizer = build_training_components(self.model, class_weights, target_params, self.device)

        train_loader = DataLoader(
            train_ds, batch_size=target_params["batch_size"],
            sampler=build_sampler(train_ds.tensors[1], class_weights) if self.hpo_config.get("use_sampler",
                                                                                             True) else None
        )

        if final_early_stop and val_loader:
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
        """Generates raw sequence window binary predictions (0 or 1)."""
        X_ds, _ = baseline_down_select(X, self.all_features, self.best_params["baseline_method"])
        dummy_y = np.zeros(len(X_ds))
        dataset, _, _, _, _, _ = train_test_split_trials(
            X = X_ds,
            Y = dummy_y,
            window_size = self.best_params["sequence_length"],
            step_size = 10,
            test_ratio = None,
            end_label = True
        )
        loader = DataLoader(dataset, batch_size = self.best_params["batch_size"], shuffle = False)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for x_batch, _ in loader:
                outputs = self.model(x_batch.to(self.device))
                preds = (outputs.reshape(-1) >= self.best_params["threshold"]).float()
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def _build_model(self, model_hyperparameters: Optional[Dict[str, Any]]) -> Any:
        """Safe placeholder initialization.

        Deep learning architectures cannot build their network graphs until data
        dimensions and Optuna hyperparameters are known.
        """
        if not model_hyperparameters or "sequence_length" not in model_hyperparameters:
            return None

        # Optional: If fully defined params are passed (e.g. loading from JSON config),
        # you could build it here if you track your input feature dimension.
        return None

    def get_model_parameters(self) -> Dict[str, Any]:
        """Returns the current optimized or configured hyperparameters."""
        return self.best_params

    def set_model_parameters(self, model_hyperparameters: Dict[str, Any]) -> None:
        """Updates the hyperparameter configuration dictionary."""
        self.best_params = model_hyperparameters