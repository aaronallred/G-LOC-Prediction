import copy
import os
import joblib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np


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
    TRADITIONAL_HYPERPARAMETERS: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.best_params = {}
        self.split_info = {}
        self.is_traditional = False

    @abstractmethod
    def train(self, X, y, params: Dict = None) -> None:
        """Train final model with specific params"""
        pass

    @abstractmethod
    def evaluate(self, X, y) -> Dict[str, float]:
        """Return metrics"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save artifacts (params, split info, model weights)"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load artifacts"""
        pass

    def is_traditiona(self) -> bool:
        """Return True if model is traditional (non-deep learning)"""
        return self.is_traditional

    def get_traditional_hyperparameters(self) -> Dict[str, Any]:
        """Return the traditional data-pipeline hyperparameters for this model."""
        traditional_hyperparameters = copy.deepcopy(self.TRADITIONAL_HYPERPARAMETERS)
        overrides = self.config.get("traditional_hyperparameters")
        if isinstance(overrides, dict):
            traditional_hyperparameters.update(copy.deepcopy(overrides))
        return traditional_hyperparameters

    def hpo_defaults(self) -> Dict[str, Any]:
        """Return model-local HPO defaults.

        This is intentionally simple and model implementations should override
        to exactly match legacy search defaults (n_trials, metric, timeout,
        sampler_seed, etc.). The returned dict will be used by the
        cross-validation driver and must contain at least the following keys:
        - enabled: bool
        - n_trials: int
        - timeout: Optional[int]
        - metric: str
        - train_fraction: float
        """
        return {
            "enabled": True,
            "n_trials": 10,
            "timeout": None,
            "metric": "f1",
            "train_fraction": 0.8,
            "sampler_seed": None,
            "pruner_startup_trials": 3,
            "pruner_warmup_steps": 0,
        }

    def build_hpo_search_space(self, trial, X_train: np.ndarray, random_seed: int) -> Dict[str, Any]:
        """Build model-specific Optuna search space.

        Subclasses should implement this method and return a dictionary of
        hyperparameters suitable for passing into the model.train(...) method.
        """
        raise NotImplementedError("Model must implement build_hpo_search_space(trial, X_train, random_seed)")

    def _initialize_model_for_classification(
        self,
        strategy: ModelInitStrategy,
        x_train,
        y_train,
        class_weight_imb,
        random_state,
        save_folder: str = None,
        model_name: str = None,
        best_params: Dict[str, Any] = None,
    ):
        """Initialize and return a fitted sklearn estimator based on initialization strategy.
        
        This method consolidates the initialization logic that was previously duplicated
        across every model's classify_traditional() implementation.
        
        Args:
            strategy: ModelInitStrategy enum specifying how to initialize the model.
            x_train: Training feature matrix.
            y_train: Training labels.
            class_weight_imb: Class weight strategy (e.g., 'balanced', None).
            random_state: Random seed for reproducibility.
            save_folder: Directory containing saved model (required for LOAD_SAVED_MODEL).
            model_name: Filename of saved model (required for LOAD_SAVED_MODEL).
            best_params: Hyperparameters dict (required for RETRAIN_WITH_CONFIG_PARAMS).
        
        Returns:
            Fitted sklearn estimator ready for prediction.
            
        Raises:
            NotImplementedError: If subclass does not override _build_sklearn_estimator().
            ValueError: If strategy-specific requirements not met (e.g., best_params missing).
        """
        y_train_flat = y_train.ravel() if hasattr(y_train, 'ravel') else y_train
        
        if strategy == ModelInitStrategy.RETRAIN_WITH_DEFAULTS:
            # Train with default hyperparameters
            return self._build_sklearn_estimator(
                class_weight=class_weight_imb,
                random_state=random_state,
                params=None,
            ).fit(x_train, y_train_flat)
        
        elif strategy == ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS:
            # Train with provided hyperparameters from config/JSON
            if best_params is None:
                raise ValueError(
                    "RETRAIN_WITH_CONFIG_PARAMS strategy requires best_params to be provided."
                )
            runtime_params = dict(best_params)
            # Don't set n_jobs here - let subclasses handle it if they support it
            return self._build_sklearn_estimator(
                class_weight=class_weight_imb,
                random_state=random_state,
                params=runtime_params,
            ).fit(x_train, y_train_flat)
        
        elif strategy == ModelInitStrategy.LOAD_SAVED_MODEL:
            # Load pre-trained model from disk
            if save_folder is None or model_name is None:
                raise ValueError(
                    "LOAD_SAVED_MODEL strategy requires both save_folder and model_name."
                )
            model_path = os.path.join(save_folder, model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            return joblib.load(model_path)
        
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")

    def _build_sklearn_estimator(
        self,
        class_weight: str = None,
        random_state: int = None,
        params: Dict[str, Any] = None,
    ):
        """Build an unfitted sklearn estimator.
        
        Subclasses should override this method to return their specific sklearn classifier
        configured with the given parameters.
        
        Args:
            class_weight: Class weight strategy to apply.
            random_state: Random seed.
            params: Additional hyperparameters to apply (overrides class_weight and random_state).
        
        Returns:
            Unfitted sklearn estimator (e.g., RandomForestClassifier, LogisticRegression, etc.).
        
        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not override _build_sklearn_estimator()."
        )

    def classify_traditional(
        self,
        x_train,
        x_test,
        y_train,
        y_test,
        class_weight_imb,
        random_state,
        save_folder,
        model_name,
        strategy: ModelInitStrategy = ModelInitStrategy.RETRAIN_WITH_DEFAULTS,
        best_params=None,
    ):
        """Legacy-compatible traditional classifier entry point.

        Replaces the confusing (retrain, temporal) flags with a semantic ModelInitStrategy enum.
        
        Args:
            x_train, x_test, y_train, y_test: Train/test splits.
            class_weight_imb: Class weight strategy.
            random_state: Random seed.
            save_folder: Folder for model persistence (used with LOAD_SAVED_MODEL).
            model_name: Model filename (used with LOAD_SAVED_MODEL).
            strategy: ModelInitStrategy specifying initialization behavior.
            best_params: Hyperparameters dict (required for RETRAIN_WITH_CONFIG_PARAMS).
        
        Returns:
            Tuple of legacy metrics: (accuracy, precision, recall, f1, specificity, g_mean)
            or variable-length tuple if model adds extra metrics (e.g., tree depths for RF).

        Traditional model wrappers should override this method and return the same
        tuple structure as the legacy classify_* functions in
        src/scripts/GLOC_classifier_traditional.py.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement classify_traditional()."
        )

    @staticmethod
    def _legacy_binary_metrics(y_true, y_pred) -> tuple[float, float, float, float, float, float]:
        """Compute legacy metric set with matching defaults and ordering."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        g_mean = geometric_mean_score(y_true, y_pred)
        return accuracy, precision, recall, f1, specificity, g_mean

    @staticmethod
    def _print_legacy_metrics(title: str, metrics: tuple[float, float, float, float, float, float]) -> None:
        """Print the legacy binary metric summary in the expected terminal format."""
        accuracy, precision, recall, f1, specificity, g_mean = metrics
        print(f"\n{title}")
        print(f"Accuracy:  {accuracy}")
        print(f"Precision:  {precision}")
        print(f"Recall:  {recall}")
        print(f"F1 Score:  {f1}")
        print(f"Specificity:  {specificity}")
        print(f"G-Mean:  {g_mean}")

    @abstractmethod
    def get_name(self) -> str:
        """Return model name"""
        pass