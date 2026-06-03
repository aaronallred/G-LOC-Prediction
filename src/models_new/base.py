from abc import ABC, abstractmethod
from enum import Enum
import joblib
import numpy as np
from sklearn.base import clone
from typing import Any, Dict

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
    def model_parameters(self) -> Dict[str, Any]:
        """Returns the model hyperparameters."""
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
    def train(self, X: Any, y: Any):
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
    def model_parameters(self) -> Dict[str, Any]:
        """Returns the model hyperparameters."""
        return self.model.get_params()

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
    


# TODO: Implement a DeepLearningModel that uses PyTorch and overrides the abstract methods accordingly.
class DeepLearningModel(BaseModel):
    @property
    def is_traditional_model(self) -> bool:
        return False