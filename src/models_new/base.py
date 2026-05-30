from abc import ABC, abstractmethod
import joblib
import numpy as np
from typing import Any, Dict

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



class TraditionalModel(BaseModel):
    @property
    def is_traditional_model(self) -> bool:
        return True
    
    @property
    def model_parameters(self) -> Dict[str, Any]:
        """Returns the model hyperparameters."""
        return self.model.get_params()
    


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
    


# TODO: Implement a DeepLearningModel that uses PyTorch and overrides the abstract methods accordingly.
class DeepLearningModel(BaseModel):
    @property
    def is_traditional_model(self) -> bool:
        return False