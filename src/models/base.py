from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.best_params = {}
        self.split_info = {}
        self.is_traditional = False

    @abstractmethod
    def tune(self, X, y, groups=None) -> None:
        """Run Optuna or BayesSearchCV"""
        pass

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