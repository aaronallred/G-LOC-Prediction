import copy
from abc import ABC, abstractmethod
from typing import Dict, Any

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class BaseModel(ABC):
    TRADITIONAL_HYPERPARAMETERS: Dict[str, Any] = {}

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
        retrain,
        temporal=False,
        best_params=None,
    ):
        """Legacy-compatible traditional classifier entry point.

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