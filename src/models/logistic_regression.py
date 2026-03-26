import joblib
import logging
from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate

from models.base import BaseModel

logger = logging.getLogger(__name__)


class LogisticRegression(BaseModel):
    """
    Logistic Regression classifier that inherits from BaseModel.
    
    Supports hyperparameter tuning via Optuna or BayesSearchCV.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LogisticRegression model.
        
        Args:
            config: Configuration dictionary containing model hyperparameters and settings.
                   Required keys: random_state (optional), class_weight (optional)
        """
        super().__init__(config)
        self.model = None
        self.is_traditional = True
        self.feature_importance = None

    def tune(self, X, y, groups=None) -> None:
        """
        Perform hyperparameter tuning using BayesSearchCV.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            groups: Group labels for cross-validation (optional)
        """
        # TODO: Implement BayesSearchCV or Optuna-based hyperparameter tuning
        # Search spaces to consider:
        # - C: float (regularization strength)
        # - penalty: str ('l2', 'l1', 'elasticnet')
        # - solver: str ('lbfgs', 'liblinear', 'saga', 'newton-cg')
        # - max_iter: int (max iterations)
        # - tol: float (tolerance for convergence)
        logger.info("Starting hyperparameter tuning for LogisticRegression")
        pass

    def train(self, X, y, params: Dict = None) -> None:
        """
        Train the LogisticRegression model with specified parameters.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            params: Dictionary of hyperparameters. If None, uses self.best_params or defaults.
        """
        # TODO: Initialize and train the model with provided or best parameters
        if params is None:
            params = self.best_params if self.best_params else self._get_default_params()
        
        logger.info(f"Training LogisticRegression with params: {params}")
        
        try:
            self.model = SklearnLogisticRegression(**params)
            self.model.fit(X, y)
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def evaluate(self, X, y) -> Dict[str, float]:
        """
        Evaluate the trained model on provided data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        # TODO: Implement evaluation with multiple metrics
        # Metrics to compute:
        # - accuracy
        # - precision
        # - recall
        # - f1
        # - specificity
        # - g_mean (geometric mean)
        
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return {}
        
        try:
            predictions = self.model.predict(X)
            
            metrics = {
                "accuracy": accuracy_score(y, predictions),
                "precision": precision_score(y, predictions, zero_division=0),
                "recall": recall_score(y, predictions, zero_division=0),
                "f1": f1_score(y, predictions, zero_division=0),
            }
            
            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {}

    def save(self, path: str) -> None:
        """
        Save model artifacts (model weights, best params, split info).
        
        Args:
            path: Directory path where artifacts will be saved.
        """
        # TODO: Implement saving of:
        # - Model weights/coefficients
        # - best_params dictionary
        # - split_info dictionary
        # - Model metadata
        
        logger.info(f"Saving LogisticRegression model to {path}")
        try:
            if self.model is not None:
                joblib.dump(self.model, f"{path}/logistic_regression_model.pkl")
            
            # Save metadata
            metadata = {
                "best_params": self.best_params,
                "split_info": self.split_info,
                "config": self.config,
            }
            joblib.dump(metadata, f"{path}/logistic_regression_metadata.pkl")
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load(self, path: str) -> None:
        """
        Load model artifacts from disk.
        
        Args:
            path: Directory path from which artifacts will be loaded.
        """
        # TODO: Implement loading of:
        # - Model weights/coefficients
        # - best_params dictionary
        # - split_info dictionary
        # - Model metadata
        
        logger.info(f"Loading LogisticRegression model from {path}")
        try:
            self.model = joblib.load(f"{path}/logistic_regression_model.pkl")
            metadata = joblib.load(f"{path}/logistic_regression_metadata.pkl")
            
            self.best_params = metadata.get("best_params", {})
            self.split_info = metadata.get("split_info", {})
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default hyperparameters for LogisticRegression.
        
        Returns:
            Dictionary with default hyperparameters.
        """
        return {
            "random_state": self.config.get("random_state", 42),
            "max_iter": self.config.get("max_iter", 1000),
            "class_weight": self.config.get("class_weight", None),
            "solver": self.config.get("solver", "lbfgs"),
            "C": self.config.get("C", 1.0),
        }

    def predict(self, X) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of predicted labels.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return np.array([])
        
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """
        Get probability predictions for each class.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of class probabilities.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return np.array([])
        
        return self.model.predict_proba(X)

    def get_name(self) -> str:
        """
        Get the name of the model.
        
        Returns:
            String name of the model.
        """
        return "LogReg"