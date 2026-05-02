"""
Deep-learning model adapter interface for G-LOC cross-validation.

This module provides the abstract DLModelAdapter base class that allows deep-learning
frameworks (PyTorch, TensorFlow, etc.) to be integrated into the cross-validation runner
without adding heavy framework dependencies to the core pipeline.

Users can subclass DLModelAdapter and implement the required methods to support their
chosen DL framework. For testing and examples, see src/modes/cross_validation.py which
includes a SyntheticDLAdapter that uses sklearn's LogisticRegression.

Example usage:
    from src.models.dl_adapter import DLModelAdapter
    import torch
    
    class MyPyTorchAdapter(DLModelAdapter):
        def fit(self, X_train, y_train, X_val, y_val, config):
            # Convert numpy to torch tensors and train
            ...
        
        def predict_proba(self, X):
            # Convert numpy to torch, run inference, return proba
            ...
        
        def save(self, path):
            # Save model checkpoint
            ...
        
        def get_name(self):
            return "MyPyTorchModel"
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class DLModelAdapter(ABC):
    """Abstract base class for deep-learning model adapters.
    
    Adapters serve as bridges between the cross-validation runner (which works with
    numpy arrays) and DL frameworks (which typically require dataset objects, tensors, etc.).
    
    By using adapters, we keep the core CV runner lightweight and framework-agnostic, while
    allowing users to integrate arbitrary DL models.
    
    Key design principles:
    - Adapters accept numpy arrays from DataPipeline
    - Adapters handle framework-specific conversions internally
    - No heavy DL deps are required by the core runner
    - Adapters expose a minimal interface (fit, predict_proba, save, get_name)
    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> None:
        """Train the DL model on training data with validation data.
        
        The model should use validation data for early stopping or other validation-based
        heuristics, but should not modify the training/validation split or use validation
        labels inappropriately.
        
        Args:
            X_train: Training features, shape (N_train, n_features), typically float32.
            y_train: Training labels, shape (N_train,), typically int or float.
            X_val: Validation features, shape (N_val, n_features), typically float32.
            y_val: Validation labels, shape (N_val,), typically int or float.
            config: Arbitrary configuration dict passed from the CV config.
                   Examples: {"epochs": 50, "batch_size": 32, "learning_rate": 1e-3}
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for input samples.
        
        Args:
            X: Features, shape (N, n_features), typically float32.
            
        Returns:
            Array of shape (N, n_classes) with class probabilities (each row sums to 1).
            For binary classification, typically shape (N, 2).
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the trained model to disk.
        
        Args:
            path: Directory path where the model should be saved.
                 The adapter may create subdirectories or files as needed.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this model.
        
        Used for logging and artifact naming in cross-validation results.
        
        Returns:
            String name, e.g., "PyTorchLSTM" or "TFTransformer".
        """
        pass
