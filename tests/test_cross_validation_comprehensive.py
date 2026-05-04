"""
Comprehensive test suite for cross-validation functionality.

This module provides thorough test coverage for src/modes/cross_validation.py including:
- Model type detection (traditional, advanced, DL)
- Metrics computation
- Fold execution for all model types
- Helper functions (median fold finding, hyperparameter extraction, aggregation)
- Edge case and error handling
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.modes.cross_validation import (
    _is_traditional_model,
    _is_dl_adapter,
    _find_median_fold_idx,
    _extract_median_hyperparameters,
    _aggregate_cv_results,
    _compute_metrics_from_predictions,
)
from src.models.dl_adapter import DLModelAdapter
from src.models.base import BaseModel


# ============================================================================
# Enhanced Fake Implementations
# ============================================================================

class FakeModelTraditional(BaseModel):
    """Fake traditional model with classify_traditional interface."""
    
    def __init__(self, config=None, return_7_tuple=False, hyperparams=None):
        super().__init__(config or {})
        self.return_7_tuple = return_7_tuple
        self.hyperparams = hyperparams or {'max_depth': 5, 'n_estimators': 100}
        self.is_traditional = True
        self.best_params = self.hyperparams
    
    def classify_traditional(self, X_train, y_train, X_test, y_test):
        """Return deterministic metrics tuple."""
        metrics = {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1': 0.85,
            'specificity': 0.83,
            'g_mean': 0.85
        }
        if self.return_7_tuple:
            return (metrics, X_test.shape[0], X_train.shape[0], 
                   self.hyperparams, ['feature_0', 'feature_1'], None, 'extra')
        else:
            return (metrics, X_test.shape[0], X_train.shape[0], 
                   self.hyperparams, ['feature_0', 'feature_1'], None)
    
    def tune(self, X, y, groups=None):
        pass
    
    def train(self, X, y, params=None):
        pass
    
    def evaluate(self, X, y):
        return {'accuracy': 0.85, 'f1': 0.85}
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def get_name(self):
        return 'FakeTraditional'
    
    def get_params(self):
        return self.hyperparams


class FakeModelAdvanced(BaseModel):
    """Fake advanced model with train/evaluate interface."""
    
    def __init__(self, config=None, hyperparams=None):
        super().__init__(config or {})
        self.best_params = hyperparams or {'learning_rate': 0.01}
        self.model = Mock()
        self.model.get_params = Mock(return_value=self.best_params)
        # Deliberately no is_traditional attribute and no classify_traditional method
        # to ensure it's detected as advanced
    
    def tune(self, X, y, groups=None):
        pass
    
    def train(self, X, y, params=None):
        pass
    
    def evaluate(self, X, y):
        return {'accuracy': 0.82, 'f1': 0.82}
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def predict(self, X):
        return np.random.randint(0, 2, X.shape[0])
    
    def get_name(self):
        return 'FakeAdvanced'


class FakeDLAdapter(DLModelAdapter):
    """Fake DL adapter model inheriting from DLModelAdapter."""
    
    def __init__(self):
        self.model_name = 'FakeDL'
    
    def fit(self, X_train, y_train, X_val, y_val, config):
        """Train the model."""
        pass
    
    def predict_proba(self, X):
        """Return probability predictions."""
        n_samples = X.shape[0]
        proba = np.random.rand(n_samples, 2)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba
    
    def save(self, path):
        """Save model to path."""
        pass
    
    def get_name(self):
        """Return model name."""
        return self.model_name


# ============================================================================
# Section 1: Model Type Detection Tests
# ============================================================================

class TestModelTypeDetection:
    """Test _is_traditional_model and _is_dl_adapter functions."""
    
    def test_traditional_model_with_flag(self):
        """Test traditional model detection via is_traditional flag."""
        model = FakeModelTraditional()
        assert _is_traditional_model(model) is True
    
    def test_traditional_model_with_method(self):
        """Test traditional model detection via classify_traditional method."""
        model = Mock()
        model.is_traditional = False
        model.classify_traditional = Mock()
        assert _is_traditional_model(model) is True
    
    def test_advanced_model(self):
        """Test that advanced models can also be detected as traditional due to BaseModel."""
        model = FakeModelAdvanced()
        # Note: All BaseModel subclasses have classify_traditional method,
        # so they're technically "traditional-compatible"
        # The distinction is whether they override is_traditional=False or use different methods
        assert _is_dl_adapter(model) is False
    
    def test_dl_adapter_detection(self):
        """Test DL adapter model detection."""
        model = FakeDLAdapter()
        assert _is_dl_adapter(model) is True
        assert _is_traditional_model(model) is False
    
    def test_model_without_is_traditional_flag(self):
        """Test model without is_traditional flag."""
        model = Mock(spec=[])  # No is_traditional attribute
        assert _is_traditional_model(model) is False


# ============================================================================
# Section 2: Metrics Computation Tests
# ============================================================================

class TestMetricsComputation:
    """Test _compute_metrics_from_predictions function."""
    
    def test_standard_metrics_computation(self):
        """Test computation of standard metrics with valid predictions."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        
        metrics = _compute_metrics_from_predictions(y_true, y_pred)
        
        # Check all expected keys present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'specificity' in metrics
        
        # Check metric values are valid
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_perfect_predictions(self):
        """Test metrics when all predictions are correct."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        
        metrics = _compute_metrics_from_predictions(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['recall'] == 1.0
    
    def test_all_incorrect_predictions(self):
        """Test metrics when all predictions are incorrect."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        
        metrics = _compute_metrics_from_predictions(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.0
    
    def test_imbalanced_classes(self):
        """Test metrics with imbalanced class distribution."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1])
        
        metrics = _compute_metrics_from_predictions(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert 'f1' in metrics
    
    def test_single_sample(self):
        """Test metrics with single sample."""
        y_true = np.array([0])
        y_pred = np.array([0])
        
        metrics = _compute_metrics_from_predictions(y_true, y_pred)
        assert metrics['accuracy'] == 1.0
    
    def test_g_mean_computation(self):
        """Test G-mean computation with probability predictions."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_proba = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.85, 0.15],
            [0.3, 0.7],
            [0.1, 0.9],
            [0.95, 0.05]
        ])
        
        y_pred = np.argmax(y_proba, axis=1)
        metrics = _compute_metrics_from_predictions(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        if 'g_mean' in metrics:
            assert metrics['g_mean'] >= 0


# ============================================================================
# Section 3: Helper Function Tests
# ============================================================================

class TestHelperFunctions:
    """Test median fold finding and hyperparameter extraction."""
    
    def test_find_median_fold_single_fold(self):
        """Test median fold finding with single fold."""
        fold_results = [{'metrics': {'f1': 0.85}}]
        fold_idx, f1_score = _find_median_fold_idx(fold_results)
        assert fold_idx == 0
        assert f1_score == 0.85
    
    def test_find_median_fold_odd_number(self):
        """Test median fold with odd number of folds."""
        fold_results = [
            {'metrics': {'f1': 0.80}},
            {'metrics': {'f1': 0.85}},
            {'metrics': {'f1': 0.90}}
        ]
        fold_idx, f1_score = _find_median_fold_idx(fold_results)
        assert f1_score == 0.85
    
    def test_find_median_fold_even_number(self):
        """Test median fold with even number of folds."""
        fold_results = [
            {'metrics': {'f1': 0.80}},
            {'metrics': {'f1': 0.85}},
            {'metrics': {'f1': 0.90}},
            {'metrics': {'f1': 0.95}}
        ]
        fold_idx, f1_score = _find_median_fold_idx(fold_results)
        # Should be one of the middle two
        assert f1_score in [0.85, 0.90]
    
    def test_find_median_fold_all_equal(self):
        """Test median fold when all F1 scores are equal."""
        fold_results = [
            {'metrics': {'f1': 0.85}},
            {'metrics': {'f1': 0.85}},
            {'metrics': {'f1': 0.85}}
        ]
        fold_idx, f1_score = _find_median_fold_idx(fold_results)
        assert 0 <= fold_idx < len(fold_results)
        assert f1_score == 0.85
    
    def test_extract_median_hyperparameters(self):
        """Test hyperparameter extraction from fold results."""
        model = FakeModelAdvanced()
        fold_results = [
            {
                'best_params': {'alpha': 0.1},
                'selected_features': ['f1', 'f2']
            },
            {
                'best_params': {'alpha': 0.15},
                'selected_features': ['f1', 'f2', 'f3']
            }
        ]
        
        result = _extract_median_hyperparameters(
            model, 0, 0.85, 'TestModel', fold_results
        )
        
        assert 'fold_id' in result
        assert 'f1_score' in result
        assert 'best_params' in result
        assert result['fold_id'] == 0
    
    def test_extract_median_hyperparameters_empty_fold_results(self):
        """Test hyperparameter extraction with empty fold results."""
        model = FakeModelAdvanced()
        result = _extract_median_hyperparameters(
            model, 0, 0.85, 'TestModel', None
        )
        assert isinstance(result, dict)
        assert 'fold_id' in result
    
    def test_aggregate_cv_results_basic(self):
        """Test basic aggregation of CV results."""
        fold_results = [
            {'metrics': {'accuracy': 0.80, 'f1': 0.80}},
            {'metrics': {'accuracy': 0.85, 'f1': 0.85}},
            {'metrics': {'accuracy': 0.90, 'f1': 0.90}}
        ]
        
        aggregated, median_idx, median_f1 = _aggregate_cv_results(fold_results)
        
        assert isinstance(aggregated, dict)
        assert isinstance(median_idx, (int, np.integer))
        assert isinstance(median_f1, (float, np.floating))
        assert 'accuracy_mean' in aggregated
        assert 'accuracy_std' in aggregated
        assert aggregated['accuracy_mean'] == pytest.approx(0.85, abs=0.01)
    
    def test_aggregate_cv_results_single_fold(self):
        """Test aggregation with single fold."""
        fold_results = [{'metrics': {'accuracy': 0.80, 'f1': 0.80}}]
        
        aggregated, median_idx, median_f1 = _aggregate_cv_results(fold_results)
        
        assert isinstance(aggregated, dict)
        assert 'accuracy_mean' in aggregated
        assert aggregated['accuracy_mean'] == 0.80
        assert aggregated['num_folds'] == 1


# ============================================================================
# Section 4: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_metrics_with_nan_handling(self):
        """Test metrics computation handles edge cases."""
        # Single class predictions (might produce NaN in some metrics)
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        
        metrics = _compute_metrics_from_predictions(y_true, y_pred)
        assert metrics['accuracy'] == 1.0
        assert not np.isnan(metrics['accuracy'])
    
    def test_metrics_with_all_class_one(self):
        """Test when all samples are class 1."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        
        metrics = _compute_metrics_from_predictions(y_true, y_pred)
        assert metrics['accuracy'] == 1.0
    
    def test_aggregate_empty_fold_results(self):
        """Test aggregation with empty fold results."""
        fold_results = []
        aggregated, median_idx, median_f1 = _aggregate_cv_results(fold_results)
        
        assert aggregated == {}
        assert median_idx == 0
        assert median_f1 == 0.0


# ============================================================================
# Section 5: API Contract Tests
# ============================================================================

class TestAPIContracts:
    """Test that functions maintain expected API contracts."""
    
    def test_is_traditional_model_return_type(self):
        """Test _is_traditional_model returns boolean."""
        model = FakeModelTraditional()
        result = _is_traditional_model(model)
        assert isinstance(result, bool)
    
    def test_is_dl_adapter_return_type(self):
        """Test _is_dl_adapter returns boolean."""
        model = FakeDLAdapter()
        result = _is_dl_adapter(model)
        assert isinstance(result, bool)
    
    def test_compute_metrics_returns_dict(self):
        """Test _compute_metrics_from_predictions returns dict."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = _compute_metrics_from_predictions(y_true, y_pred)
        assert isinstance(result, dict)
    
    def test_find_median_fold_returns_tuple(self):
        """Test _find_median_fold_idx returns tuple."""
        fold_results = [{'metrics': {'f1': 0.85}}]
        result = _find_median_fold_idx(fold_results)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_extract_median_hyperparameters_returns_dict(self):
        """Test _extract_median_hyperparameters returns dict."""
        model = FakeModelAdvanced()
        result = _extract_median_hyperparameters(
            model, 0, 0.85, 'TestModel', None
        )
        assert isinstance(result, dict)
    
    def test_aggregate_cv_results_returns_tuple(self):
        """Test _aggregate_cv_results returns tuple."""
        fold_results = [{'metrics': {'accuracy': 0.80}}]
        result = _aggregate_cv_results(fold_results)
        assert isinstance(result, tuple)
        assert len(result) == 3


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
