"""Tests for config parser cross-validation models."""

from src.GLOC_experiment_config_parser import GLOCExperimentConfigParser


def test_get_models_for_cross_validation_returns_mode_specific_models():
    """Test that get_models_for_cross_validation returns the mode-specific CV models."""
    config = GLOCExperimentConfigParser("test.yaml")
    
    cv_models = config.get_models_for_cross_validation()
    
    # Should return the mode-specific cross_validation.models
    cv_model_names = [m.get_name() for m in cv_models]
    
    assert len(cv_model_names) > 0, "Should have at least one model"
    # For test.yaml, CV has only KNN defined
    assert "KNN" in cv_model_names, "Should include KNN (specified in cross_validation.models)"


def test_get_cross_validation_models_returns_models():
    """Test that get_cross_validation_models returns the mode-specific models."""
    config = GLOCExperimentConfigParser("test.yaml")
    
    cv_models = config.get_cross_validation_models()
    
    cv_model_names = [m.get_name() for m in cv_models]
    
    assert len(cv_model_names) > 0, "Should have at least one model"
    assert "KNN" in cv_model_names, "Should include KNN"
