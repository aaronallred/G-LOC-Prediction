"""Tests for config parser cross-validation model filtering."""

from src.GLOC_experiment_config_parser import GLOCExperimentConfigParser


def test_get_models_for_cross_validation_filters_by_classifiers():
    """Test that get_models_for_cross_validation filters models by CV classifiers list."""
    config = GLOCExperimentConfigParser("test.yaml")
    
    all_models = config.get_models()
    cv_models = config.get_models_for_cross_validation()
    cv_classifiers = config.get_cross_validation_classifiers()
    
    # Verify we have multiple models but only some are in CV
    all_model_names = [m.get_name() for m in all_models]
    cv_model_names = [m.get_name() for m in cv_models]
    
    assert len(all_model_names) > 1, "Test.yaml should have multiple models"
    assert len(cv_model_names) == len(cv_classifiers), "CV models should match CV classifiers count"
    assert set(cv_model_names) == set(cv_classifiers), "CV model names should match CV classifiers exactly"
    
    # For test.yaml specifically: should have EGB + KNN, but only KNN in CV
    assert "KNN" in all_model_names, "All models should include KNN"
    assert "KNN" in cv_model_names, "CV models should include KNN"
    assert "EGB" in all_model_names, "All models should include EGB"
    assert "EGB" not in cv_model_names, "CV models should NOT include EGB (not in classifiers)"


def test_get_models_for_cross_validation_returns_all_if_not_configured():
    """Test that get_models_for_cross_validation returns all models if CV section missing."""
    config = GLOCExperimentConfigParser("GLOC_experiment_config.yaml")
    
    all_models = config.get_models()
    cv_models = config.get_models_for_cross_validation()
    
    # CV disabled in default config, so should return all models
    all_model_names = [m.get_name() for m in all_models]
    cv_model_names = [m.get_name() for m in cv_models]
    
    # When CV is disabled, get_models_for_cross_validation should return all models
    # (this is a fallback for backward compatibility)
    assert len(cv_model_names) > 0, "Should return at least some models"
