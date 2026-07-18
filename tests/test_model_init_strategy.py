"""
Tests for ModelInitStrategy enum and model initialization logic.

The semantic initialization strategy layer that the ModelInitStrategy enum was
introduced to support (including ``_initialize_model_for_classification``,
``_build_sklearn_estimator``, and ``classify_traditional``) was never landed on
``TraditionalModel`` or its subclasses. The previously-present tests for those
methods were written against that planned-but-unimplemented API and have been
removed. The enum itself (below) remains in ``src/models/base.py``.
"""

import pytest

from src.models.base import ModelInitStrategy


class TestModelInitStrategy:
    """Test ModelInitStrategy enum values and semantics."""

    def test_enum_values_exist(self):
        """Verify all expected strategy values are defined."""
        assert hasattr(ModelInitStrategy, "RETRAIN_WITH_DEFAULTS")
        assert hasattr(ModelInitStrategy, "RETRAIN_WITH_CONFIG_PARAMS")
        assert hasattr(ModelInitStrategy, "LOAD_SAVED_MODEL")

    def test_enum_values_are_unique(self):
        """Verify enum values are distinct."""
        values = [s.value for s in ModelInitStrategy]
        assert len(values) == len(set(values))

    def test_enum_string_representation(self):
        """Verify enum string values are clear and descriptive."""
        assert ModelInitStrategy.RETRAIN_WITH_DEFAULTS.value == "retrain_with_defaults"
        assert ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS.value == "retrain_with_config_params"
        assert ModelInitStrategy.LOAD_SAVED_MODEL.value == "load_saved_model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
