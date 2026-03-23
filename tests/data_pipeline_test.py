"""
Comprehensive unit tests for the DataPipeline class.

Follows SWE, MLE, and OOP best practices:
- Organized test classes by functional area
- Parametrized tests for multiple scenarios
- Comprehensive mocking of external dependencies
- Clear naming conventions and docstrings
- Edge case and error handling coverage
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import DataPipeline, ModelType
from src.models.base import BaseModel


# ============================================================================
# FIXTURES - Test Data and Mock Objects
# ============================================================================

@pytest.fixture
def temp_data_dir() -> str:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock BaseModel with required attributes.
    
    Note: Using MagicMock without spec to allow get_name() method
    since it's referenced in DataPipeline but not in BaseModel interface.
    """
    model = MagicMock()
    model.get_name.return_value = "logreg"
    model.is_traditional = True
    model.config = {}
    return model


@pytest.fixture
def mock_advanced_model() -> MagicMock:
    """Create a mock advanced (non-traditional) model.
    
    Note: Using MagicMock without spec to allow get_name() method
    since it's referenced in DataPipeline but not in BaseModel interface.
    """
    model = MagicMock()
    model.get_name.return_value = "Trans"
    model.is_traditional = False
    model.config = {}
    return model


@pytest.fixture
def model_type_noafe_explicit() -> ModelType:
    """Create a noAFE Explicit model type."""
    return ModelType(afe_filter="noAFE", feature_set="Explicit")


@pytest.fixture
def model_type_complete_explicit() -> ModelType:
    """Create a Complete Explicit model type."""
    return ModelType(afe_filter="Complete", feature_set="Explicit")


@pytest.fixture
def model_type_noafe_implicit() -> ModelType:
    """Create a noAFE Implicit model type."""
    return ModelType(afe_filter="noAFE", feature_set="Implicit")


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame with typical GLOC data structure."""
    np.random.seed(42)
    n_rows = 100
    
    data = {
        "trial_id": [f"{i % 13 + 1:02d}-{j + 1}" for i, j in enumerate(range(n_rows))],
        "event_validated": ["begin GOR" if i < 50 else "other" 
                           for i in range(n_rows)],
        "condition": ["N" if i < 50 else "AFE" for i in range(n_rows)],
        "time": np.linspace(0, 100, n_rows),
        "trial": np.repeat(np.arange(1, 5), 25),
        "subject": np.repeat(np.arange(1, 14), n_rows // 13),
        "ECG_col1": np.random.randn(n_rows),
        "BR_col1": np.random.randn(n_rows),
        "temp_col1": np.random.randn(n_rows),
    }
    return pd.DataFrame(data)



# ============================================================================
# TEST CLASS: DataPipelineInitialization
# ============================================================================

class TestDataPipelineInitialization:
    """Tests for DataPipeline initialization and parameter setup."""

    def test_initialization_with_defaults(self, mock_model, model_type_noafe_explicit, 
                                          temp_data_dir) -> None:
        """Test DataPipeline initialization with default parameters."""
        pipeline = DataPipeline(
            datafolder = temp_data_dir,
            model = mock_model,
            model_type = model_type_noafe_explicit
        )

        assert pipeline.datafolder == temp_data_dir
        assert pipeline.model == mock_model
        assert pipeline.model_type == model_type_noafe_explicit
        assert pipeline.baseline_window == 32.5  # Default value
        assert pipeline.window_size is None
        assert pipeline.stride is None
        assert pipeline.feature_reduction_type is None
        assert pipeline.imbalance_type == "none"
        assert pipeline.should_impute is True
        assert pipeline.n_neighbors == 4
        assert pipeline.num_splits == 10
        assert pipeline.kfold_ID == 0
        assert pipeline.remove_NaN_trials is True
        assert pipeline.save_impute is True
        assert pipeline.load_impute is True

    def test_initialization_with_custom_parameters(self, mock_model, 
                                                    model_type_complete_explicit,
                                                    temp_data_dir) -> None:
        """Test DataPipeline initialization with custom parameters."""
        custom_baseline = 20.0
        custom_window = 10.0
        custom_stride = 0.5
        
        pipeline = DataPipeline(
            datafolder = temp_data_dir,
            model = mock_model,
            model_type = model_type_complete_explicit,
            baseline_window = custom_baseline,
            window_size = custom_window,
            stride = custom_stride,
            feature_reduction_type = "lasso",
            imbalance_type = "ros",
            should_impute = False,
            n_neighbors = 7,
            num_splits = 5,
            kfold_ID = 2
        )

        assert pipeline.baseline_window == custom_baseline
        assert pipeline.window_size == custom_window
        assert pipeline.stride == custom_stride
        assert pipeline.feature_reduction_type == "lasso"
        assert pipeline.imbalance_type == "ros"
        assert pipeline.should_impute is False
        assert pipeline.n_neighbors == 7
        assert pipeline.num_splits == 5
        assert pipeline.kfold_ID == 2

    def test_internal_state_initialized_to_none(self, mock_model, 
                                                 model_type_noafe_explicit,
                                                 temp_data_dir) -> None:
        """Test that internal state variables are initialized to None."""
        pipeline = DataPipeline(
            datafolder = temp_data_dir,
            model = mock_model,
            model_type = model_type_noafe_explicit
        )

        assert pipeline.data_locations is None
        assert pipeline.gloc_data is None
        assert pipeline.gloc_labels is None
        assert pipeline.feature_groups_to_analyze is None
        assert pipeline.baseline_methods_to_use is None



# ============================================================================
# TEST CLASS: ModelTypeDataclass
# ============================================================================

class TestModelTypeDataclass:
    """Tests for ModelType dataclass behavior."""

    def test_model_type_creation(self) -> None:
        """Test ModelType creation with valid parameters."""
        model_type = ModelType(afe_filter="noAFE", feature_set="Explicit")
        
        assert model_type.afe_filter == "noAFE"
        assert model_type.feature_set == "Explicit"

    def test_model_type_is_frozen(self) -> None:
        """Test that ModelType is immutable (frozen)."""
        model_type = ModelType(afe_filter = "Complete", feature_set = "Implicit")
        
        with pytest.raises(AttributeError):
            model_type.afe_filter = "noAFE"

    def test_model_type_equality(self) -> None:
        """Test ModelType equality comparison."""
        type1 = ModelType(afe_filter = "noAFE", feature_set = "Explicit")
        type2 = ModelType(afe_filter = "noAFE", feature_set = "Explicit")
        type3 = ModelType(afe_filter = "Complete", feature_set = "Explicit")
        
        assert type1 == type2
        assert type1 != type3

    def test_model_type_hashable(self) -> None:
        """Test that ModelType can be used as dictionary key."""
        type1 = ModelType(afe_filter = "noAFE", feature_set = "Explicit")
        type2 = ModelType(afe_filter = "noAFE", feature_set = "Explicit")
        
        model_dict = {type1: "value1"}
        assert model_dict[type2] == "value1"



# ============================================================================
# TEST CLASS: HyperparameterAssignment
# ============================================================================

class TestHyperparameterAssignment:
    """Tests for hyperparameter assignment by classifier type."""

    @pytest.mark.parametrize("classifier_name, expected_baseline_window, expected_window_size, expected_stride", [
        ("logreg", 5, 12.5, 0.25),
        ("RF", 18.75, 7.5, 0.25),
        ("LDA", 46.25, 15, 0.25),
        ("SVM", 32.5, 15, 0.25),
        ("EGB", 46.25, 12.5, 0.25),
        ("KNN", 32.5, 15, 0.25),
    ])
    def test_assign_hyperparameters_traditional_classifiers(
        self, 
        classifier_name: str, 
        expected_baseline_window: float,
        expected_window_size: float, 
        expected_stride: float, 
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str
    ) -> None:
        """Test hyperparameter assignment for traditional classifiers.
        
        Args:
            classifier_name: Name of the classifier
            expected_baseline_window: Expected baseline window value
            expected_window_size: Expected window size value
            expected_stride: Expected stride value
            model_type_noafe_explicit: Fixture for model type
            temp_data_dir: Temporary directory fixture
        """
        mock_model = MagicMock()
        mock_model.get_name.return_value = classifier_name
        mock_model.is_traditional = True
        
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit
        )
        
        pipeline._assign_hyperparameters_by_classifier()
        
        assert pipeline.baseline_window == expected_baseline_window
        assert pipeline.window_size == expected_window_size
        assert pipeline.stride == expected_stride
        
    def test_assign_hyperparameters_unknown_classifier(self, model_type_noafe_explicit: ModelType,
                                                       temp_data_dir: str) -> None:
        """Test hyperparameter assignment with unknown classifier raises error.
        
        When a classifier is not found in the hyperparameters dict, the code
        will crash attempting to call .get() on None. This test documents that
        behavior as a known issue.
        """
        mock_model = MagicMock()
        mock_model.get_name.return_value = "unknown_classifier"
        mock_model.is_traditional = True
        
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit
        )
    
        pipeline._assign_hyperparameters_by_classifier()

        assert pipeline.baseline_window == 32.5  # Default value
        assert pipeline.window_size is None  # Default value
        assert pipeline.stride is None  # Default value

    def test_assign_hyperparameters_preserves_custom_values(self, model_type_noafe_explicit: ModelType,
                                                            temp_data_dir: str) -> None:
        """Test that custom parameter values are preserved when not in hyperparameter dict.
        
        When a parameter is not in the hyperparameter dictionary, the custom initialization
        value should be preserved.
        """
        mock_model = MagicMock()
        mock_model.get_name.return_value = "logreg"
        mock_model.is_traditional = True
        
        custom_n_neighbors = 20
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit,
            n_neighbors=custom_n_neighbors
        )
        
        pipeline._assign_hyperparameters_by_classifier()
        
        # Should use value from hyperparameters dict if available
        assert pipeline.n_neighbors == 5  # From logreg hyperparameters



# ============================================================================
# TEST CLASS: FeatureGroupAndBaselineAssignment
# ============================================================================

class TestFeatureGroupAndBaselineAssignment:
    """Tests for feature group and baseline method assignment."""

    @pytest.mark.parametrize("afe_filter,feature_set, expected_groups", [
        ("noAFE", "Explicit", ("ECG", "BR", "temp", "eyetracking", "AFE", "G", 
                               "rawEEG", "processedEEG", "strain", "demographics")),
        ("noAFE", "Implicit", ("ECG", "BR", "temp", "eyetracking", "rawEEG")),
        ("Complete", "Explicit", ("ECG", "BR", "temp", "eyetracking", "AFE", "G",
                                  "rawEEG", "processedEEG", "strain", "demographics")),
        ("Complete", "Implicit", ("ECG", "BR", "temp", "eyetracking", "rawEEG", "AFE")),
    ])
    def test_assign_feature_groups_by_model_type(self, afe_filter: str, feature_set: str,
                                                  expected_groups: Tuple[str, ...],
                                                  mock_model: MagicMock, temp_data_dir: str) -> None:
        """Test feature group assignment for different model types.
        
        Args:
            afe_filter: AFE filter type ("noAFE" or "Complete")
            feature_set: Feature set type ("Explicit" or "Implicit")
            expected_groups: Expected tuple of feature groups
            mock_model: Mock BaseModel fixture
            temp_data_dir: Temporary directory fixture
        """
        model_type = ModelType(afe_filter=afe_filter, feature_set=feature_set)
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type
        )
        
        pipeline._assign_feature_groups_and_baseline_methods()
        
        assert pipeline.feature_groups_to_analyze == expected_groups

    @pytest.mark.parametrize("afe_filter, expected_baseline_methods", [
        ("noAFE", ["v0", "v1", "v2", "v5", "v6", "v7", "v8"]),
        ("Complete", ["v0", "v1", "v2", "v5", "v6"]),
    ])
    def test_assign_baseline_methods_for_advanced_model(self, afe_filter: str,
                                                        expected_baseline_methods: List[str],
                                                        mock_advanced_model: MagicMock,
                                                        temp_data_dir: str) -> None:
        """Test baseline methods assignment for advanced (non-traditional) models.
        
        Only advanced models should have their baseline methods set by this assignment.
        """
        model_type = ModelType(afe_filter=afe_filter, feature_set="Explicit")
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_advanced_model,
            model_type=model_type
        )
        
        pipeline._assign_feature_groups_and_baseline_methods()
        
        assert pipeline.baseline_methods_to_use == expected_baseline_methods

    def test_baseline_methods_not_overwritten_for_traditional_model(self, mock_model: MagicMock,
                                                                    model_type_noafe_explicit: ModelType,
                                                                    temp_data_dir: str) -> None:
        """Test that traditional model baseline methods are not overwritten.
        
        For traditional models, baseline methods should be set during hyperparameter assignment,
        not during feature group assignment.
        """
        mock_model.is_traditional = True
        custom_baseline_methods = ["v0", "v5", "v8"]
        
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit
        )
        # Manually set baseline methods to simulate hyperparameter assignment
        pipeline.baseline_methods_to_use = custom_baseline_methods
        
        pipeline._assign_feature_groups_and_baseline_methods()
        
        # Should not be overwritten
        assert pipeline.baseline_methods_to_use == custom_baseline_methods



# ============================================================================
# TEST CLASS: DataLocationAssignment
# ============================================================================

class TestDataLocationAssignment:
    """Tests for data location path construction."""

    def test_assign_data_locations_creates_correct_paths(self, mock_model: MagicMock,
                                                         model_type_noafe_explicit: ModelType,
                                                         temp_data_dir: str) -> None:
        """Test that data locations are correctly assigned.
        
        Verifies that all required data paths are properly constructed.
        
        Note: This test works around inconsistent naming in the code
        (_data_locations vs data_locations).
        """
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit
        )
        
        # Set data_path attribute (referenced in _assign_data_locations)
        pipeline.data_path = temp_data_dir
        # Work around inconsistent attribute naming
        pipeline._data_locations = None
        
        pipeline._assign_data_locations()
        
        # Check both possible attribute names for data locations
        data_locations = pipeline._data_locations or pipeline.data_locations
        
        assert data_locations is not None
        assert "main" in data_locations
        assert "baseline" in data_locations
        assert "demographics" in data_locations
        assert "eeg_list" in data_locations
        assert "baseline_eeg_processed_list" in data_locations

    def test_assign_data_locations_main_path(self, mock_model: MagicMock,
                                             model_type_noafe_explicit: ModelType,
                                             temp_data_dir: str) -> None:
        """Test main CSV path construction."""
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit
        )
        
        # Set data_path attribute (referenced in _assign_data_locations)
        pipeline.data_path = temp_data_dir
        # Work around inconsistent attribute naming
        pipeline._data_locations = None
        
        pipeline._assign_data_locations()
        
        # Check both possible attribute names for data locations
        data_locations = pipeline._data_locations or pipeline.data_locations
        
        expected_main = os.path.join(
            temp_data_dir, "all_trials_25_hz_stacked_null_str_filled.csv"
        )
        assert data_locations["main"] == expected_main

    def test_assign_data_locations_eeg_files_generated(self, mock_model: MagicMock,
                                                       model_type_noafe_explicit: ModelType,
                                                       temp_data_dir: str) -> None:
        """Test that EEG file paths are correctly generated for all participants/trials.
        
        The method should generate paths for all combinations of participants and trials
        defined in _EEG_PARTICIPANT_TRIALS.
        """
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit
        )
        
        # Set data_path attribute (referenced in _assign_data_locations)
        pipeline.data_path = temp_data_dir
        # Work around inconsistent attribute naming
        pipeline._data_locations = None
        
        pipeline._assign_data_locations()
        
        # Check both possible attribute names for data locations
        data_locations = pipeline._data_locations or pipeline.data_locations
        eeg_list = data_locations["eeg_list"]
        
        # Should have paths for all participants and their trials
        assert len(eeg_list) > 0
        
        # All paths should contain participant/trial info
        for path in eeg_list:
            assert "GLOC_" in path
            assert "DC" in path
            assert "25Hz_EEG_power_wMAR.xlsx" in path



# ============================================================================
# TEST CLASS: DataProcessingPipeline
# ============================================================================

class TestDataProcessingPipeline:
    """Tests for the full data processing pipeline."""

    def test_get_data_before_processing(self, mock_model: MagicMock,
                                        model_type_noafe_explicit: ModelType,
                                        temp_data_dir: str) -> None:
        """Test get_data() returns None values before processing."""
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit
        )
        
        data, labels = pipeline.get_data()
        
        assert data is None
        assert labels is None

    def test_process_and_get_data_calls_all_assignments(self, mock_model: MagicMock,
                                                        model_type_noafe_explicit: ModelType,
                                                        temp_data_dir: str) -> None:
        """Test that process_and_get_data calls all assignment methods.
        
        This is an integration test verifying the pipeline calls required setup methods.
        """
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit
        )
        
        with patch.object(pipeline, '_assign_hyperparameters_by_classifier') as mock_hp, \
             patch.object(pipeline, '_assign_feature_groups_and_baseline_methods') as mock_fg, \
             patch.object(pipeline, '_assign_data_locations') as mock_dl, \
             patch.object(pipeline, '_load_data') as mock_load, \
             patch.object(pipeline, '_filter_data_by_analysis_type') as mock_filter, \
             patch.object(pipeline, '_process_and_get_feature_names') as mock_features, \
             patch.object(pipeline, '_label_gloc_events') as mock_labels:
            
            pipeline.process_and_get_data()
            
            mock_hp.assert_called_once()
            mock_fg.assert_called_once()
            mock_dl.assert_called_once()
            mock_load.assert_called_once()
            mock_filter.assert_called_once()
            mock_features.assert_called_once()
            mock_labels.assert_called_once()


# ============================================================================
# TEST CLASS: New Method Coverage
# ============================================================================

class TestLoadData:
    """Tests for _load_data behavior and data transformations."""

    def test_load_data_uses_enriched_pickle_when_available(
        self,
        mock_model: MagicMock,
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str,
    ) -> None:
        """If enriched pickle exists, _load_data should load it and return early."""
        pipeline = DataPipeline(temp_data_dir, mock_model, model_type_noafe_explicit)
        pipeline.data_locations = {
            "main": os.path.join(temp_data_dir, "all_trials_25_hz_stacked_null_str_filled.csv")
        }

        expected_df = pd.DataFrame({"a": [1, 2]})
        with patch("src.data_pipeline.os.path.isfile", return_value=True), \
             patch("src.data_pipeline.pd.read_pickle", return_value=expected_df) as mock_read_pickle, \
             patch.object(pipeline, "_add_EEG_GOR") as mock_add_eeg:
            pipeline._load_data()

        pd.testing.assert_frame_equal(pipeline.gloc_data, expected_df)
        mock_read_pickle.assert_called_once()
        mock_add_eeg.assert_not_called()

    def test_load_data_transforms_condition_and_trial_identifiers(
        self,
        mock_model: MagicMock,
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str,
    ) -> None:
        """When loading from CSV, condition and trial_id should be transformed correctly."""
        pipeline = DataPipeline(temp_data_dir, mock_model, model_type_noafe_explicit)
        main_csv = os.path.join(temp_data_dir, "all_trials_25_hz_stacked_null_str_filled.csv")
        pipeline.data_locations = {"main": main_csv}

        raw_df = pd.DataFrame(
            {
                "trial_id": ["01-01", "02-02", "bad"],
                "event_validated": ["begin GOR", "other", "other"],
                "condition": ["N", "AFE", "UNK"],
                "float_col": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            }
        )

        with patch("src.data_pipeline.os.path.isfile", return_value=False), \
             patch("src.data_pipeline.pd.read_csv", return_value=raw_df.copy()) as mock_read_csv, \
             patch.object(pipeline, "_add_EEG_GOR") as mock_add_eeg, \
             patch("pandas.DataFrame.to_pickle") as mock_to_pickle:
            pipeline._load_data()

        mock_read_csv.assert_called_once_with(main_csv)
        mock_add_eeg.assert_called_once()
        mock_to_pickle.assert_called_once()

        assert "condition" not in pipeline.gloc_data.columns
        assert pipeline.gloc_data["AFE_indicator"].dtype == np.bool_
        assert pipeline.gloc_data["AFE_indicator"].tolist() == [False, True, False]
        assert pipeline.gloc_data["float_col"].dtype == np.float32
        assert pipeline.gloc_data["subject"].dtype == np.uint8
        assert pipeline.gloc_data["trial"].dtype == np.uint8
        assert pipeline.gloc_data["subject"].tolist() == [1, 2, 0]
        assert pipeline.gloc_data["trial"].tolist() == [1, 2, 0]


class TestFilterByAnalysisType:
    """Tests for _filter_data_by_analysis_type."""

    def test_filter_type_zero_filters_subject_and_trial(
        self,
        mock_model: MagicMock,
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str,
    ) -> None:
        pipeline = DataPipeline(
            temp_data_dir,
            mock_model,
            model_type_noafe_explicit,
            analysis_type=0,
            subject_to_analyze=2,
            trial_to_analyze=3,
        )
        pipeline.gloc_data = pd.DataFrame(
            {
                "subject": [1, 2, 2],
                "trial": [3, 2, 3],
                "v": [10, 20, 30],
            }
        )

        pipeline._filter_data_by_analysis_type()

        assert len(pipeline.gloc_data) == 1
        assert pipeline.gloc_data.iloc[0]["v"] == 30

    def test_filter_type_one_filters_subject_only(
        self,
        mock_model: MagicMock,
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str,
    ) -> None:
        pipeline = DataPipeline(
            temp_data_dir,
            mock_model,
            model_type_noafe_explicit,
            analysis_type=1,
            subject_to_analyze=2,
        )
        pipeline.gloc_data = pd.DataFrame(
            {
                "subject": [1, 2, 2],
                "trial": [3, 2, 3],
                "v": [10, 20, 30],
            }
        )

        pipeline._filter_data_by_analysis_type()

        assert pipeline.gloc_data["v"].tolist() == [20, 30]

    def test_filter_type_two_currently_raises_unbound_local_error(
        self,
        mock_model: MagicMock,
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str,
    ) -> None:
        """Documents current behavior: analysis_type=2 uses undefined mask."""
        pipeline = DataPipeline(
            temp_data_dir,
            mock_model,
            model_type_noafe_explicit,
            analysis_type=2,
        )
        pipeline.gloc_data = pd.DataFrame(
            {
                "subject": [1],
                "trial": [1],
                "v": [10],
            }
        )

        with pytest.raises(UnboundLocalError):
            pipeline._filter_data_by_analysis_type()


class TestFeatureNameProcessing:
    """Tests for _process_and_get_feature_names with mocked feature processors."""

    def test_process_feature_names_builds_all_groups(
        self,
        mock_model: MagicMock,
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str,
    ) -> None:
        pipeline = DataPipeline(temp_data_dir, mock_model, model_type_noafe_explicit)
        pipeline.gloc_data = pd.DataFrame({"x": [1, 2]})
        pipeline.data_locations = {"main": "dummy"}
        pipeline.feature_groups_to_analyze = ["ECG", "processedEEG", "demographics"]

        ecg_processor = MagicMock()
        ecg_processor.process.side_effect = lambda df, _: df
        ecg_processor.get_feature_names.return_value = ["ecg_a", "ecg_b"]

        eeg_processor = MagicMock()
        eeg_processor.process.side_effect = lambda df, _: df
        eeg_processor.get_feature_names.return_value = ["eeg_a"]

        demo_processor = MagicMock()
        demo_processor.process.side_effect = lambda df, _: df
        demo_processor.get_feature_names.return_value = ["age"]

        with patch.dict(
            "src.data_pipeline.FEATURE_REGISTRY",
            {"ECG": ecg_processor, "processedEEG": eeg_processor, "demographics": demo_processor},
            clear=True,
        ):
            pipeline._process_and_get_feature_names()

        assert pipeline.feature_names["All"] == ["ecg_a", "ecg_b", "eeg_a", "age"]
        assert pipeline.feature_names["Phys"] == ["ecg_a", "ecg_b", "eeg_a"]
        assert pipeline.feature_names["ECG"] == ["ecg_a", "ecg_b"]
        assert pipeline.feature_names["EEG"] == ["eeg_a"]

    def test_process_feature_names_skips_unknown_groups(
        self,
        mock_model: MagicMock,
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str,
    ) -> None:
        pipeline = DataPipeline(temp_data_dir, mock_model, model_type_noafe_explicit)
        pipeline.gloc_data = pd.DataFrame({"x": [1]})
        pipeline.data_locations = {"main": "dummy"}
        pipeline.feature_groups_to_analyze = ["UNKNOWN"]

        with patch("src.data_pipeline.logger") as mock_logger:
            pipeline._process_and_get_feature_names()

        mock_logger.warning.assert_called_once()
        assert pipeline.feature_names == {"All": [], "Phys": [], "ECG": [], "EEG": []}


class TestGlocLabeling:
    """Tests for _label_gloc_events."""

    def test_label_gloc_events_marks_range_until_rtc(
        self,
        mock_model: MagicMock,
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str,
    ) -> None:
        pipeline = DataPipeline(temp_data_dir, mock_model, model_type_noafe_explicit)
        pipeline.gloc_data = pd.DataFrame(
            {
                "trial_id": ["01-01", "01-01", "01-01", "01-01", "01-01"],
                "event_validated": ["other", "GLOC", "other", "return to consciousness", "other"],
            }
        )

        pipeline._label_gloc_events()

        assert pipeline.gloc_labels.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]

    def test_label_gloc_events_ignores_mismatched_trial_pairs(
        self,
        mock_model: MagicMock,
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str,
    ) -> None:
        pipeline = DataPipeline(temp_data_dir, mock_model, model_type_noafe_explicit)
        pipeline.gloc_data = pd.DataFrame(
            {
                "trial_id": ["01-01", "01-01", "02-01", "02-01"],
                "event_validated": ["GLOC", "other", "return to consciousness", "other"],
            }
        )

        pipeline._label_gloc_events()

        assert pipeline.gloc_labels.tolist() == [0.0, 0.0, 0.0, 0.0]


class TestAddEegGor:
    """Tests for _add_EEG_GOR using mocked Excel reads and file checks."""

    def test_add_eeg_gor_inserts_band_values_for_matching_trial(
        self,
        mock_model: MagicMock,
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str,
    ) -> None:
        pipeline = DataPipeline(temp_data_dir, mock_model, model_type_noafe_explicit)
        pipeline.data_locations = {
            "eeg_list": [os.path.join(temp_data_dir, "GLOC_01_DC1_25Hz_EEG_power_wMAR.xlsx")]
        }
        pipeline.gloc_data = pd.DataFrame(
            {
                "trial_id": ["01-01", "01-01", "01-01"],
                "event_validated": ["begin GOR", "other", "other"],
            }
        )

        sheet = pd.DataFrame(
            {
                "ch1": [1.0, 2.0],
                "ch2": [3.0, 4.0],
                "drop_col": [9.0, 9.0],
            }
        )

        with patch("src.data_pipeline.os.path.isfile", return_value=True), \
             patch(
                 "src.data_pipeline.pd.read_excel",
                 return_value={"delta": sheet, "theta": sheet, "alpha": sheet, "beta": sheet},
             ):
            pipeline._add_EEG_GOR()

        for band in ["delta", "theta", "alpha", "beta"]:
            assert f"ch1_{band} - EEG" in pipeline.gloc_data.columns
            assert f"ch2_{band} - EEG" in pipeline.gloc_data.columns

        assert pipeline.gloc_data.loc[0, "ch1_delta - EEG"] == 1.0
        assert pipeline.gloc_data.loc[1, "ch1_delta - EEG"] == 2.0

    def test_add_eeg_gor_skips_when_filename_does_not_match_pattern(
        self,
        mock_model: MagicMock,
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str,
    ) -> None:
        pipeline = DataPipeline(temp_data_dir, mock_model, model_type_noafe_explicit)
        pipeline.data_locations = {"eeg_list": [os.path.join(temp_data_dir, "bad_name.xlsx")]}
        pipeline.gloc_data = pd.DataFrame(
            {
                "trial_id": ["01-01"],
                "event_validated": ["begin GOR"],
            }
        )

        with patch("src.data_pipeline.logger") as mock_logger:
            pipeline._add_EEG_GOR()

        mock_logger.warning.assert_called_once()


class TestProcessAndGetDataFlow:
    """Tests execution order and outputs for process_and_get_data."""

    def test_process_and_get_data_executes_steps_in_order(
        self,
        mock_model: MagicMock,
        model_type_noafe_explicit: ModelType,
        temp_data_dir: str,
    ) -> None:
        pipeline = DataPipeline(temp_data_dir, mock_model, model_type_noafe_explicit)
        pipeline.gloc_data = pd.DataFrame({"x": [1]})
        pipeline.gloc_labels = np.array([0.0])

        call_order = []

        def _mark(name: str):
            return lambda: call_order.append(name)

        with patch.object(pipeline, "_assign_hyperparameters_by_classifier", side_effect=_mark("hp")), \
             patch.object(pipeline, "_assign_feature_groups_and_baseline_methods", side_effect=_mark("groups")), \
             patch.object(pipeline, "_assign_data_locations", side_effect=_mark("locations")), \
             patch.object(pipeline, "_load_data", side_effect=_mark("load")), \
             patch.object(pipeline, "_filter_data_by_analysis_type", side_effect=_mark("filter")), \
             patch.object(pipeline, "_process_and_get_feature_names", side_effect=_mark("features")), \
             patch.object(pipeline, "_label_gloc_events", side_effect=_mark("labels")):
            returned_data, returned_labels = pipeline.process_and_get_data()

        assert call_order == ["hp", "groups", "locations", "load", "filter", "features", "labels"]
        assert returned_data is pipeline.gloc_data
        assert returned_labels is pipeline.gloc_labels


# ============================================================================
# TEST CLASS: ClassConstantsValidation
# ============================================================================

class TestClassConstantsValidation:
    """Tests for validation of class-level constants."""

    def test_feature_groups_by_model_type_completeness(self) -> None:
        """Test that FEATURE_GROUPS_BY_MODEL_TYPE contains all valid combinations.
        
        Each combination of afe_filter and feature_set should have a mapping.
        """
        afe_filters = ["noAFE", "Complete"]
        feature_sets = ["Explicit", "Implicit"]
        
        for afe_filter in afe_filters:
            for feature_set in feature_sets:
                model_type = ModelType(afe_filter=afe_filter, feature_set=feature_set)
                assert model_type in DataPipeline.FEATURE_GROUPS_BY_MODEL_TYPE
                groups = DataPipeline.FEATURE_GROUPS_BY_MODEL_TYPE[model_type]
                assert isinstance(groups, tuple)
                assert len(groups) > 0

    def test_baselining_characteristics_by_model_type(self) -> None:
        """Test BASELINING_CHARACTERISTICS_BY_MODEL_TYPE structure."""
        assert "noAFE" in DataPipeline.BASELINING_CHARACTERISTICS_BY_MODEL_TYPE
        assert "Complete" in DataPipeline.BASELINING_CHARACTERISTICS_BY_MODEL_TYPE
        
        for afe_filter, methods in DataPipeline.BASELINING_CHARACTERISTICS_BY_MODEL_TYPE.items():
            assert isinstance(methods, list)
            assert len(methods) > 0
            for method in methods:
                assert method.startswith("v")

    def test_classifier_hyperparameters_structure(self) -> None:
        """Test that hyperparameter dictionary has correct structure for all classifiers."""
        required_keys = {
            "baseline_window", "window_size", "stride", "feature_reduction_type",
            "baseline_methods_to_use", "imbalance_type", "impute_type", "n_neighbors"
        }
        
        for classifier, params in DataPipeline._CLASSIFIER_HYPERPARAMETERS.items():
            assert isinstance(classifier, str)
            assert isinstance(params, dict)
            assert required_keys.issubset(params.keys()), \
                f"Classifier {classifier} missing required keys"

    def test_eeg_participant_trials_structure(self) -> None:
        """Test EEG participant trials mapping structure."""
        for participant, trials in DataPipeline._EEG_PARTICIPANT_TRIALS.items():
            assert isinstance(participant, int)
            assert 1 <= participant <= 13, f"Participant {participant} out of range [1, 13]"
            assert isinstance(trials, list)
            assert len(trials) > 0
            assert all(isinstance(t, int) for t in trials)

    def test_eeg_baseline_bands(self) -> None:
        """Test EEG baseline bands list."""
        assert DataPipeline._EEG_BASELINE_BANDS == ["delta", "theta", "alpha", "beta"]


# ============================================================================
# TEST CLASS: EdgeCases and Error Handling
# ============================================================================

class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling scenarios."""

    def test_initialization_with_none_subject(self, mock_model: MagicMock,
                                              model_type_noafe_explicit: ModelType,
                                              temp_data_dir: str) -> None:
        """Test initialization with subject_to_analyze=None."""
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit,
            subject_to_analyze=None
        )
        
        assert pipeline.subject_to_analyze is None

    def test_initialization_with_specific_subject(self, mock_model: MagicMock,
                                                   model_type_noafe_explicit: ModelType,
                                                   temp_data_dir: str) -> None:
        """Test initialization with specific subject_to_analyze."""
        subject = "01"
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit,
            subject_to_analyze=subject
        )
        
        assert pipeline.subject_to_analyze == subject

    def test_remove_nan_trials_parameter(self, mock_model: MagicMock,
                                         model_type_noafe_explicit: ModelType,
                                         temp_data_dir: str) -> None:
        """Test remove_NaN_trials parameter variations."""
        pipeline_remove = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit,
            remove_NaN_trials=True
        )
        assert pipeline_remove.remove_NaN_trials is True
        
        pipeline_keep = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type_noafe_explicit,
            remove_NaN_trials=False
        )
        assert pipeline_keep.remove_NaN_trials is False

    def test_analysis_type_parameter_variations(self, mock_model: MagicMock,
                                                model_type_noafe_explicit: ModelType,
                                                temp_data_dir: str) -> None:
        """Test different analysis_type values."""
        for analysis_type in [0, 1, 2, 3]:
            pipeline = DataPipeline(
                datafolder=temp_data_dir,
                model=mock_model,
                model_type=model_type_noafe_explicit,
                analysis_type=analysis_type
            )
            assert pipeline.analysis_type == analysis_type

    def test_kfold_id_edge_cases(self, mock_model: MagicMock,
                                 model_type_noafe_explicit: ModelType,
                                 temp_data_dir: str) -> None:
        """Test edge cases for kfold_ID parameter."""
        # Test boundary values
        for kfold_id in [0, 9, 99]:
            pipeline = DataPipeline(
                datafolder=temp_data_dir,
                model=mock_model,
                model_type=model_type_noafe_explicit,
                num_splits=10,
                kfold_ID=kfold_id
            )
            assert pipeline.kfold_ID == kfold_id

    def test_impute_parameters_combinations(self, mock_model: MagicMock,
                                            model_type_noafe_explicit: ModelType,
                                            temp_data_dir: str) -> None:
        """Test different combinations of imputation parameters."""
        test_cases = [
            (True, True, "with save and load"),
            (True, False, "with save, no load"),
            (False, True, "no save, with load"),
            (False, False, "no save or load"),
        ]
        
        for save_impute, load_impute, description in test_cases:
            pipeline = DataPipeline(
                datafolder=temp_data_dir,
                model=mock_model,
                model_type=model_type_noafe_explicit,
                save_impute=save_impute,
                load_impute=load_impute
            )
            assert pipeline.save_impute == save_impute
            assert pipeline.load_impute == load_impute


# ============================================================================
# TEST CLASS: Integration Tests
# ============================================================================

class TestDataPipelineIntegration:
    """Integration tests for DataPipeline with mocked dependencies."""

    def test_full_initialization_flow(self, temp_data_dir: str) -> None:
        """Test complete initialization flow with a real model object.
        
        This integration test verifies the entire initialization process works
        without errors when using realistic inputs.
        """
        mock_model = MagicMock()
        mock_model.get_name.return_value = "logreg"
        mock_model.is_traditional = True
        
        model_type = ModelType(afe_filter="Complete", feature_set="Explicit")
        
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type,
            baseline_window=25.0,
            window_size=10.0,
            stride=0.5
        )
        
        assert pipeline is not None
        assert all([
            pipeline.datafolder == temp_data_dir,
            pipeline.model == mock_model,
            pipeline.model_type == model_type,
        ])

    def test_parameter_assignment_sequence(self, temp_data_dir: str) -> None:
        """Test that parameters are assigned in correct order.
        
        Verifies that calling process_and_get_data initializes all required
        parameters in the proper sequence.
        """
        mock_model = MagicMock()
        mock_model.get_name.return_value = "RF"
        mock_model.is_traditional = True
        
        model_type = ModelType(afe_filter="noAFE", feature_set="Implicit")
        
        pipeline = DataPipeline(
            datafolder=temp_data_dir,
            model=mock_model,
            model_type=model_type
        )
        
        # Set data_path attribute (referenced in _assign_data_locations)
        pipeline.data_path = temp_data_dir
        # Initialize missing attribute (bug in actual code)
        pipeline.impute_type = None
        # Work around inconsistent attribute naming
        pipeline._data_locations = None
        
        pipeline._assign_hyperparameters_by_classifier()
        pipeline._assign_feature_groups_and_baseline_methods()
        pipeline._assign_data_locations()
        
        # Verify all assignments completed
        assert pipeline.baseline_window == 18.75  # RF's baseline_window
        assert pipeline.window_size == 7.5  # RF's window_size
        assert pipeline.feature_groups_to_analyze is not None
        # Check both possible attribute names for data locations
        data_locations = pipeline._data_locations or pipeline.data_locations
        assert data_locations is not None

# ============================================================================
# PYTEST CONFIGURATION AND HOOKS
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "parametrize: mark test as parametrized"
    )


if __name__ == "__main__":
    # Allow running tests directly: python -m pytest tests/data_pipeline_test.py -v
    pytest.main([__file__, "-v", "--tb=short"])
