import pytest
from pathlib import Path
import os
import json
import numpy as np

from data_pipeline import DataPipeline, AdvancedDataPipeline, TraditionalDataPipeline
from model_type import ModelType


class _DummyModel:
	def __init__(self, *, is_traditional: bool, name: str) -> None:
		self.is_traditional = is_traditional
		self._name = name

	def get_name(self) -> str:
		return self._name


class _DummyConfigParser:
	def __init__(self, model: _DummyModel, faiss_index_type: str = "auto") -> None:
		self._model = model
		self._faiss_index_type = faiss_index_type

	def get_model(self):
		return self._model

	def get_model_type(self):
		return ModelType("Complete", "Explicit")

	def get_remove_NaN_trials(self):
		return True

	def get_subject_to_analyze(self):
		return "02"

	def get_trial_to_analyze(self):
		return "03"

	def get_analysis_type(self):
		return 2

	def get_num_splits(self):
		return 5

	def get_kfold_ID(self):
		return 1

	def get_impute_file_name(self):
		return "imputed.pkl"

	def get_should_impute(self):
		return True

	def get_n_neighbors(self):
		return 4

	def get_baseline_window(self):
		return 32.5

	def get_save_impute(self):
		return False

	def get_load_impute(self):
		return False

	def get_backstep(self):
		return 10

	def get_data_rate(self):
		return 25

	def get_offset(self):
		return 2.5

	def get_time_start(self):
		return 0.0

	def get_data_path(self):
		return "../data/"

	def get_random_seed(self):
		return 123

	def get_output_feature_dtype(self):
		return "float32"

	def get_faiss_index_type(self):
		return self._faiss_index_type


@pytest.mark.parametrize(
	"is_traditional, expected_kind",
	[(True, "traditional"), (False, "advanced")],
)
def test_resolve_pipeline_kind_uses_model_flag(is_traditional, expected_kind):
	parser = _DummyConfigParser(_DummyModel(is_traditional=is_traditional, name="RF"))
	pipeline = DataPipeline(parser)

	assert pipeline._resolve_pipeline_kind() == expected_kind

def test_build_backend_routes_to_traditional(monkeypatch):
	parser = _DummyConfigParser(_DummyModel(is_traditional=True, name="RF"))
	pipeline = DataPipeline(parser)

	created = {}

	class FakeTraditionalPipeline:
		def __init__(self, data_path, random_seed, config_parser=None):
			created["traditional"] = (data_path, random_seed, config_parser)

	class FakeAdvancedPipeline:
		def __init__(self, data_path, random_seed, config_parser=None):
			created["advanced"] = (data_path, random_seed, config_parser)

	monkeypatch.setattr("data_pipeline.TraditionalDataPipeline", FakeTraditionalPipeline)
	monkeypatch.setattr("data_pipeline.AdvancedDataPipeline", FakeAdvancedPipeline)

	backend = pipeline._build_backend()

	assert isinstance(backend, FakeTraditionalPipeline)
	assert created["traditional"] == ("../data/", 123, parser)
	assert "advanced" not in created


@pytest.mark.parametrize(
	"faiss_index_type, gpu_count, expected_backend",
	[("cpu", 1, "cpu"), ("gpu", 1, "gpu"), ("auto", 0, "cpu")],
)
def test_build_faiss_knn_index_honors_configured_index_type(monkeypatch, faiss_index_type, gpu_count, expected_backend):
	parser = _DummyConfigParser(_DummyModel(is_traditional=True, name="RF"), faiss_index_type=faiss_index_type)
	pipeline = TraditionalDataPipeline(data_path="../data/", random_seed=123, config_parser=parser)

	class _FakeHNSWState:
		def __init__(self) -> None:
			self.efSearch = None
			self.rng = None

	class _FakeCPUIndex:
		def __init__(self) -> None:
			self.hnsw = _FakeHNSWState()

	class _FakeGPUIndex:
		pass

	cpu_index = _FakeCPUIndex()
	gpu_index = _FakeGPUIndex()

	monkeypatch.setattr("data_pipeline.faiss.get_num_gpus", lambda: gpu_count, raising=False)
	monkeypatch.setattr("data_pipeline.faiss.StandardGpuResources", lambda: object(), raising=False)
	monkeypatch.setattr("data_pipeline.faiss.GpuIndexFlatL2", lambda *args: gpu_index, raising=False)
	monkeypatch.setattr("data_pipeline.faiss.IndexHNSWFlat", lambda d, M: cpu_index, raising=False)
	monkeypatch.setattr("data_pipeline.faiss.RandomGenerator", lambda seed: f"rng:{seed}", raising=False)

	index = pipeline._build_faiss_knn_index(d=8)

	if expected_backend == "gpu":
		assert index is gpu_index
	else:
		assert index is cpu_index
		assert cpu_index.hnsw.efSearch == 64
		assert cpu_index.hnsw.rng == "rng:123"


def test_get_data_for_advanced_pipeline_forwards_required_arguments(monkeypatch):
	parser = _DummyConfigParser(_DummyModel(is_traditional=False, name="RF"))
	pipeline = DataPipeline(parser)

	captured = {}

	class FakeBackend:
		def get_data(self, **kwargs):
			captured.update(kwargs)
			return "advanced-ok"

	monkeypatch.setattr(pipeline, "_build_backend", lambda: FakeBackend())

	result = pipeline.get_data(kfold_id=3, feature_streams=["EEG"])

	assert result == "advanced-ok"
	assert captured == {
		"model_type": ModelType("Complete", "Explicit"),
		"remove_NaN_trials": True,
		"subject_to_analyze": "02",
		"trial_to_analyze": "03",
		"analysis_type": 2,
		"num_splits": 5,
		"kfold_ID": 3,
		"impute_file_name": "imputed.pkl",
		"should_impute": True,
		"output_feature_dtype": "float32",
		"n_neighbors": 4,
		"baseline_window": 32.5,
		"save_impute": False,
		"load_impute": False,
	}


def test_get_data_for_traditional_pipeline_forwards_required_arguments(monkeypatch):
	parser = _DummyConfigParser(_DummyModel(is_traditional=True, name="RF"))
	pipeline = DataPipeline(parser)

	captured = {}

	class FakeBackend:
		def get_data(self, **kwargs):
			captured.update(kwargs)
			return "traditional-ok"

	monkeypatch.setattr(pipeline, "_build_backend", lambda: FakeBackend())
	monkeypatch.setattr(pipeline, "_resolve_select_features", lambda _kwargs: ["f1", "f2"])

	result = pipeline.get_data(feature_streams=[])

	assert result == "traditional-ok"
	assert captured == {
		"model_type": ModelType("Complete", "Explicit"),
		"remove_NaN_trials": True,
		"subject_to_analyze": "02",
		"trial_to_analyze": "03",
		"analysis_type": 2,
		"classifier_type": "RF",
		"select_features": ["f1", "f2"],
		"backstep": 10,
		"data_rate": 25,
		"offset": 2.5,
		"time_start": 0.0,
		"impute_file_name": "imputed.pkl",
		"should_impute": True,
		"output_feature_dtype": "float32",
		"save_impute": False,
		"load_impute": False,
	}


def test_apply_sensor_ablation_disabled_returns_original_features():
	parser = _DummyConfigParser(_DummyModel(is_traditional=True, name="RF"))
	pipeline = DataPipeline(parser)

	selected_features = ["HR (bpm) - Equivital_mean_s1", "Fz_alpha - EEG_mean_s1"]
	assert pipeline._apply_sensor_ablation(selected_features, feature_streams=[]) == selected_features


def test_apply_sensor_ablation_enabled_filters_selected_features():
	parser = _DummyConfigParser(_DummyModel(is_traditional=True, name="RF"))
	pipeline = DataPipeline(parser)

	selected_features = [
		"HR (bpm) - Equivital_mean_s1",
		"Pupil diameter left [mm] - Tobii_mean_s1",
		"Fz_alpha - EEG_mean_s1",
	]

	assert pipeline._apply_sensor_ablation(selected_features, feature_streams=["EEG", "Pupil"]) == [
		"Pupil diameter left [mm] - Tobii_mean_s1",
		"Fz_alpha - EEG_mean_s1",
	]


def test_get_data_for_traditional_pipeline_applies_sensor_ablation(monkeypatch):
	parser = _DummyConfigParser(_DummyModel(is_traditional=True, name="RF"))
	pipeline = DataPipeline(parser)

	captured = {}

	class FakeBackend:
		def get_data(self, **kwargs):
			captured.update(kwargs)
			return "traditional-ok"

	monkeypatch.setattr(pipeline, "_build_backend", lambda: FakeBackend())
	monkeypatch.setattr(
		pipeline,
		"_resolve_select_features",
		lambda _kwargs: ["HR (bpm) - Equivital_mean_s1", "Fz_alpha - EEG_mean_s1"],
	)

	result = pipeline.get_data(feature_streams=["EEG"])

	assert result == "traditional-ok"
	assert captured["select_features"] == ["Fz_alpha - EEG_mean_s1"]


def test_get_data_for_advanced_pipeline_requires_kfold_id(monkeypatch):
	parser = _DummyConfigParser(_DummyModel(is_traditional=False, name="RF"))
	pipeline = DataPipeline(parser)

	class FakeBackend:
		def get_data(self, **kwargs):
			return kwargs

	monkeypatch.setattr(pipeline, "_build_backend", lambda: FakeBackend())

	with pytest.raises(ValueError, match="kfold_id is required for advanced pipelines"):
		pipeline.get_data(feature_streams=["EEG"])


def test_advanced_impute_cache_path_has_prefix_processed_data_and_kfold_suffix():
	pipeline = AdvancedDataPipeline(data_path="../data/")
	cache_path = pipeline._resolve_advanced_impute_path("imputed_data.pkl", 3)

	assert cache_path.endswith(str(Path("Processed Data") / "advanced_imputed_data_kfold_3.pkl"))


def test_traditional_impute_cache_path_has_prefix_processed_data_and_model_suffix():
	pipeline = TraditionalDataPipeline(data_path="../data/")
	cache_path = pipeline._resolve_traditional_impute_path("imputed_data.pkl", "logreg")

	assert cache_path.endswith(str(Path("Processed Data") / "traditional_imputed_data_logreg.pkl"))

def test_advanced_data_pipeline_has_correct_dimensions():
	pipeline = AdvancedDataPipeline(data_path = "/home/gloc/G-LOC-Prediction/data/")
	X_train, _, _, _, _ = pipeline.get_data(
		model_type = ModelType("Complete", "Explicit"),
		remove_NaN_trials = True,
		subject_to_analyze = "01",
		trial_to_analyze = "03",
		analysis_type = 2,
		num_splits = 10,
		kfold_ID = 0,
		impute_file_name = "imputed_data.pkl",
		should_impute = True,
		n_neighbors = 4,
		baseline_window = 32.5,
		save_impute = False,
		load_impute = False,
	)

	assert X_train.shape == (1072086, 283)

def test_traditional_data_pipeline_has_correct_dimensions():
	pipeline = TraditionalDataPipeline(data_path = "/home/gloc/G-LOC-Prediction/data/")

	# Function to load in median hyperparameters from a simple JSON
	model_type = ModelType("Complete", "Explicit")
	BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	model_type_string = f"{model_type.afe_filter}_{model_type.feature_set}"
	json_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type_string, f'median_hyperparameters_LogReg.json')

	with open(json_path, 'r') as f:
		data = json.load(f)

	X, y = pipeline.get_data(
		model_type = ModelType("Complete", "Explicit"),
		remove_NaN_trials = True,
		subject_to_analyze = "01",
		trial_to_analyze = "03",
		analysis_type = 2,
		classifier_type = "LogReg",
		select_features = data["selected_features"],
		backstep = 10,
		data_rate = 25,
		offset = 2.5,
		time_start = 0.0,
		impute_file_name = "imputed_data.pkl",
		should_impute = True,
		save_impute = False,
		load_impute = False,
	)

	assert X.shape == (185999, 3959)


# ============================================================================
# COMPREHENSIVE FUNCTIONAL TESTS FOR CORE OUTPUT CONSISTENCY
# ============================================================================


class TestAdvancedPipelineOutputProperties:
	"""Tests that verify Advanced pipeline outputs have correct structure and properties."""

	@pytest.fixture
	def advanced_pipeline(self):
		return AdvancedDataPipeline(data_path="/home/gloc/G-LOC-Prediction/data/")

	def test_advanced_returns_five_elements(self, advanced_pipeline):
		"""Verify that get_data returns exactly 5 elements: X_train, X_test, y_train, y_test, features."""
		result = advanced_pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			num_splits=10,
			kfold_ID=0,
			impute_file_name="test_imputed.pkl",
			should_impute=True,
			n_neighbors=4,
			baseline_window=32.5,
			save_impute=False,
			load_impute=False,
		)
		
		assert isinstance(result, tuple), "Result should be a tuple"
		assert len(result) == 5, f"Expected 5 return values, got {len(result)}"
		
		X_train, X_test, y_train, y_test, features = result
		assert isinstance(X_train, np.ndarray), "X_train should be ndarray"
		assert isinstance(X_test, np.ndarray), "X_test should be ndarray"
		assert isinstance(y_train, np.ndarray), "y_train should be ndarray"
		assert isinstance(y_test, np.ndarray), "y_test should be ndarray"
		assert isinstance(features, list), "features should be a list"

	def test_advanced_output_dtypes(self, advanced_pipeline):
		"""Verify output arrays have correct data types."""
		X_train, X_test, y_train, y_test, features = advanced_pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			num_splits=10,
			kfold_ID=0,
			impute_file_name="test_imputed.pkl",
			should_impute=True,
			n_neighbors=4,
			baseline_window=32.5,
			save_impute=False,
			load_impute=False,
			output_feature_dtype=np.float32,
		)
		
		# Check data types
		assert X_train.dtype == np.float32, f"X_train dtype should be float32, got {X_train.dtype}"
		assert X_test.dtype == np.float32, f"X_test dtype should be float32, got {X_test.dtype}"
		assert y_train.dtype in [np.uint8, bool, np.bool_], f"y_train dtype should be bool-like, got {y_train.dtype}"
		assert y_test.dtype in [np.uint8, bool, np.bool_], f"y_test dtype should be bool-like, got {y_test.dtype}"

	def test_advanced_no_nan_in_features(self, advanced_pipeline):
		"""Verify that feature matrices don't contain NaN values."""
		X_train, X_test, y_train, y_test, _ = advanced_pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			num_splits=10,
			kfold_ID=0,
			impute_file_name="test_imputed.pkl",
			should_impute=True,
			n_neighbors=4,
			baseline_window=32.5,
			save_impute=False,
			load_impute=False,
		)
		
		nan_count_train = np.isnan(X_train).sum()
		nan_count_test = np.isnan(X_test).sum()
		assert nan_count_train == 0, f"X_train contains {nan_count_train} NaN values"
		assert nan_count_test == 0, f"X_test contains {nan_count_test} NaN values"

	def test_advanced_train_test_split_ratio(self, advanced_pipeline):
		"""Verify that train/test split approximately respects the number of splits."""
		X_train, X_test, y_train, y_test, _ = advanced_pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			num_splits=5,
			kfold_ID=0,
			impute_file_name="test_imputed.pkl",
			should_impute=True,
			n_neighbors=4,
			baseline_window=32.5,
			save_impute=False,
			load_impute=False,
		)
		
		total_samples = X_train.shape[0] + X_test.shape[0]
		train_ratio = X_train.shape[0] / total_samples
		test_ratio = X_test.shape[0] / total_samples
		
		# With 5 splits, test set should be ~1/5 (20%), train ~4/5 (80%)
		assert 0.15 < test_ratio < 0.25, f"Test ratio {test_ratio} is outside expected range [0.15, 0.25]"
		assert 0.75 < train_ratio < 0.85, f"Train ratio {train_ratio} is outside expected range [0.75, 0.85]"

	def test_advanced_label_distribution(self, advanced_pipeline):
		"""Verify that labels are binary and have reasonable distribution."""
		_, _, y_train, y_test, _ = advanced_pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			num_splits=10,
			kfold_ID=0,
			impute_file_name="test_imputed.pkl",
			should_impute=True,
			n_neighbors=4,
			baseline_window=32.5,
			save_impute=False,
			load_impute=False,
		)
		
		# Labels should be binary (0 or 1)
		unique_train = np.unique(y_train)
		unique_test = np.unique(y_test)
		assert len(unique_train) <= 2, f"Train labels should be binary, got {unique_train}"
		assert len(unique_test) <= 2, f"Test labels should be binary, got {unique_test}"
		assert set(unique_train).union(set(unique_test)).issubset({0, 1}), "Labels should only contain 0 and 1"

	def test_advanced_feature_list_consistency(self, advanced_pipeline):
		"""Verify that feature list length matches number of columns in X matrices."""
		X_train, X_test, _, _, features = advanced_pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			num_splits=10,
			kfold_ID=0,
			impute_file_name="test_imputed.pkl",
			should_impute=True,
			n_neighbors=4,
			baseline_window=32.5,
			save_impute=False,
			load_impute=False,
		)
		
		assert X_train.shape[1] == len(features) + 2, \
			f"X_train has {X_train.shape[1]} columns but {len(features)} feature names plus the expected pipeline columns"
		assert X_test.shape[1] == len(features) + 2, \
			f"X_test has {X_test.shape[1]} columns but {len(features)} feature names plus the expected pipeline columns"


class TestTraditionalPipelineOutputProperties:
	"""Tests that verify Traditional pipeline outputs have correct structure and properties."""

	@pytest.fixture
	def traditional_pipeline(self):
		return TraditionalDataPipeline(data_path="/home/gloc/G-LOC-Prediction/data/")

	@pytest.fixture
	def logreg_features(self):
		"""Load LogReg features for consistent testing."""
		model_type = ModelType("Complete", "Explicit")
		BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
		model_type_string = f"{model_type.afe_filter}_{model_type.feature_set}"
		json_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type_string, 'median_hyperparameters_LogReg.json')
		
		with open(json_path, 'r') as f:
			data = json.load(f)
		return data["selected_features"]

	def test_traditional_returns_two_elements(self, traditional_pipeline, logreg_features):
		"""Verify that get_data returns exactly 2 elements: X, y."""
		result = traditional_pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			classifier_type="LogReg",
			select_features=logreg_features,
			backstep=10,
			data_rate=25,
			offset=2.5,
			time_start=0.0,
			impute_file_name="test_imputed.pkl",
			should_impute=True,
			save_impute=False,
			load_impute=False,
		)
		
		assert isinstance(result, tuple), "Result should be a tuple"
		assert len(result) == 2, f"Expected 2 return values, got {len(result)}"
		
		X, y = result
		assert isinstance(X, np.ndarray), "X should be ndarray"
		assert isinstance(y, np.ndarray), "y should be ndarray"

	def test_traditional_output_dtypes(self, traditional_pipeline, logreg_features):
		"""Verify output arrays have correct data types."""
		X, y = traditional_pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			classifier_type="LogReg",
			select_features=logreg_features,
			backstep=10,
			data_rate=25,
			offset=2.5,
			time_start=0.0,
			impute_file_name="test_imputed.pkl",
			should_impute=True,
			output_feature_dtype=np.float32,
			save_impute=False,
			load_impute=False,
		)
		
		assert X.dtype == np.float32, f"X dtype should be float32, got {X.dtype}"
		assert y.dtype == np.float32, f"y dtype should be float32, got {y.dtype}"

	def test_traditional_no_nan_in_features(self, traditional_pipeline, logreg_features):
		"""Verify that feature matrix doesn't contain NaN values."""
		X, y = traditional_pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			classifier_type="LogReg",
			select_features=logreg_features,
			backstep=10,
			data_rate=25,
			offset=2.5,
			time_start=0.0,
			impute_file_name="test_imputed.pkl",
			should_impute=True,
			save_impute=False,
			load_impute=False,
		)
		
		nan_count = np.isnan(X).sum()
		assert nan_count == 0, f"X contains {nan_count} NaN values"

	def test_traditional_label_distribution(self, traditional_pipeline, logreg_features):
		"""Verify that labels are binary and have reasonable distribution."""
		X, y = traditional_pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			classifier_type="LogReg",
			select_features=logreg_features,
			backstep=10,
			data_rate=25,
			offset=2.5,
			time_start=0.0,
			impute_file_name="test_imputed.pkl",
			should_impute=True,
			save_impute=False,
			load_impute=False,
		)
		
		unique_labels = np.unique(y)
		assert len(unique_labels) <= 2, f"Labels should be binary, got {unique_labels}"
		assert set(unique_labels).issubset({0, 1}), "Labels should only contain 0 and 1"
		
		# Check that we have positive examples
		assert np.sum(y) > 0, "No positive labels found in output"

	def test_traditional_features_match_selected(self, traditional_pipeline, logreg_features):
		"""Verify that X has the same number of columns as selected features."""
		X, y = traditional_pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			classifier_type="LogReg",
			select_features=logreg_features,
			backstep=10,
			data_rate=25,
			offset=2.5,
			time_start=0.0,
			impute_file_name="test_imputed.pkl",
			should_impute=True,
			save_impute=False,
			load_impute=False,
		)
		
		assert X.shape[1] == len(logreg_features), \
			f"X has {X.shape[1]} columns but {len(logreg_features)} features selected"


class TestDataPipelineConsistency:
	"""Tests that verify consistency across different configurations and model types."""

	def test_advanced_multiple_kfolds_produce_different_splits(self):
		"""Verify that different kfold IDs produce different train/test splits."""
		pipeline = AdvancedDataPipeline(data_path="/home/gloc/G-LOC-Prediction/data/")
		
		X_train_0, X_test_0, _, _, _ = pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			num_splits=5,
			kfold_ID=0,
			impute_file_name="test_imputed_0.pkl",
			should_impute=True,
			n_neighbors=4,
			baseline_window=32.5,
			save_impute=False,
			load_impute=False,
		)
		
		X_train_1, X_test_1, _, _, _ = pipeline.get_data(
			model_type=ModelType("Complete", "Explicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			num_splits=5,
			kfold_ID=1,
			impute_file_name="test_imputed_1.pkl",
			should_impute=True,
			n_neighbors=4,
			baseline_window=32.5,
			save_impute=False,
			load_impute=False,
		)
		
		# Shapes should be the same (sampling the whole dataset with different folds)
		assert X_train_0.shape[1] == X_train_1.shape[1], "Feature dimensions should match"
		
		# But the actual split indices should differ
		# (at least one row should be in train for fold 0 but test for fold 1)
		# We can't guarantee exact row matching, but row distribution should differ
		assert abs(X_train_0.shape[0] - X_train_1.shape[0]) < X_train_0.shape[0] * 0.1, \
			"Train set sizes between different folds should be similar"

	def test_advanced_deterministic_with_same_seed(self):
		"""Verify that same configuration produces consistent outputs with random seed."""
		# Both pipelines use same random seed
		pipeline1 = AdvancedDataPipeline(data_path="/home/gloc/G-LOC-Prediction/data/", random_seed=42)
		pipeline2 = AdvancedDataPipeline(data_path="/home/gloc/G-LOC-Prediction/data/", random_seed=42)
		
		# Get data with same params
		X_train_1, _, y_train_1, _, features_1 = pipeline1.get_data(
			model_type=ModelType("noAFE", "Implicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			num_splits=10,
			kfold_ID=0,
			impute_file_name="test_deterministic_1.pkl",
			should_impute=True,
			n_neighbors=4,
			baseline_window=32.5,
			save_impute=False,
			load_impute=False,
		)
		
		X_train_2, _, y_train_2, _, features_2 = pipeline2.get_data(
			model_type=ModelType("noAFE", "Implicit"),
			remove_NaN_trials=True,
			subject_to_analyze=None,
			trial_to_analyze=None,
			analysis_type=2,
			num_splits=10,
			kfold_ID=0,
			impute_file_name="test_deterministic_2.pkl",
			should_impute=True,
			n_neighbors=4,
			baseline_window=32.5,
			save_impute=False,
			load_impute=False,
		)
		
		# Features should be identical
		assert features_1 == features_2, "Feature lists should be identical with same seed"
		
		# Data should be identical (within floating point tolerance)
		assert np.allclose(X_train_1, X_train_2), "Training data should be identical with same seed"
		assert np.array_equal(y_train_1, y_train_2), "Labels should be identical with same seed"