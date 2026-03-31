import pytest
from pathlib import Path

from src.data_pipeline import DataPipeline, AdvancedDataPipeline, TraditionalDataPipeline
from model_type import ModelType


class _DummyModel:
	def __init__(self, *, is_traditional: bool, name: str) -> None:
		self.is_traditional = is_traditional
		self._name = name

	def get_name(self) -> str:
		return self._name


class _DummyConfigParser:
	def __init__(self, model: _DummyModel) -> None:
		self._model = model

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
		def __init__(self, data_path, random_seed):
			created["traditional"] = (data_path, random_seed)

	class FakeAdvancedPipeline:
		def __init__(self, data_path, random_seed):
			created["advanced"] = (data_path, random_seed)

	monkeypatch.setattr("src.data_pipeline.TraditionalDataPipeline", FakeTraditionalPipeline)
	monkeypatch.setattr("src.data_pipeline.AdvancedDataPipeline", FakeAdvancedPipeline)

	backend = pipeline._build_backend()

	assert isinstance(backend, FakeTraditionalPipeline)
	assert created["traditional"] == ("../data/", 123)
	assert "advanced" not in created


def test_get_data_for_advanced_pipeline_forwards_required_arguments(monkeypatch):
	parser = _DummyConfigParser(_DummyModel(is_traditional=False, name="RF"))
	pipeline = DataPipeline(parser)

	captured = {}

	class FakeBackend:
		def get_data(self, **kwargs):
			captured.update(kwargs)
			return "advanced-ok"

	monkeypatch.setattr(pipeline, "_build_backend", lambda: FakeBackend())

	result = pipeline.get_data()

	assert result == "advanced-ok"
	assert captured == {
		"model_type": ModelType("Complete", "Explicit"),
		"remove_NaN_trials": True,
		"subject_to_analyze": "02",
		"trial_to_analyze": "03",
		"analysis_type": 2,
		"num_splits": 5,
		"kfold_ID": 1,
		"impute_file_name": "imputed.pkl",
		"should_impute": True,
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

	result = pipeline.get_data()

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
		"save_impute": False,
		"load_impute": False,
	}


def test_advanced_impute_cache_path_has_prefix_processed_data_and_kfold_suffix():
	pipeline = AdvancedDataPipeline(data_path="../data/")
	cache_path = pipeline._resolve_advanced_impute_path("imputed_data.pkl", 3)

	assert cache_path.endswith(str(Path("Processed Data") / "advanced_imputed_data_kfold_3.pkl"))


def test_traditional_impute_cache_path_has_prefix_processed_data_and_model_suffix():
	pipeline = TraditionalDataPipeline(data_path="../data/")
	cache_path = pipeline._resolve_traditional_impute_path("imputed_data.pkl", "Logistic Regression")

	assert cache_path.endswith(str(Path("Processed Data") / "traditional_imputed_data_Logistic_Regression.pkl"))

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
	X, y = pipeline.get_data(
		model_type = ModelType("Complete", "Explicit"),
		remove_NaN_trials = True,
		subject_to_analyze = "01",
		trial_to_analyze = "03",
		analysis_type = 2,
		classifier_type = "LogReg",
		select_features = ["f1", "f2"],
		backstep = 10,
		data_rate = 25,
		offset = 2.5,
		time_start = 0.0,
		impute_file_name = "imputed_data.pkl",
		should_impute = True,
		save_impute = False,
		load_impute = False,
	)

	assert X.shape == (186539, 3959)