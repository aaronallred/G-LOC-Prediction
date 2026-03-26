import json
from unittest.mock import MagicMock, patch
import pytest

from data_pipeline import DataPipeline


@pytest.fixture
def config_parser() -> MagicMock:
	parser = MagicMock()
	parser.get_data_path.return_value = "data"
	parser.get_model_type.return_value = MagicMock(afe_filter = "Complete", feature_set = "Explicit")
	parser.get_random_seed.return_value = 42
	parser.get_num_splits.return_value = 7
	parser.get_kfold_ID.return_value = 2
	parser.get_impute_path.return_value = "cached_data/Trans/imputed_Complete_Explicit_fold2.pkl"
	parser.get_impute_type.return_value = 1
	parser.get_n_neighbors.return_value = 5
	parser.get_baseline_window.return_value = 2.0
	parser.get_save_impute.return_value = False
	parser.get_load_impute.return_value = True
	parser.get_remove_NaN_trials.return_value = False
	parser.get_subject_to_analyze.return_value = None
	parser.get_trial_to_analyze.return_value = None
	parser.get_analysis_type.return_value = 2
	parser.get_backstep.return_value = 0
	parser.get_data_rate.return_value = 25
	parser.get_offset.return_value = 0.0
	parser.get_time_start.return_value = 0.0
	return parser


def test_resolve_pipeline_kind_infers_traditional_from_model(config_parser: MagicMock) -> None:
	model = MagicMock()
	model.is_traditional = True
	config_parser.get_model.return_value = model

	pipeline = DataPipeline(config_parser = config_parser)

	assert pipeline._resolve_pipeline_kind() == "traditional"


def test_resolve_pipeline_kind_infers_advanced_from_model(config_parser: MagicMock) -> None:
	model = MagicMock()
	model.is_traditional = False
	config_parser.get_model.return_value = model

	pipeline = DataPipeline(config_parser = config_parser)

	assert pipeline._resolve_pipeline_kind() == "advanced"


def test_resolve_pipeline_kind_raises_without_model(config_parser: MagicMock) -> None:
	config_parser.get_model.return_value = None
	pipeline = DataPipeline(config_parser = config_parser)

	with pytest.raises(ValueError, match = "is_traditional"):
		pipeline._resolve_pipeline_kind()


def test_get_data_advanced_sets_expected_kwargs(config_parser: MagicMock) -> None:
	model = MagicMock()
	model.is_traditional = False
	config_parser.get_model.return_value = model

	backend = MagicMock()
	backend.get_data.return_value = ("x_train", "x_test", "y_train", "y_test", ["f1"])

	pipeline = DataPipeline(config_parser = config_parser)
	pipeline._build_backend = MagicMock(return_value = backend)

	output = pipeline.get_data()

	assert output == ("x_train", "x_test", "y_train", "y_test", ["f1"])
	backend.get_data.assert_called_once()

	kwargs = backend.get_data.call_args.kwargs
	assert kwargs["model_type"] == config_parser.get_model_type.return_value
	assert kwargs["num_splits"] == 7
	assert kwargs["kfold_ID"] == 2
	assert kwargs["impute_path"] == "cached_data/Trans/imputed_Complete_Explicit_fold2.pkl"
	assert kwargs["impute_type"] == 1
	assert kwargs["n_neighbors"] == 5
	assert kwargs["baseline_window"] == 2.0
	assert kwargs["save_impute"] is False
	assert kwargs["load_impute"] is True


def test_get_data_traditional_uses_model_name_and_median_hyperparams(config_parser: MagicMock) -> None:
	model = MagicMock()
	model.is_traditional = True
	model.get_name.return_value = "KNN"
	config_parser.get_model.return_value = model

	backend = MagicMock()
	backend.get_data.return_value = ("X", "y")

	pipeline = DataPipeline(config_parser = config_parser)
	pipeline._build_backend = MagicMock(return_value = backend)

	median_payload = {
		"best_params": [["n_neighbors", 7]],
		"selected_features": ["a", "b"],
		"f1_score": 0.9,
		"fold_id": 3,
	}

	with patch("data_pipeline.open", create = True) as mocked_open:
		mocked_open.return_value.__enter__.return_value.read.return_value = json.dumps(median_payload)
		with patch("data_pipeline.json.load", return_value = median_payload):
			output = pipeline.get_data()

	assert output == ("X", "y")
	backend.get_data.assert_called_once()

	kwargs = backend.get_data.call_args.kwargs
	assert kwargs["model_type"] == config_parser.get_model_type.return_value
	assert kwargs["classifier_type"] == "KNN"
	selected_features = kwargs["select_features"]
	assert selected_features == ["a", "b"]
	assert kwargs["backstep"] == 0
	assert kwargs["data_rate"] == 25
	assert kwargs["analysis_type"] == 2
	assert "impute_type" not in kwargs


def test_get_data_traditional_raises_if_median_hyperparams_missing(config_parser: MagicMock) -> None:
	model = MagicMock()
	model.is_traditional = True
	model.get_name.return_value = "KNN"
	config_parser.get_model.return_value = model

	backend = MagicMock()
	pipeline = DataPipeline(config_parser = config_parser)
	pipeline._build_backend = MagicMock(return_value = backend)

	with patch("data_pipeline.open", side_effect = FileNotFoundError("missing median file")):
		with pytest.raises(FileNotFoundError):
			pipeline.get_data()


def test_data_pipeline_get_data_advanced_accepts_inputs_and_completes(config_parser: MagicMock) -> None:
	model = MagicMock()
	model.is_traditional = False
	config_parser.get_model.return_value = model

	expected = ("x_train", "x_test", "y_train", "y_test", ["f1"])

	def strict_advanced_get_data(
			*,
			model_type,
			num_splits,
			kfold_ID,
			impute_path,
			subject_to_analyze,
			trial_to_analyze,
			impute_type,
			n_neighbors,
			baseline_window,
			analysis_type,
			remove_NaN_trials,
			save_impute,
			load_impute,
	):
		assert model_type == config_parser.get_model_type.return_value
		assert num_splits == 7
		assert kfold_ID == 2
		assert impute_path == "cached_data/Trans/imputed_Complete_Explicit_fold2.pkl"
		assert impute_type == 1
		assert n_neighbors == 5
		assert baseline_window == 2.0
		assert analysis_type == 2
		assert remove_NaN_trials is False
		assert save_impute is False
		assert load_impute is True
		assert subject_to_analyze is None
		assert trial_to_analyze is None
		return expected

	backend = MagicMock()
	backend.get_data = MagicMock(side_effect = strict_advanced_get_data)

	pipeline = DataPipeline(config_parser = config_parser)
	pipeline._build_backend = MagicMock(return_value = backend)

	actual = pipeline.get_data()

	assert actual == expected
	backend.get_data.assert_called_once()


def test_data_pipeline_get_data_traditional_accepts_inputs_and_completes(config_parser: MagicMock) -> None:
	model = MagicMock()
	model.is_traditional = True
	model.get_name.return_value = "KNN"
	config_parser.get_model.return_value = model

	median_payload = {
		"best_params": [["n_neighbors", 7]],
		"selected_features": ["a", "b"],
		"f1_score": 0.9,
		"fold_id": 3,
	}

	expected = ("X", "y")

	def strict_traditional_get_data(
			*,
			backstep,
			data_rate,
			classifier_type,
			model_type,
			select_features,
			remove_NaN_trials,
			offset,
			time_start,
			subject_to_analyze,
			trial_to_analyze,
			analysis_type,
	):
		assert backstep == 0
		assert data_rate == 25
		assert classifier_type == "KNN"
		assert model_type == config_parser.get_model_type.return_value
		assert select_features == ["a", "b"]
		assert remove_NaN_trials is False
		assert offset == 0.0
		assert time_start == 0.0
		assert subject_to_analyze is None
		assert trial_to_analyze is None
		assert analysis_type == 2
		return expected

	backend = MagicMock()
	backend.get_data = MagicMock(side_effect = strict_traditional_get_data)

	pipeline = DataPipeline(config_parser = config_parser)
	pipeline._build_backend = MagicMock(return_value = backend)

	with patch("data_pipeline.open", create = True) as mocked_open:
		mocked_open.return_value.__enter__.return_value.read.return_value = json.dumps(median_payload)
		with patch("data_pipeline.json.load", return_value = median_payload):
			actual = pipeline.get_data()

	assert actual == expected
	backend.get_data.assert_called_once()
