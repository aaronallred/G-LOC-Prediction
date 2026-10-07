import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import src.modes.temporal_evaluation as temporal_evaluation
from src.model_type import ModelType
from src.models.model_factory import ModelFactory
from src.modes.temporal_evaluation import run_temporal_evaluation_review, run_temporal_evaluation_training


class FakeFoldPipeline:
	"""Returns per-fold splits with per-horizon labels, like DataPipeline.get_data(..., horizons=...)."""

	def __init__(self, X, y_by_horizon, feature_names):
		self.X = X
		self.y_by_horizon = y_by_horizon
		self.feature_names = feature_names
		self.calls = []

	def set_random_seed(self, seed):
		self.seed = seed

	def set_model_type(self, model_type):
		self.model_type = model_type

	def get_data(self, **kwargs):
		self.calls.append(kwargs)
		folds = np.array_split(np.arange(len(self.X)), kwargs["num_splits"])
		test_idx = folds[kwargs["kfold_id"]]
		train_idx = np.setdiff1d(np.arange(len(self.X)), test_idx)
		y_train = {h: self.y_by_horizon[h][train_idx] for h in kwargs["horizons"]}
		y_test = {h: self.y_by_horizon[h][test_idx] for h in kwargs["horizons"]}
		return self.X[train_idx], self.X[test_idx], y_train, y_test, self.feature_names


def _config(horizons, models=("LogReg",)):
	model_type = ModelType("Complete", "Explicit")
	return {
		"traditional_data_parameters": {"backstep": 0, "data_rate": 25},
		"advanced_data_parameters": {"horizon": 0},
		"temporal_evaluation": {
			"training": {
				"enabled": True,
				"models": list(models),
				"model_type": model_type,
				"horizons": horizons,
				"num_splits": 5,
				"random_seed": 42,
				"class_weight": None,
				"median_hyperparameters_folder": "Results/CV",
				"save_results_folder": "Results/Temporal",
				"save_models": False,
			},
			"review": {
				"enabled": True,
				"models": list(models),
				"model_type": model_type,
				"save_results_folder": "Results/Temporal",
				"error_type": "std",
				"show_plots": False,
			},
		},
	}


def _write_median_json(root, model_name, selected_features=("f0", "f1", "f2", "f3")):
	median_dir = root / "Results" / "CV" / "Complete_Explicit" / model_name
	median_dir.mkdir(parents=True)
	(median_dir / "median_hyperparameters.json").write_text(json.dumps({
		"best_params": {},
		"selected_features": list(selected_features),
		"fold_id": 3,
		"f1_score": 0.8,
	}))


def _fake_data():
	rng = np.random.default_rng(0)
	X = rng.normal(size=(300, 4)).astype(np.float32)
	return X, {0: (X[:, 0] > 1.0).astype(np.float32), 2: (X[:, 0] > 0.8).astype(np.float32)}


def test_traditional_training_loads_each_fold_once_and_writes_temporal_summary(tmp_path):
	_write_median_json(tmp_path, "LogReg")
	X, y_by_horizon = _fake_data()
	pipeline = FakeFoldPipeline(X, y_by_horizon, ["f0", "f1", "f2", "f3"])

	run_temporal_evaluation_training(_config([0, 2]), pipeline, ModelFactory(), tmp_path)

	assert [call["kfold_id"] for call in pipeline.calls] == [0, 1, 2, 3, 4]
	assert all(call["horizons"] == {0: 0, 2: 50} for call in pipeline.calls)
	assert all(call["traditional_feature_selection"] == "raw" for call in pipeline.calls)

	model_dir = tmp_path / "Results" / "Temporal" / "Complete_Explicit" / "LogReg"
	summary = json.loads((model_dir / "temporal_summary.json").read_text())
	assert summary["pipeline"] == "traditional"
	assert summary["horizons"] == [0, 2]
	assert summary["horizon_samples"] == {"0": 0, "2": 50}
	assert summary["hyperparameters_source"]["fold_id"] == 3
	assert summary["num_features"] == 4
	for key in ("0", "2"):
		result = summary["results"][key]
		assert result["num_folds"] == 5
		assert len(result["fold_metrics"]["f1"]) == 5
		assert result["n_total"] == 300
		assert (model_dir / f"horizon_{key}" / "summary.json").exists()
		for fold in range(5):
			assert (model_dir / f"horizon_{key}" / f"fold_{fold}" / "fold_result.json").exists()
	assert summary["results"]["2"]["n_positive"] > summary["results"]["0"]["n_positive"]
	assert (tmp_path / "Results" / "Temporal" / "Complete_Explicit" / "run_config.yaml").exists()


def test_traditional_training_keeps_only_cv_selected_features(tmp_path):
	_write_median_json(tmp_path, "LogReg", selected_features=("f2", "f0"))
	X, y_by_horizon = _fake_data()
	pipeline = FakeFoldPipeline(X, y_by_horizon, ["f0", "f1", "f2", "f3"])

	run_temporal_evaluation_training(_config([0]), pipeline, ModelFactory(), tmp_path)

	summary = json.loads(
		(tmp_path / "Results" / "Temporal" / "Complete_Explicit" / "LogReg" / "temporal_summary.json").read_text()
	)
	assert summary["num_features"] == 2


def test_traditional_training_skips_missing_cv_features(tmp_path, caplog):
    _write_median_json(tmp_path, "LogReg", selected_features=("f0", "not_a_feature"))
    X, y_by_horizon = _fake_data()
    pipeline = FakeFoldPipeline(X, y_by_horizon, ["f0", "f1", "f2", "f3"])

    with caplog.at_level("WARNING"):
        run_temporal_evaluation_training(_config([0]), pipeline, ModelFactory(), tmp_path)

    summary = json.loads(
        (tmp_path / "Results" / "Temporal" / "Complete_Explicit" / "LogReg" / "temporal_summary.json").read_text()
    )
    assert summary["num_features"] == 1
    assert "Skipping 1 of 2 CV-selected features" in caplog.text


def test_traditional_training_errors_when_no_cv_features_present(tmp_path):
    _write_median_json(tmp_path, "LogReg", selected_features=("missing_a", "missing_b"))
    X, y_by_horizon = _fake_data()
    pipeline = FakeFoldPipeline(X, y_by_horizon, ["f0", "f1", "f2", "f3"])

    with pytest.raises(ValueError, match="None of the 2 CV-selected features"):
        run_temporal_evaluation_training(_config([0]), pipeline, ModelFactory(), tmp_path)


def test_advanced_training_loads_each_fold_once_and_evaluates_every_horizon(tmp_path, monkeypatch):
	_write_median_json(tmp_path, "LSTM")
	X, y_by_horizon = _fake_data()
	pipeline = FakeFoldPipeline(X, y_by_horizon, ["a", "b"])
	fold_calls = []

	def fake_advanced_fold(proto_model, model_factory, best_params, X_train, X_test, y_train, y_test, features,
						   horizon, kfold_id, *args):
		fold_calls.append((kfold_id, horizon))
		return {
			"fold": kfold_id, "horizon": horizon,
			"metrics": {key: 0.5 for key in ("accuracy", "precision", "recall", "f1", "specificity", "g_mean")},
			"n_train": len(y_train), "n_test": len(y_test), "n_test_positive": 1,
		}

	monkeypatch.setattr(temporal_evaluation, "_run_advanced_fold", fake_advanced_fold)

	run_temporal_evaluation_training(_config([0, 2], models=("LSTM",)), pipeline, ModelFactory(), tmp_path)

	assert [call["kfold_id"] for call in pipeline.calls] == [0, 1, 2, 3, 4]
	assert all("traditional_feature_selection" not in call for call in pipeline.calls)
	assert len(fold_calls) == 10

	summary = json.loads(
		(tmp_path / "Results" / "Temporal" / "Complete_Explicit" / "LSTM" / "temporal_summary.json").read_text()
	)
	assert summary["pipeline"] == "advanced"
	assert summary["results"]["2"]["n_total"] == 300
	assert summary["results"]["0"]["f1_mean"] == 0.5


def test_review_writes_metric_and_comparison_plots(tmp_path):
	for model_name in ("LogReg", "LDA"):
		_write_median_json(tmp_path, model_name)
	X, y_by_horizon = _fake_data()
	pipeline = FakeFoldPipeline(X, y_by_horizon, ["f0", "f1", "f2", "f3"])
	config = _config([0, 2], models=("LogReg", "LDA"))

	run_temporal_evaluation_training(config, pipeline, ModelFactory(), tmp_path)
	run_temporal_evaluation_review(config, tmp_path)

	results_root = tmp_path / "Results" / "Temporal" / "Complete_Explicit"
	assert (results_root / "LogReg" / "metrics_vs_horizon.png").exists()
	assert (results_root / "LDA" / "metrics_vs_horizon.png").exists()
	assert (results_root / "f1_vs_horizon_all_models.png").exists()


def test_review_errors_when_summary_missing(tmp_path):
	with pytest.raises(FileNotFoundError, match="temporal_summary.json"):
		run_temporal_evaluation_review(_config([0]), tmp_path)