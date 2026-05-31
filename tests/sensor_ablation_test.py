import pickle
from pathlib import Path

import numpy as np
import pytest

from src.model_type import ModelType
from src.models.base import ModelInitStrategy
from src.modes import sensor_ablation as sensor_ablation_mod
from src.modes.sensor_ablation import (
    apply_stream_label_aliases,
    build_ranked_sensor_ablation_review_results,
    filter_sensor_ablation_review_results,
    load_sensor_ablation_f1_results,
    run_sensor_ablation_review,
    run_sensor_ablation_training,
    save_model_stream_f1_scores,
)


class TinyTraditionalModel:
    def __init__(self, name: str = "TinyKNN") -> None:
        self._name = name
        self.is_traditional = True
        self.calls = []

    def get_name(self) -> str:
        return self._name

    def classify_traditional(
        self,
        x_train,
        x_test,
        y_train,
        y_test,
        class_weight_imb,
        random_state,
        save_folder,
        model_name,
        strategy=ModelInitStrategy.RETRAIN_WITH_DEFAULTS,
        best_params=None,
    ):
        self.calls.append(
            {
                "x_train": x_train,
                "x_test": x_test,
                "y_train": y_train,
                "y_test": y_test,
                "class_weight_imb": class_weight_imb,
                "random_state": random_state,
                "save_folder": save_folder,
                "model_name": model_name,
                "strategy": strategy,
                "best_params": best_params,
            }
        )
        return (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed

    def set_model_type(self, model_type):
        self.model_type = model_type

    def get_data(self, model=None, feature_streams=None):
        self.calls.append({"model": model.get_name(), "feature_streams": list(feature_streams or [])})
        return np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1, 0, 1])


def _make_config(tmp_path: Path, review: bool = False, sort_streams_by_median: bool = False) -> dict:
    return {
        "data_path": str(tmp_path / "data"),
        "shared_data_parameters": {
            "subject_to_analyze": None,
            "trial_to_analyze": None,
            "analysis_type": 2,
            "remove_NaN_trials": True,
            "impute_file_name": "imputed.pkl",
            "save_impute": False,
            "load_impute": False,
            "impute_phase": "pre_feature",
            "output_feature_dtype": "float32",
        },
        "advanced_data_parameters": {"n_neighbors": 4, "baseline_window": 32.5},
        "traditional_data_parameters": {"backstep": 0, "data_rate": 25, "offset": 0, "time_start": 0},
        "sensor_ablation": {
            "training": {
                "enabled": True,
                "save_results_folder": str(tmp_path / "Results" / "Sensor_Ablation"),
                "models": ["KNN"],
                "model_type": ModelType("Complete", "Explicit"),
                "random_seed": 13,
                "num_splits": 2,
                "median_hyperparameters_folder": "ModelSave/CV",
                "streams": [["ECG"], ["EEG"]],
            },
            "review": {
                "enabled": review,
                "save_results_folder": str(tmp_path / "Results" / "Sensor_Ablation"),
                "models": ["KNN"],
                "model_type": ModelType("Complete", "Explicit"),
                "stream_group": ["EEG"],
                "sort_streams_by_median": sort_streams_by_median,
            },
        },
    }


def test_sensor_ablation_config_uses_model_type_objects(tmp_path):
    config = _make_config(tmp_path, review=True)

    assert config["sensor_ablation"]["training"]["model_type"] == ModelType("Complete", "Explicit")
    assert config["sensor_ablation"]["review"]["model_type"] == ModelType("Complete", "Explicit")
    assert config["sensor_ablation"]["review"]["stream_group"] == ["EEG"]
    assert config["sensor_ablation"]["review"]["sort_streams_by_median"] is False


def test_run_sensor_ablation_training_saves_stream_scores_and_calls_plotter(tmp_path):
    config = _make_config(tmp_path)
    pipeline = FakePipeline()
    plot_calls = []
    split_calls = []

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        sensor_ablation_mod.ModelFactory,
        "create_model",
        staticmethod(lambda model_name, model_hyperparameters=None: TinyTraditionalModel("TinyKNN")),
    )

    def get_hyperparameters_from_json_fn(model_name, model_type_name):
        assert model_name == "TinyKNN"
        assert model_type_name == "Complete_Explicit"
        return ({"k": 3}, [], [], [])

    def stratified_kfold_split_fn(**kwargs):
        split_calls.append(kwargs)
        x = np.asarray(kwargs["X"])
        y = np.asarray(kwargs["Y"])
        midpoint = len(y) // 2
        return x[:midpoint], x[midpoint:], y[:midpoint], y[midpoint:]

    def plot_f1_violin_by_stream_fn(**kwargs):
        plot_calls.append(kwargs)

    run_sensor_ablation_training(
        config=config,
        pipeline=pipeline,
        project_root=tmp_path,
        get_hyperparameters_from_json_fn=get_hyperparameters_from_json_fn,
        stratified_kfold_split_fn=stratified_kfold_split_fn,
        plot_f1_violin_by_stream_fn=plot_f1_violin_by_stream_fn,
        save_model_stream_f1_scores_fn=save_model_stream_f1_scores,
    )

    monkeypatch.undo()

    assert len(split_calls) == 4
    assert pipeline.calls == [
        {"model": "TinyKNN", "feature_streams": ["ECG"]},
        {"model": "TinyKNN", "feature_streams": ["EEG"]},
    ]
    assert plot_calls and "TinyKNN" in plot_calls[0]["f1_results_by_stream"]
    assert (tmp_path / "Results" / "Sensor_Ablation" / "Complete_Explicit" / "TinyKNN" / "ECG.pkl").exists()
    assert (tmp_path / "Results" / "Sensor_Ablation" / "Complete_Explicit" / "TinyKNN" / "EEG.pkl").exists()


def test_sensor_ablation_review_filters_and_ranks_cached_results(tmp_path):
    results_root = tmp_path / "Results" / "Sensor_Ablation" / "Complete_Explicit"
    (results_root / "KNN").mkdir(parents=True)
    (results_root / "RF").mkdir(parents=True)
    with open(results_root / "KNN" / "EEG.pkl", "wb") as handle:
        pickle.dump(np.asarray([0.5, 0.7]), handle)
    with open(results_root / "RF" / "EEG.pkl", "wb") as handle:
        pickle.dump(np.asarray([0.6, 0.8]), handle)
    with open(results_root / "KNN" / "ECG-HR-BR-Temperature.pkl", "wb") as handle:
        pickle.dump(np.asarray([0.2, 0.4]), handle)
    with open(results_root / "RF" / "ECG-HR-BR-Temperature.pkl", "wb") as handle:
        pickle.dump(np.asarray([0.3, 0.9]), handle)

    loaded = load_sensor_ablation_f1_results(results_root, ["KNN", "RF"], stream_group=["EEG"])
    assert set(loaded.keys()) == {"KNN", "RF"}
    filtered = filter_sensor_ablation_review_results(loaded, ["EEG"])
    assert set(filtered["KNN"].keys()) == {"EEG"}

    ranked = build_ranked_sensor_ablation_review_results(
        load_sensor_ablation_f1_results(results_root, ["KNN", "RF"]),
        classifiers=["KNN", "RF"],
        stream_label_aliases={
            "ECG-HR-BR-Temperature": "Equivital",
            "Participant": "Demographics",
            "Centrifuge": "G Force",
        },
    )
    assert "Equivital" in ranked["KNN"]


def test_run_sensor_ablation_review_uses_stream_group_filter_branch(tmp_path):
    results_root = tmp_path / "Results" / "Sensor_Ablation" / "Complete_Explicit"
    (results_root / "KNN").mkdir(parents=True)
    with open(results_root / "KNN" / "EEG.pkl", "wb") as handle:
        pickle.dump(np.asarray([0.5, 0.7]), handle)

    config = _make_config(tmp_path)
    plot_calls = []

    run_sensor_ablation_review(
        config=config,
        project_root=tmp_path,
        load_sensor_ablation_f1_results_fn=load_sensor_ablation_f1_results,
        build_ranked_sensor_ablation_review_results_fn=build_ranked_sensor_ablation_review_results,
        filter_sensor_ablation_review_results_fn=filter_sensor_ablation_review_results,
        plot_f1_violin_with_stream_matrix_fn=lambda **kwargs: plot_calls.append(("matrix", kwargs)),
        plot_f1_violin_by_stream_fn=lambda **kwargs: plot_calls.append(("by_stream", kwargs)),
        stream_label_aliases={
            "ECG-HR-BR-Temperature": "Equivital",
            "Participant": "Demographics",
            "Centrifuge": "G Force",
        },
    )

    assert plot_calls and plot_calls[0][0] == "by_stream"


def test_run_sensor_ablation_review_uses_ranked_branch(tmp_path):
    results_root = tmp_path / "Results" / "Sensor_Ablation" / "Complete_Explicit"
    (results_root / "KNN").mkdir(parents=True)
    with open(results_root / "KNN" / "ECG-HR-BR-Temperature.pkl", "wb") as handle:
        pickle.dump(np.asarray([0.5, 0.8]), handle)
    with open(results_root / "KNN" / "Participant.pkl", "wb") as handle:
        pickle.dump(np.asarray([0.4, 0.6]), handle)

    config = _make_config(tmp_path)
    config["sensor_ablation"]["review"]["sort_streams_by_median"] = True
    config["sensor_ablation"]["review"]["stream_group"] = []
    plot_calls = []

    run_sensor_ablation_review(
        config=config,
        project_root=tmp_path,
        load_sensor_ablation_f1_results_fn=load_sensor_ablation_f1_results,
        build_ranked_sensor_ablation_review_results_fn=build_ranked_sensor_ablation_review_results,
        filter_sensor_ablation_review_results_fn=filter_sensor_ablation_review_results,
        plot_f1_violin_with_stream_matrix_fn=lambda **kwargs: plot_calls.append(("matrix", kwargs)),
        plot_f1_violin_by_stream_fn=lambda **kwargs: plot_calls.append(("by_stream", kwargs)),
        stream_label_aliases={
            "ECG-HR-BR-Temperature": "Equivital",
            "Participant": "Demographics",
            "Centrifuge": "G Force",
        },
    )

    assert plot_calls and plot_calls[0][0] == "matrix"


def test_sensor_ablation_training_requires_median_hyperparameters_folder(tmp_path):
    config = _make_config(tmp_path)
    del config["sensor_ablation"]["training"]["median_hyperparameters_folder"]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        sensor_ablation_mod.ModelFactory,
        "create_model",
        staticmethod(lambda model_name, model_hyperparameters=None: TinyTraditionalModel("TinyKNN")),
    )

    with pytest.raises(KeyError):
        run_sensor_ablation_training(
            config=config,
            pipeline=FakePipeline(),
            project_root=tmp_path,
            get_hyperparameters_from_json_fn=None,
            stratified_kfold_split_fn=lambda **kwargs: (np.array([[1.0]]), np.array([[2.0]]), np.array([0]), np.array([1])),
            plot_f1_violin_by_stream_fn=lambda **kwargs: None,
            save_model_stream_f1_scores_fn=save_model_stream_f1_scores,
        )

    monkeypatch.undo()


def test_run_sensor_ablation_training_uses_parser_median_folder(tmp_path):
    # Prepare a temporary median hyperparameters folder and JSON for TinyKNN
    model_type_folder = "Complete_Explicit"
    median_base = tmp_path / "ModelSave" / "CV"
    target_folder = median_base / model_type_folder / "TinyKNN"
    target_folder.mkdir(parents=True, exist_ok=True)

    median_file = target_folder / "median_hyperparameters.json"
    median_file.write_text(
        '{"best_params": {"k": 5}, "selected_features": [], "fold_id": 0, "f1_score": 0.5}',
        encoding="utf-8",
    )

    config = _make_config(tmp_path)
    config["sensor_ablation"]["training"]["median_hyperparameters_folder"] = str(median_base)
    pipeline = FakePipeline()
    plot_calls = []
    split_calls = []

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        sensor_ablation_mod.ModelFactory,
        "create_model",
        staticmethod(lambda model_name, model_hyperparameters=None: TinyTraditionalModel("TinyKNN")),
    )

    def stratified_kfold_split_fn(**kwargs):
        split_calls.append(kwargs)
        x = np.asarray(kwargs["X"])
        y = np.asarray(kwargs["Y"])
        midpoint = len(y) // 2
        return x[:midpoint], x[midpoint:], y[:midpoint], y[midpoint:]

    def plot_f1_violin_by_stream_fn(**kwargs):
        plot_calls.append(kwargs)

    # Call with get_hyperparameters_from_json_fn=None so default resolver is used
    run_sensor_ablation_training(
        config=config,
        pipeline=pipeline,
        project_root=tmp_path,
        get_hyperparameters_from_json_fn=None,
        stratified_kfold_split_fn=stratified_kfold_split_fn,
        plot_f1_violin_by_stream_fn=plot_f1_violin_by_stream_fn,
        save_model_stream_f1_scores_fn=save_model_stream_f1_scores,
    )

    monkeypatch.undo()

    # Verify that splits were run and outputs created
    assert len(split_calls) == 4
    assert (tmp_path / "Results" / "Sensor_Ablation" / "Complete_Explicit" / "TinyKNN" / "ECG.pkl").exists()
    assert (tmp_path / "Results" / "Sensor_Ablation" / "Complete_Explicit" / "TinyKNN" / "EEG.pkl").exists()
