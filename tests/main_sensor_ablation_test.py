import numpy as np
import pickle
from pathlib import Path
import builtins

from src.model_type import ModelType
import src.main as main_module


class DummyModel:
    def __init__(self, name: str):
        self._name = name
        self.calls = []

    def get_name(self):
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
        retrain,
        temporal=False,
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
                "retrain": retrain,
                "temporal": temporal,
                "best_params": best_params,
            }
        )
        return (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


class DummyConfigParser:
    def __init__(self, enabled: bool, model, review_enabled: bool = False, review_models=None, review_stream_group=None):
        self._enabled = enabled
        self._model = model
        self._review_enabled = review_enabled
        self._review_models = [] if review_models is None else review_models
        self._review_stream_group = [] if review_stream_group is None else review_stream_group

    def get_sensor_ablation_enabled(self):
        return self._enabled

    def get_sensor_ablation_streams(self):
        return [["EEG"]] if self._enabled else [["EEG"]]

    def get_sensor_ablation_review_enabled(self):
        return self._review_enabled

    def get_sensor_ablation_review_models(self):
        return list(self._review_models)

    def get_sensor_ablation_review_stream_group(self):
        return list(self._review_stream_group)

    def get_models(self):
        return [self._model]

    def get_num_splits(self):
        return 2

    def get_model_type(self):
        return ModelType("Complete", "Explicit")

    def get_random_seed(self):
        return 42

    def get_data_path(self):
        return "/tmp/data"

    def get_remove_NaN_trials(self):
        return True

    def get_subject_to_analyze(self):
        return None

    def get_trial_to_analyze(self):
        return None

    def get_analysis_type(self):
        return 2

    def get_output_feature_dtype(self):
        return np.dtype("float32")

    def get_impute_file_name(self):
        return "imputed.pkl"

    def get_should_impute(self):
        return True

    def get_save_impute(self):
        return False

    def get_load_impute(self):
        return False

    def get_n_neighbors(self):
        return 4

    def get_baseline_window(self):
        return 32.5

    def get_backstep(self):
        return 0

    def get_data_rate(self):
        return 25

    def get_offset(self):
        return 0

    def get_time_start(self):
        return 0


class DummyPipeline:
    def __init__(self, config_parser):
        self.config_parser = config_parser
        self.calls = []

    def get_data(self, **kwargs):
        self.calls.append(kwargs)
        return (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([0, 1]),
        )


def test_run_uses_traditional_classification_when_sensor_ablation_enabled(monkeypatch):
    model = DummyModel("KNN")
    config_parser = DummyConfigParser(enabled=True, model=model)
    pipeline = DummyPipeline(config_parser)

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "GLOCExperimentConfigParser", lambda config_location=None: config_parser)
    monkeypatch.setattr(main_module, "DataPipeline", lambda config_parser: pipeline)
    monkeypatch.setattr(main_module, "get_hyperparameters_from_json", lambda *args, **kwargs: ({"n_neighbors": 5}, None, None, None))
    monkeypatch.setattr(main_module, "plot_f1_violin_by_stream", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main_module,
        "stratified_kfold_split",
        lambda Y, X, num_splits, kfold_ID, random_state: (
            np.array([[1.0, 2.0]]),
            np.array([[3.0, 4.0]]),
            np.array([0]),
            np.array([1]),
        ),
    )

    main_module.run()

    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["feature_streams"] == ["EEG"]
    assert len(model.calls) == 2
    assert model.calls[0]["model_name"] == "knn_feature_study.pkl"
    assert model.calls[0]["temporal"] is True
    assert model.calls[0]["best_params"] == {"n_neighbors": 5}


def test_run_no_enabled_modes_logs_and_exits(monkeypatch):
    model = DummyModel("RF")
    config_parser = DummyConfigParser(enabled=False, model=model, review_enabled=False)

    logs = []

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "GLOCExperimentConfigParser", lambda config_location=None: config_parser)
    monkeypatch.setattr(main_module.logging, "info", lambda msg, *args: logs.append(msg % args if args else msg))

    main_module.run()

    assert any("No runnable mode enabled" in message for message in logs)


def test_load_sensor_ablation_f1_results_reads_pickles(tmp_path):
    results_root_dir = tmp_path / "Sensor_Ablation" / "Complete_Explicit"
    knn_dir = results_root_dir / "KNN"
    rf_dir = results_root_dir / "RF"
    knn_dir.mkdir(parents=True)
    rf_dir.mkdir(parents=True)

    with open(knn_dir / "EEG.pkl", "wb") as handle:
        pickle.dump(np.array([0.1, 0.2]), handle)
    with open(rf_dir / "ECG-HR-BR.pkl", "wb") as handle:
        pickle.dump(np.array([0.9]), handle)

    loaded = main_module._load_sensor_ablation_f1_results(
        results_root_dir=results_root_dir,
        classifiers=["KNN", "RF"],
    )

    assert set(loaded.keys()) == {"KNN", "RF"}
    assert np.allclose(loaded["KNN"]["EEG"], np.array([0.1, 0.2]))
    assert np.allclose(loaded["RF"]["ECG-HR-BR"], np.array([0.9]))


def test_load_sensor_ablation_f1_results_skips_non_matching_pickles(monkeypatch, tmp_path):
    results_root_dir = tmp_path / "Sensor_Ablation" / "Complete_Explicit"
    knn_dir = results_root_dir / "KNN"
    knn_dir.mkdir(parents=True)

    with open(knn_dir / "EEG.pkl", "wb") as handle:
        pickle.dump(np.array([0.1, 0.2]), handle)
    with open(knn_dir / "Pupil.pkl", "wb") as handle:
        pickle.dump(np.array([0.3]), handle)

    opened_paths = []
    original_open = open

    def _tracking_open(path, mode="r", *args, **kwargs):
        if "rb" in mode:
            opened_paths.append(Path(path).name)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _tracking_open)

    loaded = main_module._load_sensor_ablation_f1_results(
        results_root_dir=results_root_dir,
        classifiers=["KNN"],
        stream_group=["EEG"],
    )

    assert set(opened_paths) == {"EEG.pkl"}
    assert set(loaded["KNN"].keys()) == {"EEG"}


def test_filter_sensor_ablation_review_results_matches_selected_stream_labels():
    f1_results_by_stream = {
        "KNN": {
            "EEG": np.array([0.5]),
            "ECG-HR-BR": np.array([0.6]),
            "EEG-Pupil": np.array([0.7]),
            "Pupil-Centrifuge-Participant": np.array([0.8]),
        }
    }
    filtered = main_module._filter_sensor_ablation_review_results(
        f1_results_by_stream=f1_results_by_stream,
        stream_group=["Pupil", "Centrifuge", "Participant"],
    )

    assert set(filtered["KNN"].keys()) == {"Pupil-Centrifuge-Participant"}


def test_run_sensor_ablation_review_enabled_invokes_plot_with_filtered_results(monkeypatch):
    model = DummyModel("KNN")
    config_parser = DummyConfigParser(
        enabled=False,
        model=model,
        review_enabled=True,
        review_models=["KNN"],
        review_stream_group=["EEG"],
    )

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "GLOCExperimentConfigParser", lambda config_location=None: config_parser)

    monkeypatch.setattr(
        main_module,
        "_load_sensor_ablation_f1_results",
        lambda results_root_dir, classifiers, stream_group=None: {
            "KNN": {
                "EEG": np.array([0.91, 0.92]),
                "Pupil": np.array([0.51]),
            }
        },
    )

    captured = {}

    def _capture_plot(f1_results_by_stream, model_type, save_folder=None):
        captured["results"] = f1_results_by_stream
        captured["model_type"] = model_type
        captured["save_folder"] = save_folder

    monkeypatch.setattr(main_module, "plot_f1_violin_by_stream", _capture_plot)

    main_module.run()

    assert set(captured["results"].keys()) == {"KNN"}
    assert set(captured["results"]["KNN"].keys()) == {"EEG"}
    assert captured["save_folder"] is None
