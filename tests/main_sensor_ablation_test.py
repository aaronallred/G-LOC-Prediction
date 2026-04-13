import numpy as np

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
    def __init__(self, enabled: bool, model):
        self._enabled = enabled
        self._model = model

    def get_sensor_ablation_enabled(self):
        return self._enabled

    def get_sensor_ablation_streams(self):
        return [["EEG"]] if self._enabled else [["EEG"]]

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
    monkeypatch.setattr(main_module, "get_model_subfolder", lambda model_type: "Complete_Explicit")
    monkeypatch.setattr(main_module, "get_hyperparameters_from_json", lambda *args, **kwargs: ({"n_neighbors": 5}, None, None, None))
    monkeypatch.setattr(
        main_module,
        "stratified_kfold_split",
        lambda y, x, num_splits, kfold_id, random_state: (
            np.array([[1.0, 2.0]]),
            np.array([[3.0, 4.0]]),
            np.array([0]),
            np.array([1]),
        ),
    )

    main_module.run()

    assert len(pipeline.calls) == 1
    assert "kfold_id" not in pipeline.calls[0]
    assert len(model.calls) == 2
    assert model.calls[0]["model_name"] == "knn_feature_study.pkl"
    assert model.calls[0]["temporal"] is True
    assert model.calls[0]["best_params"] == {"n_neighbors": 5}


def test_run_keeps_dimension_only_flow_when_sensor_ablation_disabled(monkeypatch):
    model = DummyModel("RF")
    config_parser = DummyConfigParser(enabled=False, model=model)
    pipeline = DummyPipeline(config_parser)

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "GLOCExperimentConfigParser", lambda config_location=None: config_parser)
    monkeypatch.setattr(main_module, "DataPipeline", lambda config_parser: pipeline)

    main_module.run()

    assert len(pipeline.calls) == 2
    assert all("kfold_id" in call for call in pipeline.calls)
    assert len(model.calls) == 0
