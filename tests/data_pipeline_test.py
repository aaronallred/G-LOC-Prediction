import pytest

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.Data_Pipeline.imputation_config import ImputePhase
from src.model_type import ModelType


class DummyModel:
    def __init__(self, *, is_traditional: bool, name: str) -> None:
        self.is_traditional = is_traditional
        self.is_traditional_model = is_traditional
        self.name = name
        self._name = name

    def get_name(self) -> str:
        return self._name


def _make_config() -> dict:
    return {
        "data_path": "/tmp/data",
        "shared_data_parameters": {
            "subject_to_analyze": None,
            "trial_to_analyze": None,
            "analysis_type": 2,
            "remove_NaN_trials": True,
            "impute_file_name": "imputed.pkl",
            "save_impute": False,
            "load_impute": False,
            "impute_phase": ImputePhase.PRE_FEATURE,
            "output_feature_dtype": "float32",
        },
        "advanced_data_parameters": {"n_neighbors": 4, "baseline_window": 32.5, "horizon": 0},
        "traditional_data_parameters": {"backstep": 0, "data_rate": 25, "offset": 0, "time_start": 0},
        "sensor_ablation": {
            "training": {
                "median_hyperparameters_folder": "ModelSave/CV",
            }
        },
    }


class CapturingBackend:
    def __init__(self, return_value):
        self.return_value = return_value
        self.calls = []

    def get_data(self, **kwargs):
        self.calls.append(kwargs)
        return self.return_value


def test_resolve_pipeline_kind_uses_model_flag():
    pipeline = DataPipeline(_make_config())
    assert pipeline._resolve_pipeline_kind(DummyModel(is_traditional=True, name="RF")) == "traditional"
    assert pipeline._resolve_pipeline_kind(DummyModel(is_traditional=False, name="Trans")) == "advanced"


def test_get_data_requires_model_type():
    pipeline = DataPipeline(_make_config())
    with pytest.raises(ValueError, match="model_type must be set"):
        pipeline.get_data(model=DummyModel(is_traditional=True, name="RF"))


def test_get_data_for_advanced_model_forwards_fold_arguments(monkeypatch):
    pipeline = DataPipeline(_make_config())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))
    pipeline.set_random_seed(7)

    backend = CapturingBackend("advanced-ok")
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)

    result = pipeline.get_data(model=DummyModel(is_traditional=False, name="Trans"), kfold_id=3, num_splits=5)

    assert result == "advanced-ok"
    assert backend.calls == [
        {
            "model_type": ModelType("Complete", "Explicit"),
            "remove_NaN_trials": True,
            "subject_to_analyze": None,
            "trial_to_analyze": None,
            "analysis_type": 2,
            "output_feature_dtype": "float32",
            "impute_file_name": "imputed.pkl",
            "impute_phase": ImputePhase.PRE_FEATURE,
            "save_impute": False,
            "load_impute": False,
            "kfold_ID": 3,
            "num_splits": 5,
            "n_neighbors": 4,
            "baseline_window": 32.5,
            "horizon": 0,
        }
    ]


def test_get_data_for_traditional_model_applies_sensor_ablation(monkeypatch):
    pipeline = DataPipeline(_make_config())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))

    backend = CapturingBackend("traditional-ok")
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)
    monkeypatch.setattr(
        pipeline,
        "_resolve_select_features",
        lambda _kwargs: [
            "HR (bpm) - Equivital",
            "Pupil diameter left [mm] - Tobii",
            "magnitude - Centrifuge",
            "participant_age",
            "Fz_alpha - EEG",
        ],
    )

    result = pipeline.get_data(
        model=DummyModel(is_traditional=True, name="RF"),
        feature_streams=["g force", "participant", "EEG"],
    )

    assert result == "traditional-ok"
    assert backend.calls[0]["model"].get_name() == "RF"
    assert backend.calls[0]["classifier_type"] == "RF"
    assert backend.calls[0]["select_features"] == [
        "magnitude - Centrifuge",
        "participant_age",
        "Fz_alpha - EEG",
    ]


def test_get_data_for_traditional_model_can_return_raw_features_without_cache_lookup(monkeypatch):
    pipeline = DataPipeline(_make_config())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))

    backend = CapturingBackend(("raw-ok", ["f1", "f2", "f3"]))
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)
    monkeypatch.setattr(
        pipeline,
        "_resolve_select_features",
        lambda _kwargs: pytest.fail("traditional CV should not read cached median hyperparameters"),
    )

    result = pipeline.get_data(
        model=DummyModel(is_traditional=True, name="RF"),
        traditional_feature_selection="raw",
        return_feature_names=True,
    )

    assert result == ("raw-ok", ["f1", "f2", "f3"])
    assert backend.calls[0]["classifier_type"] == "RF"
    assert "select_features" not in backend.calls[0]


def test_apply_sensor_ablation_rejects_unknown_stream():
    pipeline = DataPipeline(_make_config())
    with pytest.raises(ValueError, match="Unknown stream\\(s\\)"):
        pipeline._apply_sensor_ablation(["Fz_alpha - EEG"], ["mystery-stream"])


def test_standardize_flag_is_forwarded_to_backend(monkeypatch):
    pipeline = DataPipeline(_make_config())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))

    backend = CapturingBackend("ok")
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)
    monkeypatch.setattr(pipeline, "_resolve_select_features", lambda _kwargs: [])

    pipeline.get_data(model=DummyModel(is_traditional=True, name="RF"))
    assert backend.calls[0]["standardize"] is True


def test_standardize_flag_is_forwarded_to_backend_false(monkeypatch):
    config = _make_config()
    config["traditional_data_parameters"]["standardize"] = False
    pipeline = DataPipeline(config)
    pipeline.set_model_type(ModelType("Complete", "Explicit"))

    backend = CapturingBackend("ok")
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)
    monkeypatch.setattr(pipeline, "_resolve_select_features", lambda _kwargs: [])

    pipeline.get_data(model=DummyModel(is_traditional=True, name="RF"))
    assert backend.calls[0]["standardize"] is False


def test_standardize_flag_defaults_true_when_omitted(monkeypatch):
    config = _make_config()
    assert "standardize" not in config["traditional_data_parameters"]

    pipeline = DataPipeline(config)
    pipeline.set_model_type(ModelType("Complete", "Explicit"))

    backend = CapturingBackend("ok")
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)
    monkeypatch.setattr(pipeline, "_resolve_select_features", lambda _kwargs: [])

    pipeline.get_data(model=DummyModel(is_traditional=True, name="RF"))
    assert backend.calls[0]["standardize"] is True


def test_advanced_backend_does_not_receive_standardize_flag(monkeypatch):
    pipeline = DataPipeline(_make_config())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))

    backend = CapturingBackend("advanced-ok")
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)

    pipeline.get_data(model=DummyModel(is_traditional=False, name="Trans"), kfold_id=3, num_splits=5)
    assert "standardize" not in backend.calls[0]