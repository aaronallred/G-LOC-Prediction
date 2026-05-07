import pytest

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.model_type import ModelType


class DummyModel:
    def __init__(self, *, is_traditional: bool, name: str) -> None:
        self.is_traditional = is_traditional
        self._name = name

    def get_name(self) -> str:
        return self._name


class DummyConfigParser:
    def __init__(self) -> None:
        self._model_type = ModelType("Complete", "Explicit")

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
        return "float32"

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


class CapturingBackend:
    def __init__(self, return_value):
        self.return_value = return_value
        self.calls = []

    def get_data(self, **kwargs):
        self.calls.append(kwargs)
        return self.return_value


def test_resolve_pipeline_kind_uses_model_flag():
    pipeline = DataPipeline(DummyConfigParser())
    assert pipeline._resolve_pipeline_kind(DummyModel(is_traditional=True, name="RF")) == "traditional"
    assert pipeline._resolve_pipeline_kind(DummyModel(is_traditional=False, name="Trans")) == "advanced"


def test_get_data_requires_model_type():
    pipeline = DataPipeline(DummyConfigParser())
    with pytest.raises(ValueError, match="model_type must be set"):
        pipeline.get_data(model=DummyModel(is_traditional=True, name="RF"))


def test_get_data_for_advanced_model_forwards_fold_arguments(monkeypatch):
    pipeline = DataPipeline(DummyConfigParser())
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
            "should_impute": True,
            "save_impute": False,
            "load_impute": False,
            "kfold_ID": 3,
            "num_splits": 5,
            "n_neighbors": 4,
            "baseline_window": 32.5,
        }
    ]


def test_get_data_for_traditional_model_applies_sensor_ablation(monkeypatch):
    pipeline = DataPipeline(DummyConfigParser())
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


def test_apply_sensor_ablation_rejects_unknown_stream():
    pipeline = DataPipeline(DummyConfigParser())
    with pytest.raises(ValueError, match="Unknown stream\\(s\\)"):
        pipeline._apply_sensor_ablation(["Fz_alpha - EEG"], ["mystery-stream"])
