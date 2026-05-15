import pytest
import yaml

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.GLOC_experiment_config_parser import GLOCExperimentConfigParser
from src.model_type import ModelType


class DummyModel:
    def __init__(self, *, is_traditional: bool, name: str = "RF") -> None:
        self.is_traditional = is_traditional
        self._name = name

    def get_name(self) -> str:
        return self._name


class DummyConfigParser:
    def get_data_path(self):
        return "/tmp/data"

    def get_remove_NaN_trials(self):
        return True

    def get_subject_to_analyze(self):
        return "01"

    def get_trial_to_analyze(self):
        return "02"

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
        return 3

    def get_data_rate(self):
        return 25

    def get_offset(self):
        return 0.0

    def get_time_start(self):
        return 0.0


class CapturingBackend:
    def __init__(self):
        self.calls = []

    def get_data(self, **kwargs):
        self.calls.append(kwargs)
        return "backend-result"


class SentinelCVBackend:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    pass


class SentinelEvalBackend:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
    pass


def _make_pipeline() -> DataPipeline:
    pipeline = DataPipeline(DummyConfigParser())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))
    return pipeline


def test_cross_validation_mode_routes_traditional_models_without_sensor_ablation(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.set_mode("cross_validation")

    backend = CapturingBackend()
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)
    monkeypatch.setattr(pipeline, "_resolve_select_features", lambda _kwargs: pytest.fail("should not be called"))
    monkeypatch.setattr(pipeline, "_apply_sensor_ablation", lambda _features, _streams: pytest.fail("should not be called"))

    result = pipeline.get_data(
        model=DummyModel(is_traditional=True),
        feature_streams=["EEG", "participant"],
    )

    assert result == "backend-result"
    assert backend.calls[0]["model_type"] == ModelType("Complete", "Explicit")
    assert backend.calls[0]["remove_NaN_trials"] is True
    assert backend.calls[0]["subject_to_analyze"] == "01"
    assert backend.calls[0]["trial_to_analyze"] == "02"
    assert backend.calls[0]["analysis_type"] == 2
    assert backend.calls[0]["output_feature_dtype"] == "float32"
    assert backend.calls[0]["impute_file_name"] == "imputed.pkl"
    assert backend.calls[0]["should_impute"] is True
    assert backend.calls[0]["save_impute"] is False
    assert backend.calls[0]["load_impute"] is False
    assert backend.calls[0]["classifier_type"] == "RF"
    assert backend.calls[0]["model"].get_name() == "RF"


def test_evaluation_mode_still_applies_sensor_ablation(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.set_mode(None)

    backend = CapturingBackend()
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)
    monkeypatch.setattr(
        pipeline,
        "_resolve_select_features",
        lambda _kwargs: ["magnitude - Centrifuge", "participant_age", "Fz_alpha - EEG"],
    )

    result = pipeline.get_data(
        model=DummyModel(is_traditional=True),
        feature_streams=["g force", "participant", "EEG"],
    )

    assert result == "backend-result"
    assert backend.calls[0]["classifier_type"] == "RF"
    assert backend.calls[0]["select_features"] == ["magnitude - Centrifuge", "participant_age", "Fz_alpha - EEG"]
    assert backend.calls[0]["backstep"] == 3
    assert backend.calls[0]["data_rate"] == 25
    assert backend.calls[0]["offset"] == 0.0
    assert backend.calls[0]["time_start"] == 0.0


def test_build_backend_switches_on_mode(monkeypatch):
    pipeline = _make_pipeline()
    monkeypatch.setattr("src.Data_Pipeline.data_pipeline.TraditionalDataPipelineCV", SentinelCVBackend)
    monkeypatch.setattr("src.Data_Pipeline.data_pipeline.TraditionalDataPipelineEvaluation", SentinelEvalBackend)

    pipeline.set_mode("cross_validation")
    assert isinstance(pipeline._build_backend(DummyModel(is_traditional=True)), SentinelCVBackend)

    pipeline.set_mode(None)
    assert isinstance(pipeline._build_backend(DummyModel(is_traditional=True)), SentinelEvalBackend)


def test_get_active_mode_prefers_cross_validation(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cross_validation": {"enabled": True},
                "sensor_ablation": {"training": {"enabled": True}, "review": {"enabled": False}},
                "feature_space_review": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    parser = GLOCExperimentConfigParser(config_location=str(config_path))
    assert parser.get_active_mode() == "cross_validation"


def test_get_active_mode_falls_back_in_order(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cross_validation": {"enabled": False},
                "sensor_ablation": {"training": {"enabled": True}, "review": {"enabled": False}},
                "feature_space_review": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    parser = GLOCExperimentConfigParser(config_location=str(config_path))
    assert parser.get_active_mode() == "sensor_ablation"
