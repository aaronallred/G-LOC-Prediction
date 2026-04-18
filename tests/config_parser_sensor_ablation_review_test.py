import yaml
import pytest

from src.GLOC_experiment_config_parser import GLOCExperimentConfigParser


def _base_config_dict():
    return {
        "models": ["KNN"],
        "model_type": ["Complete", "Explicit"],
        "random_seed": 42,
        "data_path": "/tmp/data",
        "shared_data_parameters": {
            "subject_to_analyze": None,
            "trial_to_analyze": None,
            "analysis_type": 2,
            "remove_NaN_trials": True,
            "impute_file_name": "imputed_data.pkl",
            "save_impute": False,
            "load_impute": False,
            "should_impute": True,
            "output_feature_dtype": "float32",
            "faiss_index_type": "cpu",
        },
        "advanced_data_parameters": {
            "num_splits": 10,
            "kfold_ID": 0,
            "n_neighbors": 4,
            "baseline_window": 32.5,
        },
        "traditional_data_parameters": {
            "backstep": 0,
            "data_rate": 25,
            "offset": 0,
            "time_start": 0,
        },
        "sensor_ablation_parameters": {
            "enabled": False,
            "streams": [["EEG"]],
        },
    }


def test_parser_reads_yaml_and_parses_sensor_ablation_review_parameters(tmp_path):
    config_dict = _base_config_dict()
    config_dict["sensor_ablation_review_parameters"] = {
        "enabled": True,
        "models": ["KNN"],
        "stream_group": ["EEG", "Pupil"],
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    parser = GLOCExperimentConfigParser(config_location=str(config_path))

    assert parser.get_sensor_ablation_review_enabled() is True
    assert parser.get_sensor_ablation_review_models() == ["KNN"]
    assert parser.get_sensor_ablation_review_stream_group() == ["EEG", "Pupil"]


def test_parser_sensor_ablation_review_defaults_when_not_provided(tmp_path):
    config_dict = _base_config_dict()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    parser = GLOCExperimentConfigParser(config_location=str(config_path))

    assert parser.get_sensor_ablation_review_enabled() is False
    assert parser.get_sensor_ablation_review_models() == []
    assert parser.get_sensor_ablation_review_stream_group() == []


def test_parser_rejects_json_config_extension(tmp_path):
    config_dict = _base_config_dict()

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected .yaml or .yml"):
        GLOCExperimentConfigParser(config_location=str(config_path))


def test_parser_requires_review_models_and_stream_group_when_enabled(tmp_path):
    config_dict = _base_config_dict()
    config_dict["sensor_ablation_review_parameters"] = {
        "enabled": True,
        "models": [],
        "stream_group": [],
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    with pytest.raises(ValueError, match="models must be a non-empty list"):
        GLOCExperimentConfigParser(config_location=str(config_path))
