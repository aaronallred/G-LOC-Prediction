"""Tests for real_time_equivital execution mode and latency profiling."""

import json
from pathlib import Path

import numpy as np

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.Data_Pipeline.real_time_data_pipeline import RealTimeTraditionalDataPipeline
from src.model_type import ModelType
from src.models.model_factory import ModelFactory
from src.modes.real_time_equivital import (
    EquivitalDataStreamer,
    _load_single_trial_raw_data,
    run_real_time_equivital,
)


def _build_test_config(tmp_path: Path) -> dict:
    return {
        "data_path": "data_reduced",
        "shared_data_parameters": {
            "subject_to_analyze": None,
            "trial_to_analyze": None,
            "analysis_type": 2,
            "remove_NaN_trials": True,
            "impute_file_name": "imputed_data.pkl",
            "save_impute": False,
            "load_impute": False,
            "impute_phase": "pre_feature",
            "output_feature_dtype": "float32",
        },
        "advanced_data_parameters": {
            "n_neighbors": 4,
            "baseline_window": 32.5,
            "horizon": 0,
        },
        "traditional_data_parameters": {
            "backstep": 0,
            "data_rate": 25,
            "offset": 0,
            "time_start": 0,
            "standardize_s1": True,
        },
        "real_time_equivital": {
            "enabled": True,
            "model_type": ModelType("Complete", "Explicit"),
            "random_seed": 42,
            "models": ["KNN"],
            "num_splits": 5,
            "streams": [["ECG", "Centrifuge"]],
            "manual_ablation": True,
            "saved_models_folder": str(tmp_path / "Saved_Models"),
            "save_results_folder": str(tmp_path / "Results_Latency"),
            "use_real_time_sleep": False,
        },
    }


def test_equivital_data_streamer_push_sample():
    """Test EquivitalDataStreamer initialization and sample pushing."""
    streamer = EquivitalDataStreamer(channel_names=["ch1", "ch2"], stream_rate_hz=25.0)
    sample = np.array([1.0, 2.0], dtype=np.float32)
    # Should not raise even if pylsl outlet is not active
    streamer.push_sample(sample)


def test_load_single_trial_raw_data():
    """Test loading a single continuous raw trial from data."""
    config = {
        "data_path": "data_reduced",
        "shared_data_parameters": {
            "subject_to_analyze": None,
            "trial_to_analyze": None,
            "analysis_type": 2,
            "remove_NaN_trials": True,
            "impute_file_name": "imputed_data.pkl",
            "save_impute": False,
            "load_impute": False,
            "impute_phase": "pre_feature",
            "output_feature_dtype": "float32",
        },
        "traditional_data_parameters": {
            "backstep": 0,
            "data_rate": 25,
            "offset": 0,
            "time_start": 0,
            "standardize_s1": True,
        },
    }
    pipeline = DataPipeline(config=config)
    pipeline.set_random_seed(42)
    model_type = ModelType("Complete", "Explicit")
    pipeline.set_model_type(model_type)

    model_factory = ModelFactory()
    model = model_factory.create_model("KNN")

    raw_samples, timestamps, raw_channel_names, trial_id = _load_single_trial_raw_data(
        pipeline=pipeline,
        model_instance=model,
        model_type=model_type,
        stream_group=["ECG", "Centrifuge"],
    )

    assert raw_samples.ndim == 2
    assert raw_samples.shape[0] > 0
    assert raw_samples.shape[1] == len(raw_channel_names)
    assert len(timestamps) == raw_samples.shape[0]
    assert isinstance(trial_id, str)


def test_run_real_time_equivital_generates_latency_report(tmp_path: Path):
    """Test end-to-end execution of real_time_equivital and validation of latency summaries."""
    config = _build_test_config(tmp_path)
    pipeline = DataPipeline(config=config)
    model_factory = ModelFactory()

    run_real_time_equivital(
        config=config,
        pipeline=pipeline,
        model_factory=model_factory,
        project_root_path=tmp_path,
    )

    report_path = (
        tmp_path / "Results_Latency" / "Complete_Explicit" / "KNN" / "ECG-Centrifuge" / "real_time_summary.json"
    )
    assert report_path.exists(), f"Report file {report_path} was not created"

    with open(report_path, "r") as f:
        report = json.load(f)

    # Check top-level metadata
    assert report["model"] == "KNN"
    assert report["model_type"] == "Complete_Explicit"
    assert report["streams"] == ["ECG", "Centrifuge"]
    assert report["n_raw_samples"] > 0
    assert report["n_predictions"] > 0

    # Check latency summaries
    for key in ["data_processing_latency_ms", "inference_latency_ms", "total_latency_ms"]:
        assert key in report, f"Missing key '{key}' in report"
        summary = report[key]
        assert summary["n"] == report["n_predictions"]
        assert summary["mean"] > 0.0
        assert summary["p50"] > 0.0
        assert summary["p95"] > 0.0
        assert summary["p99"] > 0.0

    # Total latency mean should be greater than data processing latency mean
    assert report["total_latency_ms"]["mean"] >= report["data_processing_latency_ms"]["mean"]
    assert report["total_latency_ms"]["mean"] >= report["inference_latency_ms"]["mean"]

    # Check per-prediction latency lists
    assert len(report["per_prediction_data_proc_latency_ms"]) == report["n_predictions"]
    assert len(report["per_prediction_inference_latency_ms"]) == report["n_predictions"]
    assert len(report["per_prediction_total_latency_ms"]) == report["n_predictions"]


def test_dynamic_model_hyperparameters_different_models():
    """Verify that RealTimeTraditionalDataPipeline resolves hyperparameters dynamically per model."""
    model_factory = ModelFactory()
    knn = model_factory.create_model("KNN")
    rf = model_factory.create_model("RF")

    pipeline_knn = RealTimeTraditionalDataPipeline(
        artifacts={"raw_feature_names": ["HR - ECG", "Gz_filtered - G"]},
        config={},
        model=knn,
    )
    pipeline_rf = RealTimeTraditionalDataPipeline(
        artifacts={"raw_feature_names": ["HR - ECG", "Gz_filtered - G"]},
        config={},
        model=rf,
    )

    # KNN uses baseline methods ["v0", "v1", "v2"] and window_size 15.0
    assert pipeline_knn.baseline_methods_to_use == ["v0", "v1", "v2"]
    assert pipeline_knn.window_size_s == 15.0
    assert pipeline_knn.baseline_window_s == 32.5

    # RF uses baseline methods ["v0", "v1", "v2", "v5", "v6", "v7", "v8"] and window_size 7.5
    assert pipeline_rf.baseline_methods_to_use == ["v0", "v1", "v2", "v5", "v6", "v7", "v8"]
    assert pipeline_rf.window_size_s == 7.5
    assert pipeline_rf.baseline_window_s == 18.75

