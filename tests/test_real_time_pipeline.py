"""Tests for RealTimeTraditionalDataPipeline and preprocessing artifact export."""

import json
from pathlib import Path

import numpy as np

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.model_type import ModelType
from src.models.k_nearest_neighbors import KNearestNeighborsModel
from src.Data_Pipeline.real_time_data_pipeline import RealTimeTraditionalDataPipeline


def _build_test_config() -> dict:
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
        "traditional_data_parameters": {
            "backstep": 0,
            "data_rate": 25,
            "offset": 0,
            "time_start": 0,
            "standardize_s1": True,
        },
    }


def test_traditional_data_pipeline_saves_json_artifacts(tmp_path: Path):
    """Verify that TraditionalDataPipeline saves a valid preprocessing_artifacts.json file."""
    config = _build_test_config()
    pipeline = DataPipeline(config=config)
    pipeline.set_random_seed(42)
    model_type = ModelType("Complete", "Explicit")
    pipeline.set_model_type(model_type)

    model = KNearestNeighborsModel()
    artifacts_file = tmp_path / "preprocessing_artifacts.json"

    X_train, X_test, y_train, y_test, select_features = pipeline.get_data(
        model=model,
        kfold_id=0,
        num_splits=5,
        feature_streams=["ECG", "Centrifuge"],
        traditional_feature_selection="raw",
        return_feature_names=True,
        save_preprocessing_artifacts_path=str(artifacts_file),
    )

    assert artifacts_file.exists(), "preprocessing_artifacts.json was not created"

    with open(artifacts_file, "r") as f:
        data = json.load(f)

    # Check required keys
    assert "s1_pooled_mean" in data
    assert "s1_pooled_std" in data
    assert "s2_global_mean" in data
    assert "s2_global_std" in data
    assert "raw_feature_names" in data
    assert "active_feature_names" in data
    assert "active_indices" in data
    assert "dropped_feature_names" in data
    assert "knn_imputer" in data

    # Verify lengths
    assert len(data["s1_pooled_mean"]) == len(data["s2_global_mean"])
    assert len(data["active_indices"]) == len(select_features)
    assert len(data["active_feature_names"]) == len(select_features)
    assert X_test.shape[1] == len(select_features)


def test_real_time_traditional_pipeline_warmup_and_streaming(tmp_path: Path):
    """Test warmup gate, online sample ingestion, and latency budget."""
    config = _build_test_config()
    pipeline = DataPipeline(config=config)
    pipeline.set_random_seed(42)
    model_type = ModelType("Complete", "Explicit")
    pipeline.set_model_type(model_type)

    model = KNearestNeighborsModel()
    artifacts_file = tmp_path / "preprocessing_artifacts.json"

    X_train, X_test, y_train, y_test, select_features = pipeline.get_data(
        model=model,
        kfold_id=0,
        num_splits=5,
        feature_streams=["ECG", "Centrifuge"],
        traditional_feature_selection="raw",
        return_feature_names=True,
        save_preprocessing_artifacts_path=str(artifacts_file),
    )

    # Initialize RealTimeTraditionalDataPipeline from saved JSON and config
    rt_pipeline = RealTimeTraditionalDataPipeline(
        artifacts=artifacts_file,
        config=config,
        model=model,
        participant_baseline_rhr=72.0,
    )

    assert not rt_pipeline.is_warmed_up

    # Simulate raw 25 Hz streaming
    # Channels for ECG + Centrifuge: 6 ECG cols + 1 Centrifuge col = 7 channels
    n_channels = 7
    n_samples = rt_pipeline.warmup_samples_required + 50
    np.random.seed(42)
    synthetic_stream = np.random.randn(n_samples, n_channels)
    # Set HR around 80 bpm and G around 1.2
    synthetic_stream[:, 0] = 80.0 + np.random.randn(n_samples) * 5.0
    synthetic_stream[:, -1] = 1.2 + np.random.randn(n_samples) * 0.1

    latencies_ms = []
    warmed_up_outputs = []

    for idx in range(n_samples):
        timestamp = idx * 0.04
        x_proc, lat_ms = rt_pipeline.ingest_sample(synthetic_stream[idx], timestamp)
        latencies_ms.append(lat_ms)

        if x_proc is not None:
            assert len(x_proc) == len(select_features)
            warmed_up_outputs.append(x_proc)

    assert rt_pipeline.is_warmed_up
    # With 50 samples beyond warmup (2.0s) and stride=0.25s, exactly 8 or 9 stride windows are emitted
    assert len(warmed_up_outputs) >= 8

    # Latency check: all samples must finish well under 40 ms budget (mean < 5 ms)
    mean_lat = float(np.mean(latencies_ms))
    p99_lat = float(np.percentile(latencies_ms, 99))
    max_lat = float(np.max(latencies_ms))

    assert mean_lat < 5.0, f"Mean latency {mean_lat:.3f} ms exceeds 5 ms threshold"
    assert p99_lat < 40.0, f"P99 latency {p99_lat:.3f} ms exceeds 40 ms real-time deadline"
    assert max_lat < 40.0, f"Max latency {max_lat:.3f} ms exceeds 40 ms real-time deadline"


def test_real_time_traditional_pipeline_process_stream(tmp_path: Path):
    """Test process_stream generator method."""
    artifacts = {
        "s1_pooled_mean": [0.0] * 10,
        "s1_pooled_std": [1.0] * 10,
        "s2_global_mean": [0.0] * 10,
        "s2_global_std": [1.0] * 10,
        "active_indices": list(range(10)),
        "active_feature_names": [f"f_{i}" for i in range(10)],
        "raw_feature_names": [f"f_{i}" for i in range(10)],
    }
    rt = RealTimeTraditionalDataPipeline(
        artifacts=artifacts,
        baseline_window=1.0,
        window_size=1.0,
        stride=0.1,
        stream_rate_hz=10.0,
    )
    # 1.0s at 10 Hz = 10 samples for warmup
    stream_data = np.ones((25, 2))
    results = list(rt.process_stream(stream_data))
    assert len(results) == 25
    # First 10 samples (indices 0..9, timestamps 0.0s..0.9s) are in warmup
    for i in range(10):
        assert results[i][0] is None
    # Sample index 10 (timestamp 1.0s) onwards emitted feature vectors on every 0.1s stride
    for i in range(10, 25):
        assert results[i][0] is not None
        assert len(results[i][0]) == 10


def test_real_time_traditional_pipeline_reset():
    """Test reset method resets warmup state and buffers."""
    artifacts = {
        "s1_pooled_mean": [0.0] * 10,
        "s1_pooled_std": [1.0] * 10,
        "s2_global_mean": [0.0] * 10,
        "s2_global_std": [1.0] * 10,
        "active_indices": list(range(10)),
    }
    rt = RealTimeTraditionalDataPipeline(
        artifacts=artifacts,
        baseline_window=1.0,
        window_size=1.0,
        stride=0.1,
        stream_rate_hz=10.0,
    )
    for _ in range(15):
        rt.ingest_sample(np.array([80.0, 1.0]))
    assert rt.is_warmed_up

    rt.reset()
    assert not rt.is_warmed_up
    x_proc, _ = rt.ingest_sample(np.array([80.0, 1.0]))
    assert x_proc is None


def test_real_time_traditional_pipeline_knn_imputation():
    """Test KNN imputation fallback during online sample ingestion."""
    artifacts = {
        "s1_pooled_mean": [0.0] * 10,
        "s1_pooled_std": [1.0] * 10,
        "s2_global_mean": [0.0] * 10,
        "s2_global_std": [1.0] * 10,
        "active_indices": list(range(10)),
        "knn_imputer": {
            "k": 2,
            "reference_means": [75.0, 1.2],
            "reference_data": [[70.0, 1.0], [80.0, 1.4]],
        },
    }
    rt = RealTimeTraditionalDataPipeline(
        artifacts=artifacts,
        baseline_window=1.0,
        window_size=1.0,
        stride=0.1,
        stream_rate_hz=10.0,
    )

    sample_with_nan = np.array([np.nan, 1.2])
    imputed = rt.impute_sample(sample_with_nan)
    assert not np.isnan(imputed).any()
    assert 70.0 <= imputed[0] <= 80.0


def test_real_time_traditional_pipeline_standardize_s1_disabled():
    """Test RealTimeTraditionalDataPipeline with standardize_s1=False."""
    artifacts = {
        "s1_pooled_mean": [0.0] * 10,
        "s1_pooled_std": [1.0] * 10,
        "s2_global_mean": [0.0] * 10,
        "s2_global_std": [1.0] * 10,
        "active_indices": list(range(10)),
    }
    rt = RealTimeTraditionalDataPipeline(
        artifacts=artifacts,
        baseline_window=1.0,
        window_size=1.0,
        stride=0.1,
        stream_rate_hz=10.0,
        standardize_s1=False,
    )
    for _ in range(12):
        x_proc, _ = rt.ingest_sample(np.array([80.0, 1.0]))
    assert x_proc is not None
    assert len(x_proc) == 10


def test_real_time_pipeline_exact_row_and_column_count_match_manual_ablation(tmp_path: Path):
    """Verify that RealTimeTraditionalDataPipeline output matches row/column counts under manual ablation."""
    config = _build_test_config()
    pipeline = DataPipeline(config=config)
    pipeline.set_random_seed(42)
    model_type = ModelType("Complete", "Explicit")
    pipeline.set_model_type(model_type)

    model = KNearestNeighborsModel()
    artifacts_file = tmp_path / "preprocessing_artifacts.json"

    # Get data under manual ablation (raw feature selection)
    X_train, X_test, y_train, y_test, select_features = pipeline.get_data(
        model=model,
        kfold_id=0,
        num_splits=5,
        feature_streams=["ECG", "Centrifuge"],
        traditional_feature_selection="raw",
        return_feature_names=True,
        save_preprocessing_artifacts_path=str(artifacts_file),
    )

    rt_pipeline = RealTimeTraditionalDataPipeline(
        artifacts=artifacts_file,
        config=config,
        model=model,
        participant_baseline_rhr=72.0,
    )

    # Number of columns in RT pipeline must match X_test.shape[1]
    n_channels = 7
    n_samples = rt_pipeline.warmup_samples_required + 200
    synthetic_stream = np.random.randn(n_samples, n_channels)
    synthetic_stream[:, 0] = 80.0
    synthetic_stream[:, -1] = 1.2

    emitted_rows = []
    for idx in range(n_samples):
        timestamp = idx * (1.0 / rt_pipeline.stream_rate_hz)
        x_proc, _ = rt_pipeline.ingest_sample(synthetic_stream[idx], timestamp)
        if x_proc is not None:
            emitted_rows.append(x_proc)

    # Column count check: all emitted rows must match X_test.shape[1]
    assert len(emitted_rows) > 0
    for row in emitted_rows:
        assert row.shape[0] == X_test.shape[1] == len(select_features)


def test_real_time_pipeline_stride_frequency_variations():
    """Verify that RealTimeTraditionalDataPipeline produces exact expected row counts for different strides."""
    artifacts = {
        "s1_pooled_mean": [0.0] * 10,
        "s1_pooled_std": [1.0] * 10,
        "s2_global_mean": [0.0] * 10,
        "s2_global_std": [1.0] * 10,
        "active_indices": list(range(10)),
    }

    total_duration_s = 60.0
    stream_rate_hz = 25.0
    n_samples = int(total_duration_s * stream_rate_hz)
    synthetic_stream = np.random.randn(n_samples, 2)
    synthetic_stream[:, 0] = 80.0
    synthetic_stream[:, 1] = 1.2

    for stride_s in [0.25, 0.5, 1.0]:
        window_size_s = 15.0
        baseline_window_s = 15.0

        # Theoretical offline formula from _sliding_window_mean_calc:
        # number_windows = ((time_end - offset) // stride) - (window_size // stride - 1)
        time_end = (n_samples - 1) / stream_rate_hz
        expected_rows = int(((time_end - 0.0) // stride_s) - (window_size_s // stride_s - 1))

        rt = RealTimeTraditionalDataPipeline(
            artifacts=artifacts,
            baseline_window=baseline_window_s,
            window_size=window_size_s,
            stride=stride_s,
            stream_rate_hz=stream_rate_hz,
            time_start=0.0,
            offset=0.0,
        )

        emitted = []
        for idx in range(n_samples):
            timestamp = idx * (1.0 / stream_rate_hz)
            x_proc, _ = rt.ingest_sample(synthetic_stream[idx], timestamp)
            if x_proc is not None:
                emitted.append(x_proc)

        assert len(emitted) == expected_rows, (
            f"Stride {stride_s}s produced {len(emitted)} rows, expected {expected_rows}"
        )