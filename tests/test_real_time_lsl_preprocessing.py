"""Targeted tests for real-time LSL streaming and preprocessing."""

import time
from pathlib import Path

import numpy as np
import pytest

from src.Data_Pipeline.real_time_data_pipeline import (
    RealTimeDataPreprocessor,
    RealTimeTraditionalDataPipeline,
    SingleSubjectLSLStreamer,
)

SESSION_142_DIR = "Extra spin data/142_HSP_Training_HSP_135_20260508_115626"


@pytest.fixture
def test_session_dir() -> Path:
    p = Path(SESSION_142_DIR)
    if not p.exists():
        pytest.skip(f"Session 142 directory not found at {SESSION_142_DIR}")
    return p


def test_single_subject_lsl_streamer_outlets(test_session_dir: Path):
    """Verify SingleSubjectLSLStreamer creates correct LSL outlets and pushes samples."""
    streamer = SingleSubjectLSLStreamer(
        session_dir=test_session_dir,
        playback_speed=0.0,
        max_rows_per_stream=50,
        source_id_prefix="TEST_STREAMER_1_",
    )

    try:
        # Check that expected streams are configured
        assert "system_data" in streamer.outlets
        assert "ecg" in streamer.outlets
        assert "summary" in streamer.outlets
        assert "accel" in streamer.outlets

        # Verify stream names match Session 142
        assert streamer.stream_names["system_data"] == "SA5_SystemData"
        assert streamer.stream_names["ecg"] == "ECG_EQ02_3118060"
        assert streamer.stream_names["summary"] == "Summary_EQ02_3118060"
        assert streamer.stream_names["accel"] == "Accel_EQ02_3118060"

        # Verify ECG has 2 channels
        assert streamer.stream_data["ecg"][1].shape[1] == 2

        # Push samples
        pushed_any = streamer.push_next()
        assert pushed_any is True

    finally:
        streamer.close()
        time.sleep(0.05)


def test_real_time_data_preprocessor_resampling_and_features(test_session_dir: Path):
    """Verify RealTimeDataPreprocessor receives LSL streams, resamples to 25Hz, and cleans vitals."""
    streamer = SingleSubjectLSLStreamer(
        session_dir=test_session_dir,
        playback_speed=0.0,
        max_rows_per_stream=2000,
        source_id_prefix="TEST_STREAMER_2_",
    )
    preprocessor = RealTimeDataPreprocessor(stream_names=streamer.stream_names)

    try:
        preprocessor.connect(timeout=2.0)
        assert len(preprocessor.inlets) >= 3

        # Push 5 seconds of stream data
        pushed_count = streamer.push_chunk(5.0)
        assert pushed_count > 0

        time.sleep(0.05)
        samples = preprocessor.poll_samples(timeout=0.1)
        assert len(samples) > 0, "Preprocessor should have emitted 25 Hz samples"

        # Check sample grid spacing
        if len(samples) > 1:
            dt_step = samples[1][1] - samples[0][1]
            assert np.isclose(dt_step, 0.04, atol=1e-5), f"Expected 0.04s spacing, got {dt_step}"

        # Verify values and bounds of emitted 25 Hz sample
        sample_arr, t_val = samples[0]
        assert sample_arr.shape == (9,)
        assert not np.isnan(sample_arr).any(), "Sample array contains NaN values"

        # Verify Centrifuge magnitude (~1.0 G at rest)
        mag_idx = preprocessor.raw_feature_names.index("magnitude - Centrifuge")
        mag_val = sample_arr[mag_idx]
        assert 0.8 <= mag_val <= 1.5, f"Expected baseline G-magnitude ~1.0, got {mag_val}"

        # Verify HR and BR physiological limits from Session 142 summary data
        hr_idx = preprocessor.raw_feature_names.index("HR (bpm) - Equivital")
        br_idx = preprocessor.raw_feature_names.index("BR (rpm) - Equivital")
        temp_idx = preprocessor.raw_feature_names.index("Skin Temperature - IR Thermometer (°C) - Equivital")

        assert 60.0 <= sample_arr[hr_idx] <= 200.0
        assert 8.0 <= sample_arr[br_idx] <= 35.0
        assert 25.0 <= sample_arr[temp_idx] <= 42.0

        # Verify Lead 1 and Lead 2 are populated with real dynamic telemetry
        lead1_idx = preprocessor.raw_feature_names.index("ECG Lead 1 - Equivital")
        lead2_idx = preprocessor.raw_feature_names.index("ECG Lead 2 - Equivital")
        assert -2.0 <= sample_arr[lead1_idx] <= 2.0
        assert -2.0 <= sample_arr[lead2_idx] <= 2.0

        all_lead2 = [s[0][lead2_idx] for s in samples]
        assert np.any(np.array(all_lead2) != 0.0), "ECG Lead 2 should contain non-zero real telemetry"
        assert 25.0 <= sample_arr[temp_idx] <= 42.0

        # Verify derived HR features
        hr_inst_idx = preprocessor.raw_feature_names.index("HR_instant - Equivital")
        hr_avg_idx = preprocessor.raw_feature_names.index("HR_average - Equivital")
        hr_w_avg_idx = preprocessor.raw_feature_names.index("HR_w_average - Equivital")

        assert 30.0 <= sample_arr[hr_inst_idx] <= 220.0
        assert 30.0 <= sample_arr[hr_avg_idx] <= 220.0
        assert 30.0 <= sample_arr[hr_w_avg_idx] <= 220.0

        # Verify return_latency=True option
        streamer.push_chunk(1.0)
        time.sleep(0.05)
        samples_lat = preprocessor.poll_samples(timeout=0.1, return_latency=True)
        if samples_lat:
            sample_arr_l, t_val_l, lat_ms = samples_lat[0]
            assert isinstance(lat_ms, float)
            assert lat_ms > 0.0
            assert sample_arr_l.shape == (9,)

    finally:
        preprocessor.close()
        time.sleep(0.05)
        streamer.close()
        time.sleep(0.05)


def test_end_to_end_real_time_pipeline_lsl_integration(test_session_dir: Path):
    """Verify full end-to-end integration: Streamer -> Preprocessor -> RealTimeTraditionalDataPipeline."""
    streamer = SingleSubjectLSLStreamer(
        session_dir=test_session_dir,
        playback_speed=0.0,
        max_rows_per_stream=10000,
        source_id_prefix="TEST_STREAMER_3_",
    )
    preprocessor = RealTimeDataPreprocessor(stream_names=streamer.stream_names)

    artifacts = {
        "s1_pooled_mean": [0.0] * 100,
        "s1_pooled_std": [1.0] * 100,
        "s2_global_mean": [0.0] * 100,
        "s2_global_std": [1.0] * 100,
        "active_indices": list(range(10)),
        "active_feature_names": [f"feat_{i}" for i in range(10)],
        "raw_feature_names": preprocessor.raw_feature_names,
    }

    rt_pipeline = RealTimeTraditionalDataPipeline(
        artifacts=artifacts,
        baseline_window=1.0,  # 1s warmup (25 samples)
        window_size=1.0,
        stride=0.1,
        stream_rate_hz=25.0,
    )

    try:
        preprocessor.connect(timeout=2.0)

        # Push 100 seconds of telemetry
        streamer.push_chunk(100.0)
        time.sleep(0.05)

        # Stream samples through pipeline
        warmed_up_emissions = []
        for x_proc, t_s in rt_pipeline.process_lsl_preprocessor(preprocessor, max_samples=60):
            if x_proc is not None:
                assert len(x_proc) == 10
                assert not np.isnan(x_proc).any()
                warmed_up_emissions.append((x_proc, t_s))

        assert rt_pipeline.is_warmed_up is True
        assert len(warmed_up_emissions) > 0, "Pipeline should have emitted processed feature vectors after warmup"

    finally:
        preprocessor.close()
        time.sleep(0.05)
        streamer.close()
        time.sleep(0.05)
