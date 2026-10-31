"""Tests for the on-site real-time demo: preprocessor waiting, sensor flags, and the mode end to end."""

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from src.Data_Pipeline.real_time_data_pipeline import (
    RealTimeDataPreprocessor,
    SingleSubjectLSLStreamer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_142_DIR = PROJECT_ROOT / "Extra spin data/142_HSP_Training_HSP_135_20260508_115626"
SAVED_DIR = PROJECT_ROOT / "Results/Sensor_Ablation_Real_Time_On_Site/Complete_Explicit/EGB/RealTimeEquivital"
ARTIFACTS_PATH = SAVED_DIR / "preprocessing_artifacts_fold_0.json"
CONFIG_PATH = PROJECT_ROOT / "configs/on_site_real_time_equivital.yaml"
NAMES = {"system_data": "SystemData", "ecg": "ECG_EQ02", "summary": "Summary_EQ02"}

needs_session = pytest.mark.skipif(not SESSION_142_DIR.exists(), reason="session 142 data not found")
needs_model = pytest.mark.skipif(not ARTIFACTS_PATH.exists(), reason="saved On_Site model not found")


@needs_session
def test_connect_require_all_waits_until_streams_exist():
    pre = RealTimeDataPreprocessor(stream_names=NAMES)
    assert pre.connect(timeout=0.2, require_all=True) is False and pre.inlets == {}
    streamer = SingleSubjectLSLStreamer(
        SESSION_142_DIR, playback_speed=0.0, max_rows_per_stream=5000, source_id_prefix="TEST_ONSITE_1_"
    )
    try:
        assert pre.connect(timeout=1.0, require_all=True) is True and set(pre.inlets) == set(NAMES)
        first = dict(pre.inlets)
        assert pre.connect(timeout=0.2, require_all=True) is True
        assert all(pre.inlets[k] is first[k] for k in NAMES)  # no duplicate inlets
    finally:
        pre.close()
        streamer.close()


@needs_session
def test_connect_survives_a_stream_that_cannot_be_opened(monkeypatch):
    import pylsl

    streamer = SingleSubjectLSLStreamer(
        SESSION_142_DIR, playback_speed=0.0, max_rows_per_stream=5000, source_id_prefix="TEST_ONSITE_3_"
    )
    pre = RealTimeDataPreprocessor(stream_names=NAMES)
    try:
        def stale(self, timeout=0.0, **kw):  # what a stream whose outlet just died looks like
            raise pylsl.util.TimeoutError("the operation failed due to a timeout.")

        monkeypatch.setattr(pylsl.StreamInlet, "open_stream", stale)
        assert pre.connect(timeout=0.5, require_all=True) is False and pre.inlets == {}
        monkeypatch.undo()
        assert pre.connect(timeout=1.0, require_all=True) is True  # retries once the stream is usable
    finally:
        pre.close()
        streamer.close()


@needs_session
@needs_model
def test_live_samples_put_each_sensor_in_its_own_column():
    raw = json.load(open(ARTIFACTS_PATH))["raw_feature_names"]
    streamer = SingleSubjectLSLStreamer(
        SESSION_142_DIR, playback_speed=0.0, max_rows_per_stream=20000, source_id_prefix="TEST_ONSITE_2_"
    )
    pre = RealTimeDataPreprocessor(raw_feature_names=raw, stream_names=NAMES)
    try:
        assert pre.connect(timeout=1.0, require_all=True)
        samples = []
        while not samples and streamer.push_chunk(1.0):
            samples = pre.poll_samples()
        # summary stream is slow (0.2 Hz): keep pushing until a vital has been seen
        while "temp" not in pre.last_valid_wall and streamer.push_chunk(5.0):
            pre.poll_samples()
        sample = samples[0][0]
        live = {raw.index(n) for n in RealTimeDataPreprocessor.DEFAULT_RAW_FEATURES}
        assert {i for i, v in enumerate(sample) if v != 0} == live  # 65 placeholders stay exactly 0
        assert sample[raw.index("magnitude - Centrifuge")] == pytest.approx(1.0, abs=0.5)
        assert "ecg" in pre.last_sample_wall and "summary" in pre.last_sample_wall
        assert set(pre.last_valid_wall) <= {"hr", "br", "temp"} and "temp" in pre.last_valid_wall
    finally:
        pre.close()
        streamer.close()


def _flag_inputs():
    from src.modes.on_site_real_time_equivital import _sensor_flags

    raw = list(RealTimeDataPreprocessor.DEFAULT_RAW_FEATURES) + [f"placeholder_{i}" for i in range(65)]
    rng = np.random.default_rng(0)
    window = np.zeros((50, len(raw)))
    for n in RealTimeDataPreprocessor.DEFAULT_RAW_FEATURES:
        window[:, raw.index(n)] = rng.normal(1.0, 0.1, 50)
    valid = {"hr": 100.0, "br": 100.0, "temp": 100.0}
    return _sensor_flags, window, raw, valid


def test_sensor_flags_follow_spec_table():
    flags, window, raw, valid = _flag_inputs()
    i = raw.index
    call = lambda w, v=valid, now=101.0: flags(w, raw, v, now, 15.0, 0.0)

    assert call(window) == ""  # healthy; 65 zero placeholders never flag

    w = window.copy()
    w[:, i("ECG Lead 2 - Equivital")] = 0.0
    assert call(w) == "ECG Lead 2:flat"

    w = window.copy()
    w[:, i("ECG Lead 2 - Equivital")] = 0.3  # constant -> zero std
    assert call(w) == "ECG Lead 2:flat"

    w = window.copy()
    w[:, i("ECG Lead 1 - Equivital")] = np.nan
    out = call(w)
    assert out.split(";") == [
        "ECG Lead 1:flat",
        "HR_instant:flat",
        "HR_average:flat",
        "HR_w_average:flat",
    ]

    w = window.copy()
    w[:, i("magnitude - Centrifuge")] = 1.0  # constant ~1 G at rest is normal
    assert call(w) == ""
    w[:, i("magnitude - Centrifuge")] = 0.0
    assert call(w) == "magnitude - Centrifuge:flat"

    assert call(window, {**valid, "br": 80.0}, now=100.0) == "BR:stale_20s"
    assert call(window, {"hr": 100.0, "temp": 100.0}, now=130.0).startswith("HR:stale_30s")  # never-seen BR counts from run_start


def _write_config(tmp_path: Path, **overrides) -> Path:
    from src.config_loader import load_experiment_config

    cfg = load_experiment_config(str(CONFIG_PATH))
    cfg["on_site_real_time_equivital"].update(overrides)
    return cfg


def test_missing_model_names_the_path(tmp_path):
    from src.Data_Pipeline.data_pipeline import DataPipeline
    from src.models.model_factory import ModelFactory
    from src.modes.on_site_real_time_equivital import run_on_site_real_time_equivital

    cfg = _write_config(tmp_path, saved_models_folder=str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="fold_0.pkl"):
        run_on_site_real_time_equivital(cfg, DataPipeline(config=cfg), ModelFactory(), tmp_path)


def _rows(out_dir: Path):
    files = list(out_dir.rglob("on_site_*.csv"))
    return list(csv.DictReader(files[0].open())) if files else []


def _wait_for(pred, out_dir: Path, timeout: float):
    end = time.time() + timeout
    while time.time() < end:
        if pred(_rows(out_dir)):
            return True
        time.sleep(0.5)
    return False


@pytest.mark.integration
@needs_session
@needs_model
def test_mode_end_to_end_with_pause_and_resume(tmp_path):
    out_dir = tmp_path / "out"
    cfg_text = CONFIG_PATH.read_text().replace("Results/On_Site_Real_Time", str(out_dir))
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(cfg_text)

    mode = subprocess.Popen(
        [sys.executable, "-m", "src.main", "--config", str(cfg_file)],
        cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    sim = None
    try:
        time.sleep(3.0)  # the mode is already waiting when the stream appears
        assert mode.poll() is None, "mode exited before any stream existed"
        sim = subprocess.Popen(
            [sys.executable, "-m", "src.real_time.simulate_streams", "--session", str(SESSION_142_DIR), "--speed", "10"],
            cwd=PROJECT_ROOT, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        assert _wait_for(lambda r: any(x["event"] == "PREDICTION" for x in r), out_dir, 120)
        first = next(x for x in _rows(out_dir) if x["event"] == "PREDICTION")
        assert float(first["stream_time_s"]) >= 46.0 and "sensor_flags" in first

        os.killpg(sim.pid, signal.SIGSTOP)  # freezes simulator and its streamer child
        time.sleep(5.0)
        os.killpg(sim.pid, signal.SIGCONT)

        def resumed(rows):
            ev = [x["event"] for x in rows]
            return "REFILLING" in ev and "PREDICTION" in ev[ev.index("REFILLING"):]

        assert _wait_for(resumed, out_dir, 120)
        ev = [x["event"] for x in _rows(out_dir)]
        assert ev.count("CONNECTION_LOST") == 1 and ev.count("CONNECTION_RESTORED") == 1
        assert ev.index("CONNECTION_LOST") < ev.index("CONNECTION_RESTORED") < ev.index("REFILLING")
        assert next(x for x in _rows(out_dir) if x["event"] == "CONNECTION_LOST")["stream"]
    finally:
        mode.send_signal(signal.SIGINT)
        try:
            out, _ = mode.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            mode.kill()
            out, _ = mode.communicate()
        if sim is not None:
            for sig in (signal.SIGCONT, signal.SIGTERM):
                try:
                    os.killpg(sim.pid, sig)
                except ProcessLookupError:
                    pass
    assert "predictions" in out


def test_pipeline_waits_for_a_full_window_after_the_buffer_is_flushed():
    from src.Data_Pipeline.real_time_data_pipeline import RealTimeTraditionalDataPipeline

    raw = RealTimeDataPreprocessor.DEFAULT_RAW_FEATURES
    artifacts = {
        "s1_pooled_mean": [0.0] * 100, "s1_pooled_std": [1.0] * 100,
        "s2_global_mean": [0.0] * 100, "s2_global_std": [1.0] * 100,
        "active_indices": list(range(10)),
        "active_feature_names": [f"{c}_v0_{st}_s2" for st in ("mean", "stddev", "max", "range") for c in raw[:2]]
        + ["HRV (SDNN)_s2", "HRV (RMSSD)_s2"],
        "raw_feature_names": raw,
    }
    rt = RealTimeTraditionalDataPipeline(artifacts=artifacts, baseline_window=1.0, window_size=1.0, stride=0.1,
                                         stream_rate_hz=25.0)
    rng = np.random.default_rng(0)
    sample = lambda: rng.normal(80.0, 1.0, len(raw))
    emitted = [rt.ingest_sample(sample(), timestamp_s=i * 0.04)[0] for i in range(60)]
    assert any(x is not None for x in emitted)

    rt._window_buffer.clear()  # what the mode does after a connection gap
    rt._raw_timestamps.clear()
    out = []
    for i in range(60, 60 + rt.n_window_samples + 5):
        out.append(rt.ingest_sample(sample(), timestamp_s=i * 0.04)[0])
    assert all(x is None for x in out[: rt.n_window_samples - 1])  # window not full yet
    assert out[-1] is not None  # resumes once it is
