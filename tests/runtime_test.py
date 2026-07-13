import json

import pytest

from src.runtime import configure_compute, runtime_summary, xgboost_device
from src.timing import RunTimer


def test_cpu_runtime_selection_and_summary():
    device = configure_compute({"device": "cpu"})

    assert device.type == "cpu"
    assert xgboost_device() == "cpu"
    assert runtime_summary()["device"] == "cpu"


def test_require_gpu_rejects_cpu():
    with pytest.raises(RuntimeError, match="require_gpu"):
        configure_compute({"device": "cpu", "require_gpu": True})


def test_run_timer_writes_stage_and_runtime(tmp_path):
    configure_compute({"device": "cpu"})
    timer = RunTimer()

    with timer.track("smoke_test", fold=0):
        pass

    output = timer.save(tmp_path / "timing.json")
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["runtime"]["device"] == "cpu"
    assert report["total_duration_seconds"] >= 0
    assert report["stages"][0]["name"] == "smoke_test"
    assert report["stages"][0]["status"] == "completed"
    assert report["stages"][0]["duration_seconds"] >= 0
