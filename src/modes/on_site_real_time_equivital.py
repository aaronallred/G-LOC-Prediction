"""On-site real-time Equivital mode.

Waits for the live LSL streams, then predicts on every stride window with a saved traditional model.
Predictions and state changes (CONNECTION_LOST, CONNECTION_RESTORED, REFILLING, sensor flags) are printed
to the console and logged to a CSV so that it is always visible what the pipeline is doing.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.Data_Pipeline.real_time_data_pipeline import (
    RealTimeDataPreprocessor,
    RealTimeTraditionalDataPipeline,
)
from src.models.model_factory import ModelFactory
from src.modes.real_time_equivital import _resolve_saved_model_path, _summarize_latencies

_LIVE = RealTimeDataPreprocessor.DEFAULT_RAW_FEATURES
_VITALS = {"hr": ("HR (bpm) - Equivital", "HR"), "br": ("BR (rpm) - Equivital", "BR"),
           "temp": ("Skin Temperature - IR Thermometer (°C) - Equivital", "Skin Temperature")}
_ECG_DERIVED = ("HR_instant - Equivital", "HR_average - Equivital", "HR_w_average - Equivital")
_CSV_COLUMNS = ["run_id", "wall_time", "stream_time_s", "event", "stream", "pred", "sensor_flags",
                "preproc_ms", "features_ms", "model_ms", "total_ms"]


def _say(text: str) -> None:
    print(text, flush=True)


def _sensor_flags(
    window: np.ndarray,
    raw_names: list[str],
    last_valid_wall: dict[str, float],
    now: float,
    stale_s: float,
    run_start: float,
) -> str:
    """Name the sensors that look broken in the current window (spec 9.2); "" when all is fine."""
    flags: list[str] = []

    def col(name: str) -> np.ndarray:
        return window[:, raw_names.index(name)]

    def dead(x: np.ndarray, need_variation: bool) -> bool:
        if np.isnan(x).all() or not x.any():
            return True
        return need_variation and float(np.nanmax(x)) == float(np.nanmin(x))  # std of a constant float isn't exactly 0

    ecg1_flat = dead(col("ECG Lead 1 - Equivital"), True)
    if ecg1_flat:
        flags.append("ECG Lead 1:flat")
    if dead(col("ECG Lead 2 - Equivital"), True):
        flags.append("ECG Lead 2:flat")
    if ecg1_flat:
        flags.extend(f"{n.split(' - ')[0]}:flat" for n in _ECG_DERIVED)
    if dead(col("magnitude - Centrifuge"), False):
        flags.append("magnitude - Centrifuge:flat")
    for key, (_, label) in _VITALS.items():
        age = now - last_valid_wall.get(key, run_start)
        if age > stale_s:
            flags.append(f"{label}:stale_{int(age)}s")
    return ";".join(flags)


def _print_summary(total_ms: list[float], n_gloc: int) -> None:
    if not total_ms:
        _say("Summary: 0 predictions")
        return
    s = _summarize_latencies(total_ms)
    _say(f"Summary: {len(total_ms)} predictions ({n_gloc} G-LOC, {len(total_ms) - n_gloc} normal); "
         f"total ms p50 {s['p50']:.1f}, p95 {s['p95']:.1f}, max {s['max']:.1f}")


def run_on_site_real_time_equivital(
    config: dict,
    pipeline: DataPipeline,
    model_factory: ModelFactory,
    project_root_path: Path,
) -> None:
    """Wait for live LSL streams and print/log a prediction for every stride window."""
    cfg: dict[str, Any] = config["on_site_real_time_equivital"]
    model_name: str = cfg["model"]
    model_type_folder = cfg["model_type"].get_folder_name()
    stream_str = "-".join(cfg["stream_group"])
    fold = int(cfg.get("fold", 0))
    timeouts: dict[str, float] = cfg["timeouts_s"]
    stale_s = float(cfg.get("stale_vital_s", 15))

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else project_root_path / path

    saved_dir = resolve(cfg["saved_models_folder"])
    model_path = _resolve_saved_model_path(saved_dir, model_type_folder, model_name, stream_str, fold)
    artifacts_path = model_path.with_name(f"preprocessing_artifacts_fold_{fold}.json")
    for p in (model_path, artifacts_path):
        if not p.exists():
            raise FileNotFoundError(f"Saved model file not found: {p}")

    model = joblib.load(model_path)
    rt = RealTimeTraditionalDataPipeline(
        artifacts=artifacts_path,
        config=config,
        model=model_factory.create_model(model_name),
        participant_baseline_rhr=72.0,
    )
    _say(f"Model {model_path}")
    _say(f"Window {rt.window_size_s}s, baseline {rt.baseline_window_s}s, stride {rt.stride_s}s: "
         f"first prediction after ~{max(rt.baseline_window_s, rt.window_size_s):.1f}s of data")

    pre = RealTimeDataPreprocessor(raw_feature_names=rt.raw_feature_names, stream_names=cfg["stream_names"])
    live_idx = [rt.raw_feature_names.index(n) for n in _LIVE]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = resolve(cfg["save_results_folder"]) / model_type_folder / model_name / stream_str / f"on_site_{run_id}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(_CSV_COLUMNS + _LIVE)
    _say(f"Logging to {csv_path}")

    last_t = 0.0
    total_ms: list[float] = []
    n_gloc = 0

    def log_event(event: str, stream: str = "", console: str | None = None, **v: Any) -> None:
        now = datetime.now()
        if console is None:
            console = f"{event}" + (f"  streams: {stream}" if stream else "")
        _say(f"{now:%H:%M:%S}.{now.microsecond // 1000:03d}  t={last_t:.2f}s  {console}")
        live = v.get("live", [""] * len(_LIVE))
        writer.writerow([run_id, now.isoformat(timespec="milliseconds"), f"{last_t:.2f}", event, stream,
                         v.get("pred", ""), v.get("sensor_flags", ""), v.get("preproc_ms", ""),
                         v.get("features_ms", ""), v.get("model_ms", ""), v.get("total_ms", ""), *live])
        csv_file.flush()

    try:
        _say(f"Waiting for LSL streams: {cfg['stream_names']}")
        while not pre.connect(timeout=1.0, require_all=True):
            pass
        _say("Streams found: " + ", ".join(f"{k}={i.info().name()}" for k, i in pre.inlets.items()) + ". Warming up...")

        run_start = time.monotonic()
        lost = False
        prev_flags = ""
        last_warm_print = -5.0
        while True:
            samples = pre.poll_samples(return_latency=True)
            now = time.monotonic()
            silent = [k for k, t in pre.last_sample_wall.items() if now - t > timeouts[k]]

            if silent and not lost:
                lost = True
                log_event("CONNECTION_LOST", stream=",".join(silent))
                if not rt.is_warmed_up:
                    rt.reset()
            elif lost and not silent:
                lost = False
                log_event("CONNECTION_RESTORED")
                rt._window_buffer.clear()
                rt._raw_timestamps.clear()
                log_event("REFILLING")  # ingest_sample returns None until the window is full again
                continue  # discard this poll: the preprocessor interpolated across the gap in it
            if lost:
                time.sleep(0.005)
                continue
            if not samples:
                time.sleep(0.005)
                continue

            for s, t, preproc_ms in samples:
                last_t = t
                x, feat_ms = rt.ingest_sample(s, t)
                if x is None:
                    if not rt.is_warmed_up and t - last_warm_print >= 5.0:
                        last_warm_print = t
                        _say(f"warming up t={t:.1f}s")
                    continue
                t0 = time.perf_counter()
                pred = int(model.predict(x.reshape(1, -1))[0])
                model_ms = (time.perf_counter() - t0) * 1000.0
                total = preproc_ms + feat_ms + model_ms
                flags = _sensor_flags(np.asarray(rt._window_buffer), rt.raw_feature_names,
                                      pre.last_valid_wall, now, stale_s, run_start)
                total_ms.append(total)
                n_gloc += pred
                label = "██ G-LOC ██" if pred else "   normal   "
                log_event("PREDICTION",
                          console=f"{'FLAG' if flags else '  ok'} | {label}  sensors {flags or 'OK'}  "
                                  f"{total:5.1f} ms = preproc {preproc_ms:.1f} + features {feat_ms:.1f} "
                                  f"+ model {model_ms:.1f}",
                          pred=f"{'F' if flags else '-'}|{'G' if pred else '-'}", sensor_flags=flags, preproc_ms=f"{preproc_ms:.2f}",
                          features_ms=f"{feat_ms:.2f}", model_ms=f"{model_ms:.2f}", total_ms=f"{total:.2f}",
                          live=[f"{s[i]:.4f}" for i in live_idx])
                if flags != prev_flags:
                    _say(f"FLAG_ON: {flags}" if flags else "FLAG_OFF")
                    prev_flags = flags
    except KeyboardInterrupt:
        _say("")
    finally:
        _print_summary(total_ms, n_gloc)
        pre.close()
        csv_file.close()
