"""Real-time Equivital streaming evaluation mode for G-LOC prediction.

This mode streams raw multi-rate sensor telemetry for a session folder (e.g. Session 100)
over native LSL outlets, ingests and preprocesses it to 25 Hz via RealTimeDataPreprocessor,
extracts online sliding-window features via RealTimeTraditionalDataPipeline, and evaluates
real-time inference latency and predictions using a trained traditional classifier.
Detailed latency statistics are saved to real_time_summary.json.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.Data_Pipeline.real_time_data_pipeline import (
    RealTimeDataPreprocessor,
    RealTimeTraditionalDataPipeline,
    SingleSubjectLSLStreamer,
    pin_process_to_core,
)
from src.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)

_LATENCY_PERCENTILES: tuple[int, ...] = (50, 95, 99)


class EquivitalDataStreamer:
    """Stream raw sensor telemetry through a local LSL outlet."""

    def __init__(
        self,
        channel_names: list[str],
        stream_name: str = "GLOC-Equivital-Raw",
        stream_type: str = "PsychoPhys",
        stream_rate_hz: float = 25.0,
        source_id: str = "gloc-equivital-raw-01",
    ) -> None:
        self.channel_names = channel_names
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.stream_rate_hz = stream_rate_hz
        self.source_id = source_id
        self._outlet: Optional[Any] = None
        self._create_outlet()

    def _create_outlet(self) -> None:
        try:
            from pylsl import StreamInfo, StreamOutlet
        except ImportError:
            logger.warning("pylsl is not installed. LSL streaming will be disabled.")
            return

        try:
            n_channels = len(self.channel_names)
            stream_info = StreamInfo(
                name=self.stream_name,
                type=self.stream_type,
                channel_count=n_channels,
                nominal_srate=self.stream_rate_hz,
                channel_format="float32",
                source_id=self.source_id,
            )
            if self.channel_names:
                chns = stream_info.desc().append_child("channels")
                for ch_name in self.channel_names:
                    ch = chns.append_child("channel")
                    ch.append_child_value("label", ch_name)

            self._outlet = StreamOutlet(stream_info)
            logger.info(
                "Created LSL outlet '%s' (%d channels at %.1f Hz)",
                self.stream_name,
                n_channels,
                self.stream_rate_hz,
            )
        except Exception as exc:
            logger.warning("Could not initialize LSL outlet: %s", exc)

    def push_sample(self, sample: np.ndarray) -> None:
        """Push a single raw sample to the LSL outlet if active."""
        if self._outlet is not None:
            try:
                self._outlet.push_sample(sample.astype(np.float32, copy=False))
            except Exception as exc:
                logger.debug("LSL push_sample error: %s", exc)

    def close(self) -> None:
        self._outlet = None


def _load_single_trial_raw_data(
    pipeline: DataPipeline,
    model_instance: Any,
    model_type: Any,
    trial_to_select: Optional[str] = None,
    output_feature_dtype: np.dtype = np.dtype(np.float32),
) -> tuple[pd.DataFrame, list[str], str]:
    """Load raw 25 Hz telemetry for a single continuous trial."""
    backend = pipeline._build_backend(model_instance)
    file_paths = backend._get_data_locations()
    gloc_data = backend._load_data(file_paths, output_feature_dtype)

    trad_hparams = backend._resolve_traditional_hyperparameters(model_instance, model_instance.name)
    baseline_methods_to_use = trad_hparams.get("baseline_methods_to_use", ["v0", "v1", "v2", "v5", "v6"])
    feature_groups_to_analyze, _ = backend._get_feature_groups_and_baseline_methods(
        model_type, baseline_methods_to_use
    )
    gloc_data, features = backend._process_and_get_feature_names(
        gloc_data, feature_groups_to_analyze, model_type, file_paths, output_feature_dtype
    )

    unique_trials = pd.unique(gloc_data["trial_id"])
    if len(unique_trials) == 0:
        raise ValueError("No trials found in data.")

    unique_trials_str = [str(t) for t in unique_trials]
    if trial_to_select is not None and str(trial_to_select) in unique_trials_str:
        trial_id = str(trial_to_select)
    else:
        trial_id = str(unique_trials[0])

    trial_df = gloc_data[gloc_data["trial_id"] == trial_id]
    return trial_df, list(features["All"]), trial_id


def _summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """Aggregate per-sample latencies (ms) into descriptive stats + percentiles."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    summary: dict[str, float] = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    for p in _LATENCY_PERCENTILES:
        summary[f"p{p}"] = float(np.percentile(arr, p))
    return summary


def _resolve_saved_model_path(
    saved_models_folder: Path,
    model_type_folder: str,
    model_name: str,
    stream_str: str,
    kfold_id: int = 0,
) -> Path:
    return saved_models_folder / model_type_folder / model_name / stream_str / f"fold_{kfold_id}.pkl"


def _write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Saved report to %s", report_path)


def _evaluate_predictions(
    y_true: Union[np.ndarray, list[int]],
    y_pred: Union[np.ndarray, list[int]],
) -> tuple[dict[str, float], list[list[int]]]:
    """Compute performance metrics and confusion matrix given ground truth and predictions."""
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()

    if len(y_p) == 0:
        metrics_dict: dict[str, float] = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f1_score": 0.0,
            "specificity": 0.0,
            "g_mean": 0.0,
        }
        return metrics_dict, [[0, 0], [0, 0]]

    accuracy = float(metrics.accuracy_score(y_t, y_p))
    precision = float(metrics.precision_score(y_t, y_p, zero_division=0.0))
    recall = float(metrics.recall_score(y_t, y_p, zero_division=0.0))
    f1 = float(metrics.f1_score(y_t, y_p, zero_division=0.0))
    specificity = float(metrics.recall_score(y_t, y_p, pos_label=0, zero_division=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g_mean = float(geometric_mean_score(y_t, y_p))

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_score": f1,
        "specificity": specificity,
        "g_mean": g_mean,
    }
    confusion_mat = metrics.confusion_matrix(y_t, y_p, labels=[0, 1]).tolist()
    return metrics_dict, confusion_mat


def _evaluate_real_time_predictions(
    predictions: list[int],
) -> tuple[dict[str, float], list[list[int]]]:
    """Calculate classification performance metrics and confusion matrix assuming all ground truth events are 0."""
    y_true = np.zeros(len(predictions), dtype=int)
    return _evaluate_predictions(y_true, predictions)


def run_real_time_equivital(
    config: dict,
    pipeline: DataPipeline,
    model_factory: ModelFactory,
    project_root_path: Path,
) -> None:
    """Run real-time streaming evaluation against saved models on multi-rate LSL telemetry.

    Orchestrates the end-to-end real-time workflow:
      1. Loads raw session telemetry into SingleSubjectLSLStreamer (publishing to native LSL outlets).
      2. Ingests data from LSL into RealTimeDataPreprocessor (resampling & preprocessing to 25 Hz).
      3. Passes 25 Hz samples into RealTimeTraditionalDataPipeline (online feature extraction).
      4. Feeds emitted feature vectors into the trained model to make real-time inferences.
      5. Measures and logs latencies to real_time_summary.json.
    """
    mode_config = config.get("real_time_equivital", {})
    if not mode_config:
        logger.warning("No 'real_time_equivital' configuration found in config.")
        return

    model_type = mode_config["model_type"]
    random_seed: int = int(mode_config.get("random_seed", 42))
    model_names: list[str] = mode_config["models"]
    num_splits: int = int(mode_config.get("num_splits", 10))
    stream_groups: list[list[str]] = mode_config.get("streams", [["ECG", "Centrifuge"]])
    manual_ablation: bool = bool(mode_config.get("manual_ablation", True))
    use_real_time_sleep: bool = bool(mode_config.get("use_real_time_sleep", False))
    max_stream_samples: Optional[int] = mode_config.get("max_stream_samples", None)
    use_preprocessor: bool = bool(mode_config.get("use_preprocessor", True))
    trial_id_cfg: Optional[str] = mode_config.get("trial_id", None)

    saved_models_folder = Path(mode_config.get("saved_models_folder", "Results/Sensor_Ablation_Real_Time"))
    if not saved_models_folder.is_absolute():
        saved_models_folder = project_root_path / saved_models_folder

    save_results_folder = Path(mode_config.get("save_results_folder", "Results/Real_Time_Prediction_Latency"))
    if not save_results_folder.is_absolute():
        save_results_folder = project_root_path / save_results_folder

    session_dir_cfg = mode_config.get(
        "session_dir",
        "Extra spin data/142_HSP_Training_HSP_135_20260508_115626",
    )
    session_dir = Path(session_dir_cfg)
    if not session_dir.is_absolute():
        if not session_dir.exists() and (project_root_path / session_dir).exists():
            session_dir = project_root_path / session_dir
        elif session_dir.exists():
            session_dir = session_dir.resolve()

    model_type_folder = model_type.get_folder_name()
    session_label = session_dir.name if use_preprocessor else f"trial={trial_id_cfg or 'first'}"

    logger.info(
        "Starting real_time_equivital: models=%s, streams=%s, model_type=%s, source=%s, use_preprocessor=%s",
        model_names,
        stream_groups,
        model_type_folder,
        session_label,
        use_preprocessor,
    )

    pipeline.set_random_seed(random_seed)
    pipeline.set_model_type(model_type)

    feature_group = "raw" if manual_ablation else "cache"

    for model_name in model_names:
        logger.info("Processing model: %s", model_name)
        model_instance = model_factory.create_model(model_name)

        if not getattr(model_instance, "is_traditional_model", False):
            raise NotImplementedError(
                f"Model '{model_name}' is not a traditional model. "
                "Real-time evaluation currently supports only traditional (sklearn) saved models."
            )

def _run_real_time_with_preprocessor(
    model_name: str,
    stream_group: list[str],
    model_instance: Any,
    model_type: Any,
    model_type_folder: str,
    pipeline: DataPipeline,
    feature_group: str,
    num_splits: int,
    saved_models_folder: Path,
    save_results_folder: Path,
    session_dir: Path,
    use_real_time_sleep: bool,
    max_stream_samples: Optional[int],
    config: dict,
) -> None:
    """Run real-time streaming evaluation using the multi-rate preprocessor and session_dir."""
    stream_str = "-".join(stream_group)
    logger.info("Stream group: %s", stream_str)

    # 1. Check/load fold model & preprocessing artifacts
    artifacts_path = (
        saved_models_folder / model_type_folder / model_name / stream_str / "preprocessing_artifacts_fold_0.json"
    )
    saved_model_path = _resolve_saved_model_path(
        saved_models_folder, model_type_folder, model_name, stream_str, kfold_id=0
    )

    if not artifacts_path.exists() or not saved_model_path.exists():
        logger.info(
            "Saved model or artifacts not found at %s. Generating fold 0 estimator and artifacts...",
            saved_model_path,
        )
        X_train, X_test, y_train, y_test, _ = pipeline.get_data(
            model=model_instance,
            kfold_id=0,
            num_splits=num_splits,
            feature_streams=stream_group,
            return_feature_names=True,
            traditional_feature_selection=feature_group,
            save_preprocessing_artifacts_path=str(artifacts_path),
        )
        saved_model_path.parent.mkdir(parents=True, exist_ok=True)
        model_instance.train(X_train, y_train)
        model_instance.save_model(str(saved_model_path))

    logger.info("Loading saved model from %s", saved_model_path)
    loaded_model = joblib.load(saved_model_path)

    orig_affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    available_cores = sorted(orig_affinity) if orig_affinity is not None else []
    logger.info(
        "CPU allocation for %s: %d core(s) available %s",
        model_name,
        len(available_cores),
        available_cores,
    )

    # 2. Instantiate RealTimeTraditionalDataPipeline
    rt_pipeline = RealTimeTraditionalDataPipeline(
        artifacts=artifacts_path,
        config=config,
        model=model_instance,
        participant_baseline_rhr=72.0,
    )

    per_sample_preproc_latencies_ms: list[float] = []
    per_sample_data_proc_latencies_ms: list[float] = []
    per_prediction_preproc_latencies_ms: list[float] = []
    data_proc_latencies_ms: list[float] = []
    inference_latencies_ms: list[float] = []
    total_latencies_ms: list[float] = []
    predictions: list[int] = []
    prediction_to_prediction_latencies_ms: list[float] = []
    per_prediction_pred_to_pred_latencies_ms: list[Optional[float]] = []
    last_prediction_time: Optional[float] = None
    n_raw_samples = 0
    resolved_trial_id = session_dir.name

    # 3. Instantiate LSL Streamer
    playback_speed = 1.0 if use_real_time_sleep else 0.0
    streamer = SingleSubjectLSLStreamer(
        session_dir=session_dir,
        playback_speed=playback_speed,
        source_id_prefix=f"RT_{model_name}_{stream_str}_",
    )

    # 4. Start background streaming process (Process 1)
    rt_pipeline.reset()
    streamer.start(wait_for_trigger=True)

    # 5. Instantiate RealTimeDataPreprocessor and start background worker (Process 2)
    sample_queue: mp.Queue = mp.Queue(maxsize=50000)
    preprocessor = RealTimeDataPreprocessor(
        raw_feature_names=rt_pipeline.raw_feature_names,
        stream_names=streamer.stream_names,
    )
    preprocessor.start(
        sample_queue=sample_queue,
        stream_finished_event=streamer.finished_event,
        max_stream_samples=max_stream_samples,
    )
    streamer.trigger()

    # 6. Pin consumer process (Process 3) after spawning child workers
    pin_process_to_core(2)

    try:
        while True:
            try:
                item = sample_queue.get(timeout=2.0)
            except queue.Empty:
                if not preprocessor.is_alive() and not streamer.is_alive():
                    break
                continue

            if item is None:
                break

            sample_25hz, t_target, preproc_lat_ms = item
            n_raw_samples += 1
            per_sample_preproc_latencies_ms.append(preproc_lat_ms)
            X_processed, data_proc_lat_ms = rt_pipeline.ingest_sample(
                sample_25hz, timestamp_s=t_target
            )
            per_sample_data_proc_latencies_ms.append(data_proc_lat_ms)

            if n_raw_samples % 2500 == 0:
                logger.info(
                    "[%s | %s] Ingested %d samples | %d predictions made",
                    model_name,
                    stream_str,
                    n_raw_samples,
                    len(predictions),
                )

            if X_processed is not None:
                t_infer_0 = time.perf_counter()
                pred = loaded_model.predict(X_processed.reshape(1, -1))
                t_infer_1 = time.perf_counter()
                infer_lat_ms = (t_infer_1 - t_infer_0) * 1000.0

                t_now = t_infer_1
                if last_prediction_time is not None:
                    pred_to_pred_ms = (t_now - last_prediction_time) * 1000.0
                    prediction_to_prediction_latencies_ms.append(pred_to_pred_ms)
                    per_prediction_pred_to_pred_latencies_ms.append(pred_to_pred_ms)
                else:
                    per_prediction_pred_to_pred_latencies_ms.append(None)
                last_prediction_time = t_now

                per_prediction_preproc_latencies_ms.append(preproc_lat_ms)
                data_proc_latencies_ms.append(data_proc_lat_ms)
                inference_latencies_ms.append(infer_lat_ms)
                total_latencies_ms.append(preproc_lat_ms + data_proc_lat_ms + infer_lat_ms)
                predictions.append(int(pred[0]))

            if max_stream_samples is not None and n_raw_samples >= max_stream_samples:
                break
    finally:
        preprocessor.close()
        streamer.close()
        try:
            sample_queue.close()
            sample_queue.cancel_join_thread()
        except Exception:
            pass
        if orig_affinity is not None:
            try:
                os.sched_setaffinity(0, orig_affinity)
            except Exception:
                pass

    # Aggregate latencies and evaluate performance
    preproc_summary = _summarize_latencies(per_sample_preproc_latencies_ms)
    data_proc_summary = _summarize_latencies(data_proc_latencies_ms)
    inference_summary = _summarize_latencies(inference_latencies_ms)
    total_latency_summary = _summarize_latencies(total_latencies_ms)
    pred_to_pred_summary = _summarize_latencies(prediction_to_prediction_latencies_ms)

    perf_metrics, confusion_mat = _evaluate_real_time_predictions(predictions)

    report = {
        "model": model_name,
        "model_type": model_type_folder,
        "streams": stream_group,
        "trial_id": resolved_trial_id,
        "use_preprocessor": True,
        "n_raw_samples": int(n_raw_samples),
        "n_predictions": int(len(predictions)),
        "use_real_time_sleep": use_real_time_sleep,
        "performance": perf_metrics,
        "accuracy": perf_metrics["accuracy"],
        "precision": perf_metrics["precision"],
        "recall": perf_metrics["recall"],
        "f1": perf_metrics["f1"],
        "f1_score": perf_metrics["f1_score"],
        "specificity": perf_metrics["specificity"],
        "g_mean": perf_metrics["g_mean"],
        "confusion_matrix": confusion_mat,
        "preprocessing_latency_ms": preproc_summary,
        "data_processing_latency_ms": data_proc_summary,
        "inference_latency_ms": inference_summary,
        "total_latency_ms": total_latency_summary,
        "prediction_to_prediction_latency_ms": pred_to_pred_summary,
        "per_sample_preprocessing_latency_ms": per_sample_preproc_latencies_ms,
        "per_sample_data_proc_latency_ms": per_sample_data_proc_latencies_ms,
        "per_prediction_preprocessing_latency_ms": per_prediction_preproc_latencies_ms,
        "per_prediction_data_proc_latency_ms": data_proc_latencies_ms,
        "per_prediction_inference_latency_ms": inference_latencies_ms,
        "per_prediction_total_latency_ms": total_latencies_ms,
        "prediction_to_prediction_latencies_ms": prediction_to_prediction_latencies_ms,
        "per_prediction_prediction_to_prediction_latency_ms": per_prediction_pred_to_pred_latencies_ms,
    }

    model_output_dir = save_results_folder / model_type_folder / model_name / stream_str
    report_path = model_output_dir / "real_time_summary.json"
    _write_report(report, report_path)

    cm_path = model_output_dir / "confusion_matrix.json"
    _write_report({"confusion_matrix": confusion_mat, "labels": [0, 1]}, cm_path)

    logger.info(
        "Completed %s | streams=%s | trial=%s | predictions=%d | "
        "preproc mean=%.4f ms | data_proc mean=%.4f ms | infer mean=%.4f ms | "
        "total mean=%.4f ms (p95=%.4f ms) | pred_to_pred mean=%.4f ms",
        model_name,
        stream_str,
        resolved_trial_id,
        len(predictions),
        preproc_summary.get("mean", 0.0),
        data_proc_summary.get("mean", 0.0),
        inference_summary.get("mean", 0.0),
        total_latency_summary.get("mean", 0.0),
        total_latency_summary.get("p95", 0.0),
        pred_to_pred_summary.get("mean", 0.0),
    )
    logger.info(
        "Performance for %s: accuracy=%.4f | precision=%.4f | recall=%.4f | "
        "f1=%.4f | specificity=%.4f | g_mean=%.4f",
        model_name,
        perf_metrics["accuracy"],
        perf_metrics["precision"],
        perf_metrics["recall"],
        perf_metrics["f1"],
        perf_metrics["specificity"],
        perf_metrics["g_mean"],
    )


def _run_real_time_without_preprocessor(
    model_name: str,
    stream_group: list[str],
    model_instance: Any,
    model_type: Any,
    model_type_folder: str,
    pipeline: DataPipeline,
    feature_group: str,
    num_splits: int,
    saved_models_folder: Path,
    save_results_folder: Path,
    use_real_time_sleep: bool,
    max_stream_samples: Optional[int],
    baseline_window_s: float = 32.5,
    backstep_s: float = 15.0,
    stream_rate_hz: float = 25.0,
    data_rate_hz: float = 25.0,
) -> None:
    """Evaluate saved fold models on out-of-fold test sets from all subjects and trials across splits,
    measuring non-zero windowing data processing latency for each prediction window."""
    stream_str = "-".join(stream_group)
    per_fold_reports: list[dict[str, Any]] = []
    all_predictions: list[int] = []
    all_y_true: list[int] = []
    all_data_proc_latencies_ms: list[float] = []
    all_inference_latencies_ms: list[float] = []
    all_total_latencies_ms: list[float] = []
    all_p2p_latencies_ms: list[float] = []
    per_prediction_p2p_latencies_ms: list[Optional[float]] = []
    total_streamed_samples = 0

    for kfold_id in range(num_splits):
        last_prediction_time: Optional[float] = None
        if max_stream_samples is not None and total_streamed_samples >= max_stream_samples:
            break

        remaining_samples = (
            max_stream_samples - total_streamed_samples
            if max_stream_samples is not None
            else None
        )
        remaining_folds = num_splits - kfold_id
        sample_budget = (
            max(1, remaining_samples // remaining_folds)
            if remaining_samples is not None
            else None
        )

        artifacts_path = (
            saved_models_folder
            / model_type_folder
            / model_name
            / stream_str
            / f"preprocessing_artifacts_fold_{kfold_id}.json"
        )
        saved_model_path = _resolve_saved_model_path(
            saved_models_folder, model_type_folder, model_name, stream_str, kfold_id=kfold_id
        )

        X_train, X_test, y_train, y_test, select_features = pipeline.get_data(
            model=model_instance,
            kfold_id=kfold_id,
            num_splits=num_splits,
            feature_streams=stream_group,
            return_feature_names=True,
            traditional_feature_selection=feature_group,
            save_preprocessing_artifacts_path=str(artifacts_path) if not artifacts_path.exists() else None,
        )

        if not saved_model_path.exists():
            saved_model_path.parent.mkdir(parents=True, exist_ok=True)
            model_instance.train(X_train, y_train)
            model_instance.save_model(str(saved_model_path))

        loaded_model = joblib.load(saved_model_path)

        if sample_budget is not None and sample_budget < X_test.shape[0]:
            X_test_fold = X_test[:sample_budget]
            y_test_fold = np.asarray(y_test[:sample_budget], dtype=int).ravel()
        else:
            X_test_fold = X_test
            y_test_fold = np.asarray(y_test, dtype=int).ravel()

        # Initialize RealTimeTraditionalDataPipeline for measuring windowing computation latency
        rt_pipeline = RealTimeTraditionalDataPipeline(
            artifacts=artifacts_path,
            stream_rate_hz=stream_rate_hz,
            baseline_window=baseline_window_s,
        )
        rt_pipeline._v1_baseline_mean = np.ones(len(rt_pipeline.raw_feature_names), dtype=np.float64)
        rt_pipeline._v2_baseline_mean = np.zeros(len(rt_pipeline.raw_feature_names), dtype=np.float64)

        # Buffer of raw features for realistic window feature extraction timing
        rng = np.random.RandomState(42 + kfold_id)
        win_buffer = rng.randn(rt_pipeline.n_window_samples, len(rt_pipeline.raw_feature_names)).astype(np.float64)
        if win_buffer.shape[1] > 0:
            win_buffer[:, 0] = np.abs(win_buffer[:, 0]) * 30.0 + 60.0  # Positive HR values

        streamer = EquivitalDataStreamer(
            channel_names=select_features,
            stream_rate_hz=stream_rate_hz,
            source_id=f"RT_{model_name}_{stream_str}_fold_{kfold_id}",
        )

        fold_predictions: list[int] = []
        fold_data_proc_latencies: list[float] = []
        fold_infer_latencies: list[float] = []
        fold_total_latencies: list[float] = []

        try:
            for idx in range(X_test_fold.shape[0]):
                sample = X_test_fold[idx]
                streamer.push_sample(sample)
                total_streamed_samples += 1

                # 1. Windowing and data processing latency (always occurs before prediction)
                t_dp_0 = time.perf_counter()
                _ = rt_pipeline._compute_window_features(win_buffer)
                t_dp_1 = time.perf_counter()
                data_proc_lat_ms = (t_dp_1 - t_dp_0) * 1000.0

                # 2. Model inference latency
                t_inf_0 = time.perf_counter()
                pred = loaded_model.predict(sample.reshape(1, -1))
                t_inf_1 = time.perf_counter()
                infer_lat_ms = (t_inf_1 - t_inf_0) * 1000.0

                total_lat_ms = data_proc_lat_ms + infer_lat_ms

                t_now = t_inf_1
                if last_prediction_time is not None:
                    p2p_ms = (t_now - last_prediction_time) * 1000.0
                    all_p2p_latencies_ms.append(p2p_ms)
                    per_prediction_p2p_latencies_ms.append(p2p_ms)
                else:
                    per_prediction_p2p_latencies_ms.append(None)
                last_prediction_time = t_now

                fold_predictions.append(int(pred[0]))
                fold_data_proc_latencies.append(data_proc_lat_ms)
                fold_infer_latencies.append(infer_lat_ms)
                fold_total_latencies.append(total_lat_ms)

                all_data_proc_latencies_ms.append(data_proc_lat_ms)
                all_inference_latencies_ms.append(infer_lat_ms)
                all_total_latencies_ms.append(total_lat_ms)

                if use_real_time_sleep:
                    time.sleep(1.0 / stream_rate_hz)
        finally:
            streamer.close()

        fold_perf, fold_cm = _evaluate_predictions(y_test_fold[: len(fold_predictions)], fold_predictions)
        fold_latency_summary = _summarize_latencies(fold_total_latencies)

        all_predictions.extend(fold_predictions)
        all_y_true.extend(y_test_fold[: len(fold_predictions)])

        per_fold_reports.append({
            "fold_id": kfold_id,
            "n_test": int(X_test_fold.shape[0]),
            "n_predictions": int(len(fold_predictions)),
            "performance": fold_perf,
            "confusion_matrix": fold_cm,
            "latency_ms": fold_latency_summary,
            "data_processing_latency_ms": _summarize_latencies(fold_data_proc_latencies),
            "inference_latency_ms": _summarize_latencies(fold_infer_latencies),
            "total_latency_ms": fold_latency_summary,
        })

    # Holistic aggregated performance across folds
    metric_names = ["accuracy", "precision", "recall", "f1", "f1_score", "specificity", "g_mean"]
    agg_perf = {
        metric: float(np.mean([f["performance"][metric] for f in per_fold_reports]))
        if per_fold_reports
        else 0.0
        for metric in metric_names
    }
    agg_cm = (
        np.sum([np.array(f["confusion_matrix"]) for f in per_fold_reports], axis=0).tolist()
        if per_fold_reports
        else [[0, 0], [0, 0]]
    )

    data_proc_summary = _summarize_latencies(all_data_proc_latencies_ms)
    inference_summary = _summarize_latencies(all_inference_latencies_ms)
    total_latency_summary = _summarize_latencies(all_total_latencies_ms)
    preproc_summary = {
        "n": total_streamed_samples,
        "mean": 0.0,
        "std": 0.0,
        "median": 0.0,
        "min": 0.0,
        "max": 0.0,
        "p50": 0.0,
        "p95": 0.0,
        "p99": 0.0,
    }
    pred_to_pred_summary = _summarize_latencies(all_p2p_latencies_ms)

    report = {
        "model": model_name,
        "model_type": model_type_folder,
        "streams": stream_group,
        "num_splits": num_splits,
        "trial_id": "All",
        "use_preprocessor": False,
        "n_raw_samples": int(total_streamed_samples),
        "n_predictions": int(len(all_predictions)),
        "use_real_time_sleep": use_real_time_sleep,
        "performance": {
            **agg_perf,
            "folds": {
                metric: [float(f["performance"][metric]) for f in per_fold_reports]
                for metric in metric_names
            },
        },
        "accuracy": agg_perf["accuracy"],
        "precision": agg_perf["precision"],
        "recall": agg_perf["recall"],
        "f1": agg_perf["f1"],
        "f1_score": agg_perf["f1_score"],
        "specificity": agg_perf["specificity"],
        "g_mean": agg_perf["g_mean"],
        "confusion_matrix": agg_cm,
        "per_fold": per_fold_reports,
        "preprocessing_latency_ms": preproc_summary,
        "data_processing_latency_ms": data_proc_summary,
        "inference_latency_ms": inference_summary,
        "total_latency_ms": total_latency_summary,
        "prediction_to_prediction_latency_ms": pred_to_pred_summary,
        "per_sample_preprocessing_latency_ms": [0.0] * total_streamed_samples,
        "per_sample_data_proc_latency_ms": all_data_proc_latencies_ms,
        "per_prediction_preprocessing_latency_ms": [0.0] * len(all_predictions),
        "per_prediction_data_proc_latency_ms": all_data_proc_latencies_ms,
        "per_prediction_inference_latency_ms": all_inference_latencies_ms,
        "per_prediction_total_latency_ms": all_total_latencies_ms,
        "prediction_to_prediction_latencies_ms": all_p2p_latencies_ms,
        "per_prediction_prediction_to_prediction_latency_ms": per_prediction_p2p_latencies_ms,
    }

    model_output_dir = save_results_folder / model_type_folder / model_name / stream_str
    report_path = model_output_dir / "real_time_summary.json"
    _write_report(report, report_path)

    cm_path = model_output_dir / "confusion_matrix.json"
    _write_report({"confusion_matrix": agg_cm, "labels": [0, 1]}, cm_path)

    logger.info(
        "Completed %s | streams=%s | folds evaluated=%d | predictions=%d | "
        "data_proc mean=%.4f ms | infer mean=%.4f ms | total mean=%.4f ms (p95=%.4f ms)",
        model_name,
        stream_str,
        len(per_fold_reports),
        len(all_predictions),
        data_proc_summary.get("mean", 0.0),
        inference_summary.get("mean", 0.0),
        total_latency_summary.get("mean", 0.0),
        total_latency_summary.get("p95", 0.0),
    )
    logger.info(
        "Holistic performance for %s: accuracy=%.4f | precision=%.4f | recall=%.4f | "
        "f1=%.4f | specificity=%.4f | g_mean=%.4f",
        model_name,
        agg_perf["accuracy"],
        agg_perf["precision"],
        agg_perf["recall"],
        agg_perf["f1"],
        agg_perf["specificity"],
        agg_perf["g_mean"],
    )


def run_real_time_equivital(
    config: dict,
    pipeline: DataPipeline,
    model_factory: ModelFactory,
    project_root_path: Path,
) -> None:
    """Run real-time streaming evaluation against saved models on multi-rate LSL telemetry.

    Orchestrates the end-to-end real-time workflow:
      1. Loads raw session telemetry into SingleSubjectLSLStreamer (publishing to native LSL outlets).
      2. Ingests data from LSL into RealTimeDataPreprocessor (resampling & preprocessing to 25 Hz).
      3. Passes 25 Hz samples into RealTimeTraditionalDataPipeline (online feature extraction).
      4. Feeds emitted feature vectors into the trained model to make real-time inferences.
      5. Measures and logs latencies to real_time_summary.json.
    """
    mode_config = config.get("real_time_equivital", {})
    if not mode_config:
        logger.warning("No 'real_time_equivital' configuration found in config.")
        return

    model_type = mode_config["model_type"]
    random_seed: int = int(mode_config.get("random_seed", 42))
    model_names: list[str] = mode_config["models"]
    num_splits: int = int(mode_config.get("num_splits", 10))
    stream_groups: list[list[str]] = mode_config.get("streams", [["ECG", "Centrifuge"]])
    manual_ablation: bool = bool(mode_config.get("manual_ablation", True))
    use_real_time_sleep: bool = bool(mode_config.get("use_real_time_sleep", False))
    max_stream_samples: Optional[int] = mode_config.get("max_stream_samples", None)
    use_preprocessor: bool = bool(mode_config.get("use_preprocessor", True))
    trial_id_cfg: Optional[str] = mode_config.get("trial_id", None)

    saved_models_folder = Path(mode_config.get("saved_models_folder", "Results/Sensor_Ablation_Real_Time"))
    if not saved_models_folder.is_absolute():
        saved_models_folder = project_root_path / saved_models_folder

    save_results_folder = Path(mode_config.get("save_results_folder", "Results/Real_Time_Prediction_Latency"))
    if not save_results_folder.is_absolute():
        save_results_folder = project_root_path / save_results_folder

    session_dir_cfg = mode_config.get(
        "session_dir",
        "Extra spin data/142_HSP_Training_HSP_135_20260508_115626",
    )
    session_dir = Path(session_dir_cfg)
    if not session_dir.is_absolute():
        if not session_dir.exists() and (project_root_path / session_dir).exists():
            session_dir = project_root_path / session_dir
        elif session_dir.exists():
            session_dir = session_dir.resolve()

    model_type_folder = model_type.get_folder_name()
    session_label = session_dir.name if use_preprocessor else f"trial={trial_id_cfg or 'first'}"

    logger.info(
        "Starting real_time_equivital: models=%s, streams=%s, model_type=%s, source=%s, use_preprocessor=%s",
        model_names,
        stream_groups,
        model_type_folder,
        session_label,
        use_preprocessor,
    )

    pipeline.set_random_seed(random_seed)
    pipeline.set_model_type(model_type)

    feature_group = "raw" if manual_ablation else "cache"

    trad_params = config.get("traditional_data_parameters", {})
    adv_params = config.get("advanced_data_parameters", {})
    baseline_window_s = float(adv_params.get("baseline_window", 32.5))
    backstep_s = float(trad_params.get("backstep", 15.0))
    stream_rate_hz = float(trad_params.get("data_rate", 25.0))
    data_rate_hz = stream_rate_hz

    for model_name in model_names:
        logger.info("Processing model: %s", model_name)
        model_instance = model_factory.create_model(model_name)

        if not getattr(model_instance, "is_traditional_model", False):
            raise NotImplementedError(
                f"Model '{model_name}' is not a traditional model. "
                "Real-time evaluation currently supports only traditional (sklearn) saved models."
            )

        for stream_group in stream_groups:
            if use_preprocessor:
                _run_real_time_with_preprocessor(
                    model_name=model_name,
                    stream_group=stream_group,
                    model_instance=model_instance,
                    model_type=model_type,
                    model_type_folder=model_type_folder,
                    pipeline=pipeline,
                    feature_group=feature_group,
                    num_splits=num_splits,
                    saved_models_folder=saved_models_folder,
                    save_results_folder=save_results_folder,
                    session_dir=session_dir,
                    use_real_time_sleep=use_real_time_sleep,
                    max_stream_samples=max_stream_samples,
                    config=config,
                )
            else:
                _run_real_time_without_preprocessor(
                    model_name=model_name,
                    stream_group=stream_group,
                    model_instance=model_instance,
                    model_type=model_type,
                    model_type_folder=model_type_folder,
                    pipeline=pipeline,
                    feature_group=feature_group,
                    num_splits=num_splits,
                    saved_models_folder=saved_models_folder,
                    save_results_folder=save_results_folder,
                    use_real_time_sleep=use_real_time_sleep,
                    max_stream_samples=max_stream_samples,
                    baseline_window_s=baseline_window_s,
                    backstep_s=backstep_s,
                    stream_rate_hz=stream_rate_hz,
                    data_rate_hz=data_rate_hz,
                )

    logger.info("real_time_equivital complete.")
