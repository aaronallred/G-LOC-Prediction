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
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.Data_Pipeline.real_time_data_pipeline import (
    RealTimeDataPreprocessor,
    RealTimeTraditionalDataPipeline,
    SingleSubjectLSLStreamer,
)
from src.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)

_LATENCY_PERCENTILES: tuple[int, ...] = (50, 95, 99)


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
    logger.info("Saved real-time summary report to %s", report_path)


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

    logger.info(
        "Starting real_time_equivital: models=%s, streams=%s, model_type=%s, session=%s",
        model_names,
        stream_groups,
        model_type_folder,
        session_dir.name,
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

        for stream_group in stream_groups:
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

            # 2. Instantiate RealTimeTraditionalDataPipeline
            rt_pipeline = RealTimeTraditionalDataPipeline(
                artifacts=artifacts_path,
                config=config,
                model=model_instance,
                participant_baseline_rhr=72.0,
            )

            # 3. Instantiate LSL Streamer
            playback_speed = 1.0 if use_real_time_sleep else 0.0
            streamer = SingleSubjectLSLStreamer(
                session_dir=session_dir,
                playback_speed=playback_speed,
                source_id_prefix=f"RT_{model_name}_{stream_str}_",
            )

            # 4. Start background streaming process (publishes LSL outlets)
            rt_pipeline.reset()
            streamer.start()

            # 5. Instantiate RealTimeDataPreprocessor and connect
            preprocessor = RealTimeDataPreprocessor(
                raw_feature_names=rt_pipeline.raw_feature_names,
                stream_names=streamer.stream_names,
            )
            preprocessor.connect(timeout=2.0)

            # 6. Stream and infer
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

            try:
                while streamer.is_alive() or True:
                    samples = preprocessor.poll_samples(timeout=0.0, return_latency=True)
                    if not samples:
                        if not streamer.is_alive():
                            break
                        time.sleep(0.002)
                        continue

                    for sample_25hz, t_target, preproc_lat_ms in samples:
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

                    if max_stream_samples is not None and n_raw_samples >= max_stream_samples:
                        break
            finally:
                preprocessor.close()
                time.sleep(0.05)
                streamer.close()
                time.sleep(0.05)

            # 6. Aggregate latencies
            preproc_summary = _summarize_latencies(per_sample_preproc_latencies_ms)
            data_proc_summary = _summarize_latencies(data_proc_latencies_ms)
            inference_summary = _summarize_latencies(inference_latencies_ms)
            total_latency_summary = _summarize_latencies(total_latencies_ms)
            pred_to_pred_summary = _summarize_latencies(prediction_to_prediction_latencies_ms)

            report = {
                "model": model_name,
                "model_type": model_type_folder,
                "streams": stream_group,
                "trial_id": session_dir.name,
                "n_raw_samples": int(n_raw_samples),
                "n_predictions": int(len(predictions)),
                "use_real_time_sleep": use_real_time_sleep,
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

            report_path = (
                save_results_folder / model_type_folder / model_name / stream_str / "real_time_summary.json"
            )
            _write_report(report, report_path)

            logger.info(
                "Completed %s | streams=%s | session=%s | predictions=%d | "
                "preproc mean=%.4f ms | data_proc mean=%.4f ms | infer mean=%.4f ms | "
                "total mean=%.4f ms (p95=%.4f ms) | pred_to_pred mean=%.4f ms",
                model_name,
                stream_str,
                session_dir.name,
                len(predictions),
                preproc_summary.get("mean", 0.0),
                data_proc_summary.get("mean", 0.0),
                inference_summary.get("mean", 0.0),
                total_latency_summary.get("mean", 0.0),
                total_latency_summary.get("p95", 0.0),
                pred_to_pred_summary.get("mean", 0.0),
            )

    logger.info("real_time_equivital complete.")
