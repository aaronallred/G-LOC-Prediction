"""Real-time Equivital streaming evaluation mode for G-LOC prediction.

This mode loads raw 25 Hz sensor telemetry for a single trial and streams it
sample-by-sample into RealTimeTraditionalDataPipeline and over an LSL outlet.
For each emitted prediction, it measures the data processing latency
(feature extraction and standardization) and the model inference latency,
saving detailed latency statistics to real_time_summary.json.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.Data_Pipeline.real_time_data_pipeline import RealTimeTraditionalDataPipeline
from src.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)

REAL_TIME_STREAM_RATE_HZ: float = 25.0
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


class EquivitalDataStreamer:
    """Stream raw sensor telemetry through an LSL outlet.

    Parameters
    ----------
    channel_names : list[str]
        Channel names pushed as LSL metadata.
    stream_name : str
        LSL stream name.
    stream_type : str
        LSL stream type.
    stream_rate_hz : float
        Nominal streaming rate in Hz.
    """

    def __init__(
        self,
        channel_names: list[str],
        stream_name: str = "GLOC-Equivital-Raw",
        stream_type: str = "PsychoPhys",
        stream_rate_hz: float = REAL_TIME_STREAM_RATE_HZ,
    ) -> None:
        self.channel_names = channel_names
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.stream_rate_hz = stream_rate_hz
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
                source_id="gloc-equivital-raw-01",
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


def _load_single_trial_raw_data(
    pipeline: DataPipeline,
    model_instance: Any,
    model_type: Any,
    stream_group: list[str],
    output_feature_dtype: np.dtype = np.dtype(np.float32),
) -> Tuple[np.ndarray, np.ndarray, list[str], str]:
    """Load raw 25 Hz telemetry for a single continuous trial matching requested streams."""
    backend = pipeline._build_backend(model_instance)
    file_paths = backend._get_data_locations()
    gloc_data = backend._load_data(file_paths, output_feature_dtype)

    trad_hparams = backend._resolve_traditional_hyperparameters(model_instance, model_instance.name)
    baseline_methods_to_use = trad_hparams.get("baseline_methods_to_use", ["v0", "v1", "v2", "v5", "v6"])
    feature_groups_to_analyze, baseline_methods_to_use = backend._get_feature_groups_and_baseline_methods(
        model_type, baseline_methods_to_use
    )
    feature_groups_to_analyze, _, _ = backend._resolve_feature_groups_for_streams(
        stream_group, feature_groups_to_analyze
    )
    gloc_data, features = backend._process_and_get_feature_names(
        gloc_data, feature_groups_to_analyze, model_type, file_paths, output_feature_dtype
    )

    unique_trials = pd.unique(gloc_data["trial_id"])
    if len(unique_trials) == 0:
        raise ValueError("No trials found in data.")

    trial_id = str(unique_trials[0])
    trial_df = gloc_data[gloc_data["trial_id"] == trial_id]

    raw_samples = trial_df[features["All"]].to_numpy(dtype=output_feature_dtype)
    timestamps = trial_df["Time (s)"].to_numpy(dtype=np.float64)
    raw_channel_names = list(features["All"])

    return raw_samples, timestamps, raw_channel_names, trial_id


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
    """Run real-time streaming evaluation against saved models on a single raw trial.

    Parameters
    ----------
    config : dict
        Loaded experiment configuration YAML mapping.
    pipeline : DataPipeline
        Data pipeline facade (must be configured with random_seed and model_type prior to use).
    model_factory : ModelFactory
        Factory for creating model instances.
    project_root_path : Path
        Absolute path to project root directory.
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

    saved_models_folder = Path(mode_config.get("saved_models_folder", "Results/Sensor_Ablation_Real_Time"))
    if not saved_models_folder.is_absolute():
        saved_models_folder = project_root_path / saved_models_folder

    save_results_folder = Path(mode_config.get("save_results_folder", "Results/Real_Time_Prediction_Latency"))
    if not save_results_folder.is_absolute():
        save_results_folder = project_root_path / save_results_folder

    model_type_folder = model_type.get_folder_name()

    logger.info(
        "Starting real_time_equivital: models=%s, streams=%s, model_type=%s",
        model_names,
        stream_groups,
        model_type_folder,
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

            # 1. Load raw single trial telemetry
            raw_samples, timestamps, raw_channel_names, trial_id = _load_single_trial_raw_data(
                pipeline=pipeline,
                model_instance=model_instance,
                model_type=model_type,
                stream_group=stream_group,
                output_feature_dtype=np.dtype(np.float32),
            )
            n_raw_samples = raw_samples.shape[0]
            logger.info(
                "Loaded trial '%s' with %d raw samples and %d channels.",
                trial_id,
                n_raw_samples,
                len(raw_channel_names),
            )

            # 2. Check/load fold model & preprocessing artifacts
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
                # Fit and save estimator
                saved_model_path.parent.mkdir(parents=True, exist_ok=True)
                model_instance.train(X_train, y_train)
                model_instance.save_model(str(saved_model_path))

            logger.info("Loading saved model from %s", saved_model_path)
            loaded_model = joblib.load(saved_model_path)

            # 3. Instantiate RealTimeTraditionalDataPipeline
            rt_pipeline = RealTimeTraditionalDataPipeline(
                artifacts=artifacts_path,
                config=config,
                model=model_instance,
                participant_baseline_rhr=72.0,
            )

            # 4. Initialize LSL Streamer
            streamer = EquivitalDataStreamer(
                channel_names=raw_channel_names,
                stream_rate_hz=rt_pipeline.stream_rate_hz,
            )

            # 5. Stream raw samples one-by-one and measure latencies
            data_proc_latencies_ms: list[float] = []
            inference_latencies_ms: list[float] = []
            total_latencies_ms: list[float] = []
            predictions: list[int] = []

            rt_pipeline.reset()

            for idx in range(n_raw_samples):
                sample = raw_samples[idx]
                timestamp = timestamps[idx]

                # Ingest sample & measure feature extraction latency
                X_processed, data_proc_lat_ms = rt_pipeline.ingest_sample(sample, timestamp_s=timestamp)

                # Push raw sample to LSL outlet
                streamer.push_sample(sample)

                # Inference on emitted prediction rows
                if X_processed is not None:
                    t_infer_0 = time.perf_counter()
                    pred = loaded_model.predict(X_processed.reshape(1, -1))
                    t_infer_1 = time.perf_counter()
                    infer_lat_ms = (t_infer_1 - t_infer_0) * 1000.0

                    data_proc_latencies_ms.append(data_proc_lat_ms)
                    inference_latencies_ms.append(infer_lat_ms)
                    total_latencies_ms.append(data_proc_lat_ms + infer_lat_ms)
                    predictions.append(int(pred[0]))

                if use_real_time_sleep:
                    time.sleep(1.0 / rt_pipeline.stream_rate_hz)

            # 6. Aggregate latencies
            data_proc_summary = _summarize_latencies(data_proc_latencies_ms)
            inference_summary = _summarize_latencies(inference_latencies_ms)
            total_latency_summary = _summarize_latencies(total_latencies_ms)

            report = {
                "model": model_name,
                "model_type": model_type_folder,
                "streams": stream_group,
                "trial_id": trial_id,
                "n_raw_samples": int(n_raw_samples),
                "n_predictions": int(len(predictions)),
                "use_real_time_sleep": use_real_time_sleep,
                "data_processing_latency_ms": data_proc_summary,
                "inference_latency_ms": inference_summary,
                "total_latency_ms": total_latency_summary,
                "per_prediction_data_proc_latency_ms": data_proc_latencies_ms,
                "per_prediction_inference_latency_ms": inference_latencies_ms,
                "per_prediction_total_latency_ms": total_latencies_ms,
            }

            report_path = (
                save_results_folder / model_type_folder / model_name / stream_str / "real_time_summary.json"
            )
            _write_report(report, report_path)

            logger.info(
                "Completed %s | streams=%s | trial=%s | predictions=%d | "
                "data_proc mean=%.4f ms | infer mean=%.4f ms | total mean=%.4f ms (p95=%.4f ms)",
                model_name,
                stream_str,
                trial_id,
                len(predictions),
                data_proc_summary.get("mean", 0.0),
                inference_summary.get("mean", 0.0),
                total_latency_summary.get("mean", 0.0),
                total_latency_summary.get("p95", 0.0),
            )

    logger.info("real_time_equivital complete.")

