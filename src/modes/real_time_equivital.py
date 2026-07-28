"""Real-time Equivital streaming evaluation mode for G-LOC prediction.

This mode mimics the data processing used to train sensor-ablation models
(see ``configs/real_time_sensor_ablation.yaml``) so that a previously saved
fold model can be evaluated on its exact, fold-matched held-out test set
(no data leakage). The test set is then streamed through an LSL outlet one
sample at a time to emulate real-time arrival, with per-sample inference
latency measured for downstream analysis.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import joblib
import numpy as np
from imblearn.metrics import geometric_mean_score
from pylsl import StreamInfo, StreamOutlet, local_clock
from sklearn import metrics

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)

REAL_TIME_STREAM_RATE_HZ: float = 25.0
_LATENCY_PERCENTILES: tuple[int, ...] = (50, 95, 99)


def _evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the same six metrics saved in the sensor-ablation summary JSON."""
    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(metrics.recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "g_mean": float(geometric_mean_score(y_true, y_pred)),
    }


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


def _aggregate_metrics(per_fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean/std across folds for every metric key."""
    if not per_fold_metrics:
        return {}
    keys = per_fold_metrics[0].keys()
    aggregated: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([m[key] for m in per_fold_metrics], dtype=np.float64)
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return aggregated


def _load_saved_summary(saved_summary_path: Path) -> Optional[dict[str, Any]]:
    """Load the sensor-ablation summary.json if it exists."""
    if not saved_summary_path.exists():
        logger.warning("Saved summary not found at %s; comparison block will be empty.", saved_summary_path)
        return None
    with saved_summary_path.open("r") as handle:
        return json.load(handle)


def _aggregate_saved_metrics(saved_summary: Optional[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate the per-fold arrays stored in the sensor-ablation summary JSON."""
    if not saved_summary:
        return {}
    performance = saved_summary.get("performance", {})
    aggregated: dict[str, dict[str, float]] = {}
    for metric, values in performance.items():
        arr = np.asarray(values, dtype=np.float64)
        aggregated[metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    return aggregated


class EquivitalDataStreamer:
    """Stream a processed test matrix through an LSL outlet one row at a time.

    Parameters
    ----------
    data_matrix : np.ndarray
        Processed feature matrix to stream (2D).
    labels : Optional[np.ndarray]
        Corresponding labels for the rows (used for logging only).
    channel_names : Optional[list[str]]
        Channel names pushed as LSL metadata.
    stream_name : str
        LSL stream name.
    stream_type : str
        LSL stream type.
    """

    STREAM_RATE_HZ: float = REAL_TIME_STREAM_RATE_HZ

    def __init__(
            self,
            data_matrix: np.ndarray,
            labels: Optional[np.ndarray] = None,
            channel_names: Optional[list[str]] = None,
            stream_name: str = "GLOC-Equivital",
            stream_type: str = "PsychoPhys",
    ) -> None:
        self.data_matrix = data_matrix
        self.labels = labels
        self.channel_names = channel_names
        self.stream_name = stream_name
        self.stream_type = stream_type
        self._outlet: Optional[Any] = None

    def _create_outlet(self, n_channels: int) -> Optional[Any]:
        if StreamInfo is None or StreamOutlet is None:
            logger.error(
                "pylsl is not installed. Cannot create LSL outlet. "
                "Install it with: pip install pylsl"
            )
            return None

        stream_info = StreamInfo(
            name=self.stream_name,
            type=self.stream_type,
            channel_count=n_channels,
            nominal_srate=self.STREAM_RATE_HZ,
            channel_format="float32",
            source_id="gloc-equivital-01",
        )

        if self.channel_names:
            chns = stream_info.desc().append_child("channels")
            for ch_name in self.channel_names:
                ch = chns.append_child("channel")
                ch.append_child_value("label", ch_name)

        return StreamOutlet(stream_info)

    def stream(
            self,
            *,
            use_real_time_sleep: bool = False,
            on_sample: Optional[Callable[[np.ndarray, int], None]] = None,
    ) -> None:
        """Push samples one at a time through LSL.

        Parameters
        ----------
        use_real_time_sleep : bool
            If True, sleep to pace at the nominal stream rate; else push as fast as possible.
        on_sample : Optional[Callable[[np.ndarray, int], None]]
            Callback invoked with (sample, sample_index) after each sample is pushed.
        """
        logger.info("Starting Equivital data streaming (real-time sleep=%s)...", use_real_time_sleep)

        n_samples = self.data_matrix.shape[0]
        if n_samples == 0:
            logger.warning("Data matrix is empty; nothing to stream.")
            return

        n_channels = self.data_matrix.shape[1]
        self._outlet = self._create_outlet(n_channels)
        if self._outlet is None:
            logger.warning("LSL outlet not available; falling back to no-op logging.")
            return

        logger.info(
            "Streaming %d samples with %d channels at %.2f Hz",
            n_samples,
            n_channels,
            self.STREAM_RATE_HZ,
        )

        sleep_interval = 1.0 / self.STREAM_RATE_HZ
        start_time = local_clock() if local_clock is not None else 0.0
        sent_samples = 0

        try:
            for i in range(n_samples):
                sample = self.data_matrix[i]
                self._outlet.push_sample(sample.astype(np.float32, copy=False))
                sent_samples += 1

                if sent_samples % 10000 == 0:
                    logger.info("Streamed sample %d/%d", sent_samples, n_samples)

                if on_sample is not None:
                    on_sample(sample, i)

                if use_real_time_sleep and local_clock is not None:
                    elapsed = local_clock() - start_time
                    required_samples = int(self.STREAM_RATE_HZ * elapsed)
                    if sent_samples >= required_samples:
                        time.sleep(sleep_interval)

        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user.")
        except Exception as exc:
            logger.exception("Error during streaming: %s", exc)
        finally:
            logger.info("Finished streaming. Sent %d/%d samples.", sent_samples, n_samples)


def _resolve_saved_model_path(
        saved_models_folder: Path,
        model_type_folder: str,
        model_name: str,
        stream_str: str,
        kfold_id: int,
) -> Path:
    return saved_models_folder / model_type_folder / model_name / stream_str / f"fold_{kfold_id}.pkl"


def _stream_and_predict(
        loaded_model: Any,
        X_test: np.ndarray,
        use_real_time_sleep: bool,
        channel_names: Optional[list[str]],
) -> tuple[np.ndarray, list[float]]:
    """Stream ``X_test`` through LSL one row at a time, predicting per sample.

    Returns ``(y_pred, latencies_ms)``.
    """
    n_samples, n_features = X_test.shape
    predictions = np.empty(n_samples, dtype=np.int64)
    latencies: list[float] = []

    streamer = EquivitalDataStreamer(
        data_matrix=X_test,
        channel_names=channel_names,
    )

    def on_sample(sample: np.ndarray, idx: int) -> None:
        t0 = time.perf_counter()
        pred = loaded_model.predict(sample.reshape(1, -1))
        t1 = time.perf_counter()
        predictions[idx] = int(pred[0])
        latencies.append((t1 - t0) * 1000.0)

        if (idx + 1) % 10000 == 0:
            logger.info("Predicted sample %d/%d (latency=%.4f ms)", idx + 1, n_samples, latencies[-1])

    streamer.stream(use_real_time_sleep=use_real_time_sleep, on_sample=on_sample)

    return predictions, latencies


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
    """Run real-time equivital streaming evaluation against saved sensor-ablation models.

    For each model and stream group, iterates all folds, loads the fold-matched saved
    sklearn estimator, rebuilds the matching test split via ``pipeline.get_data`` with
    manual ablation (mirroring ``configs/real_time_sensor_ablation.yaml``), streams the
    test set through LSL while measuring per-sample inference latency, and writes a
    summary report comparing recomputed metrics against the saved summary.

    Parameters
    ----------
    config : dict
        Loaded experiment configuration YAML mapping.
    pipeline : DataPipeline
        Data pipeline facade (must be set with random_seed and model_type prior to use).
    model_factory : ModelFactory
        Factory for creating model instances (used only for metadata/feature-stream lookup).
    project_root_path : Path
        Absolute path to the project root directory.
    """
    mode_config = config.get("real_time_equivital", {})
    if not mode_config:
        logger.warning("No 'real_time_equivital' configuration found in config.")
        return

    model_type = mode_config["model_type"]
    random_seed: int = int(mode_config.get("random_seed", 42))
    model_names: list[str] = mode_config["models"]
    num_splits: int = int(mode_config["num_splits"])
    stream_groups: list[list[str]] = mode_config.get("streams", [["ECG", "Centrifuge"]])
    manual_ablation: bool = bool(mode_config.get("manual_ablation", True))
    class_weight = mode_config.get("class_weight", None)
    use_real_time_sleep: bool = bool(mode_config.get("use_real_time_sleep", False))

    saved_models_folder = Path(mode_config.get("saved_models_folder", "Results/Sensor_Ablation_Real_Time"))
    if not saved_models_folder.is_absolute():
        saved_models_folder = project_root_path / saved_models_folder
    if not saved_models_folder.exists():
        raise FileNotFoundError(
            f"saved_models_folder not found: {saved_models_folder}. "
            "Run sensor_ablation.training first to produce per-fold saved models."
        )

    save_results_folder = Path(mode_config.get("save_results_folder", "Results/Real_Time_Equivital"))
    if not save_results_folder.is_absolute():
        save_results_folder = project_root_path / save_results_folder

    model_type_folder = model_type.get_folder_name()

    logger.info(
        "Starting real_time_equivital: models=%s, streams=%s, num_splits=%d, model_type=%s",
        model_names, stream_groups, num_splits, model_type_folder,
    )

    pipeline.set_random_seed(random_seed)
    pipeline.set_model_type(model_type)

    feature_group = "raw" if manual_ablation else "cache"
    logger.info("Feature selection strategy: %s (manual_ablation=%s)", feature_group, manual_ablation)

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

            per_fold_reports: list[dict[str, Any]] = []
            recomputed_metrics: list[dict[str, float]] = []
            all_latencies_ms: list[float] = []
            n_test_total = 0

            for kfold_id in range(num_splits):
                logger.info("Fold %d/%d", kfold_id + 1, num_splits)

                X_train, X_test, y_train, y_test, select_features = pipeline.get_data(
                    model=model_instance,
                    kfold_id=kfold_id,
                    num_splits=num_splits,
                    feature_streams=stream_group,
                    return_feature_names=True,
                    traditional_feature_selection=feature_group,
                )

                if X_test.shape[0] == 0:
                    raise RuntimeError(
                        f"Fold {kfold_id} produced an empty test split for model={model_name}, "
                        f"streams={stream_str}. Check config / data pipeline."
                    )

                saved_model_path = _resolve_saved_model_path(
                    saved_models_folder, model_type_folder, model_name, stream_str, kfold_id,
                )
                if not saved_model_path.exists():
                    raise FileNotFoundError(
                        f"Saved model not found for fold {kfold_id}: {saved_model_path}. "
                        "Sensor_ablation.training must be run first to produce per-fold estimators."
                    )

                logger.info("Loading saved model from %s", saved_model_path)
                loaded_model = joblib.load(saved_model_path)

                expected_n_features = getattr(loaded_model, "n_features_in_", None)
                if expected_n_features is not None and expected_n_features != X_test.shape[1]:
                    raise ValueError(
                        f"Saved model for fold {kfold_id} expects {expected_n_features} features, "
                        f"but rebuilt X_test has {X_test.shape[1]}. "
                        "Config / data pipeline drift detected — saved model cannot be trusted on this test set."
                    )

                y_pred, latencies = _stream_and_predict(
                    loaded_model=loaded_model,
                    X_test=X_test,
                    use_real_time_sleep=use_real_time_sleep,
                    channel_names=select_features,
                )

                fold_metrics = _evaluate_model(np.asarray(y_test).ravel(), y_pred)
                latency_summary = _summarize_latencies(latencies)

                logger.info(
                    "Fold %d | %s | streams=%s | n_test=%d | f1=%.4f | lat_mean=%.4f ms",
                    kfold_id, model_name, stream_str, X_test.shape[0],
                    fold_metrics["f1"], latency_summary.get("mean", 0.0),
                )

                recomputed_metrics.append(fold_metrics)
                all_latencies_ms.extend(latencies)
                n_test_total += int(X_test.shape[0])

                per_fold_reports.append({
                    "fold_id": kfold_id,
                    "n_test": int(X_test.shape[0]),
                    "metrics_recomputed": fold_metrics,
                    "latency_ms": latency_summary,
                    "per_sample_latency_ms": latencies,
                })

            saved_summary_path = (
                    saved_models_folder / model_type_folder / model_name / stream_str / "summary.json"
            )
            saved_summary = _load_saved_summary(saved_summary_path)

            per_fold_saved_metrics: list[dict[str, float]] = []
            if saved_summary is not None:
                performance = saved_summary.get("performance", {})
                n_folds_reported = len(performance.get("f1", []))
                for idx in range(n_folds_reported):
                    per_fold_saved_metrics.append({
                        metric: float(values[idx])
                        for metric, values in performance.items()
                    })

            for fold_report, saved_metrics in zip(per_fold_reports, per_fold_saved_metrics):
                fold_report["metrics_saved"] = saved_metrics

            aggregated_recomputed = _aggregate_metrics(recomputed_metrics)
            aggregated_saved = _aggregate_saved_metrics(saved_summary)
            latency_aggregated = _summarize_latencies(all_latencies_ms)

            report = {
                "model": model_name,
                "model_type": model_type_folder,
                "streams": stream_group,
                "num_splits": num_splits,
                "n_test_total": n_test_total,
                "use_real_time_sleep": use_real_time_sleep,
                "per_fold": per_fold_reports,
                "aggregated_recomputed": aggregated_recomputed,
                "aggregated_saved": aggregated_saved,
                "latency_ms_aggregated": latency_aggregated,
            }

            report_path = (
                    save_results_folder / model_type_folder / model_name / stream_str / "real_time_summary.json"
            )
            _write_report(report, report_path)

            logger.info(
                "Completed %s | streams=%s | recomputed f1_mean=%.4f | saved f1_mean=%.4f | "
                "latency mean=%.4f ms (p95=%.4f ms)",
                model_name,
                stream_str,
                aggregated_recomputed.get("f1", {}).get("mean", 0.0),
                aggregated_saved.get("f1", {}).get("mean", 0.0),
                latency_aggregated.get("mean", 0.0),
                latency_aggregated.get("p95", 0.0),
            )

    logger.info("real_time_equivital complete.")
