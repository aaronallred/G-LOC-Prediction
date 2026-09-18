"""Real-Time Traditional Data Pipeline for G-LOC Prediction.

Ingests raw telemetry samples (e.g. ECG, HR, BR, Temperature, Centrifuge) one at a time,
buffers samples during an initial warmup period, and upon warmup completion computes
causal derivatives, baseline transformations, sliding-window statistics, and
fold-aware pooled s1 / global s2 standardization to emit the processed feature vector
matching the trained model's input space.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pylsl

logger = logging.getLogger(__name__)


def _guard_divide(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalization with a zero-variance guard."""
    out = np.zeros_like(data, dtype=np.float64)
    non_zero = std != 0
    if np.any(non_zero):
        out[non_zero] = (data[non_zero] - mean[non_zero]) / std[non_zero]
    return out


class RealTimeTraditionalDataPipeline:
    """Online streaming feature extraction pipeline for traditional G-LOC models.

    Mimics the exact mathematical processing steps of TraditionalDataPipeline,
    loading precomputed training states (s1/s2 standardizers, KNN imputer reference data,
    active feature indices) and actively applying them sample-by-sample.

    Parameters
    ----------
    artifacts: Union[str, Path, Dict[str, Any]]
        Path to preprocessing_artifacts.json or loaded dictionary containing learned states.
    config: Optional[Dict[str, Any]]
        Optional loaded YAML experiment configuration mapping.
    model: Optional[Any]
        Optional model instance providing data_pipeline_hyperparameters.
    participant_baseline_rhr: float
        Resting heart rate (seated) for participant (used for v5/v6 baselines, default: 72.0).
    stream_rate_hz: Optional[float]
        Nominal streaming rate in Hz (default: resolved from config or 25.0 Hz).
    standardize_s1: Optional[bool]
        Whether intra-trial (s1) standardization is enabled (default: resolved from config/artifacts or True).
    window_size: Optional[float]
        Window duration in seconds (default: resolved from model/config or 15.0).
    baseline_window: Optional[float]
        Baseline window duration in seconds (default: resolved from model/config or 32.5).
    stride: Optional[float]
        Stride step in seconds (default: resolved from model/config or 0.25).
    """

    def __init__(
        self,
        artifacts: Union[str, Path, Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        model: Optional[Any] = None,
        participant_baseline_rhr: float = 72.0,
        stream_rate_hz: Optional[float] = None,
        standardize_s1: Optional[bool] = None,
        time_start: Optional[float] = None,
        offset: Optional[float] = None,
        window_size: Optional[float] = None,
        baseline_window: Optional[float] = None,
        stride: Optional[float] = None,
    ) -> None:
        if isinstance(artifacts, (str, Path)):
            with open(artifacts, "r") as f:
                self.artifacts = json.load(f)
        else:
            self.artifacts = dict(artifacts)

        self.config = config or {}
        self.model = model

        trad_params = self.config.get("traditional_data_parameters", {})

        # Resolve stream rate
        if stream_rate_hz is not None:
            self.stream_rate_hz = float(stream_rate_hz)
        elif "data_rate" in trad_params:
            self.stream_rate_hz = float(trad_params["data_rate"])
        else:
            self.stream_rate_hz = 25.0
        self.dt = 1.0 / self.stream_rate_hz

        # Resolve model hyperparameters if provided
        model_hparams: Dict[str, Any] = {}
        if self.model is not None and hasattr(self.model, "data_pipeline_hyperparameters"):
            model_hparams = self.model.data_pipeline_hyperparameters or {}

        self.baseline_methods_to_use = list(
            model_hparams.get("baseline_methods_to_use", ["v0", "v1", "v2", "v5", "v6"])
        )

        # Resolve window settings
        if baseline_window is not None:
            self.baseline_window_s = float(baseline_window)
        elif "baseline_window" in model_hparams:
            self.baseline_window_s = float(model_hparams["baseline_window"])
        elif "advanced_data_parameters" in self.config and "baseline_window" in self.config["advanced_data_parameters"]:
            self.baseline_window_s = float(self.config["advanced_data_parameters"]["baseline_window"])
        else:
            self.baseline_window_s = float(self.artifacts.get("baseline_window", 32.5))

        if window_size is not None:
            self.window_size_s = float(window_size)
        elif "window_size" in model_hparams:
            self.window_size_s = float(model_hparams["window_size"])
        else:
            self.window_size_s = float(self.artifacts.get("window_size", 15.0))

        if stride is not None:
            self.stride_s = float(stride)
        elif "stride" in model_hparams:
            self.stride_s = float(model_hparams["stride"])
        elif "stride" in trad_params:
            self.stride_s = float(trad_params["stride"])
        else:
            self.stride_s = float(self.artifacts.get("stride", 0.25))

        # Resolve time_start and offset
        if time_start is not None:
            self.time_start = float(time_start)
        elif "time_start" in trad_params:
            self.time_start = float(trad_params["time_start"])
        else:
            self.time_start = 0.0

        if offset is not None:
            self.offset = float(offset)
        elif "offset" in trad_params:
            self.offset = float(trad_params["offset"])
        else:
            self.offset = 0.0

        # Resolve s1 standardization flag
        if standardize_s1 is not None:
            self.standardize_s1 = bool(standardize_s1)
        elif "standardize_s1" in trad_params:
            self.standardize_s1 = bool(trad_params["standardize_s1"])
        else:
            self.standardize_s1 = bool(self.artifacts.get("standardize_s1", True))

        self.participant_baseline_rhr = float(participant_baseline_rhr)

        # Precomputed standardization parameters
        self.s1_pooled_mean = np.asarray(self.artifacts.get("s1_pooled_mean", []), dtype=np.float64)
        self.s1_pooled_std = np.asarray(self.artifacts.get("s1_pooled_std", []), dtype=np.float64)
        self.s2_global_mean = np.asarray(self.artifacts.get("s2_global_mean", []), dtype=np.float64)
        self.s2_global_std = np.asarray(self.artifacts.get("s2_global_std", []), dtype=np.float64)

        # Feature selection indices
        self.active_indices = np.asarray(self.artifacts.get("active_indices", []), dtype=np.int64)
        self.active_feature_names = self.artifacts.get("active_feature_names", [])
        self.raw_feature_names = self.artifacts.get("raw_feature_names", [])
        self.engineered_feature_names = self.artifacts.get(
            "engineered_feature_names", self.artifacts.get("raw_feature_names", [])
        )

        # Buffer sizing
        self.n_baseline_samples = max(1, int(round(self.baseline_window_s * self.stream_rate_hz)))
        self.n_window_samples = max(1, int(round(self.window_size_s * self.stream_rate_hz)))
        self.warmup_samples_required = max(self.n_baseline_samples, self.n_window_samples)

        # Setup FAISS KNN Imputer state if present in artifacts
        self._setup_knn_imputer()

        # Buffers and state
        self._samples_seen = 0
        self._baseline_buffer: List[np.ndarray] = []
        self._window_buffer: List[np.ndarray] = []
        self._raw_timestamps: List[float] = []

        # Stride-aligned window scheduling
        self._next_window_idx = 0
        self._target_timestamp = self.time_start + self.window_size_s

        # Frozen trial baseline values (computed once warmup finishes)
        self._v1_baseline_mean: Optional[np.ndarray] = None
        self._v2_baseline_mean: Optional[np.ndarray] = None
        self._is_warmed_up = False

    def _setup_knn_imputer(self) -> None:
        knn_info = self.artifacts.get("knn_imputer", {})
        self.knn_k = int(knn_info.get("k", 5))
        ref_means = knn_info.get("reference_means", [])
        ref_data = knn_info.get("reference_data", [])

        self.knn_ref_means = np.asarray(ref_means, dtype=np.float32) if ref_means else None
        self.knn_ref_data = np.asarray(ref_data, dtype=np.float32) if ref_data else None
        self._faiss_index = None

        if self.knn_ref_data is not None and self.knn_ref_data.size > 0:
            try:
                import faiss
                d = self.knn_ref_data.shape[1]
                self._faiss_index = faiss.IndexFlatL2(d)
                self._faiss_index.add(self.knn_ref_data)
            except Exception as exc:
                logger.warning("Could not initialize FAISS index: %s", exc)

    def impute_sample(self, raw_sample: np.ndarray) -> np.ndarray:
        """Impute missing values in an incoming raw telemetry sample using precomputed KNN reference data."""
        if not np.isnan(raw_sample).any():
            return raw_sample

        sample_out = raw_sample.copy()
        mask = np.isnan(sample_out)

        if self._faiss_index is not None and self.knn_ref_means is not None:
            # Temporary mean fill for query vector
            q = np.where(mask, self.knn_ref_means[:len(sample_out)], sample_out).astype(np.float32)
            distances, indices = self._faiss_index.search(q.reshape(1, -1), self.knn_k)
            for j in np.flatnonzero(mask):
                sample_out[j] = np.nanmean(self.knn_ref_data[indices[0], j])
        elif self.knn_ref_means is not None:
            sample_out[mask] = self.knn_ref_means[:len(sample_out)][mask]

        return sample_out

    def reset(self) -> None:
        """Reset internal buffers when starting a new streaming trial or participant session."""
        self._samples_seen = 0
        self._baseline_buffer.clear()
        self._window_buffer.clear()
        self._raw_timestamps.clear()
        self._v1_baseline_mean = None
        self._v2_baseline_mean = None
        self._is_warmed_up = False
        self._next_window_idx = 0
        self._target_timestamp = self.time_start + self.window_size_s

    @property
    def is_warmed_up(self) -> bool:
        """Whether the pipeline has completed the initial warmup period."""
        return self._is_warmed_up

    def ingest_sample(
        self,
        raw_sample: np.ndarray,
        timestamp_s: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Ingest a single raw telemetry sample at timestamp t.

        Parameters
        ----------
        raw_sample : np.ndarray
            1D float array of raw physical sensor values.
        timestamp_s : Optional[float]
            Current stream timestamp in seconds.

        Returns
        -------
        Tuple[Optional[np.ndarray], float]
            (X_processed, data_proc_latency_ms)
            - Returns (None, latency_ms) during warmup or between stride intervals.
            - Returns (X_processed_row, latency_ms) on each completed stride window.
        """
        t0 = time.perf_counter()

        # 1. Impute missing values if needed
        sample = self.impute_sample(np.asarray(raw_sample, dtype=np.float64))

        self._samples_seen += 1
        curr_time = float(timestamp_s) if timestamp_s is not None else (self._samples_seen * self.dt)
        self._raw_timestamps.append(curr_time)

        # 2. Maintain circular buffers
        if len(self._baseline_buffer) < self.n_baseline_samples:
            self._baseline_buffer.append(sample)

        self._window_buffer.append(sample)
        if len(self._window_buffer) > self.n_window_samples:
            self._window_buffer.pop(0)
            self._raw_timestamps.pop(0)

        # 3. Check warmup gate and stride-aligned target timestamp
        if self._samples_seen < self.warmup_samples_required or curr_time < (self._target_timestamp - 1e-6):
            t1 = time.perf_counter()
            return None, (t1 - t0) * 1000.0

        # 4. Freeze baseline upon completing warmup
        if not self._is_warmed_up:
            base_arr = np.asarray(self._baseline_buffer, dtype=np.float64)
            # Physical features baseline mean
            mean_base = np.nanmean(base_arr, axis=0)

            # v1 baseline: division guard
            v1_base = np.nan_to_num(mean_base, nan=1.0)
            v1_base = np.where(v1_base == 0, 1.0, v1_base)
            self._v1_baseline_mean = v1_base

            # v2 baseline: subtraction guard
            self._v2_baseline_mean = np.nan_to_num(mean_base, nan=0.0)
            self._is_warmed_up = True

        # 5. Extract features from current window buffer
        win_arr = np.asarray(self._window_buffer, dtype=np.float64)
        X_processed = self._compute_window_features(win_arr)

        # 6. Advance stride schedule to next window
        self._next_window_idx += 1
        self._target_timestamp = self.time_start + self.window_size_s + (self._next_window_idx * self.stride_s)

        t1 = time.perf_counter()
        data_proc_latency_ms = (t1 - t0) * 1000.0

        return X_processed, data_proc_latency_ms

    def _compute_window_features(self, win_arr: np.ndarray) -> np.ndarray:
        """Compute all baseline transformations, window statistics, and standardization."""
        n_samples = win_arr.shape[0]
        time_arr = np.arange(n_samples, dtype=np.float64) * self.dt

        # Helper to compute numerical derivative stack (val, 1st deriv, 2nd deriv)
        def get_deriv_stack(vals: np.ndarray) -> np.ndarray:
            d1 = np.gradient(vals, time_arr, axis=0)
            d2 = np.gradient(d1, time_arr, axis=0)
            return np.hstack([vals, d1, d2])

        blocks = []
        # v0: Raw signals + derivatives
        if "v0" in self.baseline_methods_to_use:
            blocks.append(get_deriv_stack(win_arr))

        # v1: Physical / baseline + derivatives
        if "v1" in self.baseline_methods_to_use:
            blocks.append(get_deriv_stack(win_arr / self._v1_baseline_mean))

        # v2: Physical - baseline + derivatives
        if "v2" in self.baseline_methods_to_use:
            blocks.append(get_deriv_stack(win_arr - self._v2_baseline_mean))

        # v5: ECG / Resting HR + derivatives (ECG channels are index 0..min(6, win_arr.shape[1]))
        if "v5" in self.baseline_methods_to_use:
            ecg_cols = win_arr[:, :min(6, win_arr.shape[1])]
            blocks.append(get_deriv_stack(ecg_cols / self.participant_baseline_rhr))

        # v6: ECG - Resting HR + derivatives
        if "v6" in self.baseline_methods_to_use:
            ecg_cols = win_arr[:, :min(6, win_arr.shape[1])]
            blocks.append(get_deriv_stack(ecg_cols - self.participant_baseline_rhr))

        if not blocks:
            blocks.append(get_deriv_stack(win_arr))

        # Combine all requested baseline arrays horizontally
        combined_baseline = np.hstack(blocks)

        # Sliding window summary statistics across window axis=0
        with np.errstate(all="ignore"):
            mean_stat = np.nan_to_num(np.nanmean(combined_baseline, axis=0), nan=0.0)
            std_stat = np.nan_to_num(np.nanstd(combined_baseline, axis=0), nan=0.0)
            max_stat = np.nan_to_num(np.nanmax(combined_baseline, axis=0), nan=0.0)
            min_stat = np.nan_to_num(np.nanmin(combined_baseline, axis=0), nan=0.0)
            range_stat = max_stat - min_stat

        # Additional HRV features from index 0 (HR in bpm)
        hr_window = win_arr[:, 0]
        valid_hr = (hr_window > 0) & np.isfinite(hr_window)
        if np.sum(valid_hr) >= 2:
            with np.errstate(divide="ignore", invalid="ignore"):
                rr_interval = 60000.0 / hr_window[valid_hr]
                hrv_sdnn = float(np.nanstd(rr_interval))
                diff_rr = np.diff(rr_interval)
                valid_diff = diff_rr[np.isfinite(diff_rr)]
                hrv_rmssd = float(np.sqrt(np.mean(valid_diff ** 2))) if valid_diff.size > 0 else 0.0
        else:
            hrv_sdnn = 0.0
            hrv_rmssd = 0.0

        # Stack raw feature vector X_raw: [mean, std, max, range, hrv_sdnn, hrv_rmssd]
        X_raw = np.hstack([mean_stat, std_stat, max_stat, range_stat, [hrv_sdnn, hrv_rmssd]])

        # 6. Apply fold-aware standardization
        d_expected = len(self.s2_global_mean) if len(self.s2_global_mean) > 0 else len(self.s1_pooled_mean)
        if d_expected > 0:
            if len(X_raw) > d_expected:
                X_raw = X_raw[:d_expected]
            elif len(X_raw) < d_expected:
                padded = np.zeros(d_expected, dtype=np.float64)
                padded[:len(X_raw)] = X_raw
                X_raw = padded

        if self.standardize_s1 and len(self.s1_pooled_mean) > 0:
            X_s1 = _guard_divide(X_raw, self.s1_pooled_mean, self.s1_pooled_std)
            X_s2 = _guard_divide(X_raw, self.s2_global_mean, self.s2_global_std)
            X_std = np.hstack([X_s1, X_s2])
        else:
            X_s2 = _guard_divide(X_raw, self.s2_global_mean, self.s2_global_std)
            X_std = X_s2

        # 7. Subset to active model features
        if len(self.active_indices) > 0 and max(self.active_indices) < len(X_std):
            X_processed = X_std[self.active_indices]
        else:
            X_processed = X_std

        return X_processed

    def process_stream(
        self,
        raw_data_stream: Union[np.ndarray, pd.DataFrame, Iterable[np.ndarray]],
    ) -> Iterator[Tuple[Optional[np.ndarray], float]]:
        """Process a stream of raw telemetry samples generator-style.

        Yields (X_processed_row, latency_ms) for each incoming sample.
        """
        if isinstance(raw_data_stream, pd.DataFrame):
            stream_arr = raw_data_stream.to_numpy()
        elif isinstance(raw_data_stream, np.ndarray):
            stream_arr = raw_data_stream
        else:
            stream_arr = raw_data_stream

        for idx, sample in enumerate(stream_arr):
            yield self.ingest_sample(sample, timestamp_s=idx * self.dt)

    def process_lsl_preprocessor(
        self,
        preprocessor: RealTimeDataPreprocessor,
        max_samples: Optional[int] = None,
        poll_timeout: float = 0.01,
    ) -> Iterator[Tuple[Optional[np.ndarray], float]]:
        """Process streaming 25 Hz samples emitted by RealTimeDataPreprocessor.

        Polls the preprocessor for incoming 25 Hz samples from LSL and feeds them
        into ingest_sample().

        Parameters
        ----------
        preprocessor : RealTimeDataPreprocessor
            Active preprocessor instance connected to LSL streams.
        max_samples : Optional[int]
            Maximum number of samples to process before stopping (default: None, infinite).
        poll_timeout : float
            Timeout in seconds for polling LSL chunks (default: 0.01s).

        Yields
        ------
        Tuple[Optional[np.ndarray], float]
            (X_processed_row, timestamp_s) for each 25 Hz clock sample.
        """
        emitted = 0
        while max_samples is None or emitted < max_samples:
            samples = preprocessor.poll_samples(timeout=poll_timeout)
            for sample_25hz, t_target in samples:
                x_proc, _ = self.ingest_sample(sample_25hz, timestamp_s=t_target)
                emitted += 1
                yield x_proc, t_target
                if max_samples is not None and emitted >= max_samples:
                    break
            if not samples and max_samples is not None:
                time.sleep(poll_timeout)


class SingleSubjectLSLStreamer:
    """Streams multi-rate telemetry data from a single subject session folder via native pylsl outlets.

    Matches the exact stream names, types, channel counts, and nominal sampling rates of the
    AFRL centrifuge spin data environment:
      - 'SA5_SystemData': PDAS, 48 channels @ 100.0 Hz
      - 'Equivital_ECG': ECG, 1 channel @ 256.0 Hz
      - 'Equivital_Summary': ECG, 26 channels @ 0.2 Hz (actual sample transmission rate)
      - 'Equivital_Accel': IMU, 3 channels @ 256.0 Hz

    Parameters
    ----------
    session_dir : Union[str, Path]
        Path to session folder containing CSV stream files and optional export_meta.json.
        Defaults to 'Extra spin data/100_HSP_Training_HSP_165_20260304_125314'.
    playback_speed : float
        Playback speed multiplier. 1.0 corresponds to wall-clock real time.
        <= 0 indicates as fast as possible (for testing/benchmarks).
    max_rows_per_stream : Optional[int]
        Optional row limit when loading stream CSVs (useful for fast testing).
    source_id_prefix : str
        Prefix for LSL stream source IDs.
    """

    def __init__(
        self,
        session_dir: Union[str, Path] = "Extra spin data/100_HSP_Training_HSP_165_20260304_125314",
        playback_speed: float = 0.0,
        max_rows_per_stream: Optional[int] = None,
        source_id_prefix: str = "GLOC_STREAMER_",
    ) -> None:
        self.session_dir = Path(session_dir)
        if not self.session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {self.session_dir}")

        self.playback_speed = float(playback_speed)
        self.max_rows = max_rows_per_stream
        self.source_id_prefix = source_id_prefix

        # Stream descriptors: (key, name_pattern, lsl_name, lsl_type, ch_count, nominal_srate)
        self.stream_configs = {
            "system_data": {
                "pattern": r"SystemData",
                "default_name": "SA5_SystemData",
                "type": "PDAS",
                "ch_count": 48,
                "nominal_srate": 100.0,
            },
            "ecg": {
                "pattern": r"Equivital_ECG\.csv|ECG_EQ",
                "default_name": "Equivital_ECG",
                "type": "ECG",
                "ch_count": 1,
                "nominal_srate": 256.0,
            },
            "summary": {
                "pattern": r"Summary",
                "default_name": "Equivital_Summary",
                "type": "ECG",
                "ch_count": 26,
                "nominal_srate": 0.2,
            },
            "accel": {
                "pattern": r"Accel",
                "default_name": "Equivital_Accel",
                "type": "IMU",
                "ch_count": 3,
                "nominal_srate": 256.0,
            },
        }

        self.outlets: Dict[str, pylsl.StreamOutlet] = {}
        self.stream_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # key -> (timestamps, data_array)
        self.stream_names: Dict[str, str] = {}  # key -> actual lsl stream name

        # Streaming state
        self._pointers: Dict[str, int] = {}
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_streamed_t = 0.0
        self._wall_start_t = 0.0

        self._load_and_init_outlets()

    def _load_and_init_outlets(self) -> None:
        """Discover CSV files, synchronize time anchors, and create pylsl outlets."""
        csv_files = list(self.session_dir.glob("*.csv"))

        meta_file = self.session_dir / "export_meta.json"
        meta_info: Dict[str, Any] = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta_info = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read {meta_file}: {e}")

        # Locate files for each stream
        matched_files: Dict[str, Path] = {}
        if "streams" in meta_info:
            for s in meta_info["streams"]:
                lsl_name = s.get("lsl_name", "")
                csv_name = s.get("csv_file", "")
                csv_p = self.session_dir / csv_name
                if not csv_p.exists():
                    continue
                for key, cfg in self.stream_configs.items():
                    if re.search(cfg["pattern"], lsl_name, re.IGNORECASE) or re.search(cfg["pattern"], csv_name, re.IGNORECASE):
                        if key not in matched_files:
                            matched_files[key] = csv_p
                            self.stream_names[key] = lsl_name
                            cfg["ch_count"] = s.get("channel_count", cfg["ch_count"])
                            # For summary, actual transmission rate is 0.2 Hz
                            if key != "summary":
                                cfg["nominal_srate"] = s.get("nominal_srate", cfg["nominal_srate"])

        # Fallback to directory glob matching
        for key, cfg in self.stream_configs.items():
            if key not in matched_files:
                for f in csv_files:
                    if re.search(cfg["pattern"], f.name, re.IGNORECASE):
                        if key == "ecg" and "RR" in f.name:
                            continue
                        matched_files[key] = f
                        self.stream_names[key] = cfg["default_name"]
                        break

        # Read CSVs and discover earliest UTC anchor
        raw_dfs: Dict[str, pd.DataFrame] = {}
        earliest_utc: Optional[pd.Timestamp] = None

        for key, path in matched_files.items():
            df = pd.read_csv(path, nrows=self.max_rows)
            raw_dfs[key] = df
            if "t_utc" in df.columns:
                t_utc = pd.to_datetime(df["t_utc"], errors="coerce").dropna()
                if len(t_utc) > 0:
                    first_t = t_utc.iloc[0]
                    # Anchor on earliest time among system_data, ecg, accel
                    if key in ("system_data", "ecg", "accel"):
                        if earliest_utc is None or first_t < earliest_utc:
                            earliest_utc = first_t

        # Create outlets and align relative timestamps
        for key, df in raw_dfs.items():
            cfg = self.stream_configs[key]
            lsl_name = self.stream_names.get(key, cfg["default_name"])

            # Compute relative time
            t_lsl = pd.to_numeric(df["t_lsl"], errors="coerce").to_numpy(dtype=np.float64)
            t_lsl_rel = t_lsl - t_lsl[0]

            if earliest_utc is not None and "t_utc" in df.columns:
                t_utc = pd.to_datetime(df["t_utc"], errors="coerce")
                if len(t_utc) > 0 and not pd.isna(t_utc.iloc[0]):
                    offset_s = (t_utc.iloc[0] - earliest_utc).total_seconds()
                    t_lsl_rel = t_lsl_rel + offset_s

            # Ensure non-zero timestamp for liblsl explicit timestamping
            t_lsl_rel = np.maximum(t_lsl_rel, 1e-6)

            # Data columns (everything except t_lsl and t_utc)
            non_data_cols = [c for c in ("t_lsl", "t_utc") if c in df.columns]
            data_cols = [c for c in df.columns if c not in non_data_cols]
            data_mat = df[data_cols].to_numpy(dtype=np.float32)

            ch_count = data_mat.shape[1]
            self.stream_data[key] = (t_lsl_rel, data_mat)
            self._pointers[key] = 0

            # Create LSL outlet
            sinfo = pylsl.StreamInfo(
                name=lsl_name,
                type=cfg["type"],
                channel_count=ch_count,
                nominal_srate=float(cfg["nominal_srate"]),
                channel_format=pylsl.cf_float32,
                source_id=f"{self.source_id_prefix}{key}",
            )
            self.outlets[key] = pylsl.StreamOutlet(sinfo, max_buffered=3600)

        logger.info(
            f"Initialized SingleSubjectLSLStreamer with {len(self.outlets)} streams from {self.session_dir.name}"
        )

    def push_next(self) -> bool:
        """Pushes the next earliest sample chronologically across all streams."""
        earliest_key: Optional[str] = None
        earliest_t = float("inf")

        for key, (times, _) in self.stream_data.items():
            ptr = self._pointers[key]
            if ptr < len(times):
                t_val = times[ptr]
                if t_val < earliest_t:
                    earliest_t = t_val
                    earliest_key = key

        if earliest_key is None:
            return False

        times, data = self.stream_data[earliest_key]
        ptr = self._pointers[earliest_key]
        t_sample = float(times[ptr])
        sample = data[ptr].tolist()
        self._pointers[earliest_key] += 1

        # Real-time throttle if playback_speed > 0
        if self.playback_speed > 0:
            if self._wall_start_t == 0.0:
                self._wall_start_t = time.perf_counter()
                self._last_streamed_t = t_sample
            else:
                elapsed_stream_s = (t_sample - self._last_streamed_t) / self.playback_speed
                elapsed_wall_s = time.perf_counter() - self._wall_start_t
                sleep_s = elapsed_stream_s - elapsed_wall_s
                if sleep_s > 0:
                    time.sleep(sleep_s)

        self.outlets[earliest_key].push_sample(sample, timestamp=t_sample)
        return True

    def push_chunk(self, duration_s: float) -> int:
        """Pushes samples until relative timestamp advances by duration_s."""
        start_t: Optional[float] = None
        count = 0

        while True:
            next_t = float("inf")
            for key, (times, _) in self.stream_data.items():
                ptr = self._pointers[key]
                if ptr < len(times) and times[ptr] < next_t:
                    next_t = times[ptr]

            if next_t == float("inf"):
                break

            if start_t is None:
                start_t = next_t

            if (next_t - start_t) > duration_s:
                break

            if not self.push_next():
                break
            count += 1

        return count

    def stream_all(self) -> int:
        """Pushes all remaining samples in the session."""
        count = 0
        while self.push_next():
            count += 1
        return count

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            has_more = self.push_next()
            if not has_more:
                break
            if self.playback_speed <= 0:
                time.sleep(0)

    def start(self) -> None:
        """Starts background streaming thread."""
        if self._is_running:
            return
        self._stop_event.clear()
        self._is_running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops background streaming thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._is_running = False

    def close(self) -> None:
        """Closes streamer and releases resources."""
        self.stop()
        self.outlets.clear()
        time.sleep(0.05)

    def __enter__(self) -> SingleSubjectLSLStreamer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class RealTimeDataPreprocessor:
    """Ingests multi-rate LSL streams, replicates preprocessing, and emits 25 Hz feature samples.

    Operations:
      - Hardware time anchoring: Locks session anchor t0 from stream timestamps.
      - Centrifuge resultant G-magnitude: sqrt(Gx^2 + Gy^2 + Gz^2).
      - Physiological range checks: HR [30, 220], BR [4, 60], Skin Temp [25, 42], ECG [-5, 5].
      - Online R-peak detection from raw 256 Hz ECG to compute HR_instant, HR_average, HR_w_average.
      - 25 Hz clock grid resampling: Linearly interpolates continuous signals and latches vitals
        to emit samples on a regular Delta t = 0.04s grid.
    """

    DEFAULT_RAW_FEATURES = [
        "HR (bpm) - Equivital",
        "ECG Lead 1 - Equivital",
        "ECG Lead 2 - Equivital",
        "HR_instant - Equivital",
        "HR_average - Equivital",
        "HR_w_average - Equivital",
        "BR (rpm) - Equivital",
        "Skin Temperature - IR Thermometer (°C) - Equivital",
        "magnitude - Centrifuge",
    ]

    def __init__(
        self,
        raw_feature_names: Optional[Sequence[str]] = None,
        stream_names: Optional[Dict[str, str]] = None,
        target_freq: float = 25.0,
    ) -> None:
        self.raw_feature_names = list(raw_feature_names or self.DEFAULT_RAW_FEATURES)
        self.target_freq = float(target_freq)
        self.dt_target = 1.0 / self.target_freq

        # Stream mapping: modality -> LSL stream name / pattern
        self.stream_names = stream_names or {
            "system_data": "SA5_SystemData",
            "ecg": "Equivital_ECG",
            "summary": "Equivital_Summary",
            "accel": "Equivital_Accel",
        }

        self.inlets: Dict[str, pylsl.StreamInlet] = {}

        # Buffers for continuous signals: deque of (timestamp, value)
        self._mag_buffer: deque[Tuple[float, float]] = deque()
        self._ecg_buffer: deque[Tuple[float, float]] = deque()
        self._ecg2_buffer: deque[Tuple[float, float]] = deque()

        # Latched vitals
        self._latest_hr: float = 80.0
        self._latest_br: float = 15.0
        self._latest_skin_temp: float = 33.0
        self._latest_hr_instant: float = 80.0
        self._latest_hr_average: float = 80.0
        self._latest_hr_w_average: float = 80.0

        # R-peak detection state
        self._last_peak_t: Optional[float] = None
        self._recent_hrs: deque[float] = deque(maxlen=5)
        self._recent_ecg_max: float = 0.5
        self._min_rr_interval: float = 0.25  # Max 240 bpm

        # Clock grid state
        self._clock_anchor: Optional[float] = None
        self._target_t: float = 0.0
        self._accumulated_preproc_s: float = 0.0
        self._is_connected = False

    def connect(self, timeout: float = 5.0) -> None:
        """Resolves LSL streams and initializes inlets."""
        if self._is_connected:
            return

        for modality, name in self.stream_names.items():
            streams = pylsl.resolve_byprop("name", name, timeout=timeout)
            if not streams:
                all_s = pylsl.resolve_streams(wait_time=0.5)
                for s in all_s:
                    if name.lower() in s.name().lower():
                        streams = [s]
                        break

            if streams:
                inlet = pylsl.StreamInlet(streams[0], max_buflen=3600, max_chunklen=1024)
                inlet.open_stream(timeout=timeout)
                self.inlets[modality] = inlet
                logger.info(f"Connected LSL inlet for {modality} -> '{streams[0].name()}'")
            else:
                logger.warning(f"Could not resolve LSL stream for {modality} ('{name}')")

        self._is_connected = True

    def _update_sa5(self, sample: List[float], ts: float) -> None:
        """Process SA5 sample: channel 38=Gx, 40=Gy, 42=Gz."""
        gx = sample[38] if len(sample) > 38 else 0.0
        gy = sample[40] if len(sample) > 40 else 0.0
        gz = sample[42] if len(sample) > 42 else 1.0
        mag = float(np.sqrt(gx * gx + gy * gy + gz * gz))
        self._mag_buffer.append((ts, mag))

    def _update_ecg(self, sample: List[float], ts: float) -> None:
        """Process ECG sample (Lead 1 and Lead 2) and update online R-peak detector."""
        ecg1_val = float(np.clip(sample[0], -5.0, 5.0))
        ecg2_val = float(np.clip(sample[1], -5.0, 5.0))
        self._ecg_buffer.append((ts, ecg1_val))
        self._ecg2_buffer.append((ts, ecg2_val))

        # Online R-peak detector: adaptive refractory threshold check
        threshold = max(0.02, self._recent_ecg_max * 0.5)
        if ecg1_val > threshold:
            if self._last_peak_t is None or (ts - self._last_peak_t) >= self._min_rr_interval:
                if self._last_peak_t is not None:
                    rri = ts - self._last_peak_t
                    if 0.25 <= rri <= 2.0:
                        hr_inst = 60.0 / rri
                        if 30.0 <= hr_inst <= 220.0:
                            self._latest_hr_instant = hr_inst
                            self._recent_hrs.append(hr_inst)
                            self._latest_hr_average = float(np.mean(self._recent_hrs))
                            weights = np.arange(1, len(self._recent_hrs) + 1, dtype=float)
                            self._latest_hr_w_average = float(
                                np.sum(np.array(self._recent_hrs) * weights) / np.sum(weights)
                            )
                            if self._latest_hr == 80.0:
                                self._latest_hr = self._latest_hr_average
                self._last_peak_t = ts
                self._recent_ecg_max = 0.9 * self._recent_ecg_max + 0.1 * ecg1_val

    def _update_summary(self, sample: List[float], ts: float) -> None:
        """Process Equivital Summary vitals: ch0=HR, ch2=BR, ch5=SkinTemp."""
        if len(sample) > 0:
            hr = sample[0]
            if 30.0 <= hr <= 220.0:
                self._latest_hr = float(hr)
                if len(self._recent_hrs) == 0:
                    self._latest_hr_instant = self._latest_hr
                    self._latest_hr_average = self._latest_hr
                    self._latest_hr_w_average = self._latest_hr

        if len(sample) > 2:
            br = sample[2]
            if 4.0 <= br <= 60.0:
                self._latest_br = float(br)

        if len(sample) > 5:
            temp = sample[5]
            if 25.0 <= temp <= 42.0:
                self._latest_skin_temp = float(temp)

    def _interpolate_buffer(self, buf: deque[Tuple[float, float]], t_query: float) -> float:
        """Linear interpolation of query timestamp in buffer."""
        if not buf:
            return 0.0
        if len(buf) == 1 or t_query <= buf[0][0]:
            return buf[0][1]
        if t_query >= buf[-1][0]:
            return buf[-1][1]

        for i in range(len(buf) - 1):
            t0, v0 = buf[i]
            t1, v1 = buf[i + 1]
            if t0 <= t_query <= t1:
                if t1 == t0:
                    return v0
                return v0 + (v1 - v0) * (t_query - t0) / (t1 - t0)

        return buf[-1][1]

    def poll_samples(
        self, timeout: float = 0.0, return_latency: bool = False
    ) -> Union[List[Tuple[np.ndarray, float]], List[Tuple[np.ndarray, float, float]]]:
        """Pulls available chunks from all LSL inlets and emits ready 25 Hz samples."""
        if not self._is_connected:
            self.connect(timeout=timeout)

        # 1. Non-blocking drain from all inlets (avoids idle socket wait)
        pulled_data: Dict[str, Tuple[List[Any], List[float]]] = {}
        for modality, inlet in self.inlets.items():
            samples, timestamps = inlet.pull_chunk(timeout=0.0)
            if samples:
                pulled_data[modality] = (samples, timestamps)

        # 2. Timing begins strictly for algorithmic computation
        t0 = time.perf_counter()

        for modality, (samples, timestamps) in pulled_data.items():
            for sample, ts in zip(samples, timestamps):
                if self._clock_anchor is None:
                    self._clock_anchor = ts
                    self._target_t = 0.0

                t_rel = ts - self._clock_anchor

                if modality == "system_data":
                    self._update_sa5(sample, t_rel)
                elif modality == "ecg":
                    self._update_ecg(sample, t_rel)
                elif modality == "summary":
                    self._update_summary(sample, t_rel)

        # 3. Check maximum available time across fast streams
        if not self._mag_buffer or not self._ecg_buffer:
            t1 = time.perf_counter()
            self._accumulated_preproc_s += (t1 - t0)
            return []

        latest_available_t = min(self._mag_buffer[-1][0], self._ecg_buffer[-1][0])
        emitted: List[Tuple[np.ndarray, float]] = []

        # 4. Emit 25 Hz samples
        while self._target_t <= latest_available_t:
            mag_val = self._interpolate_buffer(self._mag_buffer, self._target_t)
            ecg1_val = self._interpolate_buffer(self._ecg_buffer, self._target_t)
            ecg2_val = self._interpolate_buffer(self._ecg2_buffer, self._target_t)

            feat_map = {
                "HR (bpm) - Equivital": self._latest_hr,
                "ECG Lead 1 - Equivital": ecg1_val,
                "ECG Lead 2 - Equivital": ecg2_val,
                "HR_instant - Equivital": self._latest_hr_instant,
                "HR_average - Equivital": self._latest_hr_average,
                "HR_w_average - Equivital": self._latest_hr_w_average,
                "BR (rpm) - Equivital": self._latest_br,
                "Skin Temperature - IR Thermometer (°C) - Equivital": self._latest_skin_temp,
                "magnitude - Centrifuge": mag_val,
            }

            sample_arr = np.array([feat_map.get(k, 0.0) for k in self.raw_feature_names], dtype=np.float64)
            emitted.append((sample_arr, self._target_t))
            self._target_t += self.dt_target

        # 5. Prune old buffer samples (older than target_t - 2.0s)
        prune_cutoff = self._target_t - 2.0
        while self._mag_buffer and self._mag_buffer[0][0] < prune_cutoff:
            self._mag_buffer.popleft()
        while self._ecg_buffer and self._ecg_buffer[0][0] < prune_cutoff:
            self._ecg_buffer.popleft()
        while self._ecg2_buffer and self._ecg2_buffer[0][0] < prune_cutoff:
            self._ecg2_buffer.popleft()

        t1 = time.perf_counter()
        self._accumulated_preproc_s += (t1 - t0)

        if not emitted:
            return []

        per_sample_lat_ms = (self._accumulated_preproc_s * 1000.0) / len(emitted)
        self._accumulated_preproc_s = 0.0

        if return_latency:
            return [(sample_arr, t_val, per_sample_lat_ms) for sample_arr, t_val in emitted]
        return emitted

    def close(self) -> None:
        """Closes all LSL inlets."""
        while self.inlets:
            _, inlet = self.inlets.popitem()
            try:
                inlet.close_stream()
            except Exception:
                pass
            del inlet
        self._is_connected = False
        time.sleep(0.05)

    def __enter__(self) -> RealTimeDataPreprocessor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


