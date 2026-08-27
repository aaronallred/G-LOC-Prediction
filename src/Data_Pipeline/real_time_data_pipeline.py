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
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

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

        # v0: Raw signals + derivatives
        b_v0 = get_deriv_stack(win_arr)

        # v1: Physical / baseline + derivatives
        b_v1 = get_deriv_stack(win_arr / self._v1_baseline_mean)

        # v2: Physical - baseline + derivatives
        b_v2 = get_deriv_stack(win_arr - self._v2_baseline_mean)

        # v5: ECG / Resting HR + derivatives (ECG channels are index 0..min(6, win_arr.shape[1]))
        ecg_cols = win_arr[:, :min(6, win_arr.shape[1])]
        b_v5 = get_deriv_stack(ecg_cols / self.participant_baseline_rhr)

        # v6: ECG - Resting HR + derivatives
        b_v6 = get_deriv_stack(ecg_cols - self.participant_baseline_rhr)

        # Combine all baseline arrays horizontally: [v0, v1, v2, v5, v6]
        combined_baseline = np.hstack([b_v0, b_v1, b_v2, b_v5, b_v6])

        # Sliding window summary statistics across window axis=0
        mean_stat = np.nanmean(combined_baseline, axis=0)
        std_stat = np.nanstd(combined_baseline, axis=0)
        max_stat = np.nanmax(combined_baseline, axis=0)
        range_stat = np.nanmax(combined_baseline, axis=0) - np.nanmin(combined_baseline, axis=0)

        # Additional HRV features from index 0 (HR in bpm)
        hr_window = win_arr[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            rr_interval = 60000.0 / hr_window
        hrv_sdnn = np.nanstd(rr_interval)
        diff_rr = np.diff(rr_interval)
        hrv_rmssd = np.sqrt(np.nanmean(diff_rr ** 2)) if len(diff_rr) > 0 else 0.0

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

