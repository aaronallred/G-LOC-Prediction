"""Real-time Equivital data streaming mode for G-LOC prediction.

This module streams Equivital physiological data (HR, BR, ECG, etc.)
through LSL (Lab Streaming Layer) for real-time G-LOC detection.
It loads data directly from the main CSV file, reading only Equivital
columns to avoid memory issues.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
except ImportError:  # pragma: no cover
    StreamInfo = None  # type: ignore[misc, assignment]
    StreamOutlet = None  # type: ignore[misc, assignment]
    local_clock = None  # type: ignore[misc, assignment]

from src.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


def _get_equivital_columns(csv_file: Path) -> list[str]:
    """Read the header of the CSV and return only columns containing 'Equivital'.

    Parameters
    ----------
    csv_file : Path
        Path to the CSV file.

    Returns
    -------
    list[str]
        List of column names that contain 'Equivital'.
    """
    logger.info("Reading header from %s to find Equivital columns...", csv_file)
    df_header = pd.read_csv(csv_file, nrows=0)
    equivital_cols = [col for col in df_header.columns if "Equivital" in col]
    if not equivital_cols:
        raise ValueError(f"No columns with 'Equivital' found in {csv_file}")
    logger.info("Found %d Equivital columns.", len(equivital_cols))
    return equivital_cols


class EquivitalDataStreamer:
    """Streams a pre-processed data matrix through an LSL outlet in real time.

    Parameters
    ----------
    data_matrix : np.ndarray
        The processed feature matrix to stream.
    labels : Optional[np.ndarray], optional
        The corresponding labels, if available.
    channel_names : Optional[list[str]], optional
        List of channel names for the LSL stream metadata.
    stream_name : str, optional
        The LSL stream name (default: "GLOC-Equivital").
    stream_type : str, optional
        The LSL stream type (default: "PsychoPhys").
    """

    # Fixed sampling rate for Equivital data (Hz)
    STREAM_RATE_HZ: float = 25.0

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
        self._outlet: Optional[Any] = None  # type: ignore[valid-type]

    def _create_outlet(self, n_channels: int) -> Optional[Any]:
        """Create and return an LSL StreamOutlet.

        Parameters
        ----------
        n_channels : int
            Number of channels in the stream.

        Returns
        -------
        StreamOutlet or None
            The created outlet, or None if pylsl is not available.
        """
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

    def stream(self, *, use_real_time_sleep: bool = True) -> None:
        """Stream the data matrix through LSL in real time.

        Parameters
        ----------
        use_real_time_sleep : bool, default True
            If True, samples are pushed at the nominal 25 Hz rate using
            ``time.sleep``. If False, all samples are sent as fast as possible.
        """
        # TODO: Implement actual LSL streaming loop.
        # This is a skeleton for future real-time analysis.
        logger.info(
            "Starting Equivital data streaming (real-time sleep=%s)...", use_real_time_sleep
        )

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
        start_time = local_clock()
        sent_samples = 0

        try:
            for i in range(n_samples):
                sample = self.data_matrix[i]
                self._outlet.push_sample(sample)
                sent_samples += 1

                if sent_samples % 10000 == 0:
                    logger.info(
                        "Streamed sample %d/%d (sample values: %s)",
                        sent_samples,
                        n_samples,
                        sample,
                    )

                if use_real_time_sleep:
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


def run_real_time_equivital(
        config: dict,
        model_factory: ModelFactory,
        project_root_path: Path,
) -> None:
    """Run the real-time Equivital streaming mode.

    For each configured model, this function:
        1. Verifies it is a traditional model.
        2. Loads only Equivital columns from the main data CSV.
        3. Initializes an ``EquivitalDataStreamer`` to push the data through LSL.

    Parameters
    ----------
    config : dict
        The loaded experiment configuration YAML mapping.
    model_factory : ModelFactory
        Factory for creating model instances.
    project_root_path : Path
        Absolute path to the project root directory.
    """
    mode_config = config.get("real_time_equivital", {})
    if not mode_config:
        logger.warning("No 'real_time_equivital' configuration found in config.")
        return

    # Extract mode-specific parameters
    model_names: list[str] = mode_config["models"]
    _saved_models_folder = Path(mode_config.get("saved_models_folder", "SavedModels"))

    logger.info("Starting real_time_equivital mode with models: %s", model_names)

    # Determine the main data file path
    data_path = Path(config.get("data_path", "data"))
    if not data_path.is_absolute():
        data_path = project_root_path / data_path
    csv_file = data_path / "all_trials_25_hz_stacked_null_str_filled.csv"
    logger.info("Loading Equivital data from %s...", csv_file)

    # Load only Equivital columns to avoid OOM
    equivital_cols = _get_equivital_columns(csv_file)
    logger.info("Found Equivital columns: %s", " ".join(equivital_cols))
    df = pd.read_csv(csv_file, usecols=equivital_cols)
    data_matrix = df.to_numpy(dtype=np.float32)
    logger.info("Loaded data matrix with shape %s", data_matrix.shape)

    for model_name in model_names:
        logger.info("Processing model: %s", model_name)
        model = model_factory.create_model(model_name)

        # Only traditional models are supported
        if not getattr(model, "is_traditional_model", False):
            logger.warning(
                "Model '%s' is not a traditional model. Skipping.", model_name
            )
            continue

        # TODO: Load pre-trained model weights from saved_models_folder if they exist.
        # For the skeleton, we just create a fresh model instance.
        # saved_model_path = project_root_path / saved_models_folder / f"{model_name}.pkl"

        try:
            streamer = EquivitalDataStreamer(
                data_matrix=data_matrix,
                labels=None,
                channel_names=equivital_cols,
            )
            streamer.stream(use_real_time_sleep=False)
        except Exception as exc:
            logger.exception("Failed to run real-time equivital for model '%s': %s", model_name, exc)
            continue
