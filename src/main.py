import argparse
import logging
from pathlib import Path
from typing import Callable

from .Data_Pipeline.data_pipeline import DataPipeline
from .config_loader import load_experiment_config
from .models_new.model_factory import ModelFactory
from .modes.cross_validation import run_cross_validation
from .modes.feature_space_review import run_feature_space_review
from .modes.sensor_ablation import run_sensor_ablation_review, run_sensor_ablation_training

def _try_enable_cuml_acceleration() -> None:
    """Enable cuML sklearn acceleration when the RAPIDS stack is available."""
    try:
        from cuml.accel import install as cuml_accel_install
    except ImportError:
        logging.info("cuML acceleration is unavailable; continuing with CPU estimators.")
        return

    try:
        cuml_accel_install()
        logging.info("cuML acceleration enabled for sklearn-compatible estimators.")
    except Exception as exc:
        logging.warning(
            "cuML acceleration initialization failed (%s). Continuing with CPU estimators.",
            exc,
        )

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the G-LOC data pipeline with a configurable YAML experiment file.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to experiment config YAML (defaults to project-level GLOC_experiment_config.yaml).",
    )
    return parser.parse_args()

def run(config_path: str) -> None:
    configure_logging()
    _try_enable_cuml_acceleration()

    config = load_experiment_config(config_path)
    data_pipeline = DataPipeline(config = config)
    model_factory = ModelFactory()

    project_root_path = Path(__file__).resolve().parent.parent
    did_run_any_mode = False

    mode_runners: list[tuple[bool, Callable[[], None]]] = [
        (
            bool(config.get("cross_validation", {}).get("enabled", False)),
            lambda: run_cross_validation(
                config = config,
                pipeline = data_pipeline,
                model_factory = model_factory,
                project_root_path = project_root_path
            ),
        ),
        (
            bool(config.get("sensor_ablation", {}).get("training", {}).get("enabled", False)),
            lambda: run_sensor_ablation_training(
                config = config,
                pipeline = data_pipeline,
                model_factory = model_factory,
                project_root = project_root_path
            )
        ),
        (
            bool(config.get("sensor_ablation", {}).get("review", {}).get("enabled", False)),
            lambda: run_sensor_ablation_review(
                config = config
            ),
        ),
        (
            bool(config.get("feature_space_review", {}).get("enabled", False)),
            lambda: run_feature_space_review(
                config = config,
                model_factory = model_factory
            )
        ),
    ]

    for is_enabled, runner in mode_runners:
        if is_enabled:
            runner()
            did_run_any_mode = True

    if not did_run_any_mode:
        logging.info(
            "No runnable mode enabled. Set sensor_ablation.training.enabled, "
            "sensor_ablation.review.enabled, feature_space_review.enabled, "
            "or cross_validation.enabled to true in the config."
        )


if __name__ == "__main__":
    args = parse_args()

    config_path = None
    if args.config is not None:
        resolved_config_path = Path(args.config).expanduser().resolve()
        if not resolved_config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")
        config_path = str(resolved_config_path)

    run(config_path)