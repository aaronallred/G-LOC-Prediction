import argparse
import logging
from pathlib import Path
from typing import Callable, Any

from .config_loader import load_experiment_config
from .runtime import configure_compute
from .timing import RunTimer

def _try_enable_cuml_acceleration(*, enabled: bool, device_type: str) -> None:
    """Enable cuML sklearn acceleration when the RAPIDS stack is available."""
    if not enabled or device_type != "cuda":
        logging.info("cuML acceleration disabled or no CUDA device selected.")
        return
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
    config = load_experiment_config(config_path)
    runtime_config = config.get("runtime", {})
    device = configure_compute(runtime_config)
    _try_enable_cuml_acceleration(
        enabled=bool(runtime_config.get("enable_cuml", True)),
        device_type=device.type,
    )

    # Import estimator modules only after cuML has had an opportunity to install
    # its sklearn accelerator hooks.
    from .Data_Pipeline.data_pipeline import DataPipeline
    from .models_new.model_factory import ModelFactory
    from .modes.cross_validation import run_cross_validation
    from .modes.feature_space_review import run_feature_space_review
    from .modes.sensor_ablation import run_sensor_ablation_review, run_sensor_ablation_training

    data_pipeline = DataPipeline(config = config)
    model_factory = ModelFactory()
    timer = RunTimer()

    project_root_path = Path(__file__).resolve().parent.parent
    did_run_any_mode = False

    mode_runners: list[tuple[str, bool, Callable[[], Any]]] = [
        (
            "cross_validation",
            bool(config.get("cross_validation", {}).get("enabled", False)),
            lambda: run_cross_validation(
                config = config,
                pipeline = data_pipeline,
                model_factory = model_factory,
                project_root_path = project_root_path
            ),
        ),
        (
            "sensor_ablation_training",
            bool(config.get("sensor_ablation", {}).get("training", {}).get("enabled", False)),
            lambda: run_sensor_ablation_training(
                config = config,
                pipeline = data_pipeline,
                model_factory = model_factory,
                project_root = project_root_path
            )
        ),
        (
            "sensor_ablation_review",
            bool(config.get("sensor_ablation", {}).get("review", {}).get("enabled", False)),
            lambda: run_sensor_ablation_review(
                config = config
            ),
        ),
        (
            "feature_space_review",
            bool(config.get("feature_space_review", {}).get("enabled", False)),
            lambda: run_feature_space_review(
                config = config,
                model_factory = model_factory
            )
        ),
    ]

    try:
        for mode_name, is_enabled, runner in mode_runners:
            if is_enabled:
                with timer.track(mode_name):
                    runner()
                did_run_any_mode = True

        if not did_run_any_mode:
            logging.info(
                "No runnable mode enabled. Set sensor_ablation.training.enabled, "
                "sensor_ablation.review.enabled, feature_space_review.enabled, "
                "or cross_validation.enabled to true in the config."
            )
    finally:
        timing_output = runtime_config.get("timing_output", "Results/run_timing.json")
        if timing_output:
            timing_path = Path(timing_output).expanduser()
            if not timing_path.is_absolute():
                timing_path = project_root_path / timing_path
            timer.save(timing_path)



if __name__ == "__main__":
    args = parse_args()

    config_path = None
    if args.config is not None:
        resolved_config_path = Path(args.config).expanduser().resolve()
        if not resolved_config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")
        config_path = str(resolved_config_path)

    run(config_path)
