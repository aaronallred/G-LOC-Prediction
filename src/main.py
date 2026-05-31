import argparse
import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from upsetplot import UpSet, from_contents

from .Data_Pipeline.data_pipeline import DataPipeline
from .GLOC_experiment_config_parser import GLOCExperimentConfigParser
from .models_new.model_factory import ModelFactory
from .modes.cross_validation import run_cross_validation
from .modes.feature_space_review import (
    investigate_feature_space,
    run_feature_space_review,
)
from .modes.sensor_ablation import (
    apply_stream_label_aliases,
    build_ranked_sensor_ablation_review_results,
    extract_f1_score,
    filter_sensor_ablation_review_results,
    load_sensor_ablation_f1_results,
    run_sensor_ablation_review,
    run_sensor_ablation_training,
    save_model_stream_f1_scores,
)
from .traditional_experiment_utils import (
    plot_f1_violin_by_stream,
    plot_f1_violin_with_stream_matrix,
    stratified_kfold_split,
)


STREAM_LABEL_ALIASES = {
    "ECG-HR-BR-Temperature": "Equivital",
    "Participant": "Demographics",
    "Centrifuge": "G Force",
}


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

def run(config_path: str | None = None) -> None:
    configure_logging()
    _try_enable_cuml_acceleration()

    config_parser = GLOCExperimentConfigParser(config_location = config_path)
    data_pipeline = DataPipeline(config_parser = config_parser)
    model_factory = ModelFactory()

    project_root_path = Path(__file__).resolve().parent.parent
    did_run_any_mode = False

    mode_runners: list[tuple[bool, Callable[[], None]]] = [
        (
            config_parser.get_cross_validation_enabled(),
            lambda: run_cross_validation(
                config = config_parser,
                pipeline = data_pipeline,
                model_factory = model_factory,
                project_root_path = project_root_path
            ),
        ),
        (
            config_parser.get_sensor_ablation_enabled(),
            lambda: run_sensor_ablation_training(
                config_parser = config_parser,
                pipeline = DataPipeline(config_parser = config_parser),
                project_root = project_root_path,
                # Use the configured median_hyperparameters_folder via the config parser by
                # passing None here; run_sensor_ablation_training will build a resolver.
                get_hyperparameters_from_json_fn = None,
                stratified_kfold_split_fn = stratified_kfold_split,
                plot_f1_violin_by_stream_fn = plot_f1_violin_by_stream,
                extract_f1_score_fn = extract_f1_score,
                save_model_stream_f1_scores_fn = save_model_stream_f1_scores,
            )
        ),
        (
            config_parser.get_sensor_ablation_review_enabled(),
            lambda: run_sensor_ablation_review(
                config_parser = config_parser,
                project_root = project_root_path,
                load_sensor_ablation_f1_results_fn = load_sensor_ablation_f1_results,
                build_ranked_sensor_ablation_review_results_fn = build_ranked_sensor_ablation_review_results,
                filter_sensor_ablation_review_results_fn = filter_sensor_ablation_review_results,
                plot_f1_violin_with_stream_matrix_fn = plot_f1_violin_with_stream_matrix,
                plot_f1_violin_by_stream_fn = plot_f1_violin_by_stream,
                stream_label_aliases = STREAM_LABEL_ALIASES,
            ),
        ),
        (
            config_parser.get_feature_space_review_enabled(),
            lambda: run_feature_space_review(
                config_parser = config_parser,
                investigate_feature_space_fn = investigate_feature_space,
                get_hyperparameters_from_json_fn = None,
                venn2_fn = venn2,
                venn3_fn = venn3,
                from_contents_fn = from_contents,
                upset_cls = UpSet,
                plt_module = plt,
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