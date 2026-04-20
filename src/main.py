import logging
import argparse
import pickle
from pathlib import Path
from numpy import ravel
import numpy as np

from .GLOC_experiment_config_parser import GLOCExperimentConfigParser
from .Data_Pipeline.data_pipeline import DataPipeline
from .traditional_experiment_utils import (
    get_hyperparameters_from_json,
    stratified_kfold_split,
    plot_f1_violin_by_stream
)

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Run the G-LOC data pipeline with a configurable YAML experiment file.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help = "Path to experiment config YAML (defaults to project-level GLOC_experiment_config.yaml).",
    )
    
    return parser.parse_args()

def _extract_f1_score(metrics_tuple: tuple) -> float:
    """Extract F1 score from legacy metric tuple returned by classify_traditional."""
    if len(metrics_tuple) <= 3:
        raise ValueError(f"Unexpected metrics tuple format. Expected F1 at index 3, got: {metrics_tuple}")
    return float(metrics_tuple[3])

def _save_model_stream_f1_scores(
    results_root_dir: Path,
    model_name: str,
    stream_str: str,
    f1_scores: np.ndarray,
) -> Path:
    """Persist one model-stream F1 array as Results/Sensor_Ablation/<model>/<stream>.pkl."""
    model_dir = results_root_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / f"{stream_str}.pkl"
    with open(output_path, "wb") as handle:
        pickle.dump(np.asarray(f1_scores, dtype = float), handle)
    return output_path


def _load_sensor_ablation_f1_results(
    results_root_dir: Path,
    classifiers: list[str],
    stream_group: list[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Load cached per-stream F1 arrays from Results/Sensor_Ablation/<model_type>/<classifier>/*.pkl."""
    loaded: dict[str, dict[str, np.ndarray]] = {}
    required_stream_group = frozenset(stream_group) if stream_group else None

    for classifier in classifiers:
        classifier_dir = results_root_dir / classifier
        if not classifier_dir.exists():
            logging.warning("Skipping missing classifier directory: %s", classifier_dir)
            continue

        stream_scores: dict[str, np.ndarray] = {}
        for pkl_path in sorted(classifier_dir.glob("*.pkl")):
            if required_stream_group is not None and frozenset(pkl_path.stem.split("-")) != required_stream_group:
                continue

            with open(pkl_path, "rb") as handle:
                stream_scores[pkl_path.stem] = np.asarray(pickle.load(handle), dtype = float)

        if stream_scores:
            loaded[classifier] = stream_scores

    return loaded


def _filter_sensor_ablation_review_results(
    f1_results_by_stream: dict[str, dict[str, np.ndarray]],
    stream_group: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Filter cached F1 results using one exact stream-label selection from YAML."""
    if len(f1_results_by_stream) == 0:
        return {}

    if len(stream_group) == 0:
        raise ValueError("sensor_ablation_review_parameters.stream_group must be non-empty.")

    required_group_set = frozenset(stream_group)

    filtered: dict[str, dict[str, np.ndarray]] = {}
    for classifier, stream_dict in f1_results_by_stream.items():
        matching_streams = {
            stream_name: scores
            for stream_name, scores in stream_dict.items()
            if frozenset(stream_name.split("-")) == required_group_set
        }
        if matching_streams:
            filtered[classifier] = matching_streams

    return filtered


def _run_sensor_ablation_training(
    config_parser: GLOCExperimentConfigParser,
    pipeline: DataPipeline,
    project_root: Path,
) -> None:
    feature_stream_groups: list[list[str]] = config_parser.get_sensor_ablation_streams()
    models_to_test: list = config_parser.get_models()
    num_splits: int = config_parser.get_num_splits()
    model_type = config_parser.get_model_type()
    f1_results_by_stream: dict[str, dict[str, np.ndarray]] = {
        model.get_name(): {}
        for model in models_to_test
    }

    sensor_ablation_results_dir = project_root / "Results" / "Sensor_Ablation" / model_type.get_folder_name()
    sensor_ablation_results_dir.mkdir(parents = True, exist_ok = True)

    for group_index, feature_streams in enumerate(feature_stream_groups, start = 1):
        logging.info("Running stream group %d/%d: %s", group_index, len(feature_stream_groups), feature_streams)
        stream_str = "-".join(feature_streams)

        for model in models_to_test:
            model_name = model.get_name()
            logging.info("Running model: %s", model_name)

            hyperparameters, _, _, _ = get_hyperparameters_from_json(model_name, model_type.get_folder_name())
            x, y = pipeline.get_data(model = model, feature_streams = feature_streams)
            f1_scores = np.zeros(num_splits, dtype = float)

            for kfold_id in range(num_splits):
                logging.info("Running fold %d/%d", kfold_id + 1, num_splits)
                x_train, x_test, y_train, y_test = stratified_kfold_split(
                    Y = ravel(y),
                    X = x,
                    num_splits = num_splits,
                    kfold_ID = kfold_id,
                    random_state = config_parser.get_random_seed(),
                )

                metrics_tuple = model.classify_traditional(
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    None,
                    config_parser.get_random_seed(),
                    save_folder = "", # TODO: Implement saving and loading of trained models
                    model_name = f"{model_name.lower()}_feature_study.pkl",
                    retrain = False,
                    temporal = True,
                    best_params = hyperparameters,
                )

                fold_f1 = _extract_f1_score(metrics_tuple)
                f1_scores[kfold_id] = fold_f1

                logging.info(
                    "Metrics for %s | streams=%s | fold=%d: %s",
                    model_name,
                    feature_streams,
                    kfold_id,
                    metrics_tuple,
                )

            output_path = _save_model_stream_f1_scores(
                results_root_dir = sensor_ablation_results_dir,
                model_name = model_name,
                stream_str = stream_str,
                f1_scores = f1_scores,
            )
            f1_results_by_stream[model_name][stream_str] = f1_scores
            logging.info("Saved F1 scores to %s", output_path)

    plot_f1_violin_by_stream(
        f1_results_by_stream = f1_results_by_stream,
        model_type = model_type,
        save_folder = sensor_ablation_results_dir,
    )


def _run_sensor_ablation_review(
    config_parser: GLOCExperimentConfigParser,
    project_root: Path,
) -> None:
    """Review and re-plot saved sensor ablation F1 results using YAML-defined models and one stream group."""
    model_type = config_parser.get_model_type()
    sensor_ablation_results_dir = project_root / "Results" / "Sensor_Ablation" / model_type.get_folder_name()

    classifiers_to_load = config_parser.get_sensor_ablation_review_models()

    f1_results_by_stream = _load_sensor_ablation_f1_results(
        results_root_dir = sensor_ablation_results_dir,
        classifiers = classifiers_to_load,
        stream_group = config_parser.get_sensor_ablation_review_stream_group(),
    )
    if len(f1_results_by_stream) == 0:
        raise ValueError(
            "Sensor ablation review is enabled, but no cached F1 result files were found under "
            f"{sensor_ablation_results_dir}."
        )

    filtered_results = _filter_sensor_ablation_review_results(
        f1_results_by_stream = f1_results_by_stream,
        stream_group = config_parser.get_sensor_ablation_review_stream_group(),
    )
    if len(filtered_results) == 0:
        raise ValueError(
            "Sensor ablation review filters removed all streams. "
            "Update sensor_ablation_review_parameters.stream_group in the config."
        )

    plot_f1_violin_by_stream(
        f1_results_by_stream = filtered_results,
        model_type = model_type,
    )


def run(config_path: str | None = None) -> None:
    configure_logging()

    config_parser: GLOCExperimentConfigParser = GLOCExperimentConfigParser(config_location = config_path)

    # Install cuML sklearn acceleration after config/model imports so imblearn
    # class definitions run against unpatched sklearn internals.
    import cuml.accel
    cuml.accel.install()

    project_root = Path(__file__).resolve().parent.parent

    did_run_any_mode = False

    if config_parser.get_sensor_ablation_enabled():
        pipeline: DataPipeline = DataPipeline(config_parser = config_parser)
        _run_sensor_ablation_training(
            config_parser = config_parser,
            pipeline = pipeline,
            project_root = project_root,
        )
        did_run_any_mode = True

    if config_parser.get_sensor_ablation_review_enabled():
        _run_sensor_ablation_review(
            config_parser = config_parser,
            project_root = project_root,
        )
        did_run_any_mode = True

    if not did_run_any_mode:
        logging.info(
            "No runnable mode enabled. Set sensor_ablation_parameters.enabled or "
            "sensor_ablation_review_parameters.enabled to true in the config."
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