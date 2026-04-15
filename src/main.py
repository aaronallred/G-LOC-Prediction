import logging
import argparse
import pickle
from pathlib import Path
from numpy import ravel
import numpy as np

from .GLOC_experiment_config_parser import GLOCExperimentConfigParser
from .data_pipeline import DataPipeline
from .traditional_experiment_utils import (
    get_hyperparameters_from_json,
    stratified_kfold_split,
)

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Run the G-LOC data pipeline with a configurable JSON experiment file.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help = "Path to experiment config JSON (defaults to project-level GLOC_experiment_config.json).",
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


def run(config_path: str | None = None) -> None:
    configure_logging()

    config_parser: GLOCExperimentConfigParser = GLOCExperimentConfigParser(config_location = config_path)
    pipeline: DataPipeline = DataPipeline(config_parser = config_parser)
    feature_stream_groups: list[list[str]] = config_parser.get_sensor_ablation_streams()
    models_to_test: list = config_parser.get_models()
    num_splits: int = config_parser.get_num_splits()
    project_root = Path(__file__).resolve().parent.parent

    # Preference 7 in Legacy
    # If sensor ablation is enabled, we need to load the best hyperparameters for each model and stream group from JSON before running the classification. 
    # Otherwise, we can run the classification without loading hyperparameters.
    if config_parser.get_sensor_ablation_enabled():
        model_type = config_parser.get_model_type()

        sensor_ablation_results_dir = project_root / "Results" / "Sensor_Ablation" / model_type.get_folder_name()
        sensor_ablation_results_dir.mkdir(parents = True, exist_ok = True)

        for model in models_to_test:
            model_name = model.get_name()
            logging.info("Running model: %s", model_name)

            hyperparameters, _, _, _ = get_hyperparameters_from_json(model_name, model_type.get_folder_name())

            for group_index, feature_streams in enumerate(feature_stream_groups, start = 1):
                logging.info("Running stream group %d/%d: %s", group_index, len(feature_stream_groups), feature_streams)

                x, y = pipeline.get_data(model = model, feature_streams = feature_streams)
                stream_str = "-".join(feature_streams)
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
                logging.info("Saved F1 scores to %s", output_path)

        return

if __name__ == "__main__":
    args = parse_args()

    config_path = None
    if args.config is not None:
        resolved_config_path = Path(args.config).expanduser().resolve()
        if not resolved_config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")
        
        config_path = str(resolved_config_path)

    run(config_path)