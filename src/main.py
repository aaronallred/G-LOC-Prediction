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
    get_model_subfolder,
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


def _stream_group_key(feature_streams: list[str] | None) -> str:
    """Build a stable stream-group key for in-memory/result-file indexing."""
    if not feature_streams:
        return "all_streams"
    return "-".join(feature_streams)


def _extract_f1_score(metrics_tuple: tuple) -> float:
    """Extract F1 score from legacy metric tuple returned by classify_traditional."""
    if len(metrics_tuple) <= 3:
        raise ValueError(f"Unexpected metrics tuple format. Expected F1 at index 3, got: {metrics_tuple}")
    return float(metrics_tuple[3])


def _persist_streamwise_f1_results(
    f1_results_by_stream: dict[str, dict[str, np.ndarray]],
    subfolder_name: str,
) -> Path:
    """Persist streamwise F1 arrays per model for downstream analysis parity with legacy preference 7."""
    results_dir = Path("./feature_study") / subfolder_name / "streamwise"
    results_dir.mkdir(parents=True, exist_ok=True)

    for model_name, stream_map in f1_results_by_stream.items():
        for stream_key, f1_scores in stream_map.items():
            file_name = f"f1_results_{model_name}_{stream_key}.pkl"
            file_path = results_dir / file_name
            with open(file_path, "wb") as handle:
                pickle.dump(np.asarray(f1_scores, dtype=float), handle)

    return results_dir


def run(config_path: str | None = None) -> None:
    configure_logging()

    config_parser: GLOCExperimentConfigParser = GLOCExperimentConfigParser(config_location = config_path)
    pipeline: DataPipeline = DataPipeline(config_parser = config_parser)
    feature_stream_groups: list[list[str]] = config_parser.get_sensor_ablation_streams()
    models_to_test: list = config_parser.get_models()
    num_splits: int = config_parser.get_num_splits()
    sensor_ablation_enabled: bool = config_parser.get_sensor_ablation_enabled()

    # Preference 7 in Legacy
    # If sensor ablation is enabled, we need to load the best hyperparameters for each model and stream group from JSON before running the classification. 
    # Otherwise, we can run the classification without loading hyperparameters.
    if sensor_ablation_enabled:
        model_type: str = config_parser.get_model_type()
        subfolder_name: str = get_model_subfolder(model_type)
        f1_results_by_stream: dict[str, dict[str, np.ndarray]] = { model.get_name() for model in models_to_test }

        for model in models_to_test:
            model_name = model.get_name() if hasattr(model, "get_name") else str(model)
            logging.info("Running model: %s", model_name)

            hyperparameters, _, _, _ = get_hyperparameters_from_json(model_name, subfolder_name)
            save_folder = f"../RestrictedFeatureEval/{subfolder_name}"

            for group_index, feature_streams in enumerate(feature_stream_groups, start = 1):
                logging.info("Running stream group %d/%d: %s", group_index, len(feature_stream_groups), feature_streams)

                x, y = pipeline.get_data(model = model, feature_streams = feature_streams)
                stream_key = _stream_group_key(feature_streams)
                f1_scores = np.zeros(num_splits, dtype=float)

                for kfold_id in range(num_splits):
                    logging.info("Running fold %d/%d", kfold_id + 1, num_splits)
                    x_train, x_test, y_train, y_test = stratified_kfold_split(
                        ravel(y),
                        x,
                        num_splits,
                        kfold_id,
                        config_parser.get_random_seed(),
                    )

                    metrics_tuple = model.classify_traditional(
                        x_train,
                        x_test,
                        y_train,
                        y_test,
                        None,
                        config_parser.get_random_seed(),
                        save_folder,
                        model_name=f"{model_name.lower()}_feature_study.pkl",
                        retrain=False,
                        temporal=True,
                        best_params=hyperparameters,
                    )

                    fold_f1 = _extract_f1_score(metrics_tuple)
                    f1_scores[kfold_id] = fold_f1

                    logging.info(
                        "Metrics for %s | streams=%s | fold=%d: %s",
                        model_name,
                        feature_streams if feature_streams else ["ALL_STREAMS"],
                        kfold_id,
                        metrics_tuple,
                    )

                f1_results_by_stream[model_name][stream_key] = f1_scores

        results_dir = _persist_streamwise_f1_results(f1_results_by_stream, subfolder_name)
        logging.info("Saved streamwise F1 results to %s", results_dir)
        return

    for model in models_to_test:
        model_name = model.get_name() if hasattr(model, "get_name") else str(model)
        logging.info("Running model: %s", model_name)

        for group_index, feature_streams in enumerate(feature_stream_groups, start=1):
            logging.info(
                "Running stream group %d/%d: %s",
                group_index,
                len(feature_stream_groups),
                feature_streams if feature_streams else "ALL_STREAMS",
            )

            for kfold_id in range(num_splits):
                logging.info("Running fold %d/%d", kfold_id + 1, num_splits)
                data = pipeline.get_data(model=model, kfold_id=kfold_id, feature_streams=feature_streams)

                print("Data Dimensions:")
                print(f"model={model_name} | stream_group={feature_streams if feature_streams else ['ALL_STREAMS']} | fold={kfold_id}")
                if isinstance(data, tuple) and len(data) >= 4:
                    x_train, x_test, y_train, y_test = data[:4]
                    print(f"x_train: {x_train.shape}")
                    print(f"x_test: {x_test.shape}")
                    print(f"y_train: {y_train.shape}")
                    print(f"y_test: {y_test.shape}")
                elif isinstance(data, tuple) and len(data) == 2:
                    x_feature_matrix, y_gloc_labels = data
                    print(f"x_feature_matrix: {x_feature_matrix.shape}")
                    print(f"y_gloc_labels: {y_gloc_labels.shape}")
                else:
                    print("Unexpected data format returned from pipeline.")

if __name__ == "__main__":
    args = parse_args()

    config_path = None
    if args.config is not None:
        resolved_config_path = Path(args.config).expanduser().resolve()
        if not resolved_config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")
        
        config_path = str(resolved_config_path)
    run(config_path)