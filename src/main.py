import logging
import argparse
from pathlib import Path

from GLOC_experiment_config_parser import GLOCExperimentConfigParser
from data_pipeline import DataPipeline


def _resolve_feature_stream_groups(config_parser: GLOCExperimentConfigParser) -> list[list[str]]:
    """Resolve feature stream groups for sensor ablation runs.

    Returns one default empty group when sensor ablation is disabled.
    """
    enabled = config_parser.get_sensor_ablation_enabled()
    stream_groups = config_parser.get_sensor_ablation_streams()

    if not enabled:
        return [[]]

    if not isinstance(stream_groups, list):
        raise ValueError("sensor_ablation.streams must be a list of stream groups.")

    cleaned_groups: list[list[str]] = []
    for stream_group in stream_groups:
        if not isinstance(stream_group, list):
            raise ValueError("Each sensor ablation stream group must be a list of stream names.")

        cleaned_group = [stream.strip() for stream in stream_group if isinstance(stream, str) and stream.strip()]
        if cleaned_group:
            cleaned_groups.append(cleaned_group)

    if len(cleaned_groups) == 0:
        raise ValueError("Sensor ablation is enabled, but no valid stream groups were provided.")

    return cleaned_groups

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

if __name__ == "__main__":
    configure_logging()
    args = parse_args()

    config_path = None
    if args.config is not None:
        resolved_config_path = Path(args.config).expanduser().resolve()
        if not resolved_config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")
        
        config_path = str(resolved_config_path)

    config_parser = GLOCExperimentConfigParser(config_location=config_path)
    pipeline = DataPipeline(config_parser = config_parser)
    feature_stream_groups = _resolve_feature_stream_groups(config_parser)
    num_splits = config_parser.get_num_splits()

    for group_index, feature_streams in enumerate(feature_stream_groups, start=1):
        logging.info(
            "Running stream group %d/%d: %s",
            group_index,
            len(feature_stream_groups),
            feature_streams if feature_streams else "ALL_STREAMS",
        )

        for kfold_id in range(num_splits):
            logging.info("Running fold %d/%d", kfold_id + 1, num_splits)
            data = pipeline.get_data(kfold_id=kfold_id, feature_streams=feature_streams)

            print("Data Dimensions:")
            print(f"stream_group={feature_streams if feature_streams else ['ALL_STREAMS']} | fold={kfold_id}")
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