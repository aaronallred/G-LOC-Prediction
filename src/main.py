import logging

from GLOC_experiment_config_parser import GLOCExperimentConfigParser
from data_pipeline import DataPipeline


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

if __name__ == "__main__":
    configure_logging()

    config_parser = GLOCExperimentConfigParser()
    pipeline = DataPipeline(config_parser = config_parser)
    data = pipeline.get_data()

    print("Data Dimensions:")
    if isinstance(data, tuple) and len(data) >= 4:
        x_train, x_test, y_train, y_test = data[:4]
        print(f"x_train: {x_train.shape}")
        print(f"x_test: {x_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")
    else:
        print("Unexpected data format returned from pipeline.")