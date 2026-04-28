import logging
from pathlib import Path
from typing import Callable


def run_hyperparameter_save(
    config_parser,
    project_root: Path,
    save_median_hyperparameters_fn: Callable,
) -> None:
    """Save median fold hyperparameters and selected features to JSON for configured models."""
    model_type = config_parser.get_model_type()
    models_to_save = config_parser.get_hyperparameter_save_models()

    if len(models_to_save) == 0:
        raise ValueError("hyperparameter_save.models must be a non-empty list when enabled.")

    logging.info(
        "Saving median fold hyperparameters for %d models: %s",
        len(models_to_save),
        ", ".join(models_to_save),
    )

    model_type_folder = model_type.get_folder_name()

    for model_name in models_to_save:
        logging.info("Saving hyperparameters for %s", model_name)
        try:
            output_path = save_median_hyperparameters_fn(
                classifier = model_name,
                model_type_folder_name = model_type_folder,
                project_root = project_root,
            )
            logging.info(
                "Successfully saved hyperparameters for %s to %s",
                model_name,
                output_path,
            )
        except (FileNotFoundError, ValueError) as error:
            logging.error(
                "Failed to save hyperparameters for %s: %s",
                model_name,
                str(error),
            )
            raise
