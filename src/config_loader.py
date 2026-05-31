from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.model_type import ModelType


class ExperimentConfigLoader(yaml.SafeLoader):
    """YAML loader with safe constructors for modern G-LOC config objects."""


def _construct_model_type(loader: ExperimentConfigLoader, node: yaml.Node) -> ModelType:
    values = loader.construct_sequence(node)
    if len(values) != 2:
        raise ValueError("!ModelType must contain exactly two items: [afe_filter, feature_set].")
    return ModelType(afe_filter=values[0], feature_set=values[1])


ExperimentConfigLoader.add_constructor("!ModelType", _construct_model_type)


def load_experiment_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the experiment YAML into a plain mapping.

    If no path is provided, load the project-level default config file.
    """
    if config_path is None:
        raise ValueError("No config path provided. Please provide a path to a YAML config file.")

    resolved_path = Path(config_path).expanduser().resolve()
    with open(resolved_path, "r", encoding = "utf-8") as handle:
        loaded_config = yaml.load(handle, Loader = ExperimentConfigLoader)

    if not isinstance(loaded_config, dict):
        raise ValueError("Config file must parse into a YAML mapping/object.")

    return loaded_config