import logging
import json
import pickle
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from numpy import ravel

from src.models.base import ModelInitStrategy
from src.models_new.model_factory import ModelFactory


def _get_model_name(model) -> str:
    if hasattr(model, "get_name"):
        return model.get_name()
    return model.name


def _build_default_hyperparameter_loader(config: dict) -> Callable[[str, str], tuple]:
    """Create a loader that reads median hyperparameters from the sensor ablation config."""
    median_hyperparameters_root = Path(config["sensor_ablation"]["training"]["median_hyperparameters_folder"])

    def _load_hyperparameters(model_name: str, model_type_name: str):
        json_path = median_hyperparameters_root / model_type_name / model_name / "median_hyperparameters.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Median hyperparameters file not found for model '{model_name}' at {json_path}"
            )

        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        best_params = data.get("best_params", {})
        selected_features = data.get("selected_features", [])
        fold_id = data.get("fold_id", None)
        score = data.get("f1_score", None)
        return best_params, selected_features, fold_id, score

    return _load_hyperparameters


def extract_f1_score(metrics_tuple: tuple) -> float:
    """Extract F1 score from legacy metric tuple returned by classify_traditional."""
    if len(metrics_tuple) <= 3:
        raise ValueError(f"Unexpected metrics tuple format. Expected F1 at index 3, got: {metrics_tuple}")
    return float(metrics_tuple[3])


def save_model_stream_f1_scores(
    results_root_dir: Path,
    model_name: str,
    stream_str: str,
    f1_scores: np.ndarray,
) -> Path:
    """Persist one model-stream F1 array as Results/Sensor_Ablation/<model>/<stream>.pkl."""
    model_dir = results_root_dir / model_name
    model_dir.mkdir(parents = True, exist_ok = True)
    output_path = model_dir / f"{stream_str}.pkl"
    with open(output_path, "wb") as handle:
        pickle.dump(np.asarray(f1_scores, dtype = float), handle)
    return output_path


def load_sensor_ablation_f1_results(
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


def filter_sensor_ablation_review_results(
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


def apply_stream_label_aliases(stream_name: str, stream_label_aliases: Mapping[str, str]) -> str:
    """Apply ordered string replacements to stream labels for plot readability."""
    aliased_name = stream_name
    for source_label, target_label in stream_label_aliases.items():
        aliased_name = aliased_name.replace(source_label, target_label)
    return aliased_name


def build_ranked_sensor_ablation_review_results(
    f1_results_by_stream: dict[str, dict[str, np.ndarray]],
    classifiers: list[str],
    stream_label_aliases: Mapping[str, str],
) -> dict[str, dict[str, np.ndarray]]:
    """Replicate preference-11 review behavior using config-selected classifiers."""
    all_streams = {
        stream_name
        for classifier in classifiers
        for stream_name in f1_results_by_stream.get(classifier, {}).keys()
    }

    if len(all_streams) == 0:
        return {}

    stream_median_map: dict[str, float] = {}
    for stream_name in all_streams:
        combined_scores = []
        for classifier in classifiers:
            if stream_name in f1_results_by_stream.get(classifier, {}):
                combined_scores.extend(np.asarray(f1_results_by_stream[classifier][stream_name], dtype = float).tolist())

        if len(combined_scores) > 0:
            stream_median_map[stream_name] = float(np.median(np.asarray(combined_scores, dtype = float)))

    sorted_streams = sorted(stream_median_map, key = stream_median_map.get, reverse = True)

    ranked_results: dict[str, dict[str, np.ndarray]] = {}
    for classifier in classifiers:
        classifier_streams = f1_results_by_stream.get(classifier, {})
        renamed_streams: dict[str, np.ndarray] = {}
        for stream_name in sorted_streams:
            if stream_name not in classifier_streams:
                continue
            aliased_stream_name = apply_stream_label_aliases(stream_name, stream_label_aliases)
            renamed_streams[aliased_stream_name] = np.asarray(classifier_streams[stream_name], dtype = float)

        if len(renamed_streams) > 0:
            ranked_results[classifier] = renamed_streams

    return ranked_results


def run_sensor_ablation_training(
    config: dict,
    pipeline,
    project_root: Path,
    get_hyperparameters_from_json_fn: Callable | None,
    stratified_kfold_split_fn: Callable,
    plot_f1_violin_by_stream_fn: Callable,
    extract_f1_score_fn: Callable[[tuple], float] = extract_f1_score,
    save_model_stream_f1_scores_fn: Callable[..., Path] = save_model_stream_f1_scores,
) -> None:
    """Run sensor ablation training across configured stream groups."""
    if get_hyperparameters_from_json_fn is None:
        get_hyperparameters_from_json_fn = _build_default_hyperparameter_loader(config)

    training_config = config["sensor_ablation"]["training"]
    feature_stream_groups: list[list[str]] = training_config["streams"]
    models_to_test: list = [ModelFactory.create_model(model_name) for model_name in training_config["models"]]
    num_splits: int = training_config["num_splits"]
    model_type = training_config["model_type"]
    f1_results_by_stream: dict[str, dict[str, np.ndarray]] = {
        _get_model_name(model): {}
        for model in models_to_test
    }

    sensor_ablation_results_dir = Path(training_config["save_results_folder"]) / model_type.get_folder_name()
    sensor_ablation_results_dir.mkdir(parents = True, exist_ok = True)
    
    # Set random seed and model type for pipeline operations
    pipeline.set_random_seed(training_config["random_seed"])
    pipeline.set_model_type(model_type)

    for group_index, feature_streams in enumerate(feature_stream_groups, start = 1):
        logging.info("Running stream group %d/%d: %s", group_index, len(feature_stream_groups), feature_streams)
        stream_str = "-".join(feature_streams)

        for model in models_to_test:
            model_name = _get_model_name(model)
            logging.info("Running model: %s", model_name)

            hyperparameters, _, _, _ = get_hyperparameters_from_json_fn(model_name, model_type.get_folder_name())
            x, y = pipeline.get_data(model = model, feature_streams = feature_streams)
            f1_scores = np.zeros(num_splits, dtype = float)

            for kfold_id in range(num_splits):
                logging.info("Running fold %d/%d", kfold_id + 1, num_splits)
                x_train, x_test, y_train, y_test = stratified_kfold_split_fn(
                    Y = ravel(y),
                    X = x,
                    num_splits = num_splits,
                    kfold_ID = kfold_id,
                    random_state = training_config["random_seed"],
                )

                metrics_tuple = model.classify_traditional(
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    None,
                    training_config["random_seed"],
                    save_folder = "",  # TODO: Implement saving and loading of trained models
                    model_name = f"{model_name.lower()}_feature_study.pkl",
                    strategy = ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS,
                    best_params = hyperparameters,
                )

                fold_f1 = extract_f1_score_fn(metrics_tuple)
                f1_scores[kfold_id] = fold_f1

                logging.info(
                    "Metrics for %s | streams=%s | fold=%d: %s",
                    model_name,
                    feature_streams,
                    kfold_id,
                    metrics_tuple,
                )

            output_path = save_model_stream_f1_scores_fn(
                results_root_dir = sensor_ablation_results_dir,
                model_name = model_name,
                stream_str = stream_str,
                f1_scores = f1_scores,
            )
            f1_results_by_stream[model_name][stream_str] = f1_scores
            logging.info("Saved F1 scores to %s", output_path)

    plot_f1_violin_by_stream_fn(
        f1_results_by_stream = f1_results_by_stream,
        model_type = model_type,
        save_folder = sensor_ablation_results_dir,
    )


def run_sensor_ablation_review(
    config: dict,
    project_root: Path,
    load_sensor_ablation_f1_results_fn: Callable[..., dict[str, dict[str, np.ndarray]]],
    build_ranked_sensor_ablation_review_results_fn: Callable[..., dict[str, dict[str, np.ndarray]]],
    filter_sensor_ablation_review_results_fn: Callable[..., dict[str, dict[str, np.ndarray]]],
    plot_f1_violin_with_stream_matrix_fn: Callable,
    plot_f1_violin_by_stream_fn: Callable,
    stream_label_aliases: Mapping[str, str],
) -> None:
    """Review and re-plot saved sensor ablation F1 results using YAML-defined models and one stream group."""
    review_config = config["sensor_ablation"]["review"]
    model_type = review_config["model_type"]
    sensor_ablation_results_dir = Path(review_config["save_results_folder"]) / model_type.get_folder_name()

    models = [ModelFactory.create_model(model_name) for model_name in review_config["models"]]
    classifiers_to_load = [_get_model_name(model) for model in models]
    sort_streams_by_median = review_config["sort_streams_by_median"]
    review_stream_group = review_config["stream_group"]

    loading_stream_group = None
    if (not sort_streams_by_median) and len(review_stream_group) > 0:
        loading_stream_group = review_stream_group

    f1_results_by_stream = load_sensor_ablation_f1_results_fn(
        results_root_dir = sensor_ablation_results_dir,
        classifiers = classifiers_to_load,
        stream_group = loading_stream_group,
    )
    if len(f1_results_by_stream) == 0:
        raise ValueError(
            "Sensor ablation review is enabled, but no cached F1 result files were found under "
            f"{sensor_ablation_results_dir}."
        )

    if sort_streams_by_median:
        filtered_results = build_ranked_sensor_ablation_review_results_fn(
            f1_results_by_stream = f1_results_by_stream,
            classifiers = classifiers_to_load,
            stream_label_aliases = stream_label_aliases,
        )
    else:
        filtered_results = filter_sensor_ablation_review_results_fn(
            f1_results_by_stream = f1_results_by_stream,
            stream_group = review_stream_group,
        )

    if len(filtered_results) == 0:
        if sort_streams_by_median:
            raise ValueError(
                "Sensor ablation review is enabled with median ranking, but no streams were available "
                "for the requested classifiers."
            )
        raise ValueError(
            "Sensor ablation review filters removed all streams. "
            "Update sensor_ablation.review.stream_group in the config."
        )

    if sort_streams_by_median:
        plot_f1_violin_with_stream_matrix_fn(
            f1_results_by_stream = filtered_results,
            model_type = model_type,
        )
    else:
        plot_f1_violin_by_stream_fn(
            f1_results_by_stream = filtered_results,
            model_type = model_type,
        )
