import logging
import json
from pathlib import Path

import numpy as np
from sklearn import metrics
from imblearn.metrics import geometric_mean_score

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.models_new.model_factory import ModelFactory
from src.traditional_experiment_utils import stratified_kfold_split, get_hyperparameters_from_json, plot_f1_violin_with_stream_matrix



def run_sensor_ablation_training(
    config: dict,
    pipeline: DataPipeline,
    model_factory: ModelFactory,
    project_root: Path,
    # get_hyperparameters_from_json_fn: Callable | None,
    # stratified_kfold_split_fn: Callable,
    # plot_f1_violin_by_stream_fn: Callable,
    # extract_f1_score_fn: Callable[[tuple], float] = extract_f1_score,
    # save_model_stream_f1_scores_fn: Callable[..., Path] = save_model_stream_f1_scores,
) -> None:
    """Run sensor ablation training across configured stream groups."""
    training_config = config["sensor_ablation"]["training"]
    feature_stream_groups: list[list[str]] = training_config["streams"]
    models_to_test: list = [model_factory.create_model(model_name) for model_name in training_config["models"]]
    num_splits: int = training_config["num_splits"]
    model_type = training_config["model_type"]
    f1_results_by_stream: dict[str, dict[str, np.ndarray]] = {
        model.name: {}
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
            logging.info("Running model: %s", model.name)

            hyperparameters, _, _, _ = get_hyperparameters_from_json(Path(project_root / training_config["median_hyperparameters_folder"]), model_type, model.name)
            X, y = pipeline.get_data(model = model, feature_streams = feature_streams)
            f1_scores = np.zeros(num_splits, dtype = float)

            for kfold_id in range(num_splits):
                logging.info("Running fold %d/%d", kfold_id + 1, num_splits)
                X_train, X_test, y_train, y_test = stratified_kfold_split(X, y, num_splits, kfold_id, training_config["random_seed"])

                # Re-initialize model for each fold to ensure clean training
                model = model_factory.create_model(model.name, model_hyperparameters = hyperparameters)
                if "random_state" in model.get_model_parameters():
                    model.set_model_parameters({"random_state": training_config["random_seed"]})

                # Train and evaluate
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                fold_result = _evaluate_model(y_test, y_pred)
                f1_scores[kfold_id] = fold_result["f1"]

                logging.info(
                    "Metrics for %s | streams=%s | fold=%d: %s",
                    model.name,
                    feature_streams,
                    kfold_id,
                    fold_result,
                )

            output_path = _save_model_stream_f1_scores(
                results_root_dir = sensor_ablation_results_dir,
                model_name = model.name,
                stream_str = stream_str,
                f1_scores = f1_scores,
            )
            f1_results_by_stream[model.name][stream_str] = f1_scores
            logging.info("Saved F1 scores to %s", output_path)

def _evaluate_model(y_test: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Evaluate model predictions and return legacy metric tuple format."""
    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    specificity = metrics.recall_score(y_test, y_pred, pos_label = 0)
    g_mean = geometric_mean_score(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "g_mean": g_mean
    }

def _save_model_stream_f1_scores(
    results_root_dir: Path,
    model_name: str,
    stream_str: str,
    f1_scores: np.ndarray,
) -> Path:
    """Persist one model-stream F1 array as Results/Sensor_Ablation/<model>/<stream>.json."""
    model_dir = results_root_dir / model_name
    model_dir.mkdir(parents = True, exist_ok = True)
    output_path = model_dir / f"{stream_str}.json"
    f1_scores_dict = {"f1": f1_scores.tolist()}
    with open(output_path, "w") as handle:
        json.dump(f1_scores_dict, handle)
    return output_path



def run_sensor_ablation_review(
    config: dict
) -> None:
    """Review and re-plot saved sensor ablation F1 results using YAML-defined models and one stream group."""
    review_config = config["sensor_ablation"]["review"]
    model_type = review_config["model_type"]
    sensor_ablation_results_dir = Path(review_config["save_results_folder"]) / model_type.get_folder_name()

    models = review_config["models"]
    # models = [model_factory.create_model(model_name) for model_name in review_config["models"]]
    # classifiers_to_load = [model.name for model in models]
    sort_streams_by_median = review_config["sort_streams_by_median"]
    stream_groups = review_config["stream_groups"]

    if len(models) == 0:
        raise ValueError("No models specified for sensor ablation review. Update sensor_ablation.review.models in the config.")

    if len(stream_groups) == 0:
        raise ValueError("No stream groups specified for sensor ablation review. Update sensor_ablation.review.stream_groups in the config.")

    for model in models:
        # Load all stream-group F1 results for this model, then filter or rank as needed for plotting
        f1_results_by_stream = _load_sensor_ablation_f1_results_for_model(
            sensor_ablation_results_dir = sensor_ablation_results_dir,
            model = model,
            stream_groups = stream_groups
        )

        if sort_streams_by_median:
            f1_results_by_stream = _sort_streams_by_median_f1(f1_results_by_stream)

        # Plot the F1 distributions for this model across the requested stream groups
        plot_f1_violin_with_stream_matrix(
            f1_results_by_stream = f1_results_by_stream,
            model = model,
            model_type = model_type
        )

def _load_sensor_ablation_f1_results_for_model(
    sensor_ablation_results_dir: Path,
    model: str,
    stream_groups: list[list[str]],
) -> dict[str, dict[str, np.ndarray]]:
    """Load cached F1 results for one model across multiple stream groups."""
    model_dir = sensor_ablation_results_dir / model
    if not model_dir.exists():
        raise ValueError(f"Expected model directory not found: {model_dir}")

    f1_results_by_stream: dict[str, dict[str, np.ndarray]] = {}
    for stream_group in stream_groups:
        stream_group_str = "-".join(stream_group)
        stream_group_results_file = model_dir / f"{stream_group_str}.json"
        with open(stream_group_results_file, "r") as handle:
            data = json.load(handle)
            f1_results_by_stream[tuple(stream_group)] = np.asarray(data["f1"], dtype = float)

    return f1_results_by_stream # model -> stream group -> F1 array

def _sort_streams_by_median_f1(f1_results_by_stream: dict, descending: bool = True) -> dict:
    """
    Sorts a dictionary of stream F1 scores by their median value.
    """
    return dict(
        sorted(
            f1_results_by_stream.items(),
            key=lambda item: np.median(item[1]),
            reverse=descending
        )
    )