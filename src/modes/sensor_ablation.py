import json
import logging
from pathlib import Path

import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn import metrics

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.advanced_experiment_utils import (
    baseline_down_select,
    get_advanced_predictions_and_targets,
)
from src.models.model_factory import ModelFactory
from src.traditional_experiment_utils import stratified_kfold_split, get_hyperparameters_from_json, \
    plot_f1_violin_with_stream_matrix


def run_sensor_ablation_training(
        config: dict,
        pipeline: DataPipeline,
        model_factory: ModelFactory,
        project_root: Path,
) -> None:
    """Run sensor ablation training across configured stream groups."""
    training_config = config["sensor_ablation"]["training"]
    feature_stream_groups: list[list[str]] = training_config["streams"]
    model_names: list[str] = training_config["models"]
    num_splits: int = training_config["num_splits"]
    model_type = training_config["model_type"]
    manual_ablation = training_config["manual_ablation"]
    class_weight = training_config.get("class_weight", None)

    sensor_ablation_results_dir = Path(training_config["save_results_folder"]) / model_type.get_folder_name()
    sensor_ablation_results_dir.mkdir(parents=True, exist_ok=True)

    sensor_ablation_models_dir = (Path(
        training_config.get("save_models_folder", "ModelSave/Sensor_Ablation")) / model_type.get_folder_name())
    sensor_ablation_models_dir.mkdir(parents=True, exist_ok=True)

    pipeline.set_random_seed(training_config["random_seed"])
    pipeline.set_model_type(model_type)

    feature_group = "cache"
    if manual_ablation == True:
        feature_group = "raw"
    logging.info("Feature Group: %s", feature_group)

    for group_index, feature_streams in enumerate(feature_stream_groups, start=1):
        logging.info("Running stream group %d/%d: %s", group_index, len(feature_stream_groups), feature_streams)
        stream_str = "-".join(feature_streams)

        for model_name in model_names:
            logging.info("Running model: %s", model_name)
            model = model_factory.create_model(model_name)
            hyperparameters, _, _, _ = get_hyperparameters_from_json(
                Path(project_root / training_config["median_hyperparameters_folder"]), model_type, model.name)
            f1_scores = np.zeros(num_splits, dtype=float)

            if model.is_traditional_model:
                _run_traditional_ablation(
                    model=model,
                    model_factory=model_factory,
                    hyperparameters=hyperparameters,
                    pipeline=pipeline,
                    feature_streams=feature_streams,
                    feature_group=feature_group,
                    class_weight=class_weight,
                    num_splits=num_splits,
                    random_seed=training_config["random_seed"],
                    f1_scores=f1_scores,
                    sensor_ablation_models_dir=sensor_ablation_models_dir,
                    sensor_ablation_results_dir=sensor_ablation_results_dir,
                    stream_str=stream_str,
                )
            else:
                _run_advanced_ablation(
                    model=model,
                    model_factory=model_factory,
                    hyperparameters=hyperparameters,
                    pipeline=pipeline,
                    feature_streams=feature_streams,
                    class_weight=class_weight,
                    num_splits=num_splits,
                    random_seed=training_config["random_seed"],
                    f1_scores=f1_scores,
                    sensor_ablation_models_dir=sensor_ablation_models_dir,
                    sensor_ablation_results_dir=sensor_ablation_results_dir,
                    stream_str=stream_str,
                )

            output_path = _save_model_stream_f1_scores(
                results_root_dir=sensor_ablation_results_dir,
                model_name=model.name,
                stream_str=stream_str,
                f1_scores=f1_scores,
            )
            logging.info("Saved F1 scores to %s", output_path)

            metadata_path = _save_model_metadata(
                models_root_dir=sensor_ablation_models_dir,
                model_name=model.name,
                stream_str=stream_str,
                feature_names=[],  # Populated inside ablation runners
                hyperparameters=hyperparameters,
                fold_metrics={},
            )
            logging.info("Saved model metadata to %s", metadata_path)


def _run_traditional_ablation(
        model,
        model_factory: ModelFactory,
        hyperparameters: dict,
        pipeline: DataPipeline,
        feature_streams: list[str],
        feature_group: str,
        class_weight,
        num_splits: int,
        random_seed: int,
        f1_scores: np.ndarray,
        sensor_ablation_models_dir: Path,
        sensor_ablation_results_dir: Path,
        stream_str: str,
) -> None:
    X, y, select_features = pipeline.get_data(
        model=model, feature_streams=feature_streams,
        return_feature_names=True,
        traditional_feature_selection=feature_group,
    )

    for kfold_id in range(num_splits):
        logging.info("Running fold %d/%d", kfold_id + 1, num_splits)
        X_train, X_test, y_train, y_test = stratified_kfold_split(X, y, num_splits, kfold_id, random_seed)

        model = model_factory.create_model(model.name, model_hyperparameters=hyperparameters)
        if class_weight is not None and "class_weight" in model.get_model_parameters():
            model.set_model_parameters({"class_weight": class_weight})
        if "random_state" in model.get_model_parameters():
            model.set_model_parameters({"random_state": random_seed})

        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_result = _evaluate_model(y_test, y_pred)
        f1_scores[kfold_id] = fold_result["f1"]

        model_path = _save_trained_model(
            models_root_dir=sensor_ablation_models_dir,
            model_name=model.name,
            stream_str=stream_str,
            kfold_id=kfold_id + 1,
            model=model,
        )

        logging.info(
            "Metrics for %s | streams=%s | fold=%d: %s",
            model.name, feature_streams, kfold_id, fold_result,
        )
        logging.info("Saved trained model to %s", model_path)


def _run_advanced_ablation(
        model,
        model_factory: ModelFactory,
        hyperparameters: dict,
        pipeline: DataPipeline,
        feature_streams: list[str],
        class_weight,
        num_splits: int,
        random_seed: int,
        f1_scores: np.ndarray,
        sensor_ablation_models_dir: Path,
        sensor_ablation_results_dir: Path,
        stream_str: str,
) -> None:
    for kfold_id in range(num_splits):
        logging.info("Running fold %d/%d", kfold_id + 1, num_splits)
        X_train, X_test, y_train, y_test, features = pipeline.get_data(
            model=model, kfold_id=kfold_id, num_splits=num_splits,
            feature_streams=feature_streams,
        )

        model = model_factory.create_model(model.name, model_hyperparameters=hyperparameters)
        model.all_features = features
        if "random_state" in model.get_model_parameters():
            model.set_model_parameters({"random_state": random_seed})

        model.train(X_train, y_train, class_weight=class_weight)

        X_test_ds, _ = baseline_down_select(X_test, model.all_features, model.best_params["baseline_method"])
        actual_labels, predicted_labels = get_advanced_predictions_and_targets(
            model=model,
            X=X_test_ds,
            y=y_test,
            sequence_length=model.best_params["sequence_length"],
            step_size=10,
            batch_size=model.best_params["batch_size"],
        )
        fold_result = _evaluate_model(actual_labels, predicted_labels)
        f1_scores[kfold_id] = fold_result["f1"]

        model_path = _save_trained_model(
            models_root_dir=sensor_ablation_models_dir,
            model_name=model.name,
            stream_str=stream_str,
            kfold_id=kfold_id + 1,
            model=model,
        )

        logging.info(
            "Metrics for %s | streams=%s | fold=%d: %s",
            model.name, feature_streams, kfold_id, fold_result,
        )
        logging.info("Saved trained model to %s", model_path)


def _evaluate_model(y_test: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Evaluate model predictions and return legacy metric tuple format."""
    # Assess Performance
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    specificity = metrics.recall_score(y_test, y_pred, pos_label=0)
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
    model_dir.mkdir(parents=True, exist_ok=True)
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
        raise ValueError(
            "No models specified for sensor ablation review. Update sensor_ablation.review.models in the config.")

    if len(stream_groups) == 0:
        raise ValueError(
            "No stream groups specified for sensor ablation review. Update sensor_ablation.review.stream_groups in the config.")

    for model in models:
        # Load all stream-group F1 results for this model, then filter or rank as needed for plotting
        f1_results_by_stream = _load_sensor_ablation_f1_results_for_model(
            sensor_ablation_results_dir=sensor_ablation_results_dir,
            model=model,
            stream_groups=stream_groups
        )

        if sort_streams_by_median:
            f1_results_by_stream = _sort_streams_by_median_f1(f1_results_by_stream)

        # Plot the F1 distributions for this model across the requested stream groups
        plot_f1_violin_with_stream_matrix(
            f1_results_by_stream=f1_results_by_stream,
            model=model,
            model_type=model_type
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
            f1_results_by_stream[tuple(stream_group)] = np.asarray(data["f1"], dtype=float)

    return f1_results_by_stream  # model -> stream group -> F1 array


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


def _save_trained_model(
        models_root_dir: Path,
        model_name: str,
        stream_str: str,
        kfold_id: int,
        model,
) -> Path:
    """Save one trained fold model."""
    model_dir = models_root_dir / model_name / stream_str
    model_dir.mkdir(parents=True, exist_ok=True)

    ext = ".pkl" if model.is_traditional_model else ".pt"
    model_path = model_dir / f"fold_{kfold_id}{ext}"
    model.save_model(str(model_path))

    return model_path


def _save_model_metadata(
        models_root_dir: Path,
        model_name: str,
        stream_str: str,
        feature_names: list[str],
        hyperparameters: dict,
        fold_metrics: dict,
) -> Path:
    """Save one metadata file for all folds of one model-stream combination."""
    model_dir = models_root_dir / model_name / stream_str
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = model_dir / "metadata.json"

    metadata = {
        "model_name": model_name,
        "stream_group": stream_str,
        "feature_names": feature_names,
        "hyperparameters": hyperparameters,
        "fold_metrics": fold_metrics,
    }

    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata_path
