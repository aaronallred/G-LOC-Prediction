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

    results_root = Path(training_config["save_results_folder"]) / model_type.get_folder_name()
    results_root.mkdir(parents=True, exist_ok=True)

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
            proto_model = model_factory.create_model(model_name)
            hyperparameters, _, _, _ = get_hyperparameters_from_json(
                Path(project_root / training_config["median_hyperparameters_folder"]), model_type, proto_model.name)

            output_dir = results_root / proto_model.name / stream_str
            output_dir.mkdir(parents=True, exist_ok=True)

            if proto_model.is_traditional_model:
                fold_results = _run_traditional_ablation(
                    model=proto_model,
                    model_factory=model_factory,
                    hyperparameters=hyperparameters,
                    pipeline=pipeline,
                    feature_streams=feature_streams,
                    feature_group=feature_group,
                    class_weight=class_weight,
                    num_splits=num_splits,
                    random_seed=training_config["random_seed"],
                    output_dir=output_dir,
                )
            else:
                fold_results = _run_advanced_ablation(
                    model=proto_model,
                    model_factory=model_factory,
                    hyperparameters=hyperparameters,
                    pipeline=pipeline,
                    feature_streams=feature_streams,
                    class_weight=class_weight,
                    num_splits=num_splits,
                    random_seed=training_config["random_seed"],
                    output_dir=output_dir,
                )

            summary_path = _save_summary(output_dir, hyperparameters, fold_results)
            logging.info("Saved summary to %s", summary_path)


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
        output_dir: Path,
) -> list[dict]:
    X, y, select_features = pipeline.get_data(
        model=model, feature_streams=feature_streams,
        return_feature_names=True,
        traditional_feature_selection=feature_group,
    )

    fold_results: list[dict] = []
    ext = ".pkl" if model.is_traditional_model else ".pt"

    for kfold_id in range(num_splits):
        logging.info("Running fold %d/%d", kfold_id + 1, num_splits)
        X_train, X_test, y_train, y_test = stratified_kfold_split(X, y, num_splits, kfold_id, random_seed)

        fold_model = model_factory.create_model(model.name, model_hyperparameters=hyperparameters)
        if class_weight is not None and "class_weight" in fold_model.get_model_parameters():
            fold_model.set_model_parameters({"class_weight": class_weight})
        if "random_state" in fold_model.get_model_parameters():
            fold_model.set_model_parameters({"random_state": random_seed})

        fold_model.train(X_train, y_train)
        y_pred = fold_model.predict(X_test)
        fold_result = _evaluate_model(y_test, y_pred)
        fold_results.append(fold_result)

        model_path = output_dir / f"fold_{kfold_id}{ext}"
        fold_model.save_model(str(model_path))

        logging.info(
            "Metrics for %s | streams=%s | fold=%d: %s",
            fold_model.name, feature_streams, kfold_id, fold_result,
        )
        logging.info("Saved trained model to %s", model_path)

    return fold_results


def _run_advanced_ablation(
        model,
        model_factory: ModelFactory,
        hyperparameters: dict,
        pipeline: DataPipeline,
        feature_streams: list[str],
        class_weight,
        num_splits: int,
        random_seed: int,
        output_dir: Path,
) -> list[dict]:
    fold_results: list[dict] = []

    for kfold_id in range(num_splits):
        logging.info("Running fold %d/%d", kfold_id + 1, num_splits)
        X_train, X_test, y_train, y_test, features = pipeline.get_data(
            model=model, kfold_id=kfold_id, num_splits=num_splits,
            feature_streams=feature_streams,
        )

        fold_model = model_factory.create_model(model.name, model_hyperparameters=hyperparameters)
        fold_model.all_features = features
        if "random_state" in fold_model.get_model_parameters():
            fold_model.set_model_parameters({"random_state": random_seed})

        fold_model.train(X_train, y_train, class_weight=class_weight)

        X_test_ds, _ = baseline_down_select(X_test, fold_model.all_features, fold_model.best_params["baseline_method"])
        actual_labels, predicted_labels = get_advanced_predictions_and_targets(
            model=fold_model,
            X=X_test_ds,
            y=y_test,
            sequence_length=fold_model.best_params["sequence_length"],
            step_size=10,
            batch_size=fold_model.best_params["batch_size"],
        )
        fold_result = _evaluate_model(actual_labels, predicted_labels)
        fold_results.append(fold_result)

        model_path = output_dir / f"fold_{kfold_id}.pt"
        fold_model.save_model(str(model_path))

        logging.info(
            "Metrics for %s | streams=%s | fold=%d: %s",
            fold_model.name, feature_streams, kfold_id, fold_result,
        )
        logging.info("Saved trained model to %s", model_path)

    return fold_results


def _evaluate_model(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
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
        "g_mean": g_mean,
    }


def _save_summary(
        output_dir: Path,
        model_hyperparameters: dict,
        fold_results: list[dict],
) -> Path:
    performance: dict[str, list[float]] = {}
    for fold_result in fold_results:
        for metric_name, value in fold_result.items():
            performance.setdefault(metric_name, []).append(float(value))

    summary = {
        "model_hyperparameters": model_hyperparameters,
        "performance": performance,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    return summary_path


def run_sensor_ablation_review(
        config: dict
) -> None:
    """Review and re-plot saved sensor ablation F1 results using YAML-defined models and one stream group."""
    review_config = config["sensor_ablation"]["review"]
    model_type = review_config["model_type"]
    sensor_ablation_results_dir = Path(review_config["save_results_folder"]) / model_type.get_folder_name()

    models = review_config["models"]
    sort_streams_by_median = review_config["sort_streams_by_median"]
    stream_groups = review_config["stream_groups"]

    if len(models) == 0:
        raise ValueError(
            "No models specified for sensor ablation review. Update sensor_ablation.review.models in the config.")

    if len(stream_groups) == 0:
        raise ValueError(
            "No stream groups specified for sensor ablation review. Update sensor_ablation.review.stream_groups in the config.")

    for model in models:
        f1_results_by_stream = _load_sensor_ablation_f1_results_for_model(
            sensor_ablation_results_dir=sensor_ablation_results_dir,
            model=model,
            stream_groups=stream_groups
        )

        if sort_streams_by_median:
            f1_results_by_stream = _sort_streams_by_median_f1(f1_results_by_stream)

        plot_f1_violin_with_stream_matrix(
            f1_results_by_stream=f1_results_by_stream,
            model=model,
            model_type=model_type
        )


def _load_sensor_ablation_f1_results_for_model(
        sensor_ablation_results_dir: Path,
        model: str,
        stream_groups: list[list[str]],
) -> dict[tuple[str, ...], np.ndarray]:
    """Load cached F1 results for one model across multiple stream groups."""
    model_dir = sensor_ablation_results_dir / model
    if not model_dir.exists():
        raise ValueError(f"Expected model directory not found: {model_dir}")

    f1_results_by_stream: dict[tuple[str, ...], np.ndarray] = {}
    for stream_group in stream_groups:
        stream_group_str = "-".join(stream_group)
        summary_path = model_dir / stream_group_str / "summary.json"
        with open(summary_path, "r") as handle:
            data = json.load(handle)
            f1_results_by_stream[tuple(stream_group)] = np.asarray(data["performance"]["f1"], dtype=float)

    return f1_results_by_stream


def _sort_streams_by_median_f1(f1_results_by_stream: dict, descending: bool = True) -> dict:
    return dict(
        sorted(
            f1_results_by_stream.items(),
            key=lambda item: np.median(item[1]),
            reverse=descending
        )
    )
