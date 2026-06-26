import json
import logging
import pickle
from pathlib import Path
from typing import Sequence

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.traditional_experiment_utils import stratified_kfold_split
from src.modes.shap_grouped_plots import plot_all_grouped_shap_bars

logger = logging.getLogger(__name__)


def create_shap_explanation(
    model,
    x_train,
    x_test,
    feature_names: Sequence[str],
    random_state: int,
    nsamples_train: int = 100,
    nsamples_test: int = 50,
):
    """Create a SHAP Explanation object for a trained model."""

    logger.info("Starting SHAP analysis.")

    feature_names = list(feature_names)

    x_train = pd.DataFrame(x_train, columns=feature_names)
    x_test = pd.DataFrame(x_test, columns=feature_names)

    shap_model = _get_underlying_model(model)

    try:
        logger.info("Trying model-based SHAP explainer.")
        explainer = shap.Explainer(
            shap_model,
            x_train,
            feature_names=list(feature_names),
        )

        if explainer.__class__.__name__ == "TreeExplainer":
            explanation = explainer(x_test, check_additivity=False)
        else:
            explanation = explainer(x_test)

    except Exception as exc:
        logger.warning("Model-based SHAP explainer failed: %s", exc)
        logger.info("Falling back to predict_proba SHAP explainer.")

        predict_proba_fn = _predict_proba_numpy(model)

        background = shap.sample(
            x_train,
            nsamples_train,
            random_state=random_state,
        )

        x_test_subset = shap.sample(
            x_test,
            nsamples_test,
            random_state=random_state,
        )

        explainer = shap.Explainer(
            predict_proba_fn,
            background,
            feature_names=feature_names,
            algorithm="permutation",
        )

        num_features = x_train.shape[1]

        explanation = explainer(
            x_test_subset,
            max_evals=2 * num_features + 1,
        )

    logger.info("Training dataset shape: %s", x_train.shape)
    logger.info("Testing dataset shape: %s", x_test.shape)
    logger.info("SHAP explainer class: %s", explainer.__class__.__name__)

    return explanation


def save_shap_explanation(explanation, save_path: Path) -> None:
    """Save a SHAP Explanation object as a pickle file."""

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("wb") as file:
        pickle.dump(explanation, file)

    logger.info("Saved SHAP explanation to %s", save_path)


def run_shap_analysis(
    config: dict,
    pipeline,
    model_factory,
    project_root: Path,
) -> None:
    """Run SHAP analysis for saved sensor-ablation models."""

    shap_config = config["shap_analysis"]
    training_config = config.get("sensor_ablation", {}).get("training", {})

    model_type = shap_config.get("model_type", training_config.get("model_type"))
    if model_type is None:
        raise ValueError(
            "model_type must be provided in shap_analysis or sensor_ablation.training."
        )

    models = shap_config.get("models", training_config.get("models"))
    streams = shap_config.get("streams", training_config.get("streams"))
    num_splits = shap_config.get("num_splits", training_config.get("num_splits"))
    random_seed = shap_config.get("random_seed", training_config.get("random_seed", 42))
    manual_ablation = shap_config.get(
        "manual_ablation",
        training_config.get("manual_ablation", False),
    )

    if not models:
        raise ValueError("No models specified for SHAP analysis.")
    if not streams:
        raise ValueError("No streams specified for SHAP analysis.")
    if num_splits is None:
        raise ValueError("num_splits must be specified for SHAP analysis.")
    
    plot_saved_only = shap_config.get("plot_saved_only", False)

    if plot_saved_only:
        _plot_saved_shap_explanations(
            shap_config=shap_config,
            project_root=project_root,
            model_type=model_type,
            models=models,
            streams=streams,
            num_splits=num_splits,
        )
        return

    saved_models_dir = _resolve_path(
        project_root,
        shap_config.get("saved_models_folder", "ModelSave/Sensor_Ablation"),
    )

    save_results_dir = _resolve_path(
        project_root,
        shap_config.get("save_results_folder", "Results/SHAP_Analysis"),
    )

    saved_models_dir = _append_model_type_folder_if_needed(
        saved_models_dir,
        model_type,
    )

    save_results_dir = save_results_dir / model_type.get_folder_name()

    nsamples_train = shap_config.get("nsamples_train", 100)
    nsamples_test = shap_config.get("nsamples_test", 50)
    overwrite = shap_config.get("overwrite", False)

    logger.info("Running SHAP analysis mode.")
    logger.info("Loading models from %s", saved_models_dir)
    logger.info("Saving SHAP explanations to %s", save_results_dir)

    save_results_dir.mkdir(parents=True, exist_ok=True)

    pipeline.set_random_seed(random_seed)
    pipeline.set_model_type(model_type)

    feature_group = "raw" if manual_ablation else "cache"
    logger.info("Feature Group: %s", feature_group)

    for group_index, feature_streams in enumerate(streams, start=1):
        logger.info(
            "Running stream group %d/%d: %s",
            group_index,
            len(streams),
            feature_streams,
        )

        stream_str = "-".join(feature_streams)

        for model_name in models:
            logger.info("Running SHAP for model: %s", model_name)

            model_template = model_factory.create_model(model_name)

            X, y, select_features = pipeline.get_data(
                model=model_template,
                feature_streams=feature_streams,
                return_feature_names=True,
                traditional_feature_selection=feature_group,
            )

            metadata_path = saved_models_dir / model_name / stream_str / "metadata.json"
            if metadata_path.exists():
                metadata = _load_model_metadata(metadata_path)
                _validate_feature_names_match(
                    current_feature_names=select_features,
                    saved_feature_names=metadata.get("feature_names", []),
                    metadata_path=metadata_path,
                )

            for kfold_id in range(num_splits):
                fold_num = kfold_id + 1

                output_path = (
                    save_results_dir
                    / model_name
                    / stream_str
                    / f"fold_{fold_num}_shap_explanation.pkl"
                )

                if output_path.exists() and not overwrite:
                    logger.info(
                        "Skipping existing SHAP explanation: %s",
                        output_path,
                    )
                    continue

                model_path = (
                    saved_models_dir
                    / model_name
                    / stream_str
                    / f"fold_{fold_num}.pkl"
                )

                if not model_path.exists():
                    raise FileNotFoundError(f"Saved model not found: {model_path}")

                logger.info("Loading trained model from %s", model_path)
                trained_model = _load_trained_model(model_path)

                X_train, X_test, _, _ = stratified_kfold_split(
                    X,
                    y,
                    num_splits,
                    kfold_id,
                    random_seed,
                )

                if hasattr(trained_model, "n_features_in_"):
                    if trained_model.n_features_in_ != X_train.shape[1]:
                        raise ValueError(
                            f"Saved model expects {trained_model.n_features_in_} features, "
                            f"but recreated X_train has {X_train.shape[1]}."
                )

                explanation = create_shap_explanation(
                    model=trained_model,
                    x_train=X_train,
                    x_test=X_test,
                    feature_names=select_features,
                    random_state=random_seed,
                    nsamples_train=nsamples_train,
                    nsamples_test=nsamples_test,
                )

                _validate_explanation_feature_names(explanation, select_features)
                save_shap_explanation(explanation, output_path)


def _load_trained_model(model_path: Path):
    """Load one saved fold model."""

    with model_path.open("rb") as handle:
        return pickle.load(handle)


def _load_model_metadata(metadata_path: Path) -> dict:
    """Load metadata saved alongside trained fold models."""

    with metadata_path.open("r") as handle:
        return json.load(handle)


def _resolve_path(project_root: Path, path_value: str | Path) -> Path:
    """Resolve relative paths against the project root."""

    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def _append_model_type_folder_if_needed(path: Path, model_type) -> Path:
    """
    Add the model_type folder unless the path already appears to include it.
    """

    model_type_folder = model_type.get_folder_name()

    if path.name == model_type_folder:
        return path

    return path / model_type_folder


def _get_underlying_model(model):
    """
    Try to access the wrapped sklearn model if this is one of your custom model objects.
    Falls back to the object itself.
    """

    for attr_name in ("model", "estimator", "classifier", "clf"):
        if hasattr(model, attr_name):
            return getattr(model, attr_name)

    return model


def _get_predict_proba_fn(model):
    """
    Return a predict_proba function from either the wrapper or underlying sklearn model.
    """

    if hasattr(model, "predict_proba"):
        return model.predict_proba

    underlying_model = _get_underlying_model(model)

    if hasattr(underlying_model, "predict_proba"):
        return underlying_model.predict_proba

    raise AttributeError(
        f"Model of type {type(model)} does not have predict_proba."
    )


def _validate_feature_names_match(
    current_feature_names: Sequence[str],
    saved_feature_names: Sequence[str],
    metadata_path: Path,
) -> None:
    """Raise an error if recreated feature names do not match saved metadata."""

    if not saved_feature_names:
        logger.warning("No feature names found in metadata at %s.", metadata_path)
        return

    if list(current_feature_names) != list(saved_feature_names):
        raise ValueError(
            f"Feature names recreated from pipeline do not match metadata at {metadata_path}. "
            "Do not trust SHAP plots until the pipeline/config matches the saved model."
        )

def _predict_proba_numpy(model):
    predict_proba_fn = _get_predict_proba_fn(model)

    def wrapped_predict_proba(X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        return predict_proba_fn(X)

    return wrapped_predict_proba

def _validate_explanation_feature_names(explanation, feature_names: Sequence[str]) -> None:
    feature_names = list(feature_names)

    if explanation.feature_names is None:
        explanation.feature_names = feature_names

    if list(explanation.feature_names) != feature_names:
        raise ValueError("SHAP explanation feature names do not match selected feature names.")

    values = explanation.values

    if values.ndim == 2:
        n_explanation_features = values.shape[1]
    elif values.ndim == 3:
        n_explanation_features = values.shape[1]
    else:
        raise ValueError(f"Unexpected SHAP values shape: {values.shape}")

    if n_explanation_features != len(feature_names):
        raise ValueError(
            f"SHAP values have {n_explanation_features} features, "
            f"but feature_names has {len(feature_names)}."
        )
    
def plot_shap_violin(
    explanation,
    classifier: str,
    model_type,
    save_path: Path,
    print_vals: bool = False,
    max_display: int = 20,
    class_index: int = 1,
    plot_width: float = 20,
    plot_height: float = 10,
    left_margin: float = 0.42,
    right_margin: float = 0.92,
) -> None:
    """Plot and save the original SHAP violin plot."""

    explanation_to_plot = _select_class_explanation(
        explanation=explanation,
        class_index=class_index,
    )

    if print_vals:
        logger.info("--- Feature Importance Sum of Effect Magnitudes ---")

        feature_scores = []

        for idx, feature_name in enumerate(explanation_to_plot.feature_names):
            score = np.sum(np.abs(explanation_to_plot.values[:, idx]))
            feature_scores.append((feature_name, score))

        feature_scores = sorted(
            feature_scores,
            key=lambda item: item[1],
            reverse=True,
        )

        for name, score in feature_scores:
            logger.info("%s: %.6f", name, score)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(plot_width, plot_height))

    shap.plots.violin(
        explanation_to_plot,
        max_display=max_display,
        show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(plot_width, plot_height)

    ax = plt.gca()
    ax.set_title(
        f"Feature Importance - {classifier} {model_type.get_folder_name()}",
        fontsize=16,
    )

    plt.subplots_adjust(
        left=left_margin,
        right=right_margin,
        top=0.92,
        bottom=0.12,
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP violin plot to %s", save_path)

def _select_class_explanation(explanation, class_index: int = 1):
    """
    For binary/multiclass explanations, select one class before plotting.
    For already-2D explanations, return as-is.
    """

    if explanation.values.ndim == 2:
        return explanation

    if explanation.values.ndim == 3:
        return explanation[:, :, class_index]

    raise ValueError(f"Unexpected SHAP values shape: {explanation.values.shape}")

def _load_shap_explanation(explanation_path: Path):
    """Load a saved SHAP Explanation object."""

    with explanation_path.open("rb") as handle:
        return pickle.load(handle)
    
def _validate_plot_scope(plot_scope: str) -> None:
    """Validate requested SHAP plotting scope."""

    valid_plot_scopes = {"individual", "all", "both"}

    if plot_scope not in valid_plot_scopes:
        raise ValueError(
            f"Invalid plot_scope={plot_scope!r}. "
            f"Expected one of {sorted(valid_plot_scopes)}."
        )


def _validate_explanations_can_be_combined(explanations: list) -> None:
    """Validate that fold explanations have matching feature names and compatible shapes."""

    if not explanations:
        raise ValueError("Cannot combine an empty list of SHAP explanations.")

    reference_feature_names = list(explanations[0].feature_names)

    for explanation_index, explanation in enumerate(explanations, start=1):
        if explanation.feature_names is None:
            raise ValueError(
                f"SHAP explanation {explanation_index} has no feature names."
            )

        if list(explanation.feature_names) != reference_feature_names:
            raise ValueError(
                f"SHAP explanation {explanation_index} has different feature names. "
                "Cannot safely consolidate folds."
            )

        if explanation.values.ndim != explanations[0].values.ndim:
            raise ValueError(
                f"SHAP explanation {explanation_index} has values.ndim="
                f"{explanation.values.ndim}, but the first explanation has "
                f"values.ndim={explanations[0].values.ndim}."
            )


def _combine_shap_explanations(explanations: list):
    """Combine fold-level SHAP Explanation objects into one all-fold Explanation."""

    _validate_explanations_can_be_combined(explanations)

    first_explanation = explanations[0]

    combined_values = np.concatenate(
        [explanation.values for explanation in explanations],
        axis=0,
    )

    combined_base_values = None
    if all(explanation.base_values is not None for explanation in explanations):
        combined_base_values = np.concatenate(
            [np.asarray(explanation.base_values) for explanation in explanations],
            axis=0,
        )

    combined_data = None
    if all(explanation.data is not None for explanation in explanations):
        combined_data = np.concatenate(
            [np.asarray(explanation.data) for explanation in explanations],
            axis=0,
        )

    return shap.Explanation(
        values=combined_values,
        base_values=combined_base_values,
        data=combined_data,
        feature_names=first_explanation.feature_names,
    )


def _plot_saved_shap_explanations(
    shap_config: dict,
    project_root: Path,
    model_type,
    models: list[str],
    streams: list[list[str]],
    num_splits: int,
) -> None:
    """Load saved SHAP explanations and create individual and/or consolidated plots."""

    save_results_dir = (
        _resolve_path(
            project_root,
            shap_config.get("save_results_folder", "Results/SHAP_Analysis"),
        )
        / model_type.get_folder_name()
    )

    save_plots_dir = (
        _resolve_path(
            project_root,
            shap_config.get("save_plots_folder", "Results/SHAP_Plots"),
        )
        / model_type.get_folder_name()
    )

    max_display = shap_config.get("max_display", 20)
    class_index = shap_config.get("class_index", 1)
    print_vals = shap_config.get("print_vals", False)

    plot_scope = shap_config.get("plot_scope", "individual")
    _validate_plot_scope(plot_scope)

    plot_individual_folds = plot_scope in {"individual", "both"}
    plot_all_folds = plot_scope in {"all", "both"}

    grouped_bar_config = shap_config.get("grouped_bar_plots", {})
    grouped_bar_enabled = grouped_bar_config.get("enabled", False)

    logger.info("Plotting saved SHAP explanations from %s", save_results_dir)
    logger.info("Saving SHAP plots to %s", save_plots_dir)
    logger.info("SHAP plot scope: %s", plot_scope)

    for feature_streams in streams:
        stream_str = "-".join(feature_streams)

        for model_name in models:
            fold_explanations = []

            for kfold_id in range(num_splits):
                fold_num = kfold_id + 1

                explanation_path = (
                    save_results_dir
                    / model_name
                    / stream_str
                    / f"fold_{fold_num}_shap_explanation.pkl"
                )

                if not explanation_path.exists():
                    raise FileNotFoundError(
                        f"Saved SHAP explanation not found: {explanation_path}"
                    )

                explanation = _load_shap_explanation(explanation_path)
                fold_explanations.append(explanation)

                if plot_individual_folds:
                    plot_path = (
                        save_plots_dir
                        / model_name
                        / stream_str
                        / f"fold_{fold_num}_shap_violin.png"
                    )

                    plot_shap_violin(
                        explanation=explanation,
                        classifier=model_name,
                        model_type=model_type,
                        save_path=plot_path,
                        print_vals=print_vals,
                        max_display=max_display,
                        class_index=class_index,
                        plot_width=shap_config.get("violin_plot_width", 20),
                        plot_height=shap_config.get("violin_plot_height", 10),
                        left_margin=shap_config.get("violin_left_margin", 0.42),
                        right_margin=shap_config.get("violin_right_margin", 0.92),
                    )

                    if grouped_bar_enabled:
                        plot_all_grouped_shap_bars(
                            explanation=explanation,
                            model_name=model_name,
                            model_type=model_type,
                            save_dir=save_plots_dir / model_name / stream_str,
                            fold_num=fold_num,
                            plot_types=grouped_bar_config.get("plot_types"),
                            class_index=class_index,
                            score_mode=grouped_bar_config.get("score_mode", "sum_abs"),
                            max_display=grouped_bar_config.get("max_display", 20),
                            plot_width=grouped_bar_config.get("plot_width", 14),
                            plot_height=grouped_bar_config.get("plot_height", 8),
                            left_margin=grouped_bar_config.get("left_margin", 0.35),
                            right_margin=grouped_bar_config.get("right_margin", 0.95),
                            log_group_stats=grouped_bar_config.get("log_group_stats", True),
                        )

            if plot_all_folds:
                consolidated_explanation = _combine_shap_explanations(
                    fold_explanations
                )

                consolidated_plot_path = (
                    save_plots_dir
                    / model_name
                    / stream_str
                    / "all_folds_shap_violin.png"
                )

                plot_shap_violin(
                    explanation=consolidated_explanation,
                    classifier=model_name,
                    model_type=model_type,
                    save_path=consolidated_plot_path,
                    print_vals=print_vals,
                    max_display=max_display,
                    class_index=class_index,
                    plot_width=shap_config.get("violin_plot_width", 20),
                    plot_height=shap_config.get("violin_plot_height", 10),
                    left_margin=shap_config.get("violin_left_margin", 0.42),
                    right_margin=shap_config.get("violin_right_margin", 0.92),
                )

                if grouped_bar_enabled:
                    plot_all_grouped_shap_bars(
                        explanation=consolidated_explanation,
                        model_name=model_name,
                        model_type=model_type,
                        save_dir=save_plots_dir / model_name / stream_str,
                        fold_num="all_folds",
                        plot_types=grouped_bar_config.get("plot_types"),
                        class_index=class_index,
                        score_mode=grouped_bar_config.get("score_mode", "sum_abs"),
                        max_display=grouped_bar_config.get("max_display", 20),
                        plot_width=grouped_bar_config.get("plot_width", 14),
                        plot_height=grouped_bar_config.get("plot_height", 8),
                        left_margin=grouped_bar_config.get("left_margin", 0.35),
                        right_margin=grouped_bar_config.get("right_margin", 0.95),
                        log_group_stats=grouped_bar_config.get("log_group_stats", True),
                    )