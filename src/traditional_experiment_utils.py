from collections import OrderedDict
from itertools import islice
from pathlib import Path
from typing import Any
import logging

import json
import os

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from src.model_type import ModelType


def stratified_kfold_split(X, y, num_splits, kfold_ID, random_state=42):
    """Split arrays using a reproducible stratified k-fold index."""
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)

    n_folds = skf.get_n_splits()
    if kfold_ID < 0 or kfold_ID >= n_folds:
        raise ValueError(f"Fold index {kfold_ID} out of range (must be between 0 and {n_folds - 1})")

    train_index, test_index = next(islice(skf.split(X, y), kfold_ID, kfold_ID + 1))
    x_train, y_train = X[train_index], y[train_index]
    x_test, y_test = X[test_index], y[test_index]
    return x_train, x_test, y_train, y_test

def get_hyperparameters_from_json(
    json_path: Path
):
    """Load cached best params, selected features, fold ID, and score from the modern CV JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    best_params = OrderedDict(data.get('best_params', {}))
    selected_features = data.get('selected_features', [])
    score = data.get('f1_score', 0.0)
    fold_id = int(data.get('fold_id', 0))

    return best_params, selected_features, fold_id, score


def _sanitize_for_json(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to JSON-compatible format.
    
    Parameters
    ----------
    obj : Any
        Object to sanitize (numpy arrays, dicts, lists, or primitives).
        
    Returns
    -------
    Any
        JSON-serializable version of the object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    else:
        return obj


def save_median_hyperparameters(
    classifier: str, 
    model_type_folder_name: str,
    project_root: Path | None = None,
) -> Path:
    """Return the modern median-hyperparameters JSON path for a classifier.

    The modern cross-validation writer stores one JSON file per model under:
    ``<project_root>/Results/CrossValidation/<model_type>/<classifier>/median_hyperparameters.json``.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    else:
        project_root = Path(project_root)
    output_path = (
        project_root
        / "Results"
        / "CrossValidation"
        / model_type_folder_name
        / classifier
        / "median_hyperparameters.json"
    )

    if not output_path.exists():
        raise FileNotFoundError(f"Median hyperparameters file not found: {output_path}")

    return output_path



# Plotting
def plot_f1_violin_by_stream(f1_results_by_stream: dict, model_type: ModelType, save_folder = None):
    """
    Create faceted violin plots of F1 scores per classifier, grouped by feature stream.

    Parameters
    ----------
    f1_results_by_stream : dict
        Nested dict: classifier -> stream -> list of F1 scores.
        Example: {
            "KNN": {"EEG": [...], "HR": [...]},
            "RF": {"EEG": [...], "HR": [...]},
            "EGB": {"EEG": [...], "HR": [...]}
        }
    model_type : ModelType
        Model type specifier (e.g. ModelType("Complete", "Explicit")).
    save_folder : str, optional
        Directory used to save the plot. If omitted, the plot is only displayed.
    """

    # ------------------------------------------------------------
    # Convert nested dict structure into a long-format DataFrame
    # Each row = one F1 score, with classifier + stream labels
    # This format is required for seaborn's faceted plotting
    # ------------------------------------------------------------
    records = []
    for clf, stream_dict in f1_results_by_stream.items():
        for stream, f1_scores in stream_dict.items():
            for score in f1_scores:
                records.append({
                    "Classifier": clf,
                    "Stream": stream,
                    "F1 Score": score
                })
    df = pd.DataFrame(records)
    df['Stream'] = df['Stream'].str.replace('-', '\n')

    # ------------------------------------------------------------
    # Create faceted violin plots:
    # - One column per classifier
    # - Y-axis lists streams
    # - X-axis shows F1 scores
    # - Hue ensures consistent stream colors across classifiers
    # ------------------------------------------------------------
    g = sns.catplot(
        data = df,
        x = "Stream",
        y = "F1 Score",
        col = "Classifier",
        kind = "violin",
        orient = "v",
        inner = "box",          # Adds a small boxplot inside each violin
        hue = "Stream",         # Ensures consistent coloring across panels
        palette = "Set2",
        legend = False,         # Avoid redundant legend (streams already on y-axis)
        sharex = False,          # Lock x-axis across classifiers for comparability
        sharey = True,         # Allow each classifier to show its own stream list
        height = 6,
        aspect = 1.2,
    )

    # Force all panels to use the full F1 range (0–1)
    # g.set(ylim=(0.7, 1))
    # g.set_xticklabels(rotation=20, horizontalalignment='right')

    # Adjust spacing and add a global title
    g.fig.subplots_adjust(top = 0.85)
    g.fig.suptitle(
        f"F1 Score Distributions by Feature Stream | Model: {model_type.get_folder_name()}",
        fontsize = 14,
        fontweight = "bold"
    )

    # ------------------------------------------------------------
    # Save the plot only when the caller provides a destination.
    # ------------------------------------------------------------
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok = True)

        plot_path = os.path.join(save_folder, f"f1_violin_by_stream.png")
        g.savefig(plot_path)
        print(f"Saved faceted F1 violin plot to {plot_path}")
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_f1_violin_with_stream_matrix(
    f1_results_by_stream: dict,
    model_type: ModelType,
    save_folder = None,
):
    """Plot classifier-wise F1 violins with a stream-component matrix under each panel."""
    import matplotlib.gridspec as gridspec

    records = []
    for clf, stream_dict in f1_results_by_stream.items():
        for stream, f1_scores in stream_dict.items():
            for score in f1_scores:
                records.append(
                    {
                        "Classifier": clf,
                        "Stream": stream,
                        "F1 Score": score,
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Cannot plot empty F1 results.")

    classifiers = df["Classifier"].unique()
    unique_streams = df["Stream"].unique()

    components = set()
    for stream_name in unique_streams:
        components.update(stream_name.split("-"))
    sorted_components = sorted(components)

    matrix_rows = []
    for component in sorted_components:
        row = []
        for stream_name in unique_streams:
            is_present = 1 if component in stream_name.split("-") else 0
            row.append(is_present)
        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows, index = sorted_components, columns = unique_streams)

    num_classifiers = len(classifiers)
    fig = plt.figure(figsize = (6 * num_classifiers, 10))
    fig.suptitle(
        f"F1 Score Distributions by Feature Stream | Model: {model_type.get_folder_name()}",
        fontsize = 16,
        fontweight = "bold",
        y = 0.95,
    )

    outer_grid = gridspec.GridSpec(1, num_classifiers, wspace = 0.1)

    for idx, classifier in enumerate(classifiers):
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec = outer_grid[idx],
            height_ratios = [4, 1.5],
            hspace = 0.05,
        )

        ax_top = plt.Subplot(fig, inner_grid[0])
        ax_bottom = plt.Subplot(fig, inner_grid[1])
        fig.add_subplot(ax_top)
        fig.add_subplot(ax_bottom)

        classifier_data = df[df["Classifier"] == classifier]
        sns.violinplot(
            data = classifier_data,
            x = "Stream",
            y = "F1 Score",
            order = unique_streams,
            ax = ax_top,
            palette = "Set2",
            inner = "box",
            hue = "Stream",
            legend = False,
        )

        ax_top.set_title(classifier, fontsize = 14, fontweight = "bold")
        ax_top.set_xlabel("")
        ax_top.set_xticklabels([])

        if idx > 0:
            ax_top.set_ylabel("")
            ax_top.set_yticklabels([])
        else:
            ax_top.set_ylabel("F1 Score", fontsize = 12)

        sns.heatmap(
            matrix_df,
            ax = ax_bottom,
            cbar = False,
            cmap = "Greens",
            linewidths = 1,
            linecolor = "lightgray",
            vmin = 0,
            vmax = 1.5,
        )

        ax_bottom.set_xlabel("Data Stream Combination", fontsize = 10)
        ax_bottom.set_xticklabels([])
        ax_bottom.tick_params(left = False, bottom = False)

        if idx > 0:
            ax_bottom.set_yticklabels([])
        else:
            ax_bottom.set_yticklabels(sorted_components, rotation = 0, fontsize = 10)

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok = True)
        plot_path = os.path.join(save_folder, "f1_violin_by_stream.png")
        fig.savefig(plot_path, bbox_inches = "tight")
        print(f"Saved faceted F1 violin plot to {plot_path}")

    plt.show()