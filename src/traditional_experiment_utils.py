from collections import OrderedDict
from itertools import islice
from pathlib import Path
from typing import Any

import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from src.model_type import ModelType


def stratified_kfold_split(Y, X, num_splits, kfold_ID, random_state=42):
    """Split arrays using a reproducible stratified k-fold index."""
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)

    n_folds = skf.get_n_splits()
    if kfold_ID < 0 or kfold_ID >= n_folds:
        raise ValueError(f"Fold index {kfold_ID} out of range (must be between 0 and {n_folds - 1})")

    train_index, test_index = next(islice(skf.split(X, Y), kfold_ID, kfold_ID + 1))
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]
    return x_train, x_test, y_train, y_test

def get_hyperparameters_from_json(classifier: str, model_type: str):
    """Load cached best params, selected features, fold ID, and score from JSON."""
    base_dir = Path(__file__).resolve().parent.parent
    json_path = base_dir / 'ModelSave' / 'CV' / model_type / f'median_hyperparameters_{classifier}.json'

    with open(json_path, 'r') as f:
        data = json.load(f)

    best_params = OrderedDict(data['best_params'])
    selected_features = data['selected_features']
    score = data['f1_score']
    fold_id = int(data['fold_id'])

    return best_params, selected_features, fold_id, score



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
        Extra subfolder name for saving results.
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
    # Save the plot to the appropriate folder
    # ------------------------------------------------------------
    os.makedirs(save_folder, exist_ok = True)

    plot_path = os.path.join(save_folder, f"f1_violin_by_stream.png")
    g.savefig(plot_path)
    print(f"Saved faceted F1 violin plot to {plot_path}")
    plt.tight_layout()

    # Display the plot
    plt.show()