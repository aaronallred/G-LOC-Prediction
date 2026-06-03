from collections import OrderedDict
from itertools import islice
from pathlib import Path
from typing import Any
import logging

import json
import os

import ast
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        median_hyperparameters_folder: Path,
        model_type: ModelType,
        classifier_name: str    
    ):
    """Load cached best params, selected features, fold ID, and score from the modern CV JSON."""
    with open(median_hyperparameters_folder / model_type.get_folder_name() / classifier_name / "median_hyperparameters.json", 'r') as f:
        data = json.load(f)

    best_params = OrderedDict(data.get('best_params', {}))
    selected_features = data.get('selected_features', [])
    score = data.get('f1_score', 0.0)
    fold_id = int(data.get('fold_id', 0))

    return best_params, selected_features, fold_id, score



# Plotting
def plot_f1_violin_with_stream_matrix(
    f1_results_by_stream: dict,
    model: str,
    model_type: ModelType
):
    """Plot F1 violins with a stream-component matrix for a single model."""
    
    STREAM_LABEL_ALIASES = {
        "ECG-HR-BR-Temperature": "Equivital",
        "Participant": "Demographics",
        "Centrifuge": "G Force",
    }

    def _apply_aliases(stream_iterable) -> list:
        """Finds subset matches of stream components and replaces them with aliases."""
        components = set(stream_iterable)
        for alias_key, alias_name in STREAM_LABEL_ALIASES.items():
            alias_parts = set(alias_key.split("-"))
            if alias_parts.issubset(components):
                components -= alias_parts
                components.add(alias_name)
        return sorted(list(components))

    records = []
    ordered_streams = []
    matrix_columns = []

    for i, (stream_group, scores) in enumerate(f1_results_by_stream.items()):
        # Normalize the dictionary key to an iterable of strings
        if isinstance(stream_group, str):
            try:
                parsed = ast.literal_eval(stream_group)
                stream_group = parsed if isinstance(parsed, list) else [stream_group]
            except (ValueError, SyntaxError):
                stream_group = [stream_group]
                
        display_components = _apply_aliases(stream_group)
        stream_id = f"Group_{i}" # Unique ID to preserve exact dictionary order
        
        ordered_streams.append(stream_id)
        matrix_columns.append(display_components)
        
        for score in scores:
            records.append({
                "Stream": stream_id,
                "F1 Score": score
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Cannot plot empty F1 results.")

    # Extract all unique components for the Y-axis of the matrix
    all_components = set(comp for comps in matrix_columns for comp in comps)
    sorted_components = sorted(list(all_components))

    # Build the binary matrix mapping
    matrix_rows = [
        [1 if comp in comps else 0 for comps in matrix_columns] 
        for comp in sorted_components
    ]
    matrix_df = pd.DataFrame(matrix_rows, index=sorted_components, columns=ordered_streams)

    # Initialize Plot
    fig = plt.figure(figsize=(12, 8))
    title = (f"F1 Score Distributions by Feature Stream\n"
             f"Model: {model} | Model Type: {model_type.afe_filter} {model_type.feature_set}")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.95)

    grid = gridspec.GridSpec(2, 1, height_ratios=[4, 1.5], hspace=0.05)
    ax_top = fig.add_subplot(grid[0])
    ax_bottom = fig.add_subplot(grid[1])

    # Top: Violin Plot
    sns.violinplot(
        data=df,
        x="Stream",
        y="F1 Score",
        order=ordered_streams,
        ax=ax_top,
        palette="Set2",
        inner="box",
        hue="Stream",
        legend=False,
    )

    ax_top.set_xlabel("")
    ax_top.set_xticklabels([])
    ax_top.set_ylabel("F1 Score", fontsize=12)

    # Bottom: Component Matrix Heatmap
    sns.heatmap(
        matrix_df,
        ax=ax_bottom,
        cbar=False,
        cmap="Greens",
        linewidths=1,
        linecolor="lightgray",
        vmin=0,
        vmax=1.5,
    )

    ax_bottom.set_xlabel("Data Stream Combination", fontsize=12)
    ax_bottom.set_xticklabels([])
    ax_bottom.tick_params(left=False, bottom=False)
    ax_bottom.set_yticklabels(sorted_components, rotation=0, fontsize=10)

    plt.show()