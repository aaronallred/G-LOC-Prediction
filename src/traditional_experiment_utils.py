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
    """Save median-performing fold hyperparameters and features to JSON.
    
    This function loads F1 scores from all 10 cross-validation folds' HPO summaries,
    identifies the median-performing fold, loads its model and selected features,
    and saves them to a standardized JSON file for later retrieval.
    
    Parameters
    ----------
    classifier : str
        Type of classifier ('RF', 'KNN', 'EGB', 'logreg', 'SVM', 'LDA').
    model_type_folder_name : str
        Model type subfolder name (e.g., 'Complete_Explicit').
    project_root : Path, optional
        Root directory of the project. If None, inferred from this file's location.
        
    Returns
    -------
    Path
        Path to the saved JSON file containing median hyperparameters.
        
    Raises
    ------
    ValueError
        If classifier is not recognized or required files are not found.
    FileNotFoundError
        If fold summary or model files cannot be located.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    else:
        project_root = Path(project_root)
    
    # Validate classifier
    valid_classifiers = {'RF', 'KNN', 'EGB', 'logreg', 'SVM', 'LDA'}
    if classifier not in valid_classifiers:
        raise ValueError(
            f"Unsupported classifier: '{classifier}'. "
            f"Valid options: {', '.join(sorted(valid_classifiers))}"
        )
    
    logging.info(f"Loading F1 scores from 10 folds for {classifier} ({model_type_folder_name})")
    
    # --------
    # Step 1: Load F1 scores from each fold's HPO summary
    # --------
    fold_scores = []
    
    for fold_id in range(10):
        fold_str = str(fold_id)
        
        # Build path to the fold summary depending on classifier type
        classifier_to_hpo = {
            'RF': 'rf_hpo',
            'KNN': 'KNN_hpo',
            'EGB': 'EGB_hpo',
            'logreg': 'logreg_hpo',
            'SVM': 'SVM_hpo',
            'LDA': 'LDA_hpo',
        }
        hpo_folder = classifier_to_hpo[classifier]
        perf_path = (
            project_root / 'PerformanceSave' / 'CrossValidation' / 
            hpo_folder / model_type_folder_name / fold_str / 'FoldSummary.pkl'
        )
        
        if not perf_path.exists():
            raise FileNotFoundError(f"Fold summary not found: {perf_path}")
        
        # Load fold summary and extract F1 score
        perf_dict = joblib.load(str(perf_path))
        perf_df = perf_dict[fold_str]
        f1_score = float(perf_df['f1-score'].iloc[0])
        
        fold_scores.append({'fold_id': fold_str, 'f1_score': f1_score})
        logging.debug(f"  Fold {fold_id}: F1 = {f1_score:.4f}")
    
    # --------
    # Step 2: Identify the median-performing fold
    # --------
    fold_scores.sort(key=lambda x: x['f1_score'])  # sort by F1
    median_fold = fold_scores[len(fold_scores) // 2]
    median_fold_id = median_fold['fold_id']
    median_f1 = median_fold['f1_score']
    
    logging.info(f"Median fold: {median_fold_id} with F1 = {median_f1:.4f}")
    
    # --------
    # Step 3: Load model + selected features for the median fold
    # --------
    classifier_to_model_file = {
        'RF': 'random_forest_model.pkl',
        'KNN': 'KNN_model.pkl',
        'EGB': 'ensemble_model.pkl',
        'logreg': 'logistic_regression_model.pkl',
        'SVM': 'SVM_model.pkl',
        'LDA': 'LDA_model.pkl',
    }
    model_filename = classifier_to_model_file[classifier]
    model_path = (
        project_root / 'ModelSave' / 'CV' / model_type_folder_name / 
        median_fold_id / model_filename
    )
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine features file path
    if classifier == 'logreg':
        features_filename = 'SelectedFeaturesLR.pkl'
    else:
        features_filename = f'SelectedFeatures{classifier}.pkl'
    
    features_path = (
        project_root / 'ModelSave' / 'CV' / model_type_folder_name / 
        median_fold_id / features_filename
    )
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    # Load model and selected features
    logging.debug(f"Loading model from {model_path}")
    model = joblib.load(str(model_path))
    
    logging.debug(f"Loading features from {features_path}")
    selected_features = joblib.load(str(features_path))
    
    # --------
    # Step 4: Package and save median fold results
    # --------
    median_result = {
        'fold_id': median_fold_id,
        'f1_score': median_f1,
        'best_params': model.best_params_,
        'selected_features': selected_features
    }
    
    output_dir = project_root / 'ModelSave' / 'CV' / model_type_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'median_hyperparameters_{classifier}.json'
    
    with open(output_path, 'w') as f:
        json.dump(_sanitize_for_json(median_result), f, indent=4)
    
    logging.info(f"Saved median fold data to {output_path}")
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