from collections import OrderedDict
from itertools import islice
from pathlib import Path
from typing import Any

import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold


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


def get_model_subfolder(model_type):
    """Map the configured model_type tuple to its on-disk results folder."""
    if model_type == ['noAFE', 'phys']:
        return 'Implicit noAFE'
    elif model_type == ['noAFE', 'phys+']:
        return 'Explicit noAFE'
    elif model_type == ['combined', 'phys']:
        return 'Implicit complete'
    elif model_type == ['combined', 'phys+']:
        return 'Explicit complete'
    elif model_type == ['noAFE', 'explicit']:
        return 'Explicit_nonAFE_final'
    elif model_type == ['combined', 'implicit']:
        return 'Implicit complete'
    elif model_type == ['combined', 'explicit']:
        return 'Explicit complete'
    elif model_type == ['noAFE', 'implicit']:
        return 'Implicit noAFE'
    elif model_type == ['complete', 'implicit']:
        return 'Implicit_Complete'
    elif model_type == ['complete', 'explicit']:
        return 'Complete_Explicit'
    else:
        raise ValueError(f"Unrecognized model_type: {model_type}")


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