"""
Generates median hyperparameter JSON files for Implicit noAFE models
and saves them directly into ModelSave/CV/Implicit noAFE/median_hyperparameters_{classifier}.json.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any, Dict, List

import joblib
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_DIR = os.path.join(BASE_DIR, "finalized no AFE results")
TARGET_DIR = os.path.join(BASE_DIR, "ModelSave", "CV", "Implicit noAFE")

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "KNN": {
        "dir": "knn",
        "perf_pattern": "FoldSummary_KNN_implicit_fold{i}.pkl",
        "model_pattern": "KNN_model_implicit_fold{i}.pkl",
        "feat_pattern": "SelectedFeatures_KNN_implicit_fold{i}.pkl",
        "out_classifier": "KNN",
    },
    "LDA": {
        "dir": "LDA",
        "perf_pattern": "FoldSummary_LDA_implicit_fold{i}.pkl",
        "model_pattern": "LDA_model_implicit_fold{i}.pkl",
        "feat_pattern": "SelectedFeatures_LDA_implicit_fold{i}.pkl",
        "out_classifier": "LDA",
    },
    "logreg": {
        "dir": "logreg",
        "perf_pattern": "FoldSummary_logreg_implicit_fold{i}.pkl",
        "model_pattern": "logistic_regression_model_implicit_fold{i}.pkl",
        "feat_pattern": "SelectedFeatures_logreg_implicit_fold{i}.pkl",
        "out_classifier": "logreg",
    },
    "RF": {
        "dir": "rf",
        "perf_pattern": "FoldSummary_rf_implicit_fold{i}.pkl",
        "model_pattern": "random_forest_model_implicit_fold{i}.pkl",
        "feat_pattern": "SelectedFeatures_rf_implicit_fold{i}.pkl",
        "out_classifier": "RF",
    },
    "SVM": {
        "dir": "SVM",
        "perf_pattern": "FoldSummary_SVM_implicit_fold{i}.pkl",
        "model_pattern": "svm_model_implicit_fold{i}.pkl",
        "feat_pattern": "SelectedFeatures_SVM_implicit_fold{i}.pkl",
        "out_classifier": "SVM",
    },
    "EGB": {
        "dir": "egb",
        "perf_pattern": "FoldSummary_ensemble_implicit_fold{i}.pkl",
        "model_pattern": "ensemble_model_implicit_fold{i}.pkl",
        "feat_pattern": "SelectedFeatures_EGB_implicit_fold{i}.pkl",
        "out_classifier": "EGB",
    },
}


def sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NumPy and custom containers into JSON-serializable primitives."""
    if isinstance(obj, (np.ndarray,)):
        return [sanitize_for_json(v) for v in obj.tolist()]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (dict, OrderedDict)):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    else:
        return obj


def compute_median_hyperparameters(model_name: str) -> Dict[str, Any]:
    """Computes median hyperparameters for a specified model from finalized no AFE results."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_name]
    model_dir = os.path.join(SOURCE_DIR, cfg["dir"])

    scores: List[Dict[str, Any]] = []
    missing_folds: List[int] = []

    for fold_id in range(10):
        perf_name = cfg["perf_pattern"].format(i=fold_id)
        perf_path = os.path.join(model_dir, "performance", perf_name)
        if not os.path.exists(perf_path):
            missing_folds.append(fold_id)
            continue

        perf_data = joblib.load(perf_path)
        key = str(fold_id) if str(fold_id) in perf_data else list(perf_data.keys())[0]
        f1 = float(perf_data[key]["f1-score"].iloc[0])
        scores.append({"fold_id": str(fold_id), "f1_score": f1})

    if missing_folds or len(scores) < 10:
        raise ValueError(
            f"Model {model_name} has incomplete folds: found {len(scores)} folds, "
            f"missing {missing_folds}. Cannot compute 10-fold median."
        )

    # Sort ascending by F1 score; median fold is index len(scores) // 2 (index 5 of 10)
    scores.sort(key=lambda x: x["f1_score"])
    median_entry = scores[len(scores) // 2]
    med_fold_id = median_entry["fold_id"]
    med_f1 = median_entry["f1_score"]

    # Load model best_params_
    model_file = cfg["model_pattern"].format(i=med_fold_id)
    model_path = os.path.join(model_dir, "model_hpo", model_file)
    model_obj = joblib.load(model_path)
    if hasattr(model_obj, "best_params_"):
        best_params = model_obj.best_params_
    elif hasattr(model_obj, "get_params"):
        best_params = model_obj.get_params()
    else:
        raise AttributeError(f"Model object from {model_path} has neither best_params_ nor get_params")

    # Load selected features
    feat_file = cfg["feat_pattern"].format(i=med_fold_id)
    feat_path = os.path.join(model_dir, "selected_features", feat_file)
    feats = joblib.load(feat_path)
    if isinstance(feats, np.ndarray):
        feats = feats.tolist()
    elif not isinstance(feats, list):
        feats = list(feats)

    return {
        "fold_id": str(med_fold_id),
        "f1_score": float(med_f1),
        "best_params": sanitize_for_json(best_params),
        "selected_features": [str(f) for f in feats],
    }


def generate_implicit_noafe_median_hyperparameters() -> Dict[str, str]:
    """Generates the 5 median hyperparameter JSON files directly in ModelSave/CV/Implicit noAFE/."""
    os.makedirs(TARGET_DIR, exist_ok=True)
    target_models = ["KNN", "LDA", "logreg", "RF", "SVM"]
    generated_paths: Dict[str, str] = {}

    for model_name in target_models:
        clf_out = MODEL_CONFIGS[model_name]["out_classifier"]
        res = compute_median_hyperparameters(model_name)
        out_file = os.path.join(TARGET_DIR, f"median_hyperparameters_{clf_out}.json")
        with open(out_file, "w") as f:
            json.dump(res, f, indent=4)
        generated_paths[clf_out] = out_file

    return generated_paths


if __name__ == "__main__":
    paths = generate_implicit_noafe_median_hyperparameters()
    print(f"Generated {len(paths)} median hyperparameter files in {TARGET_DIR}:")
    for clf, p in paths.items():
        print(f"  - {clf}: {p}")
