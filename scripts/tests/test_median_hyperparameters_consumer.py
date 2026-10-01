import json
import os
import sys
import pytest

# Ensure scripts directory is in sys.path for direct imports within scripts
SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from temporal_functions import get_hyperparameters_from_json, get_model_subfolder

TARGET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ModelSave", "CV", "Implicit noAFE")
)

def test_implicit_noafe_directory_has_only_canonical_files():
    """Verifies that all files are clumped in Implicit noAFE without subdirectories."""
    assert os.path.exists(TARGET_DIR)
    entries = sorted(os.listdir(TARGET_DIR))
    expected_files = sorted([
        "median_hyperparameters_KNN.json",
        "median_hyperparameters_LDA.json",
        "median_hyperparameters_logreg.json",
        "median_hyperparameters_RF.json",
        "median_hyperparameters_SVM.json",
    ])
    assert entries == expected_files, f"Unexpected directory contents: {entries}"
    for f in entries:
        full_path = os.path.join(TARGET_DIR, f)
        assert os.path.isfile(full_path), f"Expected file, found directory: {f}"

@pytest.mark.parametrize("classifier, expected_fold, expected_feat_count", [
    ("KNN", 1, 33),
    ("LDA", 8, 32),
    ("logreg", 7, 11518),
    ("RF", 0, 19196),
    ("SVM", 4, 1277),
])
def test_get_hyperparameters_from_json_implicit_noafe(classifier, expected_fold, expected_feat_count):
    model_type = ["noAFE", "implicit"]
    subfolder = get_model_subfolder(model_type)
    assert subfolder == "Implicit noAFE"

    best_params, selected_features, fold_id, score = get_hyperparameters_from_json(
        classifier, subfolder
    )

    assert fold_id == expected_fold
    assert isinstance(score, float)
    assert score > 0.70
    assert isinstance(best_params, dict)
    assert len(best_params) > 0
    assert isinstance(selected_features, list)
    assert len(selected_features) == expected_feat_count
    assert all(isinstance(f, str) for f in selected_features)
