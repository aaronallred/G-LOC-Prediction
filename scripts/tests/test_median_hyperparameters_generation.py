import os
import pytest
import numpy as np
from scripts.generate_median_hyperparameters_noafe import (
    sanitize_for_json,
    compute_median_hyperparameters,
    generate_implicit_noafe_median_hyperparameters,
)

def test_sanitize_for_json_converts_numpy_types():
    data = {
        "int_val": np.int64(42),
        "float_val": np.float64(3.14159),
        "arr_val": np.array(["feat1", "feat2"]),
        "nested": {"bool_val": np.bool_(True), "list": [np.int32(1), np.float32(2.5)]},
    }
    sanitized = sanitize_for_json(data)
    assert type(sanitized["int_val"]) is int
    assert type(sanitized["float_val"]) is float
    assert type(sanitized["arr_val"]) is list
    assert sanitized["arr_val"] == ["feat1", "feat2"]
    assert type(sanitized["nested"]["bool_val"]) is bool
    assert type(sanitized["nested"]["list"][0]) is int
    assert type(sanitized["nested"]["list"][1]) is float

def test_compute_median_hyperparameters_knn():
    res = compute_median_hyperparameters("KNN")
    assert res["fold_id"] == "1"
    assert pytest.approx(res["f1_score"], rel=1e-4) == 0.989583
    assert "n_neighbors" in res["best_params"]
    assert len(res["selected_features"]) == 33
    assert all(isinstance(f, str) for f in res["selected_features"])

def test_compute_median_hyperparameters_logreg_is_lowercase():
    res = compute_median_hyperparameters("logreg")
    assert res["fold_id"] == "7"
    assert pytest.approx(res["f1_score"], rel=1e-4) == 0.919925
    assert "C" in res["best_params"]
    assert len(res["selected_features"]) == 11518

def test_compute_median_hyperparameters_egb_raises_error():
    with pytest.raises(ValueError, match="incomplete folds"):
        compute_median_hyperparameters("EGB")
