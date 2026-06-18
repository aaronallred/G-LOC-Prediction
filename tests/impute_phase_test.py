import numpy as np
import pytest

from src.Data_Pipeline.data_pipeline import AdvancedDataPipeline
from src.Data_Pipeline.imputation_config import ImputePhase
from src.model_type import ModelType


def make_pipeline():
    return AdvancedDataPipeline(data_path="/tmp", random_seed=42)


def test_pre_feature_imputation_called(monkeypatch):
    pipeline = make_pipeline()

    # Monkeypatch pandas.read_csv to avoid reading real baseline files
    import pandas as pd

    def fake_read_csv(path, *a, **k):
        return pd.DataFrame({"resting HR [seated]": np.arange(14)})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    # Minimal stubs for upstream steps
    monkeypatch.setattr(pipeline, "_get_data_locations", lambda: {})
    monkeypatch.setattr(pipeline, "_load_data", lambda fp, dtype: None)
    monkeypatch.setattr(pipeline, "_filter_data_by_analysis_type", lambda a, d, s, t: None)
    monkeypatch.setattr(pipeline, "_process_and_get_feature_names", lambda g, fg, mt, fp, dt: (None, {"All": []}))
    monkeypatch.setattr(pipeline, "_label_gloc_events", lambda gloc: np.array([0, 1, 0, 1]))
    # Prevent _remove_all_nan_trials from attempting to index a None dataframe in tests
    monkeypatch.setattr(pipeline, "_remove_all_nan_trials", lambda gloc_data, features, labels, verbose=False: (gloc_data, labels, None))

    # reduce_memory should return an array with NaNs so pre-feature imputer would be meaningful
    sample = np.array([[np.nan, 1, 1], [1, 2, 3], [np.nan, 4, 5], [6, 7, 8]])
    monkeypatch.setattr(pipeline, "_reduce_memory", lambda *args, **kwargs: (sample, np.array([0, 1, 0, 1]), {"AFE_indicator": np.array([0, 1, 0, 1])}))

    called = {"imputer": False}

    def fake_imputer(arr, *args, **kwargs):
        called["imputer"] = True
        return np.nan_to_num(arr, nan=0.0)

    monkeypatch.setattr(pipeline, "_impute_missing_data", fake_imputer)

    # Short-circuit downstream heavy ops
    monkeypatch.setattr(pipeline, "_get_combined_baseline_data", lambda *a, **k: (np.zeros((4, 1)), [], None, None))
    # Short-circuit feature generation to avoid baseline/metadata dependencies
    monkeypatch.setattr(pipeline, "_generate_features", lambda *a, **k: (x_feats, ["f1", "f2"]))
    # Also ensure class-level baseline helper is stubbed in case instance binding didn't apply
    monkeypatch.setattr(AdvancedDataPipeline, "_get_combined_baseline_data", lambda self, *a, **k: (np.zeros((4, 1)), [], None, None))
    # Also patch via module path to be extra robust
    monkeypatch.setattr("src.Data_Pipeline.data_pipeline.AdvancedDataPipeline._get_combined_baseline_data", lambda self, *a, **k: (np.zeros((4, 1)), [], None, None))
    # Monkeypatch pandas.read_csv to avoid reading real files
    import pandas as pd

    def fake_read_csv(path, *args, **kwargs):
        # Return a baseline dataframe with 14 rows so slicing works
        df = pd.DataFrame({"resting HR [seated]": np.arange(14)})
        return df

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(pipeline, "_generate_features", lambda *a, **k: (np.hstack([np.zeros((4, 2)), np.arange(4).reshape(-1, 1)]), ["f1", "f2"]))
    monkeypatch.setattr(pipeline, "_feature_clean_and_prep", lambda *a, **k: (np.hstack([np.zeros((4, 2)), np.arange(4).reshape(-1, 1)]), np.array([0, 1, 0, 1]), ["f1", "f2"], np.arange(4)))
    monkeypatch.setattr(pipeline, "_get_train_test_split", lambda *a, **k: (np.zeros((2, 3)), np.array([0, 1]), np.zeros((2, 3)), np.array([0, 1])))

    pipeline.get_data(model_type=ModelType("Complete", "Explicit"), num_splits=2, kfold_ID=0, impute_file_name="imputed.pkl", impute_phase=ImputePhase.PRE_FEATURE)

    assert called["imputer"], "Pre-feature imputer was not called for impute_phase=PRE_FEATURE"


def test_post_feature_remove_rows_invokes_process_NaN(monkeypatch):
    pipeline = make_pipeline()

    # Upstream stubs
    monkeypatch.setattr(pipeline, "_get_data_locations", lambda: {})
    monkeypatch.setattr(pipeline, "_load_data", lambda fp, dtype: None)
    monkeypatch.setattr(pipeline, "_filter_data_by_analysis_type", lambda a, d, s, t: None)
    monkeypatch.setattr(pipeline, "_process_and_get_feature_names", lambda g, fg, mt, fp, dt: (None, {"All": []}))
    monkeypatch.setattr(pipeline, "_label_gloc_events", lambda gloc: np.array([0, 1, 0, 1]))
    # Prevent _remove_all_nan_trials from attempting to index a None dataframe in tests
    monkeypatch.setattr(pipeline, "_remove_all_nan_trials", lambda gloc_data, features, labels, verbose=False: (gloc_data, labels, None))
    monkeypatch.setattr(pipeline, "_reduce_memory", lambda *args, **kwargs: (np.zeros((4, 3)), np.array([0, 1, 0, 1]), {"AFE_indicator": np.array([0, 1, 0, 1])}))
    monkeypatch.setattr(pipeline, "_get_combined_baseline_data", lambda *a, **k: (np.zeros((4, 1)), [], None, None))
    monkeypatch.setattr(pipeline, "_get_combined_baseline_data", lambda *a, **k: (np.zeros((4, 1)), [], None, None))
    # Short-circuit baseline computation (avoid reading real baseline CSV)
    monkeypatch.setattr(pipeline, "_get_combined_baseline_data", lambda *a, **k: (np.zeros((4, 1)), [], None, None))
    # Short-circuit baseline computation (avoid reading real baseline CSV)
    monkeypatch.setattr(pipeline, "_get_combined_baseline_data", lambda *a, **k: (np.zeros((4, 1)), [], None, None))
    # Short-circuit baseline computation
    monkeypatch.setattr(pipeline, "_get_combined_baseline_data", lambda *a, **k: (np.zeros((4, 1)), [], None, None))
    # Short-circuit baseline computation
    monkeypatch.setattr(pipeline, "_get_combined_baseline_data", lambda *a, **k: (np.zeros((4, 1)), [], None, None))
    # Prevent _remove_all_nan_trials from attempting to index a None dataframe in tests
    monkeypatch.setattr(pipeline, "_remove_all_nan_trials", lambda gloc_data, features, labels, verbose=False: (gloc_data, labels, None))

    # Generate features containing NaNs (last column must be trial ints)
    x_with_nans = np.array([[1.0, np.nan, 0], [2.0, 3.0, 1], [np.nan, 5.0, 2], [4.0, 6.0, 3]])
    monkeypatch.setattr(pipeline, "_generate_features", lambda *a, **k: (x_with_nans, ["f1", "f2"]))

    called = {"process_NaN": False}

    def fake_process_NaN(x_feature_matrix, y_labels, all_features, trials):
        called["process_NaN"] = True
        # emulate removal of NaN rows: keep only rows without NaN
        mask = ~np.isnan(x_feature_matrix).any(axis=1)
        return x_feature_matrix[mask], y_labels[mask], all_features, trials[mask]

    monkeypatch.setattr(pipeline, "_process_NaN", fake_process_NaN)

    # Short-circuit downstream
    monkeypatch.setattr(pipeline, "_get_combined_baseline_data", lambda *a, **k: (np.zeros((4, 1)), [], None, None))
    monkeypatch.setattr(pipeline, "_get_train_test_split", lambda *a, **k: (np.zeros((2, 3)), np.array([0, 1]), np.zeros((2, 3)), np.array([0, 1])))

    pipeline.get_data(model_type=ModelType("Complete", "Explicit"), num_splits=2, kfold_ID=0, impute_file_name="imputed.pkl", impute_phase=ImputePhase.POST_FEATURE_REMOVE_ROWS)

    assert called["process_NaN"], "_process_NaN was not invoked for impute_phase=POST_FEATURE_REMOVE_ROWS"


def test_post_feature_knn_calls_faster_knn_imputer(monkeypatch):
    pipeline = make_pipeline()

    # Stubs upstream
    monkeypatch.setattr(pipeline, "_get_data_locations", lambda: {"baseline": "dummy", "baseline_eeg_processed_list": []})
    monkeypatch.setattr(pipeline, "_load_data", lambda fp, dtype: None)
    monkeypatch.setattr(pipeline, "_filter_data_by_analysis_type", lambda a, d, s, t: None)
    monkeypatch.setattr(pipeline, "_process_and_get_feature_names", lambda g, fg, mt, fp, dt: (None, {"All": []}))
    monkeypatch.setattr(pipeline, "_label_gloc_events", lambda gloc: np.array([0, 1, 0, 1]))
    # Prevent _remove_all_nan_trials from attempting to index a None dataframe in tests
    monkeypatch.setattr(pipeline, "_remove_all_nan_trials", lambda gloc_data, features, labels, verbose=False: (gloc_data, labels, None))
    monkeypatch.setattr(pipeline, "_reduce_memory", lambda *args, **kwargs: (np.zeros((4, 3)), np.array([0, 1, 0, 1]), {"AFE_indicator": np.array([0, 1, 0, 1])}))

    # Provide feature matrix with trial column as last column
    x_feats = np.hstack([np.zeros((4, 2)), np.arange(4).reshape(-1, 1)])
    monkeypatch.setattr(pipeline, "_feature_clean_and_prep", lambda *a, **k: (x_feats, np.array([0, 1, 0, 1]), ["f1", "f2"], np.arange(4)))

    # Ensure train/test split runs without error
    monkeypatch.setattr(pipeline, "_get_train_test_split", lambda *a, **k: (np.zeros((2, 3)), np.array([0, 1]), np.zeros((2, 3)), np.array([0, 1])))

    # grouped kfold split should return train and test indices at positions 4 and 5
    monkeypatch.setattr(pipeline, "_groupedtrial_kfold_split", lambda *a, **k: (None, None, None, None, np.array([0, 1]), np.array([2, 3])))

    called = {"faster": False}

    def fake_faster_knn_impute_train_test(X, train_idx, test_idx, n_neighbors):
        called["faster"] = True
        # return X unchanged
        return X

    monkeypatch.setattr(pipeline, "_faster_knn_impute_train_test", fake_faster_knn_impute_train_test)

    # Short-circuit baseline computation to avoid file IO
    monkeypatch.setattr(pipeline, "_get_combined_baseline_data", lambda *a, **k: (np.zeros((4, 1)), [], None, None))
    # Ensure feature generation is also short-circuited for this test
    monkeypatch.setattr(pipeline, "_generate_features", lambda *a, **k: (x_feats, ["f1", "f2"]))

    pipeline.get_data(model_type=ModelType("Complete", "Explicit"), num_splits=2, kfold_ID=0, impute_file_name="imputed.pkl", impute_phase=ImputePhase.POST_FEATURE_KNN)

    assert called["faster"], "Post-feature KNN imputer was not called for impute_phase=POST_FEATURE_KNN"
