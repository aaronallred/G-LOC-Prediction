import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.Data_Pipeline.data_pipeline import BaseGLOCDataPipeline, DataPipeline, TraditionalDataPipelineCV, TraditionalDataPipelineEvaluation
from src.model_type import ModelType
from src.models.base import ModelInitStrategy
from src.models.extreme_gradient_boosting import ExtremeGradientBoostingModel
from src.models.k_nearest_neighbors import KNearestNeighborsModel
from src.models.linear_discriminant_analysis import LinearDiscriminantAnalysisModel
from src.models.logistic_regression import LogisticRegression
from src.models.random_forest import RandomForestModel
from src.models.support_vector_machine import SupportVectorMachineModel
from src.scripts import GLOC_classifier_traditional as legacy
from src.scripts import temporal_functions_traditional as legacy_traditional


@pytest.fixture
def binary_split_data():
    X, y = make_classification(
        n_samples=120,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


@pytest.fixture(autouse=True)
def patch_legacy_confusion_matrix(monkeypatch):
    monkeypatch.setattr(legacy, "create_confusion_matrix", lambda *args, **kwargs: None)


def _assert_float_tuple_close(actual, expected, atol=1e-12):
    assert len(actual) == len(expected)
    assert np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), atol=atol)


@pytest.mark.parametrize(
    "model_cls, expected_hyperparameters",
    [
        (
            LogisticRegression,
            {
                "baseline_window": 5,
                "window_size": 12.5,
                "stride": 0.25,
                "feature_reduction_type": "lasso",
                "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 5,
            },
        ),
        (
            RandomForestModel,
            {
                "baseline_window": 18.75,
                "window_size": 7.5,
                "stride": 0.25,
                "feature_reduction_type": "none",
                "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 3,
            },
        ),
        (
            LinearDiscriminantAnalysisModel,
            {
                "baseline_window": 46.25,
                "window_size": 15,
                "stride": 0.25,
                "feature_reduction_type": "lasso",
                "baseline_methods_to_use": ["v0", "v1", "v2"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 3,
            },
        ),
        (
            SupportVectorMachineModel,
            {
                "baseline_window": 32.5,
                "window_size": 15,
                "stride": 0.25,
                "feature_reduction_type": "ridge",
                "baseline_methods_to_use": ["v0", "v1", "v2"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 3,
            },
        ),
        (
            ExtremeGradientBoostingModel,
            {
                "baseline_window": 46.25,
                "window_size": 12.5,
                "stride": 0.25,
                "feature_reduction_type": "lasso",
                "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
                "imbalance_type": "none",
                "impute_type": 1,
                "n_neighbors": 3,
            },
        ),
        (
            KNearestNeighborsModel,
            {
                "baseline_window": 32.5,
                "window_size": 15,
                "stride": 0.25,
                "feature_reduction_type": "performance",
                "baseline_methods_to_use": ["v0", "v1", "v2"],
                "imbalance_type": "ros",
                "impute_type": 1,
                "n_neighbors": 5,
            },
        ),
    ],
)
def test_traditional_models_expose_legacy_hyperparameters(model_cls, expected_hyperparameters):
    model = model_cls(config={})
    assert model.get_traditional_hyperparameters() == expected_hyperparameters


def _call_legacy_classifier(model_cls, legacy_fn, x_train, x_test, y_train, y_test, legacy_model_name, best_params, strategy):
    """Call legacy classifier with appropriate parameters based on strategy."""
    # Convert strategy enum to the old retrain/temporal flags for legacy comparison
    retrain = strategy == ModelInitStrategy.RETRAIN_WITH_DEFAULTS
    temporal = strategy == ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS

    if model_cls in {LogisticRegression, SupportVectorMachineModel}:
        return legacy_fn(
            x_train,
            x_test,
            y_train,
            y_test,
            None,
            42,
            ".",
            legacy_model_name,
            retrain,
            temporal,
            best_params,
        )

    return legacy_fn(
        x_train,
        x_test,
        y_train,
        y_test,
        42,
        ".",
        legacy_model_name,
        retrain,
        temporal,
        best_params,
    )


@pytest.mark.parametrize(
    "model_cls,legacy_fn,best_params,legacy_model_name",
    [
        (
            LogisticRegression,
            legacy.classify_logistic_regression,
            {"solver": "lbfgs", "C": 1.0, "max_iter": 1000},
            "logreg.pkl",
        ),
        (
            KNearestNeighborsModel,
            legacy.classify_knn,
            {"n_neighbors": 7, "weights": "uniform", "metric": "minkowski"},
            "knn.pkl",
        ),
        (
            LinearDiscriminantAnalysisModel,
            legacy.classify_lda,
            {"solver": "svd"},
            "lda.pkl",
        ),
        (
            SupportVectorMachineModel,
            legacy.classify_svm,
            {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
            "svm.pkl",
        ),
        (
            ExtremeGradientBoostingModel,
            legacy.classify_ensemble_with_gradboost,
            {"n_estimators": 60, "learning_rate": 0.05, "max_depth": 3},
            "egb.pkl",
        ),
    ],
)
def test_traditional_model_temporal_eval_matches_legacy(
    binary_split_data,
    model_cls,
    legacy_fn,
    best_params,
    legacy_model_name,
):
    x_train, x_test, y_train, y_test = binary_split_data
    model = model_cls(config={})

    model_output = model.classify_traditional(
        x_train,
        x_test,
        y_train,
        y_test,
        class_weight_imb=None,
        random_state=42,
        save_folder=".",
        model_name=legacy_model_name,
        strategy=ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS,
        best_params=best_params,
    )

    legacy_output = _call_legacy_classifier(
        model_cls,
        legacy_fn,
        x_train,
        x_test,
        y_train,
        y_test,
        legacy_model_name,
        best_params,
        strategy=ModelInitStrategy.RETRAIN_WITH_CONFIG_PARAMS,
    )

    _assert_float_tuple_close(model_output, legacy_output)


class _TinyTraditionalConfigParser:
    def get_data_path(self):
        return "/tmp/data"

    def get_remove_NaN_trials(self):
        return False

    def get_subject_to_analyze(self):
        return None

    def get_trial_to_analyze(self):
        return None

    def get_analysis_type(self):
        return 2

    def get_output_feature_dtype(self):
        return "float32"

    def get_impute_file_name(self):
        return "imputed.pkl"

    def get_should_impute(self):
        return True

    def get_save_impute(self):
        return False

    def get_load_impute(self):
        return False

    def get_n_neighbors(self):
        return 3

    def get_baseline_window(self):
        return 32.5

    def get_backstep(self):
        return 0

    def get_data_rate(self):
        return 25

    def get_offset(self):
        return 0

    def get_time_start(self):
        return 0


class _TinyTraditionalModel:
    def __init__(self):
        self.is_traditional = True

    def get_name(self):
        return "TinyTrad"

    def get_traditional_hyperparameters(self):
        return {
            "baseline_window": 32.5,
            "window_size": 15.0,
            "stride": 0.25,
            "impute_type": 1,
            "n_neighbors": 3,
        }


def test_traditional_cv_data_pipeline_matches_legacy_oracle_and_uses_imputed_matrix(monkeypatch):
    raw_matrix = np.asarray(
        [
            [1.0, np.nan],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )
    imputed_matrix = np.asarray(
        [
            [1.0, 9.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    gloc_data = pd.DataFrame(
        {
            "trial_id": ["T1", "T1", "T1"],
            "Time (s)": [0.0, 1.0, 2.0],
            "event_validated": ["GLOC", "middle", "return to consciousness"],
            "subject": ["01", "01", "01"],
            "feat_a": raw_matrix[:, 0],
            "feat_b": raw_matrix[:, 1],
        }
    )
    feature_names = ["feat_a", "feat_b"]

    def _legacy_data_locations(_datafolder):
        return {
            "filename": "ignored.csv",
            "baseline": "baseline.csv",
            "demographic": "demographic.csv",
            "eeg": [],
            "baseline_eeg_processed_list": [],
        }

    def _legacy_analysis_driven_csv_processing(*_args, **_kwargs):
        return (
            gloc_data.copy(),
            raw_matrix.copy(),
            raw_matrix.copy(),
            raw_matrix.copy(),
            raw_matrix.copy(),
            feature_names.copy(),
            feature_names.copy(),
            feature_names.copy(),
            feature_names.copy(),
        )

    def _legacy_label_gloc_events(_gloc_data_reduced):
        return labels.copy()

    def _legacy_afe_subset(*_args, **_kwargs):
        return (
            gloc_data.copy(),
            raw_matrix.copy(),
            raw_matrix.copy(),
            raw_matrix.copy(),
            raw_matrix.copy(),
            labels.copy(),
        )

    def _legacy_remove_all_nan_trials(*_args, **_kwargs):
        return (
            gloc_data.copy(),
            raw_matrix.copy(),
            raw_matrix.copy(),
            raw_matrix.copy(),
            raw_matrix.copy(),
            labels.copy(),
            pd.DataFrame(),
        )

    def _legacy_faster_knn_impute(_features, _n_neighbors):
        return imputed_matrix.copy()

    def _legacy_baseline_data(*_args, **_kwargs):
        return (
            imputed_matrix.copy(),
            feature_names.copy(),
            imputed_matrix.copy(),
            feature_names.copy(),
        )

    def _legacy_feature_generation(*_args, **_kwargs):
        return labels.copy(), imputed_matrix.copy(), feature_names.copy()

    def _legacy_remove_constant_columns(x_feature_matrix, all_features):
        return x_feature_matrix.copy(), all_features.copy()

    def _legacy_process_nan_temporal(y_gloc_labels, x_feature_matrix, all_features):
        return y_gloc_labels.copy(), x_feature_matrix.copy(), all_features.copy(), np.asarray([], dtype=int)

    def _new_get_data_locations(self):
        return _legacy_data_locations("/tmp/data")

    def _new_load_data(self, *_args, **_kwargs):
        return gloc_data.copy()

    def _new_filter_data_by_analysis_type(self, *_args, **_kwargs):
        return gloc_data.copy()

    def _new_process_and_get_feature_names(self, gloc_data_in, *_args, **_kwargs):
        return gloc_data_in.copy(), {
            "All": feature_names.copy(),
            "Phys": feature_names.copy(),
            "ECG": feature_names[:1],
            "EEG": feature_names[1:],
        }

    def _new_label_gloc_events(_self, _gloc_data_in):
        return labels.copy()

    def _new_afe_subset(_self, gloc_data_in, gloc_labels_in):
        return gloc_data_in.copy(), gloc_labels_in.copy()

    def _new_eeg_specific_imputation(_self, gloc_data_in, _features):
        return gloc_data_in.copy()

    def _new_remove_all_nan_trials(_self, gloc_data_in, _features, gloc_labels_in):
        return gloc_data_in.copy(), gloc_labels_in.copy(), pd.DataFrame()

    def _new_faster_knn_impute(_self, *_args, **_kwargs):
        return imputed_matrix.copy()

    def _new_reduce_memory(_self, gloc_data_in, gloc_labels_in, features_in, _output_feature_dtype):
        experiment_metadata = {
            "trial_id": gloc_data_in["trial_id"].to_numpy(),
            "Time (s)": gloc_data_in["Time (s)"].to_numpy(dtype=np.float32),
            "AFE_indicator": np.zeros((len(gloc_data_in), 1), dtype=np.bool_),
        }
        return gloc_data_in[features_in["All"]].to_numpy(dtype=np.float32), gloc_labels_in.copy(), experiment_metadata

    def _new_get_combined_baseline_data(
        _self,
        gloc_data_all_features_numpy,
        _experiment_metadata,
        _baseline_window,
        _baseline_methods_to_use,
        _features,
        _file_paths,
        _model_type,
    ):
        return (
            gloc_data_all_features_numpy.copy(),
            feature_names.copy(),
            gloc_data_all_features_numpy.copy(),
            feature_names.copy(),
        )

    def _new_feature_generation(_self, *args):
        combined_baseline = args[4]
        gloc_labels_in = args[5]
        combined_baseline_names = args[8]
        return gloc_labels_in.copy(), combined_baseline.copy(), combined_baseline_names.copy()

    def _new_remove_constant_columns(_self, x_feature_matrix, all_features):
        return x_feature_matrix.copy(), all_features.copy()

    def _new_sliding_window_max(_self, *_args, **_kwargs):
        return np.zeros((len(gloc_data), 1), dtype=np.float32), None, np.arange(len(gloc_data))

    def _new_process_nan(_self, x_feature_matrix, y_gloc_labels, all_features, trials):
        return y_gloc_labels.copy(), x_feature_matrix.copy(), all_features.copy(), trials.copy()

    monkeypatch.setattr(legacy_traditional, "data_locations", _legacy_data_locations)
    monkeypatch.setattr(legacy_traditional, "analysis_driven_csv_processing", _legacy_analysis_driven_csv_processing)
    monkeypatch.setattr(legacy_traditional, "label_gloc_events", _legacy_label_gloc_events)
    monkeypatch.setattr(legacy_traditional, "afe_subset", _legacy_afe_subset)
    monkeypatch.setattr(legacy_traditional, "remove_all_nan_trials", _legacy_remove_all_nan_trials)
    monkeypatch.setattr(legacy_traditional, "faster_knn_impute", _legacy_faster_knn_impute)
    monkeypatch.setattr(legacy_traditional, "baseline_data", _legacy_baseline_data)
    monkeypatch.setattr(legacy_traditional, "feature_generation", _legacy_feature_generation)
    monkeypatch.setattr(legacy_traditional, "remove_constant_columns", _legacy_remove_constant_columns)
    monkeypatch.setattr(legacy_traditional, "process_NaN_temporal", _legacy_process_nan_temporal)

    monkeypatch.setattr(BaseGLOCDataPipeline, "_get_data_locations", _new_get_data_locations)
    monkeypatch.setattr(BaseGLOCDataPipeline, "_load_data", _new_load_data)
    monkeypatch.setattr(BaseGLOCDataPipeline, "_filter_data_by_analysis_type", _new_filter_data_by_analysis_type)
    monkeypatch.setattr(BaseGLOCDataPipeline, "_process_and_get_feature_names", _new_process_and_get_feature_names)
    monkeypatch.setattr(BaseGLOCDataPipeline, "_label_gloc_events", _new_label_gloc_events)
    monkeypatch.setattr(BaseGLOCDataPipeline, "_afe_subset", _new_afe_subset)
    monkeypatch.setattr(BaseGLOCDataPipeline, "_eeg_specific_imputation", _new_eeg_specific_imputation)
    monkeypatch.setattr(BaseGLOCDataPipeline, "_remove_all_nan_trials", _new_remove_all_nan_trials)
    monkeypatch.setattr(BaseGLOCDataPipeline, "_reduce_memory", _new_reduce_memory)
    monkeypatch.setattr(BaseGLOCDataPipeline, "_get_combined_baseline_data", _new_get_combined_baseline_data)
    monkeypatch.setattr(BaseGLOCDataPipeline, "_remove_constant_columns", _new_remove_constant_columns)
    monkeypatch.setattr(TraditionalDataPipelineCV, "_process_NaN", _new_process_nan, raising=False)
    monkeypatch.setattr(TraditionalDataPipelineEvaluation, "_faster_knn_impute", _new_faster_knn_impute)
    monkeypatch.setattr(TraditionalDataPipelineEvaluation, "_feature_generation", _new_feature_generation)
    monkeypatch.setattr(TraditionalDataPipelineEvaluation, "_sliding_window_max", _new_sliding_window_max)

    def _run_legacy_reference():
        _file_paths = legacy_traditional.data_locations("/tmp/data")
        legacy_gloc_data_reduced, legacy_features, legacy_features_phys, legacy_features_ecg, legacy_features_eeg, legacy_all_features, legacy_all_features_phys, legacy_all_features_ecg, legacy_all_features_eeg = legacy_traditional.analysis_driven_csv_processing(
            2,
            _file_paths["filename"],
            ["ECG", "BR", "temp", "eyetracking", "rawEEG", "processedEEG"],
            _file_paths["demographic"],
            ["noAFE", "implicit"],
            _file_paths["eeg"],
            None,
            None,
        )
        legacy_gloc = legacy_traditional.label_gloc_events(legacy_gloc_data_reduced)
        legacy_gloc_data_reduced, legacy_features, legacy_features_phys, legacy_features_ecg, legacy_features_eeg, legacy_gloc = legacy_traditional.afe_subset(
            ["noAFE", "implicit"],
            legacy_gloc_data_reduced,
            legacy_all_features,
            legacy_features,
            legacy_features_phys,
            legacy_features_ecg,
            legacy_features_eeg,
            legacy_gloc,
        )
        legacy_features = legacy_traditional.faster_knn_impute(legacy_features, 3)
        legacy_trial_column = legacy_gloc_data_reduced["trial_id"]
        legacy_time_column = legacy_gloc_data_reduced["Time (s)"]
        legacy_event_validated_column = legacy_gloc_data_reduced["event_validated"]
        legacy_subject_column = legacy_gloc_data_reduced["subject"]
        legacy_combined_baseline, legacy_combined_baseline_names, legacy_baseline_v0, legacy_baseline_names_v0 = legacy_traditional.baseline_data(
            ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
            legacy_trial_column,
            legacy_time_column,
            legacy_event_validated_column,
            legacy_subject_column,
            legacy_features,
            legacy_all_features,
            legacy_gloc,
            32.5,
            legacy_features_phys,
            legacy_all_features_phys,
            legacy_features_ecg,
            legacy_all_features_ecg,
            legacy_features_eeg,
            legacy_all_features_eeg,
            {},
            [],
            ["noAFE", "implicit"],
        )
        legacy_y_gloc_labels, legacy_x_feature_matrix, legacy_all_features = legacy_traditional.feature_generation(
            0,
            0,
            0.25,
            15.0,
            legacy_combined_baseline,
            legacy_gloc,
            legacy_trial_column,
            legacy_time_column,
            legacy_combined_baseline_names,
            legacy_baseline_names_v0,
            legacy_baseline_v0,
            ["ECG", "BR", "temp", "eyetracking", "rawEEG", "processedEEG"],
        )
        legacy_x_feature_matrix, legacy_all_features = legacy_traditional.remove_constant_columns(
            legacy_x_feature_matrix,
            legacy_all_features,
        )
        legacy_y_gloc_labels_noNaN, legacy_x_feature_matrix_noNaN, legacy_all_features, _removed_ind = legacy_traditional.process_NaN_temporal(
            legacy_y_gloc_labels,
            legacy_x_feature_matrix,
            legacy_all_features,
        )
        return legacy_x_feature_matrix_noNaN, legacy_y_gloc_labels_noNaN

    def _run_new_pipeline():
        pipeline = DataPipeline(_TinyTraditionalConfigParser())
        pipeline.set_mode("cross_validation")
        pipeline.set_model_type(ModelType("NoAFE", "Implicit"))
        return pipeline.get_data(model=_TinyTraditionalModel())

    legacy_x, legacy_y = _run_legacy_reference()
    new_x, new_y = _run_new_pipeline()

    assert np.allclose(new_x, imputed_matrix)
    assert np.allclose(legacy_x, imputed_matrix)
    assert np.allclose(new_x, legacy_x)
    assert np.array_equal(new_y, legacy_y)
    assert not np.allclose(new_x, raw_matrix, equal_nan=True)

