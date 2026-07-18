import numpy as np
import pytest

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.Data_Pipeline.fold_standardizer import GlobalStandardizer, TrialAwareStandardizer
from src.Data_Pipeline.imputation_config import ImputePhase
from src.model_type import ModelType


class DummyModel:
    def __init__(self, *, is_traditional: bool, name: str) -> None:
        self.is_traditional = is_traditional
        self.is_traditional_model = is_traditional
        self.name = name
        self._name = name

    def get_name(self) -> str:
        return self._name


def _make_config() -> dict:
    return {
        "data_path": "/tmp/data",
        "shared_data_parameters": {
            "subject_to_analyze": None,
            "trial_to_analyze": None,
            "analysis_type": 2,
            "remove_NaN_trials": True,
            "impute_file_name": "imputed.pkl",
            "save_impute": False,
            "load_impute": False,
            "impute_phase": ImputePhase.PRE_FEATURE,
            "output_feature_dtype": "float32",
        },
        "advanced_data_parameters": {"n_neighbors": 4, "baseline_window": 32.5, "horizon": 0},
        "traditional_data_parameters": {"backstep": 0, "data_rate": 25, "offset": 0, "time_start": 0},
        "sensor_ablation": {
            "training": {
                "median_hyperparameters_folder": "ModelSave/CV",
            }
        },
    }


class CapturingBackend:
    def __init__(self, return_value):
        self.return_value = return_value
        self.calls = []

    def get_data(self, **kwargs):
        self.calls.append(kwargs)
        return self.return_value


def test_resolve_pipeline_kind_uses_model_flag():
    pipeline = DataPipeline(_make_config())
    assert pipeline._resolve_pipeline_kind(DummyModel(is_traditional=True, name="RF")) == "traditional"
    assert pipeline._resolve_pipeline_kind(DummyModel(is_traditional=False, name="Trans")) == "advanced"


def test_get_data_requires_model_type():
    pipeline = DataPipeline(_make_config())
    with pytest.raises(ValueError, match="model_type must be set"):
        pipeline.get_data(model=DummyModel(is_traditional=True, name="RF"))


def test_get_data_for_advanced_model_forwards_fold_arguments(monkeypatch):
    pipeline = DataPipeline(_make_config())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))
    pipeline.set_random_seed(7)

    backend = CapturingBackend("advanced-ok")
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)

    result = pipeline.get_data(model=DummyModel(is_traditional=False, name="Trans"), kfold_id=3, num_splits=5)

    assert result == "advanced-ok"
    assert backend.calls == [
        {
            "model_type": ModelType("Complete", "Explicit"),
            "remove_NaN_trials": True,
            "subject_to_analyze": None,
            "trial_to_analyze": None,
            "analysis_type": 2,
            "output_feature_dtype": "float32",
            "impute_file_name": "imputed.pkl",
            "impute_phase": ImputePhase.PRE_FEATURE,
            "save_impute": False,
            "load_impute": False,
            "kfold_ID": 3,
            "num_splits": 5,
            "n_neighbors": 4,
            "baseline_window": 32.5,
            "horizon": 0,
        }
    ]


def test_get_data_for_traditional_model_applies_sensor_ablation(monkeypatch):
    pipeline = DataPipeline(_make_config())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))

    backend = CapturingBackend("traditional-ok")
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)
    monkeypatch.setattr(
        pipeline,
        "_resolve_select_features",
        lambda _kwargs: [
            "HR (bpm) - Equivital",
            "Pupil diameter left [mm] - Tobii",
            "magnitude - Centrifuge",
            "participant_age",
            "Fz_alpha - EEG",
        ],
    )

    result = pipeline.get_data(
        model=DummyModel(is_traditional=True, name="RF"),
        feature_streams=["g force", "participant", "EEG"],
    )

    assert result == "traditional-ok"
    assert backend.calls[0]["model"].get_name() == "RF"
    assert backend.calls[0]["classifier_type"] == "RF"
    assert backend.calls[0]["select_features"] == [
        "magnitude - Centrifuge",
        "participant_age",
        "Fz_alpha - EEG",
    ]


def test_get_data_for_traditional_model_can_return_raw_features_without_cache_lookup(monkeypatch):
    pipeline = DataPipeline(_make_config())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))

    backend = CapturingBackend(("raw-ok", ["f1", "f2", "f3"]))
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)
    monkeypatch.setattr(
        pipeline,
        "_resolve_select_features",
        lambda _kwargs: pytest.fail("traditional CV should not read cached median hyperparameters"),
    )

    result = pipeline.get_data(
        model=DummyModel(is_traditional=True, name="RF"),
        kfold_id=0,
        num_splits=5,
        traditional_feature_selection="raw",
        return_feature_names=True,
    )

    assert result == ("raw-ok", ["f1", "f2", "f3"])
    assert backend.calls[0]["classifier_type"] == "RF"
    assert backend.calls[0]["kfold_id"] == 0
    assert backend.calls[0]["num_splits"] == 5
    assert "select_features" not in backend.calls[0]


def test_apply_sensor_ablation_rejects_unknown_stream():
    pipeline = DataPipeline(_make_config())
    with pytest.raises(ValueError, match="Unknown stream\\(s\\)"):
        pipeline._apply_sensor_ablation(["Fz_alpha - EEG"], ["mystery-stream"])


# ---------------------------------------------------------------------------
# Fold-aware standardization tests
# ---------------------------------------------------------------------------


class TestGlobalStandardizer:
    """Tests for src.Data_Pipeline.fold_standardizer.GlobalStandardizer."""

    def test_fit_transform_matches_sklearn_normalization(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 4))
        std = GlobalStandardizer().fit(X).transform(X)
        # StandardScaler-equivalent z-score; per-column zero mean and unit std.
        np.testing.assert_allclose(std.mean(axis=0), np.zeros(4), atol=1e-9)
        np.testing.assert_allclose(std.std(axis=0), np.ones(4), atol=1e-9)

    def test_train_only_statistics_exclude_test_rows(self):
        """Fitting on a subset must not see test rows."""
        X = np.array([[1.0], [2.0], [3.0], [100.0]])  # last row is the "test outlier"
        std = GlobalStandardizer().fit(X[:3]).transform(X)
        # First three rows z-scored using μ=2,σ≈0.816
        np.testing.assert_allclose(std[0], [-1.2247], atol=1e-3)
        np.testing.assert_allclose(std[1], [0.0], atol=1e-6)
        np.testing.assert_allclose(std[2], [1.2247], atol=1e-3)
        # The outlier test row should yield a large magnitude z-score; NOT zero.
        assert abs(std[3, 0]) > 50.0

    def test_zero_std_columns_remain_zero_after_transform(self):
        X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        std = GlobalStandardizer().fit(X).transform(X)
        # Column 0 has zero σ → should stay zero (no NaNs from divide-by-zero).
        np.testing.assert_array_equal(std[:, 0], np.zeros(3))
        np.testing.assert_allclose(std[:, 1].mean(), 0.0, atol=1e-9)

    def test_transform_without_fit_raises(self):
        with pytest.raises(RuntimeError, match="transform called before fit"):
            GlobalStandardizer().transform(np.zeros((2, 2)))


class TestTrialAwareStandardizer:
    """Tests for src.Data_Pipeline.fold_standardizer.TrialAwareStandardizer."""

    def test_straddling_trial_z_scored_against_its_own_train_rows(self):
        """A trial whose windows are split into train/test must use its own train-window μ/σ for test rows."""
        X = np.array([
            [1.0],   # t1 train
            [3.0],   # t1 train
            [5.0],   # t1 test   -> should be z-scored using t1 train mean=2, std=1
            [100.0], # t2 train
            [200.0], # t2 train  -> μ_t2=150, std=50
        ])
        trial = np.array(["t1", "t1", "t1", "t2", "t2"])
        train_mask = np.array([True, True, False, True, True])
        std = TrialAwareStandardizer().fit(X, trial, train_mask).transform(X, trial)
        # t1 test row: (5-2)/1 = 3
        np.testing.assert_allclose(std[2, 0], [3.0])
        # t1 train rows: (1-2)/1=-1, (3-2)/1=1
        np.testing.assert_allclose(std[0, 0], [-1.0])
        np.testing.assert_allclose(std[1, 0], [1.0])
        # t2: z-scored with t2 μ=150, σ=50
        np.testing.assert_allclose(std[3, 0], [-1.0])
        np.testing.assert_allclose(std[4, 0], [1.0])

    def test_fully_test_trial_uses_pooled_training_statistics(self):
        """A trial entirely in the test fold must be z-scored using pooled training μ/σ across all trials."""
        X = np.array([
            [1.0],   # t1 train
            [3.0],   # t1 train  (μ_t1=2, σ_t1=1, μ_pooled=2, σ_pooled=1)
            [10.0],  # t2 test   -> only this row exists for t2, so it has NO train rows of its own
            [100.0], # t3 train  (μ_t3=100, σ_t3=0, falls back to σ=0 guard)
        ])
        trial = np.array(["t1", "t1", "t2", "t3"])
        train_mask = np.array([True, True, False, True])
        std = TrialAwareStandardizer().fit(X, trial, train_mask).transform(X, trial)

        # t2 has no training rows of its own → use pooled training μ/σ.
        # Pooled training rows are [1, 3, 100] → μ=34.667, σ=46.187.
        # Test row (10) → (10-34.667)/46.187 ≈ -0.534
        assert abs(std[2, 0] - (-0.534)) < 0.01

        # t3's training row has σ=0 → zero the z-score (legacy NaN guard).
        np.testing.assert_allclose(std[3, 0], [0.0])

    def test_zero_std_columns_remain_zero(self):
        X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        trial = np.array(["t", "t", "t"])
        train_mask = np.array([True, False, True])
        std = TrialAwareStandardizer().fit(X, trial, train_mask).transform(X, trial)
        # Column 0 has zero σ.
        np.testing.assert_array_equal(std[:, 0], np.zeros(3))


def test_traditional_get_data_requires_fold_info():
    """The facade must raise ValueError for traditional models when fold info is missing."""
    from src.models.k_nearest_neighbors import KNearestNeighborsModel
    pipeline = DataPipeline(_make_config())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))

    with pytest.raises(ValueError, match="kfold_id and num_splits"):
        pipeline.get_data(model=DummyModel(is_traditional=True, name="KNN"))

    with pytest.raises(ValueError, match="kfold_id and num_splits"):
        pipeline.get_data(
            model=DummyModel(is_traditional=True, name="KNN"),
            kfold_id=0,
        )


def test_traditional_get_data_forwards_fold_kwargs():
    """When correctly populated, the facade forwards kfold_id and num_splits to the backend."""
    pipeline = DataPipeline(_make_config())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))

    backend = CapturingBackend("ok")
    monkeypatch_kwarg = None  # type: ignore[name-defined]

    class _RecordingBackend(CapturingBackend):
        def get_data(self, **kwargs):
            nonlocal monkeypatch_kwarg
            monkeypatch_kwarg = kwargs
            return super().get_data(**kwargs)

    pipeline._build_backend = lambda _model: _RecordingBackend("ok")
    pipeline._resolve_select_features = lambda _kwargs: []

    pipeline.get_data(
        model=DummyModel(is_traditional=True, name="KNN"),
        kfold_id=2,
        num_splits=5,
        traditional_feature_selection="raw",
    )
    assert monkeypatch_kwarg is not None
    assert monkeypatch_kwarg["kfold_id"] == 2
    assert monkeypatch_kwarg["num_splits"] == 5