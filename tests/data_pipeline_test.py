import numpy as np
import pytest

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.Data_Pipeline.features import FEATURE_REGISTRY, DEMOGRAPHIC_NAMES
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
            "num_splits": 5,
            "kfold_ID": 3,
            "n_neighbors": 4,
            "baseline_window": 32.5,
            "horizon": 0,
            "feature_streams": None,
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
        kfold_id=0,
        num_splits=5,
        feature_streams=["g force", "participant", "EEG"],
    )

    assert result == "traditional-ok"
    assert backend.calls[0]["model"].get_name() == "RF"
    assert backend.calls[0]["classifier_type"] == "RF"
    # Sensor ablation runs inside the backend, not the facade, so the facade
    # forwards the unfiltered features that _resolve_select_features returned.
    assert backend.calls[0]["select_features"] == [
        "HR (bpm) - Equivital",
        "Pupil diameter left [mm] - Tobii",
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


def test_resolve_feature_groups_for_streams_passthrough_when_no_streams():
    # _resolve_feature_groups_for_streams lives on the backend
    # TraditionalDataPipeline (also on AdvancedDataPipeline via the shared
    # base class), not the DataPipeline facade.
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline
    pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
    default_groups = ("ECG", "BR", "temp", "eyetracking", "G", "rawEEG",
                       "processedEEG", "strain", "demographics")
    filtered, applied, filter_substrings = pipeline._resolve_feature_groups_for_streams(
        None, default_groups
    )
    assert filtered == default_groups
    assert applied is False
    assert filter_substrings is None


def test_resolve_feature_groups_for_streams_unknown_stream_skipped(caplog):
    # Unknown streams are logged and skipped — NOT raised — to keep the
    # pipeline robust to config typos (see plan rationale). When ALL
    # requested streams are unknown, the resolver falls back to the default
    # groups with applied=False.
    import logging
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline
    pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
    default_groups = ("ECG", "BR", "temp", "eyetracking", "G", "rawEEG",
                       "processedEEG", "strain", "demographics")

    with caplog.at_level(logging.WARNING, logger="src.Data_Pipeline.data_pipeline"):
        filtered, applied, filter_substrings = pipeline._resolve_feature_groups_for_streams(
            ["mystery-stream"], default_groups
        )

    assert filtered == default_groups
    assert applied is False
    assert filter_substrings is None
    assert any("Unknown stream" in rec.message or "No usable streams" in rec.message
               for rec in caplog.records)


def test_resolve_feature_groups_for_streams_filters_to_eeg_groups():
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline
    pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
    default_groups = ("ECG", "BR", "temp", "eyetracking", "G", "rawEEG",
                       "processedEEG", "strain", "demographics")
    # Stream "EEG" -> {rawEEG, processedEEG}. Default ordering preserved.
    # ``filter_substrings`` is the lowercased user-provided stream keyword
    # (here ``"eeg"``) emitted for the union-substring post-filter. The
    # substring matches every column produced by both EEG groups, so the
    # post-filter is a no-op for EEG-only requests.
    filtered, applied, filter_substrings = pipeline._resolve_feature_groups_for_streams(
        ["EEG"], default_groups
    )
    assert filtered == ("rawEEG", "processedEEG")
    assert applied is True
    assert filter_substrings == ["eeg"]


def test_resolve_feature_groups_for_streams_ecg_matches_legacy_substring():
    # Stream "ECG" must select only the ECG feature group, matching the legacy
    # ``restrict_feature_space`` substring matcher (which matched columns whose
    # names contained the "ecg" substring only). BR and Temperature are
    # independent streams with their own group keys.
    #
    # ``filter_substrings == ["ecg"]`` ensures the union-substring post-filter
    # (``_apply_substring_filter``) drops the ECG group's bundled HR-derived
    # columns and the ``HRV`` columns emitted by
    # ``_sliding_window_other_features`` (none of which contain the ``"ecg"``
    # substring).
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline
    pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
    default_groups = ("ECG", "BR", "temp", "eyetracking", "G", "rawEEG",
                       "processedEEG", "strain", "demographics")
    filtered, applied, filter_substrings = pipeline._resolve_feature_groups_for_streams(
        ["ECG"], default_groups
    )
    assert filtered == ("ECG",)
    assert applied is True
    assert filter_substrings == ["ecg"]


def test_resolve_feature_groups_for_streams_hr_sets_filter_substrings():
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline
    pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
    default_groups = ("ECG", "BR", "temp", "eyetracking", "G", "rawEEG",
                       "processedEEG", "strain", "demographics")
    # HR spans the ECG feature group (HR-named columns like
    # ``HR (bpm) - Equivital*``) AND ``demographics`` (``participant_HR_*``).
    # Pre-filter expands directly to those feature groups; the sub-stream
    # column narrowing is via the union-substring post-filter with the
    # ``"hr"`` keyword (``_apply_substring_filter``).
    filtered, applied, filter_substrings = pipeline._resolve_feature_groups_for_streams(
        ["HR"], default_groups
    )
    assert set(filtered) == {"ECG", "demographics"}
    assert applied is True
    assert filter_substrings == ["hr"]


def test_apply_substring_filter_keeps_hr_names_only():
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline
    pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
    names = [
        "HR (bpm) - Equivital_v0_mean_s1",
        "ECG Lead 1 - Equivital_v0_mean_s1",
        "BR (rpm) - Equivital_v0_mean_s1",
        "HRV (SDNN)_s1",
        "HRV (RMSSD)_s1",
        "participant_HR_seated_v0_mean_s1",
        "participant_age_v0_mean_s1",
    ]
    filtered = pipeline._apply_substring_filter(names, ["hr"])
    # Substring `hr` (case-insensitive) matches every HR-derived name,
    # including HRV-derived columns the legacy word-boundary regex dropped.
    # This matches the legacy ``restrict_feature_space(['HR'])`` substring
    # behavior.
    assert filtered == [
        "HR (bpm) - Equivital_v0_mean_s1",
        "HRV (SDNN)_s1",
        "HRV (RMSSD)_s1",
        "participant_HR_seated_v0_mean_s1",
    ]


def test_apply_substring_filter_union_for_ecg_plus_hr():
    """Union-substring narrowing reproduces legacy multi-stream union semantics.

    The legacy ``restrict_feature_space(['ECG','HR'])`` matched any column
    whose name contained ``"ecg"`` OR ``"hr"``. NEW's union-substring filter
    with ``["ecg", "hr"]`` returns the same union: ECG Lead variants (matched
    by ``"ecg"``) + HR-derived + HRV + participant_HR_* variants (matched by
    ``"hr"``).
    """
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline
    pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
    names = [
        "HR (bpm) - Equivital_v0_mean_s1",
        "ECG Lead 1 - Equivital_v0_mean_s1",
        "ECG Lead 2 - Equivital_v0_mean_s1",
        "BR (rpm) - Equivital_v0_mean_s1",
        "HRV (SDNN)_s1",
        "HRV (RMSSD)_s1",
        "participant_HR_seated_v0_mean_s1",
        "participant_age_v0_mean_s1",
    ]
    filtered = pipeline._apply_substring_filter(names, ["ecg", "hr"])
    assert filtered == [
        "HR (bpm) - Equivital_v0_mean_s1",
        "ECG Lead 1 - Equivital_v0_mean_s1",
        "ECG Lead 2 - Equivital_v0_mean_s1",
        "HRV (SDNN)_s1",
        "HRV (RMSSD)_s1",
        "participant_HR_seated_v0_mean_s1",
    ]


def test_apply_substring_filter_noop_when_filter_substrings_none():
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline
    pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
    names = ["HR (bpm) - Equivital_v0_mean_s1", "ECG Lead 1 - Equivital_v0_mean_s1"]
    # When filter_substrings is None (no stream filter in effect), the
    # filter is a no-op.
    assert pipeline._apply_substring_filter(names, None) == names


# ---------------------------------------------------------------------------
# Stream-substring parity test: NEW FEATURE_GROUPS-pre-filter + union-substring
# post-filter vs OLD restrict_feature_space substring matcher.
# Verifies every supported stream keyword produces the same column set.
# ---------------------------------------------------------------------------

# Enumerate the engineered variant suffixes applied by the sliding-window feature
# generation step. Mirrors ``_feature_generation`` / ``_sliding_window_other_features``
# behavior: every raw column produces 5 (v-methods) x 12 (stats) x 2 (planes) = 120
# engineered variants.
_V_METHODS = ("v0", "v1", "v2", "v5", "v6")
_STATS = (
    "mean", "stddev", "max", "range",
    "derivative_mean", "derivative_stddev", "derivative_max", "derivative_range",
    "2derivative_mean", "2derivative_stddev", "2derivative_max", "2derivative_range",
)
_PLANES = ("s1", "s2")

# Eye-tracking special-case features emitted by ``_sliding_window_other_features``.
# All eight names contain "pupil" so they survive the substring filter for
# `["Pupil"]` requests.
_EYETRACKING_SPECIAL_FEATURES = (
    "Left Pupil Integral (Non-Baseline)",
    "Right Pupil Integral (Non-Baseline)",
    "Left Pupil Mean of Consecutive Difference (Non-Baseline)",
    "Right Pupil Mean of Consecutive Difference (Non-Baseline)",
    "Left Pupil Max of Consecutive Difference (Non-Baseline)",
    "Right Pupil Max of Consecutive Difference (Non-Baseline)",
    "Left Pupil Sum of Consecutive Difference (Non-Baseline)",
    "Right Pupil Sum of Consecutive Difference (Non-Baseline)",
)

# ECG-group special-case features emitted by ``_sliding_window_other_features``.
# Both names contain "hr" so they survive the substring filter for `["HR"]` and
# `["ECG","HR"]` requests but NOT `["ECG"]` alone (matches legacy matcher).
_ECG_HRV_FEATURES = ("HRV (SDNN)", "HRV (RMSSD)")


def _expand_engineered(raw_names, groups_present):
    """Build the full set of engineered column names from raw names.

    Mirrors the sliding-window feature generation: each raw column
    produces ``<raw>_<vN>_<stat>_<plane>`` variants (5 x 12 x 2 = 120 per
    raw column). Eye-tracking and ECG groups additionally emit per-plane
    special-case columns whose base names contain no ``_vN_`` infix.
    """
    out = []
    for raw in raw_names:
        for v in _V_METHODS:
            for st in _STATS:
                for pl in _PLANES:
                    out.append(f"{raw}_{v}_{st}_{pl}")
    if "eyetracking" in groups_present:
        for n in _EYETRACKING_SPECIAL_FEATURES:
            for pl in _PLANES:
                out.append(f"{n}_{pl}")
    if "ECG" in groups_present:
        for n in _ECG_HRV_FEATURES:
            for pl in _PLANES:
                out.append(f"{n}_{pl}")
    return out


def _get_group_names(group, model_type):
    """Get a feature group's raw feature names without requiring process() to run.

    DemographicsGroup stores ``self.demographic_names`` only after ``process()``
    is invoked on real CSV data, so for tests we use the static
    ``DEMOGRAPHIC_NAMES`` constant the group writes to that attribute.
    """
    if group == "demographics":
        return list(DEMOGRAPHIC_NAMES)
    return FEATURE_REGISTRY[group].get_feature_names(model_type)


def _legacy_restrict_feature_space(streams, universe):
    """Reference implementation of the OLD substring matcher (case-insensitive
    union of stream keywords). Mirrors ``src/scripts/feature_study_main.py``
    restrict_feature_space at commit 88c9f07."""
    sl = [s.lower() for s in streams]
    return {n for n in universe if any(s in n.lower() for s in sl)}


# Every supported single-stream keyword used in shipped configs +
# canonical MISPELL aliases. The parity test covers each individually to
# guard against regressions in the FEATURE_GROUPS pre-filter or the
# substring post-filter.
_SINGLE_STREAMS = (
    "ECG", "HR", "BR", "Temperature", "Pupil", "Centrifuge", "EEG", "Strain",
    "Participant",
)

# Multi-stream combos — covers shipped configs (sensor_ablation_review.yaml,
# shap_generate.yaml, master.yaml) plus the tricky `['ECG','HR']` / `['HR','ECG']`
# union cases that motivated the substring-union filter design.
_MULTI_STREAMS = (
    ("ECG", "HR"),
    ("HR", "ECG"),
    ("ECG", "BR"),
    ("ECG", "Temperature"),
    ("EEG", "Pupil"),
    ("EEG", "Pupil", "Participant"),
    ("EEG", "HR"),
    ("EEG", "ECG"),
    ("EEG", "ECG", "HR"),
    ("EEG", "Pupil", "ECG"),
    ("ECG", "EEG", "Centrifuge", "Participant", "Pupil"),
)


def _build_full_universe(default_groups, model_type):
    """Build the FULL engineered-variant universe across the default feature
    groups (the universe the legacy substring matcher operated over)."""
    all_raw = []
    for g in default_groups:
        all_raw.extend(_get_group_names(g, model_type))
    universe = set(_expand_engineered(all_raw, default_groups))
    # ``AFE_indicator_windowed`` auto-appended for Complete+Explicit model
    # types by both pipelines; no stream keyword contains "afe" so the legacy
    # matcher dropped it for any stream ablation request.
    universe.add("AFE_indicator_windowed")
    return universe


def _run_new_matcher(pipeline, streams, default_groups, model_type):
    """Run the NEW matcher: pre-filter feature groups + AFE-drop + substring-union.

    Mirrors the three legacy call sites in the pipeline (advanced get_data,
    traditional cache branch, traditional non-cache branch)."""
    filtered, applied, filter_substrings = pipeline._resolve_feature_groups_for_streams(
        list(streams), default_groups
    )
    new_raw = []
    for g in filtered:
        new_raw.extend(_get_group_names(g, model_type))
    new_universe = set(_expand_engineered(new_raw, filtered))
    if "ECG" in filtered:  # auto-append AFE for Complete+Explicit
        new_universe.add("AFE_indicator_windowed")
    # Apply AFE drop when stream filter is active — exact parity with both
    # pipeline branches.
    if applied:
        names_list = list(new_universe)
        X_dummy = np.zeros((2, len(names_list)), dtype=np.float32)
        _, names_after_drop = pipeline._drop_afe_indicator_columns(
            X_dummy, names_list, applied
        )
        new_universe = set(names_after_drop)
    if filter_substrings:
        new_universe = set(
            pipeline._apply_substring_filter(list(new_universe), filter_substrings)
        )
    return new_universe


@pytest.mark.parametrize("stream", _SINGLE_STREAMS, ids=lambda s: f"stream={s}")
def test_each_sensor_stream_matches_legacy_substring(stream):
    """Per-stream parity: NEW group pre-filter + union-substring narrowing
    must select exactly the same column set as the legacy
    ``restrict_feature_space`` substring matcher.

    Covers every single stream keyword shipped in configs (ECG, HR, BR,
    Temperature, Pupil, Centrifuge, EEG, Strain, Participant). The
    ECG-without-HR divergence (NEW ECG group bundles HR-derived + HRV
    columns the legacy ``"ecg"`` substring did not match) is corrected by
    the union-substring post-filter.
    """
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline, BaseGLOCDataPipeline
    pipeline = TraditionalDataPipeline(
        data_path="/tmp/nonexistent_data_path", random_seed=42
    )
    mt = ModelType("Complete", "Explicit")
    default_groups = BaseGLOCDataPipeline.FEATURE_GROUPS_BY_MODEL_TYPE[mt]
    universe = _build_full_universe(default_groups, mt)

    old = _legacy_restrict_feature_space([stream], universe)
    new = _run_new_matcher(pipeline, [stream], default_groups, mt)

    assert new == old, (
        f"stream={stream!r} parity mismatch: "
        f"OLD={len(old)}, NEW={len(new)}, "
        f"OLD-only[:3]={sorted(old - new)[:3]}, "
        f"NEW-only[:3]={sorted(new - old)[:3]}"
    )


@pytest.mark.parametrize("streams", _MULTI_STREAMS, ids=lambda s: "+".join(s))
def test_multi_stream_combos_match_legacy_substring_union(streams):
    """Multi-stream parity: NEW union-substring narrowing reproduces the legacy
    ``restrict_feature_space`` union semantics (any column whose name contains
    ANY requested stream keyword).

    Specifically covers the tricky ``['ECG','HR']`` case where the legacy
    union keeps both ``ECG Lead`` variants (matched by ``"ecg"``) and the
    ECG-group-bundled HR-derived columns (matched by ``"hr"``); NEW's prior
    HR-only narrowing approach dropped the ``ECG Lead`` variants (240-column
    divergence). The union-substring filter resolves this.
    """
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline, BaseGLOCDataPipeline
    pipeline = TraditionalDataPipeline(
        data_path="/tmp/nonexistent_data_path", random_seed=42
    )
    mt = ModelType("Complete", "Explicit")
    default_groups = BaseGLOCDataPipeline.FEATURE_GROUPS_BY_MODEL_TYPE[mt]
    universe = _build_full_universe(default_groups, mt)

    old = _legacy_restrict_feature_space(list(streams), universe)
    new = _run_new_matcher(pipeline, streams, default_groups, mt)

    assert new == old, (
        f"streams={list(streams)} parity mismatch: "
        f"OLD={len(old)}, NEW={len(new)}, "
        f"OLD-only[:3]={sorted(old - new)[:3]}, "
        f"NEW-only[:3]={sorted(new - old)[:3]}"
    )


def test_drop_afe_indicator_columns_only_when_filtering_applied():
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline
    pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
    X = np.arange(6).reshape(2, 3).astype(np.float32)
    names = ["Fz - EEG_v0_mean_s1", "AFE_indicator_windowed", "trial_ints"]

    # No-filter case: AFE column is preserved.
    X_out, names_out = pipeline._drop_afe_indicator_columns(X, names, applied=False)
    assert names_out == names
    assert X_out.shape == X.shape

    # Filter case: AFE column dropped from both matrix and name list.
    X_out, names_out = pipeline._drop_afe_indicator_columns(X, names, applied=True)
    assert names_out == ["Fz - EEG_v0_mean_s1", "trial_ints"]
    assert X_out.shape == (2, 2)
    # Column values preserved in the surviving columns.
    np.testing.assert_array_equal(X_out[:, 0], X[:, 0])
    np.testing.assert_array_equal(X_out[:, 1], X[:, 2])


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