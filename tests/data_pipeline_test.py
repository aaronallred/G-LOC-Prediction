import numpy as np
import json
import pytest

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.Data_Pipeline.features import DEMOGRAPHIC_NAMES, FEATURE_REGISTRY
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
		"traditional_data_parameters": {
			"backstep": 0,
			"data_rate": 25,
			"offset": 0,
			"time_start": 0,
			"standardize_s1": True,
		},
		"sensor_ablation": {"training": {"median_hyperparameters_folder": "ModelSave/CV"}},
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
	assert (
		pipeline._resolve_pipeline_kind(DummyModel(is_traditional=True, name="RF")) == "traditional"
	)
	assert (
		pipeline._resolve_pipeline_kind(DummyModel(is_traditional=False, name="Trans"))
		== "advanced"
	)


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

	result = pipeline.get_data(
		model=DummyModel(is_traditional=False, name="Trans"), kfold_id=3, num_splits=5
	)

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
	assert "select_features" not in backend.calls[0]


def test_apply_sensor_ablation_passthrough_when_none_or_empty():
	from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline

	pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
	features = ["ECG Lead 1", "Fz_alpha - EEG", "magnitude - Centrifuge"]
	assert pipeline._apply_sensor_ablation(features, None) == features
	assert pipeline._apply_sensor_ablation(features, []) == features


def test_apply_sensor_ablation_rejects_unknown_stream():
	from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline

	pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
	with pytest.raises(ValueError, match="Unknown stream\\(s\\)"):
		pipeline._apply_sensor_ablation(["Fz_alpha - EEG"], ["mystery-stream"])


def test_apply_sensor_ablation_single_and_multi_stream():
	from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline

	pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
	features = [
		"ECG Lead 1_v0_mean_s1",
		"HR (bpm) - Equivital_v0_mean_s1",
		"HRV (SDNN)_s1",
		"BR (rpm) - Equivital_v0_mean_s1",
		"Skin Temperature - Equivital_v0_mean_s1",
		"Pupil diameter left [mm] - Tobii_v0_mean_s1",
		"magnitude - Centrifuge_v0_mean_s1",
		"Fz_alpha - EEG_v0_mean_s1",
		"participant_age_v0_mean_s1",
		"participant_HR_seated_v0_mean_s1",
		"AFE_indicator_windowed",
	]
	# ECG alone keeps only ECG Lead (not HR, HRV, BR, Temp)
	ecg_res = pipeline._apply_sensor_ablation(features, ["ECG"])
	assert ecg_res == ["ECG Lead 1_v0_mean_s1"]

	# HR alone keeps HR, HRV, and participant_HR
	hr_res = pipeline._apply_sensor_ablation(features, ["HR"])
	assert hr_res == [
		"HR (bpm) - Equivital_v0_mean_s1",
		"HRV (SDNN)_s1",
		"participant_HR_seated_v0_mean_s1",
	]

	# ECG + HR union
	ecg_hr_res = pipeline._apply_sensor_ablation(features, ["ECG", "HR"])
	assert ecg_hr_res == [
		"ECG Lead 1_v0_mean_s1",
		"HR (bpm) - Equivital_v0_mean_s1",
		"HRV (SDNN)_s1",
		"participant_HR_seated_v0_mean_s1",
	]

	# Aliases: demographics -> participant_, g force -> centrifuge
	alias_res = pipeline._apply_sensor_ablation(features, ["demographics", "g force"])
	assert alias_res == [
		"magnitude - Centrifuge_v0_mean_s1",
		"participant_age_v0_mean_s1",
		"participant_HR_seated_v0_mean_s1",
	]


def test_advanced_sensor_ablation_filters_at_end():
	from src.Data_Pipeline.data_pipeline import AdvancedDataPipeline

	pipeline = AdvancedDataPipeline.__new__(AdvancedDataPipeline)
	all_features = [
		"ECG Lead 1_v0_mean",
		"HR (bpm) - Equivital_v0_mean",
		"Pupil diameter left [mm] - Tobii_v0_mean",
		"magnitude - Centrifuge_v0_mean",
	]
	# Simulate matrix with 4 features + 1 trial id column
	x_dummy = np.arange(10 * 5).reshape(10, 5)
	filtered_features = pipeline._apply_sensor_ablation(all_features, ["Pupil", "Centrifuge"])
	col_indices = [all_features.index(name) for name in filtered_features]
	x_ablated = np.hstack([x_dummy[:, col_indices], x_dummy[:, -1:]])

	assert filtered_features == [
		"Pupil diameter left [mm] - Tobii_v0_mean",
		"magnitude - Centrifuge_v0_mean",
	]
	assert x_ablated.shape == (10, 3)


def test_traditional_sensor_ablation_consistent_row_counts():
	"""Verify that get_data produces identical train and test row counts across all sensor streams."""
	from src.config_loader import load_experiment_config
	from src.Data_Pipeline.data_pipeline import DataPipeline
	from src.models.model_factory import ModelFactory
	from src.model_type import ModelType

	cfg = load_experiment_config("configs/test.yaml")
	pipeline = DataPipeline(cfg)
	pipeline.set_model_type(ModelType("Complete", "Explicit"))
	pipeline.set_random_seed(42)

	model = ModelFactory.create_model("KNN")

	# Full data (no ablation)
	X_tr_full, X_te_full, y_tr_full, y_te_full = pipeline.get_data(
		model=model, kfold_id=0, num_splits=2, traditional_feature_selection="raw"
	)
	expected_tr_rows = X_tr_full.shape[0]
	expected_te_rows = X_te_full.shape[0]

	# Test individual stream groups
	for streams in [["EEG"], ["Pupil"], ["Centrifuge"], ["ECG"]]:
		X_tr, X_te, y_tr, y_te = pipeline.get_data(
			model=model,
			kfold_id=0,
			num_splits=2,
			feature_streams=streams,
			traditional_feature_selection="raw",
		)
		assert X_tr.shape[0] == expected_tr_rows, (
			f"Stream {streams} train rows {X_tr.shape[0]} != expected {expected_tr_rows}"
		)
		assert X_te.shape[0] == expected_te_rows, (
			f"Stream {streams} test rows {X_te.shape[0]} != expected {expected_te_rows}"
		)


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
	"mean",
	"stddev",
	"max",
	"range",
	"derivative_mean",
	"derivative_stddev",
	"derivative_max",
	"derivative_range",
	"2derivative_mean",
	"2derivative_stddev",
	"2derivative_max",
	"2derivative_range",
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
	"ECG",
	"HR",
	"BR",
	"Temperature",
	"Pupil",
	"Centrifuge",
	"EEG",
	"Strain",
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


@pytest.mark.parametrize("stream", _SINGLE_STREAMS, ids=lambda s: f"stream={s}")
def test_each_sensor_stream_matches_legacy_substring(stream):
	"""Per-stream parity: _apply_sensor_ablation must select exactly the same
	column set as the legacy restrict_feature_space substring matcher.
	"""
	from src.Data_Pipeline.data_pipeline import BaseGLOCDataPipeline, TraditionalDataPipeline

	pipeline = TraditionalDataPipeline(data_path="/tmp/nonexistent_data_path", random_seed=42)
	mt = ModelType("Complete", "Explicit")
	default_groups = BaseGLOCDataPipeline.FEATURE_GROUPS_BY_MODEL_TYPE[mt]
	universe = _build_full_universe(default_groups, mt)

	old = _legacy_restrict_feature_space([stream], universe)
	new = set(pipeline._apply_sensor_ablation(list(universe), [stream]))

	assert new == old, (
		f"stream={stream!r} parity mismatch: "
		f"OLD={len(old)}, NEW={len(new)}, "
		f"OLD-only[:3]={sorted(old - new)[:3]}, "
		f"NEW-only[:3]={sorted(new - old)[:3]}"
	)


@pytest.mark.parametrize("streams", _MULTI_STREAMS, ids=lambda s: "+".join(s))
def test_multi_stream_combos_match_legacy_substring_union(streams):
	"""Multi-stream parity: _apply_sensor_ablation reproduces the legacy
	restrict_feature_space union semantics (any column whose name contains
	ANY requested stream keyword).
	"""
	from src.Data_Pipeline.data_pipeline import BaseGLOCDataPipeline, TraditionalDataPipeline

	pipeline = TraditionalDataPipeline(data_path="/tmp/nonexistent_data_path", random_seed=42)
	mt = ModelType("Complete", "Explicit")
	default_groups = BaseGLOCDataPipeline.FEATURE_GROUPS_BY_MODEL_TYPE[mt]
	universe = _build_full_universe(default_groups, mt)

	old = _legacy_restrict_feature_space(list(streams), universe)
	new = set(pipeline._apply_sensor_ablation(list(universe), list(streams)))

	assert new == old, (
		f"streams={list(streams)} parity mismatch: "
		f"OLD={len(old)}, NEW={len(new)}, "
		f"OLD-only[:3]={sorted(old - new)[:3]}, "
		f"NEW-only[:3]={sorted(new - old)[:3]}"
	)


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
		X = np.array(
			[
				[1.0],  # t1 train
				[3.0],  # t1 train
				[5.0],  # t1 test   -> should be z-scored using t1 train mean=2, std=1
				[100.0],  # t2 train
				[200.0],  # t2 train  -> μ_t2=150, std=50
			]
		)
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
		X = np.array(
			[
				[1.0],  # t1 train
				[3.0],  # t1 train  (μ_t1=2, σ_t1=1, μ_pooled=2, σ_pooled=1)
				[
					10.0
				],  # t2 test   -> only this row exists for t2, so it has NO train rows of its own
				[100.0],  # t3 train  (μ_t3=100, σ_t3=0, falls back to σ=0 guard)
			]
		)
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
	pipeline = DataPipeline(_make_config())
	pipeline.set_model_type(ModelType("Complete", "Explicit"))

	with pytest.raises(ValueError, match="kfold_id and num_splits"):
		pipeline.get_data(model=DummyModel(is_traditional=True, name="KNN"))

	with pytest.raises(ValueError, match="kfold_id and num_splits"):
		pipeline.get_data(model=DummyModel(is_traditional=True, name="KNN"), kfold_id=0)


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


def test_traditional_get_data_forwards_standardize_s1():
	"""The facade forwards standardize_s1 from traditional_data_parameters to the backend."""
	cfg = _make_config()
	cfg["traditional_data_parameters"]["standardize_s1"] = False
	pipeline = DataPipeline(cfg)
	pipeline.set_model_type(ModelType("Complete", "Explicit"))

	received_kwargs = None

	class _RecordingBackend(CapturingBackend):
		def get_data(self, **kwargs):
			nonlocal received_kwargs
			received_kwargs = kwargs
			return super().get_data(**kwargs)

	pipeline._build_backend = lambda _model: _RecordingBackend("ok")
	pipeline.get_data(
		model=DummyModel(is_traditional=True, name="KNN"),
		kfold_id=0,
		num_splits=2,
		traditional_feature_selection="raw",
	)
	assert received_kwargs is not None
	assert received_kwargs["standardize_s1"] is False


def test_traditional_get_data_defaults_standardize_s1_to_true():
	"""When standardize_s1 is omitted, the facade defaults it to True."""
	cfg = _make_config()
	del cfg["traditional_data_parameters"]["standardize_s1"]
	pipeline = DataPipeline(cfg)
	pipeline.set_model_type(ModelType("Complete", "Explicit"))

	received_kwargs = None

	class _RecordingBackend(CapturingBackend):
		def get_data(self, **kwargs):
			nonlocal received_kwargs
			received_kwargs = kwargs
			return super().get_data(**kwargs)

	pipeline._build_backend = lambda _model: _RecordingBackend("ok")
	pipeline.get_data(
		model=DummyModel(is_traditional=True, name="KNN"),
		kfold_id=0,
		num_splits=2,
		traditional_feature_selection="raw",
	)
	assert received_kwargs is not None
	assert received_kwargs["standardize_s1"] is True


class TestStandardizeRawBehavior:
	"""Unit tests for TraditionalDataPipeline._standardize_raw with standardize_s1 flag."""

	def test_standardize_raw_s1_enabled_produces_s1_and_s2(self):
		from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline

		pipeline = TraditionalDataPipeline(data_path="/tmp/data", config=_make_config())
		X_raw = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
		trial_id = np.array(["t1", "t1", "t2", "t2"])
		train_mask = np.array([True, True, True, False])

		out = pipeline._standardize_raw(X_raw, trial_id, train_mask, standardize_s1=True)
		assert out.shape == (4, 4)
		assert pipeline._last_trial_standardizer is not None
		assert pipeline._last_global_standardizer is not None

		s1_expected = (
			TrialAwareStandardizer().fit(X_raw, trial_id, train_mask).transform(X_raw, trial_id)
		)
		s2_expected = GlobalStandardizer().fit(X_raw[train_mask]).transform(X_raw)
		np.testing.assert_allclose(out[:, :2], s1_expected)
		np.testing.assert_allclose(out[:, 2:], s2_expected)

	def test_standardize_raw_s1_disabled_produces_only_s2(self):
		from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline

		pipeline = TraditionalDataPipeline(data_path="/tmp/data", config=_make_config())
		X_raw = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
		trial_id = np.array(["t1", "t1", "t2", "t2"])
		train_mask = np.array([True, True, True, False])

		out = pipeline._standardize_raw(X_raw, trial_id, train_mask, standardize_s1=False)
		assert out.shape == (4, 2)
		assert pipeline._last_trial_standardizer is None
		assert pipeline._last_global_standardizer is not None

		s2_expected = GlobalStandardizer().fit(X_raw[train_mask]).transform(X_raw)
		np.testing.assert_allclose(out, s2_expected)

	def test_feature_generation_returns_only_s2_when_standardize_s1_false(self):
		from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline

		pipeline = TraditionalDataPipeline(data_path="/tmp/data", config=_make_config())

		time_start = 0.0
		offset = 0.0
		stride = 1.0
		window_size = 2.0
		n_samples = 100
		combined_baseline = {"t1": np.ones((n_samples, 2))}
		gloc = np.zeros(n_samples)
		trial_column = np.array(["t1"] * n_samples)
		time_column = np.arange(0, n_samples * 0.1, 0.1)
		combined_baseline_names = ["featA", "featB"]

		# 8 windows produced for t1
		train_mask = np.ones(8, dtype=bool)

		y_labels, X_feat, all_feats, trial_ids = pipeline._feature_generation(
			time_start=time_start,
			offset=offset,
			stride=stride,
			window_size=window_size,
			combined_baseline=combined_baseline,
			gloc=gloc,
			trial_column=trial_column,
			time_column=time_column,
			combined_baseline_names=combined_baseline_names,
			baseline_names_v0=combined_baseline_names,
			baseline_v0=combined_baseline,
			feature_groups_to_analyze=[],
			train_mask=train_mask,
			standardize_s1=False,
		)

		assert len(all_feats) == X_feat.shape[1]
		assert len(all_feats) > 0
		assert all(f.endswith("_s2") for f in all_feats)
		assert not any(f.endswith("_s1") for f in all_feats)

	def test_feature_generation_returns_s1_and_s2_when_standardize_s1_true(self):
		from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline

		pipeline = TraditionalDataPipeline(data_path="/tmp/data", config=_make_config())

		time_start = 0.0
		offset = 0.0
		stride = 1.0
		window_size = 2.0
		n_samples = 100
		combined_baseline = {"t1": np.ones((n_samples, 2))}
		gloc = np.zeros(n_samples)
		trial_column = np.array(["t1"] * n_samples)
		time_column = np.arange(0, n_samples * 0.1, 0.1)
		combined_baseline_names = ["featA", "featB"]

		train_mask = np.ones(8, dtype=bool)

		y_labels, X_feat, all_feats, trial_ids = pipeline._feature_generation(
			time_start=time_start,
			offset=offset,
			stride=stride,
			window_size=window_size,
			combined_baseline=combined_baseline,
			gloc=gloc,
			trial_column=trial_column,
			time_column=time_column,
			combined_baseline_names=combined_baseline_names,
			baseline_names_v0=combined_baseline_names,
			baseline_v0=combined_baseline,
			feature_groups_to_analyze=[],
			train_mask=train_mask,
			standardize_s1=True,
		)

		assert len(all_feats) == X_feat.shape[1]
		assert any(f.endswith("_s1") for f in all_feats)
		assert any(f.endswith("_s2") for f in all_feats)


def test_traditional_data_pipeline_does_not_have_faster_knn_impute():
	from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline

	assert not hasattr(TraditionalDataPipeline, "_faster_knn_impute")
	assert not hasattr(TraditionalDataPipeline, "_resolve_traditional_impute_path")


def test_traditional_preprocessing_artifacts_contain_no_knn_imputer(tmp_path, monkeypatch):
	import json
	from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline
	from src.models.random_forest import RandomForestModel

	pipeline = TraditionalDataPipeline(data_path="/tmp/data", random_seed=42)
	monkeypatch.setattr(
		pipeline,
		"_gen_windowed_label_metadata",
		lambda *a, **kw: (np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])),
	)
	monkeypatch.setattr(
		pipeline, "_load_data", lambda *a, **kw: (None, None, None, None, None, None, ["f1", "f2"])
	)
	monkeypatch.setattr(pipeline, "_filter_data_by_analysis_type", lambda a, d, s, t: d)
	monkeypatch.setattr(
		pipeline,
		"_process_and_get_feature_names",
		lambda *a, **kw: (None, {"All": ["f1_s2", "f2_s2"]}),
	)
	monkeypatch.setattr(pipeline, "_label_gloc_events", lambda *a, **kw: np.array([0, 0, 1, 1]))
	monkeypatch.setattr(pipeline, "_afe_subset", lambda d, l: (d, l))
	monkeypatch.setattr(pipeline, "_remove_all_nan_trials", lambda *a, **kw: (None, np.array([0, 0, 1, 1]), []))
	monkeypatch.setattr(
		pipeline,
		"_reduce_memory",
		lambda *a, **kw: (
			None,
			np.array([0, 0, 1, 1]),
			{"trial_id": np.array(["t1", "t1", "t2", "t2"]), "Time (s)": np.array([0, 1, 2, 3])},
		),
	)
	monkeypatch.setattr(
		pipeline,
		"_get_combined_baseline_data",
		lambda *a, **kw: ({"t1": np.zeros((1, 2))}, ["f1", "f2"], {"t1": np.zeros((1, 2))}, ["f1", "f2"]),
	)
	monkeypatch.setattr(
		pipeline,
		"_feature_generation",
		lambda *a, **kw: (
			np.array([0, 0, 1, 1]),
			np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
			["f1_s2", "f2_s2"],
			np.array([0, 0, 1, 1]),
		),
	)
	monkeypatch.setattr(
		pipeline,
		"_process_NaN_temporal",
		lambda labels, data, feats: (labels, data, feats, np.array([], dtype=int)),
	)
	monkeypatch.setattr(pipeline, "_ready_outputs", lambda data, labels: (data, labels))

	artifact_path = tmp_path / "preprocessing_artifacts.json"
	pipeline.get_data(
		backstep=0,
		data_rate=250,
		remove_NaN_trials=True,
		offset=0.0,
		time_start=0.0,
		subject_to_analyze=None,
		trial_to_analyze=None,
		analysis_type=2,
		classifier_type="RF",
		model=RandomForestModel(),
		model_type=ModelType("noAFE", "Explicit"),
		kfold_id=0,
		num_splits=2,
		traditional_feature_selection="raw",
		save_preprocessing_artifacts_path=str(artifact_path),
	)

	with open(artifact_path) as f:
		artifacts = json.load(f)

	assert "knn_imputer" not in artifacts
	assert "s2_global_mean" in artifacts
	assert "active_feature_names" in artifacts

def test_get_data_forwards_traditional_horizons(monkeypatch):
    pipeline = DataPipeline(_make_config())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))
    backend = CapturingBackend("temporal-ok")
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)

    pipeline.get_data(model=DummyModel(is_traditional=True, name="RF"), kfold_id=0, num_splits=5,
                      traditional_feature_selection="raw", horizons={0: 0, 2: 50})

    assert backend.calls[0]["horizons"] == {0: 0, 2: 50}


def test_get_data_forwards_advanced_horizons(monkeypatch):
    pipeline = DataPipeline(_make_config())
    pipeline.set_model_type(ModelType("Complete", "Explicit"))
    backend = CapturingBackend("advanced-ok")
    monkeypatch.setattr(pipeline, "_build_backend", lambda _model: backend)

    pipeline.get_data(model=DummyModel(is_traditional=False, name="LSTM"), kfold_id=0, num_splits=5,
                      horizons={0: 0, 1.5: 37})

    assert backend.calls[0]["horizons"] == {0: 0, 1.5: 37}


def test_window_labels_only_matches_gen_windowed_label_metadata():
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline
    pipeline = TraditionalDataPipeline(data_path="/tmp/data", config=_make_config())
    data_rate, seconds = 25, 60
    n = data_rate * seconds
    trial_ids = ["01-01", "01-02"]
    trial_column = np.repeat(trial_ids, n)
    time_column = np.tile(np.arange(n) / data_rate, len(trial_ids))
    rng = np.random.default_rng(0)
    combined_baseline = {trial_id: rng.normal(size=(n, 3)) for trial_id in trial_ids}
    gloc = np.zeros(n * len(trial_ids))
    gloc[40 * data_rate:45 * data_rate] = 1
    gloc[n + 30 * data_rate:n + 33 * data_rate] = 1
    shifted = pipeline._shift_labels_by_samples(gloc, trial_column, 2 * data_rate)

    for labels in (gloc, shifted):
        expected, _ = pipeline._gen_windowed_label_metadata(
            0.0, 0.0, 0.25, 12.5, combined_baseline, labels, trial_column, time_column, ["a", "b", "c"]
        )
        actual = pipeline._window_labels_only(0.0, 0.0, 0.25, 12.5, labels, trial_column, time_column)
        np.testing.assert_array_equal(actual, expected)

    assert not np.array_equal(
        pipeline._window_labels_only(0.0, 0.0, 0.25, 12.5, gloc, trial_column, time_column),
        pipeline._window_labels_only(0.0, 0.0, 0.25, 12.5, shifted, trial_column, time_column),
    )

