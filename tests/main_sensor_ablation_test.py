import numpy as np
import pickle
from pathlib import Path
import builtins

from src.model_type import ModelType
import src.main as main_module

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from upsetplot import UpSet, from_contents
from src.scripts.temporal_functions_traditional import get_hyperparameters_from_json

class DummyModel:
    def __init__(self, name: str):
        self._name = name
        self.calls = []

    def get_name(self):
        return self._name

    def classify_traditional(
        self,
        x_train,
        x_test,
        y_train,
        y_test,
        class_weight_imb,
        random_state,
        save_folder,
        model_name,
        retrain,
        temporal=False,
        best_params=None,
    ):
        self.calls.append(
            {
                "x_train": x_train,
                "x_test": x_test,
                "y_train": y_train,
                "y_test": y_test,
                "class_weight_imb": class_weight_imb,
                "random_state": random_state,
                "save_folder": save_folder,
                "model_name": model_name,
                "retrain": retrain,
                "temporal": temporal,
                "best_params": best_params,
            }
        )
        return (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


class DummyConfigParser:
    def __init__(
        self,
        enabled: bool,
        model,
        review_enabled: bool = False,
        review_models=None,
        review_stream_group=None,
        review_sort_streams_by_median: bool = False,
        feature_space_review_enabled: bool = False,
        feature_space_review_models=None,
    ):
        self._enabled = enabled
        self._model = model
        self._review_enabled = review_enabled
        self._review_models = [] if review_models is None else review_models
        self._review_stream_group = [] if review_stream_group is None else review_stream_group
        self._review_sort_streams_by_median = review_sort_streams_by_median
        self._feature_space_review_enabled = feature_space_review_enabled
        self._feature_space_review_models = ["KNN", "EGB", "RF"] if feature_space_review_models is None else feature_space_review_models

    def get_sensor_ablation_enabled(self):
        return self._enabled

    def get_sensor_ablation_streams(self):
        return [["EEG"]] if self._enabled else [["EEG"]]

    def get_sensor_ablation_review_enabled(self):
        return self._review_enabled

    def get_sensor_ablation_review_models(self):
        return list(self._review_models)

    def get_sensor_ablation_review_stream_group(self):
        return list(self._review_stream_group)

    def get_sensor_ablation_review_sort_streams_by_median(self):
        return self._review_sort_streams_by_median

    def get_feature_space_review_enabled(self):
        return self._feature_space_review_enabled

    def get_feature_space_review_models(self):
        return list(self._feature_space_review_models)

    def get_models(self):
        return [self._model]

    def get_num_splits(self):
        return 2

    def get_model_type(self):
        return ModelType("Complete", "Explicit")

    def get_random_seed(self):
        return 42

    def get_data_path(self):
        return "/tmp/data"

    def get_remove_NaN_trials(self):
        return True

    def get_subject_to_analyze(self):
        return None

    def get_trial_to_analyze(self):
        return None

    def get_analysis_type(self):
        return 2

    def get_output_feature_dtype(self):
        return np.dtype("float32")

    def get_impute_file_name(self):
        return "imputed.pkl"

    def get_should_impute(self):
        return True

    def get_save_impute(self):
        return False

    def get_load_impute(self):
        return False

    def get_n_neighbors(self):
        return 4

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

    def get_cross_validation_enabled(self):
        return False

    def get_cross_validation_num_splits(self):
        return 5

    def get_cross_validation_random_seed(self):
        return 42

    def get_cross_validation_class_weight(self):
        return "balanced"

    def get_cross_validation_save_results_folder(self):
        return "CrossValidation"

    def get_cross_validation_support_deep_learning(self):
        return True

    def get_cross_validation_save_median_hyperparameters(self):
        return True

    def get_models_for_cross_validation(self):
        return [self._model]


class DummyPipeline:
    def __init__(self, config_parser):
        self.config_parser = config_parser
        self.calls = []

    def get_data(self, **kwargs):
        self.calls.append(kwargs)
        return (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([0, 1]),
        )


def test_run_uses_traditional_classification_when_sensor_ablation_enabled(monkeypatch):
    model = DummyModel("KNN")
    config_parser = DummyConfigParser(enabled=True, model=model)
    pipeline = DummyPipeline(config_parser)
    captured = {}

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "GLOCExperimentConfigParser", lambda config_location=None: config_parser)
    monkeypatch.setattr(main_module, "_try_enable_cuml_acceleration", lambda: None)
    monkeypatch.setattr(main_module, "DataPipeline", lambda config_parser: pipeline)
    monkeypatch.setattr(
        main_module,
        "run_sensor_ablation_training",
        lambda **kwargs: captured.update(kwargs=kwargs),
    )

    main_module.run()

    assert "kwargs" in captured
    assert captured["kwargs"]["config_parser"] is config_parser
    assert captured["kwargs"]["pipeline"] is pipeline
    assert captured["kwargs"]["project_root"] == Path(main_module.__file__).resolve().parent.parent
    assert captured["kwargs"]["get_hyperparameters_from_json_fn"] is main_module.get_hyperparameters_from_json
    assert captured["kwargs"]["stratified_kfold_split_fn"] is main_module.stratified_kfold_split
    assert captured["kwargs"]["plot_f1_violin_by_stream_fn"] is main_module.plot_f1_violin_by_stream
    assert captured["kwargs"]["extract_f1_score_fn"] is main_module.extract_f1_score
    assert captured["kwargs"]["save_model_stream_f1_scores_fn"] is main_module.save_model_stream_f1_scores


def test_run_sensor_ablation_training_passes_expected_parameters(tmp_path):
    model = DummyModel("KNN")
    config_parser = DummyConfigParser(enabled=True, model=model)
    pipeline = DummyPipeline(config_parser)

    hyperparameter_calls = []
    split_calls = []
    save_calls = []
    plot_calls = []

    def _get_hyperparameters_from_json_fn(classifier, model_type_name):
        hyperparameter_calls.append((classifier, model_type_name))
        return ({"alpha": 1.0}, ["EEG"], None, None)

    def _stratified_kfold_split_fn(**kwargs):
        split_calls.append(kwargs)
        return (
            np.array([[10.0]]),
            np.array([[20.0]]),
            np.array([0]),
            np.array([1]),
        )

    def _save_model_stream_f1_scores_fn(**kwargs):
        save_calls.append(kwargs)
        return kwargs["results_root_dir"] / kwargs["model_name"] / f"{kwargs['stream_str']}.pkl"

    def _plot_f1_violin_by_stream_fn(**kwargs):
        plot_calls.append(kwargs)

    project_root = tmp_path / "project"

    main_module.run_sensor_ablation_training(
        config_parser=config_parser,
        pipeline=pipeline,
        project_root=project_root,
        get_hyperparameters_from_json_fn=_get_hyperparameters_from_json_fn,
        stratified_kfold_split_fn=_stratified_kfold_split_fn,
        plot_f1_violin_by_stream_fn=_plot_f1_violin_by_stream_fn,
        save_model_stream_f1_scores_fn=_save_model_stream_f1_scores_fn,
    )

    expected_results_dir = project_root / "Results" / "Sensor_Ablation" / config_parser.get_model_type().get_folder_name()

    assert hyperparameter_calls == [("KNN", config_parser.get_model_type().get_folder_name())]
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0] == {"model": model, "feature_streams": ["EEG"]}
    assert len(split_calls) == config_parser.get_num_splits()
    assert [call["kfold_ID"] for call in split_calls] == [0, 1]
    assert all(call["num_splits"] == config_parser.get_num_splits() for call in split_calls)
    assert all(call["random_state"] == config_parser.get_random_seed() for call in split_calls)
    assert len(model.calls) == config_parser.get_num_splits()
    assert model.calls[0]["class_weight_imb"] is None
    assert model.calls[0]["random_state"] == config_parser.get_random_seed()
    assert model.calls[0]["save_folder"] == ""
    assert model.calls[0]["model_name"] == "knn_feature_study.pkl"
    assert model.calls[0]["retrain"] is False
    assert model.calls[0]["temporal"] is True
    assert model.calls[0]["best_params"] == {"alpha": 1.0}
    assert len(save_calls) == 1
    assert save_calls[0]["results_root_dir"] == expected_results_dir
    assert save_calls[0]["model_name"] == "KNN"
    assert save_calls[0]["stream_str"] == "EEG"
    assert np.allclose(save_calls[0]["f1_scores"], np.array([0.6, 0.6]))
    assert len(plot_calls) == 1
    assert plot_calls[0]["model_type"] == config_parser.get_model_type()
    assert plot_calls[0]["save_folder"] == expected_results_dir
    assert set(plot_calls[0]["f1_results_by_stream"].keys()) == {"KNN"}
    assert set(plot_calls[0]["f1_results_by_stream"]["KNN"].keys()) == {"EEG"}
    assert np.allclose(plot_calls[0]["f1_results_by_stream"]["KNN"]["EEG"], np.array([0.6, 0.6]))


def test_run_no_enabled_modes_logs_and_exits(monkeypatch):
    model = DummyModel("RF")
    config_parser = DummyConfigParser(enabled=False, model=model, review_enabled=False)

    logs = []

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "GLOCExperimentConfigParser", lambda config_location=None: config_parser)
    monkeypatch.setattr(main_module, "_try_enable_cuml_acceleration", lambda: None)
    monkeypatch.setattr(main_module.logging, "info", lambda msg, *args: logs.append(msg % args if args else msg))

    main_module.run()

    assert any("No runnable mode enabled" in message for message in logs)


def test_load_sensor_ablation_f1_results_reads_pickles(tmp_path):
    results_root_dir = tmp_path / "Sensor_Ablation" / "Complete_Explicit"
    knn_dir = results_root_dir / "KNN"
    rf_dir = results_root_dir / "RF"
    knn_dir.mkdir(parents=True)
    rf_dir.mkdir(parents=True)

    with open(knn_dir / "EEG.pkl", "wb") as handle:
        pickle.dump(np.array([0.1, 0.2]), handle)
    with open(rf_dir / "ECG-HR-BR.pkl", "wb") as handle:
        pickle.dump(np.array([0.9]), handle)

    loaded = main_module.load_sensor_ablation_f1_results(
        results_root_dir=results_root_dir,
        classifiers=["KNN", "RF"],
    )

    assert set(loaded.keys()) == {"KNN", "RF"}
    assert np.allclose(loaded["KNN"]["EEG"], np.array([0.1, 0.2]))
    assert np.allclose(loaded["RF"]["ECG-HR-BR"], np.array([0.9]))


def test_load_sensor_ablation_f1_results_skips_non_matching_pickles(monkeypatch, tmp_path):
    results_root_dir = tmp_path / "Sensor_Ablation" / "Complete_Explicit"
    knn_dir = results_root_dir / "KNN"
    knn_dir.mkdir(parents=True)

    with open(knn_dir / "EEG.pkl", "wb") as handle:
        pickle.dump(np.array([0.1, 0.2]), handle)
    with open(knn_dir / "Pupil.pkl", "wb") as handle:
        pickle.dump(np.array([0.3]), handle)

    opened_paths = []
    original_open = open

    def _tracking_open(path, mode="r", *args, **kwargs):
        if "rb" in mode:
            opened_paths.append(Path(path).name)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _tracking_open)

    loaded = main_module.load_sensor_ablation_f1_results(
        results_root_dir=results_root_dir,
        classifiers=["KNN"],
        stream_group=["EEG"],
    )

    assert set(opened_paths) == {"EEG.pkl"}
    assert set(loaded["KNN"].keys()) == {"EEG"}


def test_filter_sensor_ablation_review_results_matches_selected_stream_labels():
    f1_results_by_stream = {
        "KNN": {
            "EEG": np.array([0.5]),
            "ECG-HR-BR": np.array([0.6]),
            "EEG-Pupil": np.array([0.7]),
            "Pupil-Centrifuge-Participant": np.array([0.8]),
        }
    }
    filtered = main_module.filter_sensor_ablation_review_results(
        f1_results_by_stream=f1_results_by_stream,
        stream_group=["Pupil", "Centrifuge", "Participant"],
    )

    assert set(filtered["KNN"].keys()) == {"Pupil-Centrifuge-Participant"}


def test_run_sensor_ablation_review_enabled_invokes_plot_with_filtered_results(monkeypatch):
    model = DummyModel("KNN")
    config_parser = DummyConfigParser(
        enabled=False,
        model=model,
        review_enabled=True,
        review_models=["KNN"],
        review_stream_group=["EEG"],
    )

    captured = {}

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "GLOCExperimentConfigParser", lambda config_location=None: config_parser)
    monkeypatch.setattr(main_module, "_try_enable_cuml_acceleration", lambda: None)
    monkeypatch.setattr(
        main_module,
        "run_sensor_ablation_review",
        lambda **kwargs: captured.update(kwargs=kwargs),
    )

    main_module.run()

    assert captured["kwargs"]["config_parser"] is config_parser
    assert captured["kwargs"]["load_sensor_ablation_f1_results_fn"] is main_module.load_sensor_ablation_f1_results
    assert captured["kwargs"]["build_ranked_sensor_ablation_review_results_fn"] is main_module.build_ranked_sensor_ablation_review_results
    assert captured["kwargs"]["filter_sensor_ablation_review_results_fn"] is main_module.filter_sensor_ablation_review_results
    assert captured["kwargs"]["plot_f1_violin_with_stream_matrix_fn"] is main_module.plot_f1_violin_with_stream_matrix
    assert captured["kwargs"]["plot_f1_violin_by_stream_fn"] is main_module.plot_f1_violin_by_stream
    assert captured["kwargs"]["stream_label_aliases"] == main_module.STREAM_LABEL_ALIASES


def test_run_sensor_ablation_review_passes_expected_parameters(tmp_path):
    model = DummyModel("KNN")
    config_parser = DummyConfigParser(
        enabled=False,
        model=model,
        review_enabled=True,
        review_models=["KNN"],
        review_stream_group=["EEG"],
    )

    load_calls = []
    filter_calls = []
    plot_calls = []

    def _load_sensor_ablation_f1_results_fn(**kwargs):
        load_calls.append(kwargs)
        return {"KNN": {"EEG": np.array([0.2, 0.4])}}

    def _build_ranked_sensor_ablation_review_results_fn(**kwargs):
        raise AssertionError("Ranked review should not be used when sort_streams_by_median is disabled")

    def _filter_sensor_ablation_review_results_fn(**kwargs):
        filter_calls.append(kwargs)
        return {"KNN": {"EEG": np.array([0.2, 0.4])}}

    def _plot_f1_violin_by_stream_fn(**kwargs):
        plot_calls.append(("by_stream", kwargs))

    def _plot_f1_violin_with_stream_matrix_fn(**kwargs):
        plot_calls.append(("matrix", kwargs))

    main_module.run_sensor_ablation_review(
        config_parser=config_parser,
        project_root=tmp_path,
        load_sensor_ablation_f1_results_fn=_load_sensor_ablation_f1_results_fn,
        build_ranked_sensor_ablation_review_results_fn=_build_ranked_sensor_ablation_review_results_fn,
        filter_sensor_ablation_review_results_fn=_filter_sensor_ablation_review_results_fn,
        plot_f1_violin_with_stream_matrix_fn=_plot_f1_violin_with_stream_matrix_fn,
        plot_f1_violin_by_stream_fn=_plot_f1_violin_by_stream_fn,
        stream_label_aliases=main_module.STREAM_LABEL_ALIASES,
    )

    expected_results_dir = tmp_path / "Results" / "Sensor_Ablation" / config_parser.get_model_type().get_folder_name()

    assert load_calls == [
        {
            "results_root_dir": expected_results_dir,
            "classifiers": ["KNN"],
            "stream_group": ["EEG"],
        }
    ]
    assert len(filter_calls) == 1
    assert filter_calls[0]["stream_group"] == ["EEG"]
    assert set(filter_calls[0]["f1_results_by_stream"].keys()) == {"KNN"}
    assert set(filter_calls[0]["f1_results_by_stream"]["KNN"].keys()) == {"EEG"}
    assert np.allclose(filter_calls[0]["f1_results_by_stream"]["KNN"]["EEG"], np.array([0.2, 0.4]))
    assert len(plot_calls) == 1
    assert plot_calls[0][0] == "by_stream"
    assert plot_calls[0][1]["model_type"] == config_parser.get_model_type()
    assert set(plot_calls[0][1]["f1_results_by_stream"].keys()) == {"KNN"}
    assert np.allclose(plot_calls[0][1]["f1_results_by_stream"]["KNN"]["EEG"], np.array([0.2, 0.4]))


def test_build_ranked_sensor_ablation_review_results_sorts_and_aliases_streams():
    f1_results_by_stream = {
        "KNN": {
            "EEG": np.array([0.8, 0.9]),
            "ECG-HR-BR-Temperature": np.array([0.95]),
            "EEG-Centrifuge-Participant": np.array([0.7]),
        },
        "RF": {
            "EEG": np.array([0.6]),
            "EEG-Centrifuge-Participant": np.array([0.85]),
        },
    }
    ranked = main_module.build_ranked_sensor_ablation_review_results(
        f1_results_by_stream = f1_results_by_stream,
        classifiers = ["KNN", "RF"],
        stream_label_aliases = main_module.STREAM_LABEL_ALIASES,
    )

    assert list(ranked["KNN"].keys()) == [
        "Equivital",
        "EEG",
        "EEG-G Force-Demographics",
    ]
    assert list(ranked["RF"].keys()) == [
        "EEG",
        "EEG-G Force-Demographics",
    ]


def test_run_sensor_ablation_review_ranked_mode_invokes_matrix_plot(monkeypatch):
    model = DummyModel("KNN")
    config_parser = DummyConfigParser(
        enabled = False,
        model = model,
        review_enabled = True,
        review_models = ["KNN", "RF"],
        review_stream_group = [],
        review_sort_streams_by_median = True,
    )

    captured = {}

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "GLOCExperimentConfigParser", lambda config_location=None: config_parser)
    monkeypatch.setattr(main_module, "_try_enable_cuml_acceleration", lambda: None)
    monkeypatch.setattr(
        main_module,
        "run_sensor_ablation_review",
        lambda **kwargs: captured.update(kwargs=kwargs),
    )

    main_module.run()

    assert captured["kwargs"]["config_parser"] is config_parser
    assert captured["kwargs"]["load_sensor_ablation_f1_results_fn"] is main_module.load_sensor_ablation_f1_results
    assert captured["kwargs"]["build_ranked_sensor_ablation_review_results_fn"] is main_module.build_ranked_sensor_ablation_review_results
    assert captured["kwargs"]["filter_sensor_ablation_review_results_fn"] is main_module.filter_sensor_ablation_review_results
    assert captured["kwargs"]["plot_f1_violin_with_stream_matrix_fn"] is main_module.plot_f1_violin_with_stream_matrix
    assert captured["kwargs"]["plot_f1_violin_by_stream_fn"] is main_module.plot_f1_violin_by_stream
    assert captured["kwargs"]["stream_label_aliases"] == main_module.STREAM_LABEL_ALIASES


def test_investigate_feature_space_uses_venn3_and_returns_expected_sets(monkeypatch, capsys):
    model_type = ModelType("Complete", "Explicit")

    feature_map = {
        "KNN": ["A", "B", "C"],
        "EGB": ["B", "C", "D"],
        "RF": ["C", "D", "E"],
    }

    monkeypatch.setattr(
        main_module,
        "get_hyperparameters_from_json",
        lambda classifier, model_type_name: ({}, feature_map[classifier], None, None),
    )

    venn3_calls = {}

    def _capture_venn3(sets, set_labels=None):
        venn3_calls["sets"] = sets
        venn3_calls["set_labels"] = set_labels

    monkeypatch.setattr(main_module, "venn3", _capture_venn3)
    monkeypatch.setattr(main_module.plt, "show", lambda: None)

    shared_features, unique_features = main_module.investigate_feature_space(
        model_type = model_type,
        classifiers = ["KNN", "EGB", "RF"],
        get_hyperparameters_from_json_fn = main_module.get_hyperparameters_from_json,
        venn2_fn = main_module.venn2,
        venn3_fn = main_module.venn3,
        from_contents_fn = main_module.from_contents,
        upset_cls = main_module.UpSet,
        plt_module = main_module.plt,
    )

    assert shared_features == {"C"}
    assert unique_features == {
        "KNN": {"A", "B"},
        "EGB": {"B", "D"},
        "RF": {"D", "E"},
    }
    assert venn3_calls["set_labels"] == ("KNN", "EGB", "RF")
    assert venn3_calls["sets"][0] == {"A", "B", "C"}
    assert venn3_calls["sets"][1] == {"B", "C", "D"}
    assert venn3_calls["sets"][2] == {"C", "D", "E"}

    captured = capsys.readouterr().out
    assert "Shared features across all (1):" in captured
    assert "Features unique to KNN (2):" in captured


def test_investigate_feature_space_uses_upset_for_four_or_more_classifiers(monkeypatch):
    model_type = ModelType("Complete", "Explicit")

    feature_map = {
        "KNN": ["A", "B"],
        "EGB": ["B", "C"],
        "RF": ["C", "D"],
        "LDA": ["D", "E"],
    }

    monkeypatch.setattr(
        main_module,
        "get_hyperparameters_from_json",
        lambda classifier, model_type_name: ({}, feature_map[classifier], None, None),
    )

    upset_calls = {}

    def _capture_from_contents(contents):
        upset_calls["contents"] = contents
        return contents

    class _FakeUpSet:
        def __init__(self, data):
            upset_calls["data"] = data

        def plot(self):
            upset_calls["plotted"] = True

    monkeypatch.setattr(main_module, "from_contents", _capture_from_contents)
    monkeypatch.setattr(main_module, "UpSet", _FakeUpSet)
    monkeypatch.setattr(main_module.plt, "show", lambda: None)

    shared_features, unique_features = main_module.investigate_feature_space(
        model_type = model_type,
        classifiers = ["KNN", "EGB", "RF", "LDA"],
        get_hyperparameters_from_json_fn = main_module.get_hyperparameters_from_json,
        venn2_fn = main_module.venn2,
        venn3_fn = main_module.venn3,
        from_contents_fn = main_module.from_contents,
        upset_cls = main_module.UpSet,
        plt_module = main_module.plt,
    )

    assert shared_features == set()
    assert unique_features["KNN"] == {"A", "B"}
    assert unique_features["EGB"] == {"B", "C"}
    assert unique_features["RF"] == {"C", "D"}
    assert unique_features["LDA"] == {"D", "E"}
    assert upset_calls["contents"] == {
        "KNN": {"A", "B"},
        "EGB": {"B", "C"},
        "RF": {"C", "D"},
        "LDA": {"D", "E"},
    }
    assert upset_calls["data"] == upset_calls["contents"]
    assert upset_calls["plotted"] is True


def test_run_feature_space_review_enabled_invokes_feature_space_review(monkeypatch):
    model = DummyModel("KNN")
    config_parser = DummyConfigParser(
        enabled=False,
        model=model,
        review_enabled=False,
        feature_space_review_enabled=True,
        feature_space_review_models=["KNN", "EGB", "RF"],
    )

    captured = {}

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "GLOCExperimentConfigParser", lambda config_location=None: config_parser)
    monkeypatch.setattr(main_module, "_try_enable_cuml_acceleration", lambda: None)
    monkeypatch.setattr(
        main_module,
        "run_feature_space_review",
        lambda **kwargs: captured.update(kwargs=kwargs),
    )

    main_module.run()

    assert captured["kwargs"]["config_parser"] is config_parser
    assert captured["kwargs"]["investigate_feature_space_fn"] is main_module.investigate_feature_space


def test_run_feature_space_review_passes_expected_parameters():
    model = DummyModel("KNN")
    config_parser = DummyConfigParser(
        enabled=False,
        model=model,
        review_enabled=False,
        feature_space_review_enabled=True,
        feature_space_review_models=["KNN", "EGB", "RF"],
    )

    captured = {}

    def _investigate_feature_space_fn(*args):
        captured["args"] = args
        return set(), {}

    main_module.run_feature_space_review(
        config_parser=config_parser,
        investigate_feature_space_fn=_investigate_feature_space_fn,
        get_hyperparameters_from_json_fn=main_module.get_hyperparameters_from_json,
        venn2_fn=main_module.venn2,
        venn3_fn=main_module.venn3,
        from_contents_fn=main_module.from_contents,
        upset_cls=main_module.UpSet,
        plt_module=main_module.plt,
    )

    assert captured["args"][0] == config_parser.get_model_type()
    assert captured["args"][1] == ["KNN", "EGB", "RF"]
    assert captured["args"][2] is main_module.get_hyperparameters_from_json
    assert captured["args"][3] is main_module.venn2
    assert captured["args"][4] is main_module.venn3
    assert captured["args"][5] is main_module.from_contents
    assert captured["args"][6] is main_module.UpSet
    assert captured["args"][7] is main_module.plt
