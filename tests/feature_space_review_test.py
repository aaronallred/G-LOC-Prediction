from pathlib import Path

import pytest

from src.config_loader import load_experiment_config
from src.model_type import ModelType
from src.modes.feature_space_review import investigate_feature_space, run_feature_space_review


class FeatureModel:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


def _feature_loader(features_by_name):
    def _load(classifier, model_type_name):
        return ({"hp": classifier}, features_by_name[classifier], None, None)

    return _load


def test_investigate_feature_space_two_models_returns_shared_and_unique_features():
    plotted = []

    shared, unique = investigate_feature_space(
        model_type=ModelType("Complete", "Explicit"),
        classifiers=["KNN", "RF"],
        get_hyperparameters_from_json_fn=_feature_loader({
            "KNN": ["f1", "f2", "f3"],
            "RF": ["f2", "f4"],
        }),
        venn2_fn=lambda *args, **kwargs: plotted.append("venn2"),
        venn3_fn=None,
        from_contents_fn=None,
        upset_cls=None,
        plt_module=type("Plot", (), {"figure": lambda *a, **k: None, "title": lambda *a, **k: None, "show": lambda *a, **k: None})(),
    )

    assert shared == {"f2"}
    assert unique["KNN"] == {"f1", "f3"}
    assert unique["RF"] == {"f4"}
    assert plotted == ["venn2"]


def test_investigate_feature_space_three_models_uses_venn3():
    plotted = []

    investigate_feature_space(
        model_type=ModelType("Complete", "Explicit"),
        classifiers=["KNN", "RF", "LDA"],
        get_hyperparameters_from_json_fn=_feature_loader({
            "KNN": ["f1", "f2"],
            "RF": ["f2", "f3"],
            "LDA": ["f2", "f4"],
        }),
        venn2_fn=None,
        venn3_fn=lambda *args, **kwargs: plotted.append("venn3"),
        from_contents_fn=None,
        upset_cls=None,
        plt_module=type("Plot", (), {"figure": lambda *a, **k: None, "title": lambda *a, **k: None, "show": lambda *a, **k: None})(),
    )

    assert plotted == ["venn3"]


def test_investigate_feature_space_four_models_uses_upset():
    plotted = []

    class FakeUpSet:
        def __init__(self, data):
            self.data = data

        def plot(self):
            plotted.append(sorted(self.data.keys()))

    investigate_feature_space(
        model_type=ModelType("Complete", "Explicit"),
        classifiers=["KNN", "RF", "LDA", "SVM"],
        get_hyperparameters_from_json_fn=_feature_loader({
            "KNN": ["f1"],
            "RF": ["f2"],
            "LDA": ["f3"],
            "SVM": ["f4"],
        }),
        venn2_fn=None,
        venn3_fn=None,
        from_contents_fn=lambda features: features,
        upset_cls=FakeUpSet,
        plt_module=type("Plot", (), {"figure": lambda *a, **k: None, "title": lambda *a, **k: None, "show": lambda *a, **k: None})(),
    )

    assert plotted == [["KNN", "LDA", "RF", "SVM"]]


def test_run_feature_space_review_delegates_configured_model_names(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
                """data_path: /tmp/data
shared_data_parameters:
    subject_to_analyze: null
    trial_to_analyze: null
    analysis_type: 2
    remove_NaN_trials: true
    impute_file_name: imputed.pkl
    save_impute: false
    load_impute: false
    impute_phase: pre_feature
    output_feature_dtype: float32
advanced_data_parameters:
    n_neighbors: 4
    baseline_window: 32.5
traditional_data_parameters:
    backstep: 0
    data_rate: 25
    offset: 0
    time_start: 0
feature_space_review:
    enabled: true
    models:
        - KNN
        - RF
    model_type: !ModelType [Complete, Explicit]
""",
        encoding="utf-8",
    )
    config = load_experiment_config(config_path)
    captured = {}

    run_feature_space_review(
        config=config,
        investigate_feature_space_fn=lambda *args, **kwargs: captured.update(
            classifiers=args[1],
            model_type=args[0],
        ),
        get_hyperparameters_from_json_fn=_feature_loader({"KNN": ["f1"], "RF": ["f2"]}),
        venn2_fn=None,
        venn3_fn=None,
        from_contents_fn=None,
        upset_cls=None,
        plt_module=type("Plot", (), {"figure": lambda *a, **k: None, "title": lambda *a, **k: None, "show": lambda *a, **k: None})(),
    )

    assert captured["classifiers"] == ["KNN", "RF"]
    assert captured["model_type"] == ModelType("Complete", "Explicit")

