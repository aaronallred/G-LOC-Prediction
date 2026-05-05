from typing import Callable


def investigate_feature_space(
    model_type,
    classifiers: list[str],
    get_hyperparameters_from_json_fn: Callable,
    venn2_fn,
    venn3_fn,
    from_contents_fn,
    upset_cls,
    plt_module,
):
    """Inspect feature overlap across classifiers and render the matching set visualization."""
    features_dict = {}
    for clf in classifiers:
        _, selected_features, _, _ = get_hyperparameters_from_json_fn(clf, model_type.get_folder_name())
        features_dict[clf] = set(selected_features)

    if len(features_dict) == 0:
        raise ValueError("At least one classifier is required to inspect feature overlap.")

    shared_features = set.intersection(*features_dict.values())
    unique_features = {
        clf: features - shared_features
        for clf, features in features_dict.items()
    }

    print(f"\nShared features across all ({len(shared_features)}):")
    for feat in sorted(shared_features):
        print(f"  - {feat}")

    for clf, feats in unique_features.items():
        print(f"\nFeatures unique to {clf} ({len(feats)}):")
        for feat in sorted(feats):
            print(f"  - {feat}")

    num_classifiers = len(classifiers)
    if num_classifiers == 2:
        if venn2_fn is None:
            raise ImportError("matplotlib_venn is required for two-classifier overlap plots.")
        clf1, clf2 = classifiers
        plt_module.figure(figsize=(6, 5))
        venn2_fn([features_dict[clf1], features_dict[clf2]], set_labels=(clf1, clf2))
        plt_module.title(f"Feature Overlap: {clf1} vs {clf2}")
        plt_module.show()
    elif num_classifiers == 3:
        if venn3_fn is None:
            raise ImportError("matplotlib_venn is required for three-classifier overlap plots.")
        clf1, clf2, clf3 = classifiers
        plt_module.figure(figsize=(6, 5))
        venn3_fn(
            [features_dict[clf1], features_dict[clf2], features_dict[clf3]],
            set_labels=(clf1, clf2, clf3),
        )
        plt_module.title(f"Feature Overlap: {clf1} vs {clf2} vs {clf3}")
        plt_module.show()
    elif num_classifiers >= 4:
        if from_contents_fn is None or upset_cls is None:
            raise ImportError("upsetplot is required for four-or-more-classifier overlap plots.")
        upset_data = from_contents_fn(features_dict)
        upset_cls(upset_data).plot()
        plt_module.title("Feature Overlap Across Classifiers")
        plt_module.show()

    return shared_features, unique_features


def run_feature_space_review(
    config_parser,
    investigate_feature_space_fn: Callable,
    get_hyperparameters_from_json_fn: Callable,
    venn2_fn,
    venn3_fn,
    from_contents_fn,
    upset_cls,
    plt_module,
) -> None:
    model_type = config_parser.get_feature_space_review_model_type()
    classifiers = config_parser.get_feature_space_review_models()

    if len(classifiers) == 0:
        raise ValueError("feature_space_review.models must be a non-empty list when enabled.")

    investigate_feature_space_fn(
        model_type,
        classifiers,
        get_hyperparameters_from_json_fn,
        venn2_fn,
        venn3_fn,
        from_contents_fn,
        upset_cls,
        plt_module,
    )
