import json
from typing import Callable

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from upsetplot import from_contents, UpSet

from src.models_new.model_factory import ModelFactory
from src.traditional_experiment_utils import get_hyperparameters_from_json



def run_feature_space_review(
    config: dict,
    model_factory: ModelFactory
) -> None:
    """
    Performs a review of the feature space by analyzing the selected features across different classifiers. 
    It identifies shared and unique features among the classifiers and visualizes the overlap using Venn 
    diagrams (for 2-3 classifiers) or an UpSet plot (for 4 or more classifiers).

    Parameters:
    - config (dict): The YAML config containing settings for the feature space review.
    - model_factory (ModelFactory): An instance of the ModelFactory to create model instances based on the configuration.

    Returns:
    - None
    """
    feature_space_review_config = config["feature_space_review"]
    model_type = feature_space_review_config["model_type"]
    models = [model_factory.create_model(model_name) for model_name in feature_space_review_config["models"]]

    if len(models) == 0:
        raise ValueError("feature_space_review.models must be a non-empty list when enabled.")
    
    # Inspect feature overlap across classifiers and render the matching set visualization
    features_dict = {}
    for model in models:
        median_hyperparameters_folder = Path(feature_space_review_config["median_hyperparameters_folder"])
        _, selected_features, _, _ = get_hyperparameters_from_json(median_hyperparameters_folder, model_type, model.name)
        features_dict[model.name] = set(selected_features)

    shared_features = set.intersection(*features_dict.values())
    unique_features = {
        classifier: features - shared_features
        for classifier, features in features_dict.items()
    }

    print(f"\nShared features across all ({len(shared_features)}):")
    for feat in sorted(shared_features):
        print(f"  - {feat}")

    for classifier, feats in unique_features.items():
        print(f"\nFeatures unique to {classifier} ({len(feats)}):")
        for feat in sorted(feats):
            print(f"  - {feat}")

    num_models = len(models)
    if num_models == 2:
        # For 2 models, use a Venn diagram to show shared vs unique features
        clf1, clf2 = models[0].name, models[1].name
        plt.figure(figsize=(6, 5))
        venn2([features_dict[clf1], features_dict[clf2]], set_labels=(clf1, clf2))
        plt.title(f"Feature Overlap: {clf1} vs {clf2}")
        plt.show()
    elif num_models == 3:
        # For 3 models, use a 3-set Venn diagram
        clf1, clf2, clf3 = models[0].name, models[1].name, models[2].name
        plt.figure(figsize=(6, 5))
        venn3(
            [features_dict[clf1], features_dict[clf2], features_dict[clf3]],
            set_labels=(clf1, clf2, clf3),
        )
        plt.title(f"Feature Overlap: {clf1} vs {clf2} vs {clf3}")
        plt.show()
    elif num_models >= 4:
        # For 4 or more models, use an UpSet plot to show all intersections
        upset_data = from_contents(features_dict)
        UpSet(upset_data).plot()
        plt.title("Feature Overlap Across Classifiers")
        plt.show()