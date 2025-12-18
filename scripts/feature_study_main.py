import numpy as np
import os
import joblib  # For saving the model
from numpy import ravel
import time
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from upsetplot import from_contents, UpSet


from GLOC_data_processing import *
from scripts.GLOC_classifier import stratified_kfold_split, classify_logistic_regression, classify_random_forest, \
    classify_lda, classify_svm, classify_knn, classify_ensemble_with_gradboost
from scripts.imbalance_techniques import resample_ros
from scripts.temporal_functions import plotting_offset_models, data_with_prediction, \
    plot_f1_scores_across_classifiers, get_model_subfolder, \
    data_with_prediction_verification, get_median_hyperparameters, get_hyperparameters_from_json, \
    plot_metrics_from_cache
import pickle
import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

######## This file will be used to do feature focused studies
######## Will only be looking at classification, not temporal instances

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_f1_violin_by_stream(f1_results_by_stream, model_type, subfolder2=None):
    """
    Create faceted violin plots of F1 scores per classifier, grouped by feature stream.

    Parameters
    ----------
    f1_results_by_stream : dict
        Nested dict: classifier -> stream -> list of F1 scores.
        Example: {
            "KNN": {"EEG": [...], "HR": [...]},
            "RF": {"EEG": [...], "HR": [...]},
            "EGB": {"EEG": [...], "HR": [...]}
        }
    model_type : list of str
        Model type specifier (e.g. ["complete", "explicit"]).
    subfolder2 : str, optional
        Extra subfolder name for saving results.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Build long-format DataFrame
    records = []
    for clf, stream_dict in f1_results_by_stream.items():
        for stream, f1_scores in stream_dict.items():
            for score in f1_scores:
                records.append({
                    "Classifier": clf,
                    "Stream": stream,
                    "F1 Score": score
                })
    df = pd.DataFrame(records)

    # Plot using seaborn catplot (faceted violin plots)
    g = sns.catplot(
        data=df,
        x="F1 Score",
        y="Stream",
        col="Classifier",
        kind="violin",
        orient="h",
        inner="box",
        hue="Stream",
        palette="Set2",
        legend = False,
        sharex=True,
        sharey=False,
        height=5,
        aspect=1.2,
    )

    g.set(xlim=(0, 1))

    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(f"F1 Score Distributions by Feature Stream | Model: {get_model_subfolder(model_type)}",
                   fontsize=14, fontweight="bold")

    ############################ Save plot ############################
    subfolder = get_model_subfolder(model_type)
    results_folder = os.path.join('./feature_study', subfolder, subfolder2 or "")
    os.makedirs(results_folder, exist_ok=True)

    plot_path = os.path.join(results_folder, "f1_violin_by_stream.png")
    g.savefig(plot_path)
    print(f"Saved faceted F1 violin plot to {plot_path}")

    plt.show()


def restrict_feature_space(select_features, streams):
    """
    Restrict selected features to those corresponding to specified data streams.

    Parameters
    ----------
    select_features : list of str
        Features already selected by HPO (e.g. ["HR_mean", "ECG_var", "BR_slope"]).
    streams : list of str
        Data streams to keep (e.g. ["HR", "ECG"]).
        Valid options include: "HR", "ECG", "BR", "Temperature", "Pupil",
        "Centrifuge", "EEG", "Strain", "Participant".

    Returns
    -------
    usable_features : list of str
        Subset of select_features that correspond to the specified streams.
    """

    # Normalize streams for case-insensitive matching
    streams_lower = [s.lower() for s in streams]

    usable_features = []
    for feat in select_features:
        feat_lower = feat.lower()
        # Check if any stream keyword is contained in the feature name
        if any(stream in feat_lower for stream in streams_lower):
            usable_features.append(feat)


    # Diagnostic message if nothing found
    if not usable_features:
        print("No usable features were found overall after restricting by streams.")

    return usable_features


offset_ranges = (0,1,1) # No longer doing any offset (no temporal eval)
data_rate = 25 # (hz)
preference = 7 # Which section of the code do we want to run
random_state = 42
class_weight_imb = None

if preference == 7:
    # Restrict feature space after features have been selected
    model_type = ['complete', 'explicit']

    # Load selected features from JSON for each classifier
    _, select_featuresEGB, _, _ = get_hyperparameters_from_json('EGB', get_model_subfolder(model_type))
    _, select_featuresKNN, _, _ = get_hyperparameters_from_json('KNN', get_model_subfolder(model_type))
    _, select_featuresRF, _, _ = get_hyperparameters_from_json('RF', get_model_subfolder(model_type))

    # Define stream combinations to iterate through
    # All different kinds: HR, ECG, EEG, Temperature, Pupil, Centrifuge, Strain, Participant
    # Organized as follows:
        # ECG: ECG + HR
        # EEG: EEG
        # Phys: HR + ECG + BR + Temperature (skin) + Pupil + EEG
        # Non Phys: Centrifuge + Participant
    # Further organized into specific data streams, sourced from same data
        # ECG: ECG + HR + BR?
        # EEG: EEG
        # FNIRS: Pupil
        # Temperature: Temperature
        # Non phys: Centrifuge, Strain, Participant
    stream_combos = [
        ['ECG', 'HR', 'BR'],
        ['EEG'],
        ['Pupil'],
        ['Centrifuge', 'Strain', 'Participant', 'Temperature'],
        ['ECG', 'HR', 'BR', 'EEG'],
        ['ECG', 'HR', 'BR', 'Pupil'],
        ['ECG', 'HR', 'BR', 'Centrifuge', 'Strain', 'Participant', 'Temperature'],
        ['EEG', 'Pupil'],
        ['EEG', 'Centrifuge', 'Strain', 'Participant', 'Temperature'],
        ['Pupil', 'Centrifuge', 'Strain', 'Participant', 'Temperature'],
        ['ECG', 'HR', 'BR', 'EEG', 'Pupil'],
        ['ECG', 'HR', 'BR', 'EEG', 'Centrifuge', 'Strain', 'Participant', 'Temperature'],
        ['ECG', 'HR', 'BR', 'Pupil', 'Centrifuge', 'Strain', 'Participant', 'Temperature'],
        ['EEG', 'Pupil', 'Centrifuge', 'Strain', 'Participant', 'Temperature'],
        ['ECG', 'HR', 'BR', 'EEG', 'Pupil', 'Centrifuge', 'Strain', 'Participant', 'Temperature']
    ]

    classifiers_to_test = ['KNN','EGB','RF']

    # Nested dict: classifier -> stream -> F1 scores
    f1_results_by_stream = {clf: {} for clf in classifiers_to_test}

    for streams_of_interest in stream_combos:
        print(f"\n=== Evaluating streams: {streams_of_interest} ===")

        # Restrict features per classifier
        usable_featuresKNN = restrict_feature_space(select_featuresKNN, streams_of_interest)
        usable_featuresRF = restrict_feature_space(select_featuresRF, streams_of_interest)
        usable_featuresEGB = restrict_feature_space(select_featuresEGB, streams_of_interest)

        usable_features_dict = {
            "KNN": usable_featuresKNN,
            "RF": usable_featuresRF,
            "EGB": usable_featuresEGB
        }

        stream_str = "-".join(streams_of_interest)

        for classifier in classifiers_to_test:
            start_time = time.time()
            num_kfold = 10

            f1_model = np.zeros(num_kfold)

            subfolder_name = get_model_subfolder(model_type)
            feature_subfolder = stream_str

            hyperparameters, select_features, _, _ = get_hyperparameters_from_json(classifier, subfolder_name)

            # Override features with restricted ones
            select_features = usable_features_dict[classifier]

            print('Starting eval for ', classifier)
            save_folder = os.path.join("../RestrictedFeatureEval", subfolder_name)
            os.makedirs(save_folder, exist_ok=True)

            # Always pass offset=0
            x, y = data_with_prediction(0, data_rate, classifier, model_type, select_features)

            for k in range(num_kfold):
                x_train, x_test, y_train, y_test = stratified_kfold_split(ravel(y), x, num_kfold, k, random_state)

                if classifier == 'RF':
                    _, _, _, f1, _, _, _ = classify_random_forest(
                        x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                        save_folder, model_name="rf_feature_study.pkl", retrain=False,
                        temporal=True, best_params=hyperparameters)
                elif classifier == 'KNN':
                    ros_x_train, ros_y_train = resample_ros(x_train, y_train, random_state)
                    _, _, _, f1, _, _ = classify_knn(
                        x_train, x_test, y_train, y_test, random_state,
                        save_folder, model_name="knn_feature_study.pkl", retrain=False,
                        temporal=True, best_params=hyperparameters)
                elif classifier == 'EGB':
                    _, _, _, f1, _, _ = classify_ensemble_with_gradboost(
                        x_train, x_test, y_train, y_test, random_state,
                        save_folder, model_name="egb_feature_study.pkl", retrain=False,
                        temporal=True, best_params=hyperparameters)

                f1_model[k] = f1

            print('Success for classifier:', classifier)
            f1_results_by_stream[classifier][stream_str] = f1_model.flatten()

            end_time = time.time()
            print(f"Total time for classifier '{classifier}': {end_time - start_time:.2f} seconds")

    # Call new faceted plotting function
    plot_f1_violin_by_stream(f1_results_by_stream, model_type, subfolder2="streamwise")


def investigate_feature_space(model_type, classifiers):
    """
    Investigate feature space overlap across classifiers.
    Automatically adapts visualization depending on number of classifiers.
    """

    # Step 1: Load selected features
    features_dict = {}
    for clf in classifiers:
        _, selected, _, _ = get_hyperparameters_from_json(clf, get_model_subfolder(model_type))
        features_dict[clf] = set(selected)

    # Step 2: Compute shared + unique features
    shared_features = set.intersection(*features_dict.values())
    unique_features = {clf: feats - shared_features for clf, feats in features_dict.items()}

    print(f"\nShared features across all ({len(shared_features)}):")
    for feat in sorted(shared_features):
        print(f"  - {feat}")

    for clf, feats in unique_features.items():
        print(f"\nFeatures unique to {clf} ({len(feats)}):")
        for feat in sorted(feats):
            print(f"  - {feat}")

    # Step 3: Visualization
    n = len(classifiers)

    if n == 2:
        clf1, clf2 = classifiers
        plt.figure(figsize=(6, 5))
        venn2([features_dict[clf1], features_dict[clf2]], set_labels=(clf1, clf2))
        plt.title(f'Feature Overlap: {clf1} vs {clf2}')
        plt.show()

    elif n == 3:
        clf1, clf2, clf3 = classifiers
        plt.figure(figsize=(6, 5))
        venn3([features_dict[clf1], features_dict[clf2], features_dict[clf3]],
              set_labels=(clf1, clf2, clf3))
        plt.title(f'Feature Overlap: {clf1} vs {clf2} vs {clf3}')
        plt.show()

    elif n >= 4:
        upset_data = from_contents(features_dict)
        UpSet(upset_data).plot()
        plt.title('Feature Overlap Across Classifiers')
        plt.show()

    return shared_features, unique_features


if preference == 8:

    # Preference to plot overlap of features ONLY
    # Agnostic to how every many classifiers we want to look at
    # Just ensure that saved hyperparameters and selected features are in JSON format
    model_type = ['noAFE', 'explicit']

    # Try with all 6 classifiers
    investigate_feature_space(model_type, ['KNN', 'LDA', 'logreg'])


if preference == 9:
    # Save Hyperparameters to JSON
    model_type = ['noAFE', 'explicit']
    classifier = 'logreg'
    get_median_hyperparameters(classifier,get_model_subfolder(model_type))
