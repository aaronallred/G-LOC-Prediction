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


offset_ranges = (0,1,1) # No longer doing any offset (no temporal eval)
data_rate = 25 # (hz)
preference = 7 # Which section of the code do we want to run
random_state = 42
class_weight_imb = None

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

    # ------------------------------------------------------------
    # Convert nested dict structure into a long-format DataFrame
    # Each row = one F1 score, with classifier + stream labels
    # This format is required for seaborn's faceted plotting
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Create faceted violin plots:
    # - One column per classifier
    # - Y-axis lists streams
    # - X-axis shows F1 scores
    # - Hue ensures consistent stream colors across classifiers
    # ------------------------------------------------------------
    g = sns.catplot(
        data=df,
        x="F1 Score",
        y="Stream",
        col="Classifier",
        kind="violin",
        orient="h",
        inner="box",          # Adds a small boxplot inside each violin
        hue="Stream",         # Ensures consistent coloring across panels
        palette="Set2",
        legend=False,         # Avoid redundant legend (streams already on y-axis)
        sharex=True,          # Lock x-axis across classifiers for comparability
        sharey=False,         # Allow each classifier to show its own stream list
        height=5,
        aspect=1.2,
    )

    # Force all panels to use the full F1 range (0–1)
    g.set(xlim=(0, 1))

    # Adjust spacing and add a global title
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(
        f"F1 Score Distributions by Feature Stream | Model: {get_model_subfolder(model_type)}",
        fontsize=14,
        fontweight="bold"
    )

    # ------------------------------------------------------------
    # Save the plot to the appropriate folder
    # ------------------------------------------------------------
    subfolder = get_model_subfolder(model_type)
    results_folder = os.path.join('./feature_study', subfolder, subfolder2 or "")
    os.makedirs(results_folder, exist_ok=True)

    plot_path = os.path.join(results_folder, "f1_violin_by_stream.png")
    g.savefig(plot_path)
    print(f"Saved faceted F1 violin plot to {plot_path}")

    # Display the plot
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

    # Convert stream names to lowercase so matching is case-insensitive.
    # This ensures "HR" matches "hr_mean", "Hr_std", etc.
    streams_lower = [s.lower() for s in streams]

    # List to store features that belong to the requested streams
    usable_features = []

    # Iterate through all selected features from HPO
    for feat in select_features:
        feat_lower = feat.lower()  # normalize feature name for matching

        # Keep the feature if ANY stream keyword appears in the feature name.
        # Example: "HR_mean" contains "hr", so it matches the "HR" stream.
        if any(stream in feat_lower for stream in streams_lower):
            usable_features.append(feat)

    # If no features matched the requested streams, notify the user.
    # This helps catch cases where stream names or feature names don't align.
    if not usable_features:
        print("No usable features were found overall after restricting by streams.")

    # Return the filtered feature list
    return usable_features



if preference == 7:
    # Restrict feature space after features have been selected

    # This defines which model folder to load hyperparameters from
    model_type = ['complete', 'explicit']

    # Load selected features from JSON for each classifier
    # These represent the *full* feature sets before restriction of data streams
    _, select_featuresEGB, _, _ = get_hyperparameters_from_json('EGB', get_model_subfolder(model_type))
    _, select_featuresKNN, _, _ = get_hyperparameters_from_json('KNN', get_model_subfolder(model_type))
    _, select_featuresRF, _, _ = get_hyperparameters_from_json('RF', get_model_subfolder(model_type))

    # Define stream combinations to iterate through
    # Each entry is a list of data streams that will be used to restrict the feature space
    # These represent all combinations of the four source groups
    stream_combos = [
        ['ECG', 'HR', 'BR'], # Source group 1
        ['EEG'], # Source group 2
        ['Pupil'], # Source group 3
        ['Centrifuge', 'Strain', 'Participant', 'Temperature'], # Source group 4
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

    # Classifiers to evaluate for each stream combination
    classifiers_to_test = ['KNN','EGB','RF']

    # Nested dict: classifier -> stream -> F1 scores
    # This will store all results for later plotting
    f1_results_by_stream = {clf: {} for clf in classifiers_to_test}

    # Loop over every stream combination
    for streams_of_interest in stream_combos:
        print(f"\n=== Evaluating streams: {streams_of_interest} ===")

        # Restrict features per classifier based on the current stream subset
        usable_featuresKNN = restrict_feature_space(select_featuresKNN, streams_of_interest)
        usable_featuresRF = restrict_feature_space(select_featuresRF, streams_of_interest)
        usable_featuresEGB = restrict_feature_space(select_featuresEGB, streams_of_interest)

        # Bundle restricted features for easy lookup
        usable_features_dict = {
            "KNN": usable_featuresKNN,
            "RF": usable_featuresRF,
            "EGB": usable_featuresEGB
        }

        # Convert stream list into a string for filenames and dict keys
        stream_str = "-".join(streams_of_interest)

        # Evaluate each classifier on the restricted feature set
        for classifier in classifiers_to_test:
            start_time = time.time()
            num_kfold = 10  # Number of CV folds

            # Storage for F1 scores across folds
            f1_model = np.zeros(num_kfold)

            # Determine model folder
            subfolder_name = get_model_subfolder(model_type)
            feature_subfolder = stream_str

            # Load hyperparameters for this classifier
            hyperparameters, select_features, _, _ = get_hyperparameters_from_json(classifier, subfolder_name)

            # Override features with restricted ones
            select_features = usable_features_dict[classifier]

            print('Starting eval for ', classifier)

            # Folder for saving trained models (not the F1 results)
            save_folder = os.path.join("../RestrictedFeatureEval", subfolder_name)
            os.makedirs(save_folder, exist_ok=True)

            # Always pass offset=0 when generating data
            x, y = data_with_prediction(0, data_rate, classifier, model_type, select_features)

            # Perform stratified k-fold evaluation
            for k in range(num_kfold):
                x_train, x_test, y_train, y_test = stratified_kfold_split(ravel(y), x, num_kfold, k, random_state)

                # Classifier-specific evaluation
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

                # Store F1 score for this fold
                f1_model[k] = f1

            # Save results in memory for plotting
            f1_results_by_stream[classifier][stream_str] = f1_model.flatten()

            # ---------------- SAVE RAW F1 RESULTS ----------------
            # Save each classifier × stream F1 array to disk for preference 8
            results_folder = os.path.join('./feature_study', subfolder_name, "streamwise")
            os.makedirs(results_folder, exist_ok=True)

            filename = f"f1_results_{classifier}_{stream_str}.pkl"
            filepath = os.path.join(results_folder, filename)

            with open(filepath, "wb") as f:
                pickle.dump(f1_model.flatten(), f)

            print(f"Saved F1 results for {classifier} | streams={stream_str} → {filepath}")
            # -----------------------------------------------------

    # After all evaluations, generate faceted violin plots
    plot_f1_violin_by_stream(f1_results_by_stream, model_type, subfolder2="streamwise")


def investigate_feature_space(model_type, classifiers):
    """
    Investigate feature space overlap across classifiers.
    Automatically adapts visualization depending on number of classifiers.
    """

    # Step 1: Load selected features for each classifier
    # Store them as sets to make operations easy
    features_dict = {}
    for clf in classifiers:
        _, selected, _, _ = get_hyperparameters_from_json(clf, get_model_subfolder(model_type))
        features_dict[clf] = set(selected)

    # Step 2: Compute shared and unique features
    # Shared = intersection across all classifiers
    shared_features = set.intersection(*features_dict.values())
    # Unique = features present in one classifier but not in the shared set
    unique_features = {clf: feats - shared_features for clf, feats in features_dict.items()}

    # Print shared features
    print(f"\nShared features across all ({len(shared_features)}):")
    for feat in sorted(shared_features):
        print(f"  - {feat}")

    # Print unique features per classifier
    for clf, feats in unique_features.items():
        print(f"\nFeatures unique to {clf} ({len(feats)}):")
        for feat in sorted(feats):
            print(f"  - {feat}")

    # Step 3: Visualization
    # Choose visualization based on number of classifiers
    n = len(classifiers)

    if n == 2:
        # Two-classifier case → Venn2
        clf1, clf2 = classifiers
        plt.figure(figsize=(6, 5))
        venn2([features_dict[clf1], features_dict[clf2]], set_labels=(clf1, clf2))
        plt.title(f'Feature Overlap: {clf1} vs {clf2}')
        plt.show()

    elif n == 3:
        # Three-classifier case → Venn3
        clf1, clf2, clf3 = classifiers
        plt.figure(figsize=(6, 5))
        venn3([features_dict[clf1], features_dict[clf2], features_dict[clf3]],
              set_labels=(clf1, clf2, clf3))
        plt.title(f'Feature Overlap: {clf1} vs {clf2} vs {clf3}')
        plt.show()

    elif n >= 4:
        # Four or more classifiers → UpSet plot (generalized set visualization)
        upset_data = from_contents(features_dict)
        UpSet(upset_data).plot()
        plt.title('Feature Overlap Across Classifiers')
        plt.show()

    # Return both shared and unique feature sets for further analysis
    return shared_features, unique_features



if preference == 8:
    """
    Inspect previously saved F1 results from preference 7.
    Allows:
        - Loading all saved .pkl files
        - Viewing classifiers independently
        - Selecting which streams to examine
        - Re-plotting subsets of the results
    """

    # Define model type and locate the folder where preference 7 saved results
    model_type = ['complete', 'explicit']
    subfolder = get_model_subfolder(model_type)
    results_folder = os.path.join('./feature_study', subfolder)

    # Classifiers whose results we expect to load
    classifiers = ['KNN', 'EGB', 'RF']

    # Prepare nested dictionary:
    # f1_results_by_stream[classifier][stream_name] = array_of_f1_scores
    f1_results_by_stream = {clf: {} for clf in classifiers}

    # Iterate through all files in the results folder and load matching .pkl files
    for clf in classifiers:
        for filename in os.listdir(results_folder):
            # Only load files that match the naming pattern for this classifier
            if filename.startswith(f"f1_results_{clf}_") and filename.endswith(".pkl"):
                # Extract the stream name from the filename
                stream_str = filename.replace(f"f1_results_{clf}_", "").replace(".pkl", "")
                filepath = os.path.join(results_folder, filename)

                # Load the stored F1 score array
                with open(filepath, "rb") as f:
                    f1_scores = pickle.load(f)

                # Store results in the nested dictionary
                f1_results_by_stream[clf][stream_str] = f1_scores

    # -------------------------------
    # INTERACTIVE SELECTION SECTION
    # -------------------------------

    # Show available classifiers to the user
    print("\nAvailable classifiers:", classifiers)

    # User selects one or more classifiers (or "all")
    clf_choice = input("Enter classifier(s) to inspect (e.g., 'KNN', 'KNN RF', 'KNN,EGB', or 'all'): ").strip()

    if clf_choice.lower() == "all":
        # User wants all classifiers
        selected_classifiers = classifiers
    else:
        # Split user input on commas/spaces and keep only valid classifier names
        selected_classifiers = [
            c.strip()
            for c in clf_choice.replace(",", " ").split()
            if c.strip() in classifiers
        ]

        # If nothing valid was selected, stop execution
        if len(selected_classifiers) == 0:
            print("\nNo valid classifiers selected. Available options:", classifiers)
            raise SystemExit

    # Collect all stream names available for the selected classifiers
    all_streams = sorted({
        stream
        for clf in selected_classifiers
        for stream in f1_results_by_stream[clf].keys()
    })

    # Display available streams
    print("\nAvailable streams:")
    for s in all_streams:
        print("  ", s)

    # User selects one or more streams (or "all")
    stream_choice = input(
        "Enter stream(s) to inspect (e.g., 'EEG', 'EEG Pupil', 'EEG,Pupil', or 'all'): "
    ).strip()

    if stream_choice.lower() == "all":
        # User wants all streams
        selected_streams = all_streams
    else:
        # Normalize separators: convert commas to spaces
        raw_parts = stream_choice.replace(",", " ").split()

        # Match each token to any stream name containing that token
        selected_streams = []
        for token in raw_parts:
            token = token.strip()
            matches = [s for s in all_streams if token.lower() in s.lower()]
            selected_streams.extend(matches)

        # Remove duplicates and sort
        selected_streams = sorted(list(set(selected_streams)))

        # If no valid streams were matched, stop execution
        if len(selected_streams) == 0:
            print("\nNo valid streams selected. Available options:")
            for s in all_streams:
                print("  ", s)
            raise SystemExit

    # Build a filtered dictionary containing only the selected classifiers and streams
    filtered_results = {
        clf: {
            stream: f1_results_by_stream[clf][stream]
            for stream in selected_streams
            if stream in f1_results_by_stream[clf]
        }
        for clf in selected_classifiers
    }

    # Remove classifiers that ended up with no matching streams
    filtered_results = {clf: d for clf, d in filtered_results.items() if len(d) > 0}

    # Plot the selected subset using the faceted violin plot function
    print("\nPlotting selected results...")
    plot_f1_violin_by_stream(filtered_results, model_type, subfolder2="filtered")

if preference == 9:
    # Preference to plot overlap of features ONLY
    # Agnostic to how every many classifiers we want to look at
    # Just ensure that saved hyperparameters and selected features are in JSON format
    model_type = ['noAFE', 'explicit']

    # Try with all however many classifiers
    investigate_feature_space(model_type, ['KNN', 'LDA', 'logreg'])


if preference == 10:
    # Save Hyperparameters to JSON
    model_type = ['noAFE', 'explicit']
    classifier = 'logreg'
    get_median_hyperparameters(classifier,get_model_subfolder(model_type))
