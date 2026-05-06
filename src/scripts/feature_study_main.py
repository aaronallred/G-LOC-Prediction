import numpy as np
import os
import joblib  # For saving the model
from numpy import ravel
import time
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from upsetplot import from_contents, UpSet


from .GLOC_data_processing_traditional import *
from .GLOC_classifier_traditional import stratified_kfold_split, classify_logistic_regression, classify_random_forest, \
    classify_lda, classify_svm, classify_knn, classify_ensemble_with_gradboost
from .imbalance_techniques_traditional import resample_ros
from .temporal_functions_traditional import plotting_offset_models, data_with_prediction, \
    plot_f1_scores_across_classifiers, get_model_subfolder, \
     get_median_hyperparameters, get_hyperparameters_from_json, \
    plot_metrics_from_cache
from src.GLOC_experiment_config_parser import GLOCExperimentConfigParser
from src.data_pipeline import DataPipeline
from src.scripts.plot_shap import plot_all_shap

import sys
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

def test_violin(f1_results_by_stream, model_type, subfolder2=None):
    """
    Create faceted violin plots with a 'checkbox' style matrix x-axis.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os

    # ------------------------------------------------------------
    # 1. Prepare the Data
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

    # Identify unique classifiers and streams
    classifiers = df['Classifier'].unique()
    unique_streams = df['Stream'].unique()  # Sort by length looks nice usually

    # ------------------------------------------------------------
    # 2. Build the Matrix Data (The "Checkbox" Grid)
    # ------------------------------------------------------------
    # Identify all unique individual components (e.g., 'ECG', 'EEG', 'Pupil')
    components = set()
    for s in unique_streams:
        # Split by hyphen to get ingredients
        parts = s.split('-')
        components.update(parts)
    sorted_components = sorted(list(components))

    # Create a DataFrame where Index=Components, Columns=Stream Names
    # 1 = component is present, 0 = absent
    matrix_data = []
    for comp in sorted_components:
        row = []
        for stream_name in unique_streams:
            # Check if component is in the stream name
            # (using precise split check to avoid partial matches)
            is_present = 1 if comp in stream_name.split('-') else 0
            row.append(is_present)
        matrix_data.append(row)

    matrix_df = pd.DataFrame(matrix_data, index=sorted_components, columns=unique_streams)

    # ------------------------------------------------------------
    # 3. Setup Custom Grid Layout
    # ------------------------------------------------------------
    # We need a column for each classifier.
    # Inside each column, we need 2 rows: Top (Violin, ~80%), Bottom (Matrix, ~20%)
    num_clfs = len(classifiers)

    fig = plt.figure(figsize=(6 * num_clfs, 10))
    # Add a global title slightly higher up
    fig.suptitle(
        f"F1 Score Distributions by Feature Stream | Model: {get_model_subfolder(model_type)}",
        fontsize=16, fontweight="bold", y=0.95
    )

    # Create outer grid (1 row, N columns for classifiers)
    outer_grid = gridspec.GridSpec(1, num_clfs, wspace=0.1)

    for i, clf in enumerate(classifiers):
        # Create an inner grid for this classifier (2 rows: plot & matrix)
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer_grid[i],
            height_ratios=[4, 1.5],  # Adjust this to change relative height of matrix
            hspace=0.05
        )

        ax_top = plt.Subplot(fig, inner_grid[0])
        ax_bottom = plt.Subplot(fig, inner_grid[1])
        fig.add_subplot(ax_top)
        fig.add_subplot(ax_bottom)

        # Filter data for this classifier
        clf_data = df[df['Classifier'] == clf]

        # --- A. TOP PLOT (Violin) ---
        sns.violinplot(
            data=clf_data,
            x="Stream",
            y="F1 Score",
            order=unique_streams,  # CRITICAL: Ensure ordering matches matrix
            ax=ax_top,
            palette="Set2",
            inner="box",
            hue="Stream",
            legend=False
        )

        # Styling Top
        ax_top.set_title(clf, fontsize=14, fontweight='bold')
        ax_top.set_xlabel('')
        ax_top.set_xticklabels([])  # Hide x-text on top plot

        # Only show Y labels for the first classifier (leftmost)
        if i > 0:
            ax_top.set_ylabel('')
            ax_top.set_yticklabels([])
        else:
            ax_top.set_ylabel("F1 Score", fontsize=12)

        # Force y-limits if needed
        # ax_top.set_ylim(0.0, 1.0)

        # --- B. BOTTOM PLOT (Matrix Heatmap) ---
        # Draw the heatmap
        sns.heatmap(
            matrix_df,
            ax=ax_bottom,
            cbar=False,
            cmap="Greens",  # White=0, Black=1 (or use 'Blues', etc)
            linewidths=1,
            linecolor='lightgray',
            vmin=0, vmax=1.5  # Slight offset makes the '1's distinct grey/black
        )

        # Styling Bottom
        ax_bottom.set_xlabel("Data Stream Combination", fontsize=10)
        ax_bottom.set_xticklabels([])  # We don't need text labels on x, the dots represent them
        ax_bottom.tick_params(left=False, bottom=False)  # Remove tick marks

        # Only show Component names (rows) for the first classifier
        if i > 0:
            ax_bottom.set_yticklabels([])
        else:
            ax_bottom.set_yticklabels(sorted_components, rotation=0, fontsize=10)

    # ------------------------------------------------------------
    # Save the plot to the appropriate folder
    # ------------------------------------------------------------
    subfolder = get_model_subfolder(model_type)
    results_folder = os.path.join('./feature_study', subfolder, subfolder2 or "")
    os.makedirs(results_folder, exist_ok=True)

    plot_path = os.path.join(results_folder, "f1_violin_by_stream.png")
    fig.savefig(plot_path, bbox_inches='tight')
    print(f"Saved faceted F1 violin plot to {plot_path}")

    plt.show()

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
    df['Stream'] = df['Stream'].str.replace('-', '\n')

    # ------------------------------------------------------------
    # Create faceted violin plots:
    # - One column per classifier
    # - Y-axis lists streams
    # - X-axis shows F1 scores
    # - Hue ensures consistent stream colors across classifiers
    # ------------------------------------------------------------
    g = sns.catplot(
        data=df,
        x="Stream",
        y="F1 Score",
        col="Classifier",
        kind="violin",
        orient="v",
        inner="box",          # Adds a small boxplot inside each violin
        hue="Stream",         # Ensures consistent coloring across panels
        palette="Set2",
        legend=False,         # Avoid redundant legend (streams already on y-axis)
        sharex=False,          # Lock x-axis across classifiers for comparability
        sharey=True,         # Allow each classifier to show its own stream list
        height=6,
        aspect=1.2,
    )

    # Force all panels to use the full F1 range (0–1)
    # g.set(ylim=(0.7, 1))
    # g.set_xticklabels(rotation=20, horizontalalignment='right')

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
    plt.tight_layout()

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

    # Output any streams that have multiple keywords
    # This was used to identify participant_HR
    for feat in usable_features:
        feat_lower = feat.lower()
        matches = [s for s in streams_lower if s in feat_lower]  # substring match
        if len(matches) >= 2:
            print(feat, "=>", matches)

    # If no features matched the requested streams, notify the user.
    # This helps catch cases where stream names or feature names don't align.
    if not usable_features:
        print("No usable features were found overall after restricting by streams.")

    # Return the filtered feature list
    return usable_features



if preference == 7:
    config_parser = GLOCExperimentConfigParser(config_location = "C:/Users/savan/G-LOC-Prediction/GLOC_experiment_config.json")
    pipeline = DataPipeline(config_parser = config_parser)

    models_to_test = config_parser.get_models()
    stream_groups_to_test = config_parser.get_sensor_ablation_streams()
    model_type_obj = config_parser.get_model_type()
    model_type = [model_type_obj.afe_filter.lower(), model_type_obj.feature_set.lower()]
    num_kfold = config_parser.get_num_splits()

    # For manual ablation, find RF model
    for model in models_to_test:
        if model.get_name() =='RF':
            man_abl_model = model

    # Store results for plotting later
    f1_results_by_stream = {model.get_name(): {} for model in models_to_test}

    for stream_group in stream_groups_to_test:
        print(f"Running stream group: {stream_group}")

        # Convert stream list into a string for filenames and dict keys
        stream_str = "-".join(stream_group)

        for model in models_to_test:
            classifier = model.get_name()
            print(f"Running model: {classifier}")
            
            start_time = time.time()
            num_kfold = 10  # Number of CV folds

            # Storage for F1 scores across folds
            f1_model = np.zeros(num_kfold)

            # Determine model folder
            subfolder_name = get_model_subfolder(model_type)

            # DataPipeline already loads selected features and applies stream ablation.
            # Keep only classifier hyperparameters here for model fitting.
            hyperparameters, _, _, _ = get_hyperparameters_from_json(classifier, subfolder_name)

            print("Starting evaluation for model:", classifier)

            # Folder for saving trained models (not the F1 results)
            save_folder = os.path.join("../RestrictedFeatureEval", subfolder_name)
            os.makedirs(save_folder, exist_ok=True)

            # Pull feature matrix and labels directly from DataPipeline.
            x, y, select_features = pipeline.get_data(model=model, feature_streams=stream_group if stream_group else None)

            results_folder = os.path.join('./feature_study', subfolder_name, "streamwise")

            # Perform stratified k-fold evaluation
            for k in range(num_kfold):
                x_train, x_test, y_train, y_test = stratified_kfold_split(ravel(y), x, num_kfold, k, random_state)

                # Classifier-specific evaluation
                if classifier == 'RF':
                    _, _, _, f1, _, _, _, explanation = classify_random_forest(
                        x_train, x_test, y_train, y_test, class_weight_imb, random_state,
                        save_folder, model_name="rf_feature_study.pkl", retrain=False,
                        temporal=True, best_params=hyperparameters, select_features=select_features,
                        feat_imp=True)
                elif classifier == 'KNN':
                    ros_x_train, ros_y_train = resample_ros(x_train, y_train, random_state)
                    _, _, _, f1, _, _, explanation = classify_knn(
                        x_train, x_test, y_train, y_test, random_state,
                        save_folder, model_name="knn_feature_study.pkl", retrain=False,
                        temporal=True, best_params=hyperparameters, select_features=select_features,
                        feat_imp=True)
                elif classifier == 'EGB':
                    _, _, _, f1, _, _, explanation = classify_ensemble_with_gradboost(
                        x_train, x_test, y_train, y_test, random_state,
                        save_folder, model_name="egb_feature_study.pkl", retrain=False,
                        temporal=True, best_params=hyperparameters, select_features=select_features,
                        feat_imp=True)

                # Store F1 score for this fold
                f1_model[k] = f1

                save_path = f"{results_folder}/shap_explanation_{classifier}_fold_{k+1}.pkl"

                plot_all_shap(
                    subfolder_name=subfolder_name,
                    fold_num=k,
                    classifier=classifier,
                    explanation=explanation
                )

                with open(save_path, "wb") as f:
                    pickle.dump(explanation, f)

            # median_f1 = np.median(f1_model)
            #
            # median_fold = int(np.argmin(np.abs(f1_model - median_f1)))
            #
            # print("F1 scores:", f1_model)
            # print("Median F1:", median_f1)
            # print("Selected median fold:", median_fold)
            #
            # k = median_fold
            #
            # x_train, x_test, y_train, y_test = stratified_kfold_split(
            #     ravel(y), x, num_kfold, k, random_state)
            #
            # if classifier == 'RF':
            #     _, _, _, f1, _, _, _, explanation = classify_random_forest(
            #         x_train, x_test, y_train, y_test, class_weight_imb, random_state,
            #         save_folder, model_name="rf_feature_study.pkl", retrain=False,
            #         temporal=True, best_params=hyperparameters, select_features=select_features,
            #         feat_imp=True)
            # elif classifier == 'KNN':
            #     ros_x_train, ros_y_train = resample_ros(x_train, y_train, random_state)
            #     _, _, _, f1, _, _, explanation = classify_knn(
            #         x_train, x_test, y_train, y_test, random_state,
            #         save_folder, model_name="knn_feature_study.pkl", retrain=False,
            #         temporal=True, best_params=hyperparameters, select_features=select_features,
            #         feat_imp=True)
            # elif classifier == 'EGB':
            #     _, _, _, f1, _, _, explanation = classify_ensemble_with_gradboost(
            #         x_train, x_test, y_train, y_test, random_state,
            #         save_folder, model_name="egb_feature_study.pkl", retrain=False,
            #         temporal=True, best_params=hyperparameters, select_features=select_features,
            #         feat_imp=True)
            #
            # save_path = f"{results_folder}/shap_explanation_{classifier}_fold_{k}.pkl"
            #
            # plot_all_shap(
            #     subfolder_name=subfolder_name,
            #     fold_num=k,
            #     classifier=classifier,
            #     explanation=explanation
            # )
            #
            # with open(save_path, "wb") as f:
            #     pickle.dump(explanation, f)

            # Save results in memory for plotting
            f1_results_by_stream[classifier][stream_str] = f1_model.flatten()

            # ---------------- SAVE RAW F1 RESULTS ----------------
            # Save each classifier × stream F1 array to disk for preference 8
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
        - Selecting which streams to examine via group-combo batches
        - Re-plotting subsets of the results
    """

    # Define model type and locate the folder where preference 7 saved results
    model_type = ['complete', 'explicit']
    subfolder = get_model_subfolder(model_type)
    results_folder = os.path.join('./feature_study', subfolder, 'streamwise')

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
    clf_choice = input(
        "Enter classifier(s) to inspect (e.g., 'KNN', 'KNN RF', 'KNN,EGB', or 'all'): "
    ).strip()

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

    if len(all_streams) == 0:
        print("\nNo streams found for the selected classifiers.")
        raise SystemExit

    # -----------------------------------------
    # STREAM GROUP DEFINITIONS
    # -----------------------------------------

    # Tokens used to detect group membership inside stream names
    STREAM_GROUPS = {
        1: ["ECG", "HR", "BR"],
        2: ["EEG"],
        3: ["Pupil"],
        4: ["Strain", "Centrifuge", "Participant", "Temperature"]
    }

    print("\nStream Groups (detected by token presence in stream name):")
    for gid, items in STREAM_GROUPS.items():
        print(f"  {gid}: {', '.join(items)}")

    # Map each stream name -> set of group IDs it belongs to
    def groups_for_stream(stream_name: str) -> set:
        s_lower = stream_name.lower()
        groups = set()
        for gid, tokens in STREAM_GROUPS.items():
            for tok in tokens:
                if tok.lower() in s_lower:
                    groups.add(gid)
                    break
        return groups

    stream_to_groups = {s: groups_for_stream(s) for s in all_streams}

    # -----------------------------------------
    # USER DEFINES BATCHES AS COMBINATIONS OF GROUP IDS
    # -----------------------------------------

    raw_input_str = input(
        "\nEnter stream group batches (e.g., '1, 2, 3 4, 4'): "
    ).strip()

    if not raw_input_str:
        print("\nNo batches specified.")
        raise SystemExit

    # Example: "1, 2, 3 4, 4" -> ["1", "2", "3 4", "4"]
    batch_strings = [batch.strip() for batch in raw_input_str.split(",")]

    # Build list of required group-sets for each batch
    required_group_sets = []

    for batch in batch_strings:
        if not batch:
            continue

        if batch.lower() == "all":
            # special case: means "any group set is acceptable"
            # signal this with None and handle later
            required_group_sets.append(None)
            continue

        try:
            group_ids = [int(x) for x in batch.split()]
        except ValueError:
            print(f"\nInvalid group numbers in batch: '{batch}'")
            raise SystemExit

        invalid = [g for g in group_ids if g not in STREAM_GROUPS]
        if invalid:
            print(f"\nInvalid group IDs {invalid} in batch '{batch}'")
            raise SystemExit

        required_group_sets.append(frozenset(group_ids))

    # -----------------------------------------
    # DETERMINE WHICH STREAMS TO INCLUDE
    # -----------------------------------------

    # If any batch was 'all', then we just take all streams
    if any(rgs is None for rgs in required_group_sets):
        selected_streams = list(all_streams)
    else:
        # A stream is included if its group-set exactly matches
        # ANY of the batch group-sets
        valid_group_sets = set(required_group_sets)
        selected_streams = [
            s for s, gset in stream_to_groups.items()
            if frozenset(gset) in valid_group_sets
        ]

    selected_streams = sorted(selected_streams)

    if len(selected_streams) == 0:
        print("\nNo streams matched the specified group batches.")
        print("Available streams and their group-sets:")
        for s, gset in stream_to_groups.items():
            print(f"  {s}: groups {sorted(gset) if gset else '[]'}")
        raise SystemExit

    print("\nSelected streams to plot (across all batches):")
    for s in selected_streams:
        print("  ", s)

    # -----------------------------------------
    # BUILD FILTERED RESULTS AND PLOT
    # -----------------------------------------

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

    if len(filtered_results) == 0:
        print("\nNo matching streams found for the selected classifiers after filtering.")
        raise SystemExit

    print("\nPlotting selected results...")
    plot_f1_violin_by_stream(filtered_results, model_type, subfolder2="filtered")

    # # Median Results per Classifier Plot
    # for clf in classifiers:
    #     for stream in all_streams:
    #         if stream in f1_results_by_stream[clf]:
    #             print(f"Median F1 Score for {clf}, {stream}: {np.median(f1_results_by_stream[clf][stream])}")
    #
    # records = []
    # for clf in classifiers:
    #     for stream in all_streams:
    #         if stream in f1_results_by_stream[clf]:
    #             records.append({
    #                 "Classifier": clf,
    #                 "Stream": stream,
    #                 "Median F1 Score": np.median(f1_results_by_stream[clf][stream])
    #             })
    # df = pd.DataFrame(records)
    #
    # print(df)
    # plt.figure(figsize=(12, 6))
    # sns.lineplot(
    #     data=df,
    #     x='Stream',
    #     y='Median F1 Score',
    #     hue='Classifier',
    #     style='Classifier',
    #     markers=True,
    #     dashes=True
    # )
    # plt.xticks(rotation=15, ha='right')
    # plt.grid(True, alpha=0.5)
    # plt.tight_layout()
    # plt.show()



if preference == 9:
    # Preference to plot overlap of features ONLY
    # Agnostic to how every many classifiers we want to look at
    # Just ensure that saved hyperparameters and selected features are in JSON format
    model_type = ['complete', 'explicit']

    # Try with all however many classifiers
    investigate_feature_space(model_type, ['KNN', 'EGB', 'RF'])


if preference == 10:
    # Save Hyperparameters to JSON
    model_type = ['noAFE', 'explicit']
    classifier = 'logreg'
    get_median_hyperparameters(classifier,get_model_subfolder(model_type))

if preference == 11:
    """
    Inspect previously saved F1 results from preference 7.
    Allows:
        - Loading all saved .pkl files
        - Viewing classifiers independently
        - Re-plotting subsets of the results
    """

    # Define model type and locate the folder where preference 7 saved results
    model_type = ['complete', 'explicit']
    subfolder = get_model_subfolder(model_type)
    results_folder = os.path.join('./feature_study', subfolder, 'streamwise')

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
                    f1_results_by_stream[clf][stream_str] = pickle.load(f)

    # -------------------------------
    # INTERACTIVE SELECTION SECTION
    # -------------------------------

    # Show available classifiers to the user
    print("\nAvailable classifiers:", classifiers)

    # User selects one or more classifiers (or "all")
    clf_choice = input(
        "Enter classifier(s) to inspect (e.g., 'KNN', 'KNN RF', 'KNN,EGB', or 'all'): "
    ).strip()

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
    all_streams = {
        stream
        for clf in selected_classifiers
        for stream in f1_results_by_stream[clf].keys()
    }

    if len(all_streams) == 0:
        print("\nNo streams found for the selected classifiers.")
        raise SystemExit

    # -----------------------------------------
    # SORT STREAMS BY MEDIAN F1 SCORE
    # -----------------------------------------
    stream_median_map = {}

    for stream in all_streams:
        combined_scores = []
        for clf in selected_classifiers:
            # Check if this classifier has data for this specific stream
            if stream in f1_results_by_stream[clf]:
                combined_scores.extend(f1_results_by_stream[clf][stream])

        if combined_scores:
            stream_median_map[stream] = np.median(combined_scores)

    # Sort streams by median
    sorted_streams = sorted(stream_median_map, key=stream_median_map.get, reverse=True)

    # -----------------------------------------
    # BUILD FILTERED RESULTS AND PLOT
    # -----------------------------------------

    # Build a filtered dictionary containing only the selected classifiers and streams
    filtered_results = {
        clf: {
            # stream: f1_results_by_stream[clf][stream]
            stream.replace('ECG-HR-BR-Temperature', 'Equivital').replace('Participant', 'Demographics').replace('Centrifuge','G Force'): f1_results_by_stream[clf][stream]
            for stream in sorted_streams
            if stream in f1_results_by_stream[clf]
        }
        for clf in selected_classifiers
    }

    # Remove classifiers that ended up with no matching streams
    filtered_results = {clf: d for clf, d in filtered_results.items() if len(d) > 0}

    if len(filtered_results) == 0:
        print("\nNo matching streams found for the selected classifiers after filtering.")
        raise SystemExit

    print("\nPlotting selected results...")
    # plot_f1_violin_by_stream(filtered_results, model_type, subfolder2="filtered")
    test_violin(filtered_results, model_type, subfolder2="filtered")