from pathlib import Path
import pickle
import shap
import numpy as np
import os
import matplotlib.pyplot as plt


### SETTINGS
# Choose classifier
classifier = "EGB"

# Choose filepath
file_folder = os.path.join(
    "../..",
    "feature_study",
    "Explicit Complete",
    "feature_importance",
    "shap_all_9may"
)

### CREATE DICTIONARIES FOR CUSTOM PLOTS
# All Raw EEG Channels
channels_dict = {"F1": "F1", "Fz": "Fz", "F3": "F3", "FC3": "FC3", "FT9": "FT9", "FC5": "FC5", "FC1": "FC1", "C3": "C3",
    "C5": "C5", "TP9": "TP9", "CP5": "CP5", "CP1": "CP1", "Pz": "Pz", "P3": "P3", "P7": "P7", "P5": "P5", "P1": "P1",
    "P6": "P6", "P4": "P4", "P8": "P8", "TP10": "TP10", "CP6": "CP6", "CP2": "CP2", "Cz": "Cz", "C4": "C4", "T8": "T8",
    "FT10": "FT10", "FC6": "FC6", "FC2": "FC2", "AF4": "AF4", "FC4": "FC4", "AFz": "AFz", "O2": "O2", "F4": "F4",
    "O1": "O1", "T7": "T7"}

# EEG Bands
bands_dict = {"delta": "delta", "theta": "theta", "alpha": "alpha", "beta": "beta"}

# Baseline Methods
baseline_dict = {"v0": "v0", "v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4", "v5": "v5", "v6": "v6", "v7": "v7",
    "v8": "v8"}

# Data Modalities
modalities_dict = {"Equivital": "Equivital", "HRV": "Equivital", "EEG": "EEG", "Centrifuge": "G-Force",
    "Pupil": "Eye Tracking", "Participant": "Demographics"}

# EEG Channels with Bands
channels_bands_dict = {}

for electrode in channels_dict.values():
    for band in bands_dict.values():

        combined_name = f"{electrode}_{band} "
        raw_channel = f"{electrode} "

        # search term : displayed name
        channels_bands_dict[f"{electrode}_{band}"] = combined_name
        channels_dict[f"{electrode}"] = raw_channel

# EEG raw data and channels with bands
raw_processed_dict = channels_bands_dict | channels_dict

# EEG Raw vs PSD
raw_vs_psd_dict = {}

for electrode in channels_dict.keys():
    raw_vs_psd_dict[f"{electrode} "] = f"{electrode}_Raw"

    for band in bands_dict.keys():
        raw_vs_psd_dict[f"{electrode}_{band} "] = f"{electrode}_PSD"

def load_explanations(classifier, file_folder, num_folds=10):

    explanations = []

    for i in range(1, num_folds + 1):

        try:
            filename = os.path.join(file_folder, f"shap_explanation_{classifier}_fold_{i}.pkl")
            print(f"Loading {filename}")

            with open(filename, "rb") as f:
                exp = pickle.load(f)
        except Exception as e:
            print(f"No file found for {classifier} classifier: {e}")
            exit()

        explanations.append(exp)

    print("Loading Complete")

    combined_explanation = shap.Explanation(
        values=np.concatenate([e.values for e in explanations], axis=0),
        base_values=np.concatenate([e.base_values for e in explanations], axis=0),
        data=np.concatenate([e.data for e in explanations], axis=0),
        feature_names=explanations[0].feature_names
    )

    if classifier == "KNN" or classifier == "RF":
        combined_explanation = shap.Explanation(
            values=combined_explanation.values[:, :, 1],
            base_values=combined_explanation.base_values[:, 1],
            data=combined_explanation.data,
            feature_names=combined_explanation.feature_names
        )

    return combined_explanation, explanations


def plot_original_violin(explanation, classifier):

    plt.figure(figsize=(12, 8))

    shap.plots.violin(
        explanation,
        max_display=20,
        show=False
    )

    plt.title(f"Feature Importance - {classifier} Explicit Complete")

    plt.subplots_adjust(left=0.35)

    plt.show()

def plot_keyword_violin_dict(explanation, keyword_map, match_prefix=False):
    n_samples = explanation.values.shape[0]
    match_counts = {}

    # 1. Get unique display names in the order they appear in the map
    # We use dict.fromkeys to keep order while getting unique values
    unique_display_names = list(dict.fromkeys(keyword_map.values()))

    # 2. Initialize arrays for the grouped data
    grouped_values = np.zeros((n_samples, len(unique_display_names)))
    grouped_data = np.zeros((n_samples, len(unique_display_names)))

    for i, display_name in enumerate(unique_display_names):
        # Find ALL keywords that map to this specific display name
        associated_keywords = [k for k, v in keyword_map.items() if v == display_name]

        # Find indices of all features matching ANY of these keywords
        # We use a set to prevent double-counting if a feature matches two keywords
        matching_indices = set()
        for kw in associated_keywords:
            if match_prefix:
                indices = [
                    j for j, name in enumerate(explanation.feature_names)
                    if name.lower().startswith(f"{kw.lower()}")
                ]
            else:
                indices = [
                    j for j, name in enumerate(explanation.feature_names)
                    if kw.lower() in name.lower()
                ]
            matching_indices.update(indices)

        idx_list = list(matching_indices)

        match_counts[display_name] = len(idx_list)

        if not idx_list:
            continue

        # Sum the SHAP values
        grouped_values[:, i] = np.sum(explanation.values[:, idx_list], axis=1)
        print(f"{display_name} Mean: {np.mean(np.sum(np.abs(explanation.values[:, idx_list]), axis=1))}")

        # Take mean of the feature data
        grouped_data[:, i] = np.mean(explanation.data[:, idx_list], axis=1)

    # 3. Create the new Explanation object
    grouped_explanation = shap.Explanation(
        values=grouped_values,
        data=grouped_data,
        feature_names=unique_display_names
    )

    print("\n--- Feature Matching Info ---")
    for i, name in enumerate(unique_display_names):
        vals = grouped_values[:, i]
        print(
            f"{name} ---",
            "Matches:", match_counts.get(name, 0),
            "Min:", np.min(vals),
            "Max:", np.max(vals),
            "Std:", np.std(vals),
            "Unique:", len(np.unique(vals))
        )

    if sum(match_counts.values()) == 0:
        print("No matches for keywords\n")
        return

    # 4. Plotting
    try:
        plt.figure(figsize=(10, 8))
        shap.plots.violin(
            grouped_explanation,
            plot_type="violin",
            show=False
        )

        # Move title and formatting here
        plt.title(f"Feature Importance - {classifier} Explicit Complete")
        plt.subplots_adjust(left=0.3)
        plt.show()

    except Exception as e:
        print(f"Unable to Create Violin Plot:{e}")
        plt.close()

def score_by_fold(explanations, keyword_map, target_group):
    fold_sum_scores = []
    fold_mean_scores = []

    for exp in explanations:
        values = exp.values[:, :, 1] if len(exp.values.shape) == 3 else exp.values
        feature_names = exp.feature_names

        keywords = [k for k, v in keyword_map.items() if v == target_group]

        idx = set()
        for kw in keywords:
            idx.update([
                j for j, name in enumerate(feature_names)
                if kw.lower() in name.lower()
            ])

        idx = sorted(list(idx))

        sum_score = np.sum(np.sum(np.abs(values[:, idx]), axis=1))
        mean_score = np.mean(np.sum(np.abs(values[:, idx]), axis=1))
        fold_sum_scores.append(sum_score)
        fold_mean_scores.append(mean_score)

    return np.array(fold_mean_scores), np.array(fold_sum_scores)

def plot_all_shap(
    subfolder_name,
    fold_num,
    classifier="KNN",
    plot_types=("violin", "beeswarm", "bar"),
    base_dir=None,
    explanation=None,
):
    """
    Load a SHAP Explanation object from a pickle file and generate plots.

    Parameters
    ----------
    subfolder_name : str
        Name of the subfolder (e.g., 'Explicit Complete')
    fold_num : int
        Fold number to load
    classifier : str
        Model name (default: 'KNN')
    plot_types : tuple
        Which plots to generate ('violin', 'beeswarm', 'bar')
    base_dir : Path or None
        Base directory; defaults to two levels up from this file
    """

    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]

    results_folder = base_dir / "feature_study" / subfolder_name / "streamwise"

    data_file_path = results_folder / f"shap_{classifier}_f{fold_num}.pkl"

    # # Load explanation
    # with open(data_file_path, "rb") as f:
    #     explanation = pickle.load(f)

    # Slice class 1
    if len(explanation.values.shape) == 3:
        class1_exp = explanation[:, :, 1]
    else:
        class1_exp = explanation

    # Reattach feature names (important after pickle load)
    class1_exp.feature_names = explanation.feature_names

    # Plot mapping
    plot_funcs = {
        "violin": shap.plots.violin,
        "beeswarm": shap.plots.beeswarm,
        "bar": shap.plots.bar,
    }

    for plot_type in plot_types:
        if plot_type not in plot_funcs:
            print(f"Skipping unknown plot type: {plot_type}")
            continue

        try:
            plt.figure()
            plot_funcs[plot_type](class1_exp, show=False)

            save_path = results_folder / f"shap_{classifier}_{plot_type}_plot_f{fold_num}.png"
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

            print(f"Saved {plot_type} plot to: {save_path}")

        except Exception as e:
            print(f"Skipping {plot_type} plot due to error: {e}")
            plt.close()

def main():

    # Load Explanations
    explanation, fold_explanations = load_explanations(classifier, file_folder)

    # Look at by fold mean and sum
    modalities = ["Equivital", "EEG", "Eye Tracking", "Demographics", "G-Force"]
    for mode in modalities:
        mean_by_fold, sum_by_fold = score_by_fold(
            fold_explanations,
            modalities_dict,
            mode
        )

        print(f"{mode} Mean by fold:", mean_by_fold)
        print("Across Fold Mean:", np.mean(mean_by_fold))
        print("Mean Std:", np.std(mean_by_fold, ddof=1))
        print(f"{mode} Sum by fold:", sum_by_fold)
        print("Total Sum:", np.sum(sum_by_fold))
        print("Sum Std:", np.std(sum_by_fold, ddof=1))

    # Plot all features
    plot_original_violin(explanation, classifier)
    # Plot modalities
    plot_keyword_violin_dict(explanation, modalities_dict)
    # Plot baseline methods
    plot_keyword_violin_dict(explanation, baseline_dict)

    # EEG Plots
    # Plot EEG Channels
    plot_keyword_violin_dict(explanation, channels_dict, True)
    # Plot EEG Bands No Channels
    plot_keyword_violin_dict(explanation, bands_dict)
    # Plot EEG Channels and Bands
    plot_keyword_violin_dict(explanation, channels_bands_dict, True)
    # Plot EEG Raw Data And Processed
    plot_keyword_violin_dict(explanation, raw_processed_dict, True)
    # Plot Raw Vs PSD
    plot_keyword_violin_dict(explanation, raw_vs_psd_dict, True)


if __name__ == "__main__":
    main()