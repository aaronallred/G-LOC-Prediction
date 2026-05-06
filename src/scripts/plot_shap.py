import pickle
import shap
import numpy as np
import os
import matplotlib.pyplot as plt


### SETTINGS
classifier = "RF"

file_folder = os.path.join(
    "../..",
    "feature_study",
    "Explicit Complete",
    "feature_importance",
    "shap_rf_noeeg_6may"
)

channels_dict = {"F1": "F1", "Fz": "Fz", "F3": "F3", "FC3": "FC3", "FT9": "FT9", "FC5": "FC5", "FC1": "FC1", "C3": "C3",
    "C5": "C5", "TP9": "TP9", "CP5": "CP5", "CP1": "CP1", "Pz": "Pz", "P3": "P3", "P7": "P7", "P5": "P5", "P1": "P1",
    "P6": "P6", "P4": "P4", "P8": "P8", "TP10": "TP10", "CP6": "CP6", "CP2": "CP2", "Cz": "Cz", "C4": "C4", "T8": "T8",
    "FT10": "FT10", "FC6": "FC6", "FC2": "FC2", "AF4": "AF4", "FC4": "FC4", "AFz": "AFz", "O2": "O2", "F4": "F4",
    "O1": "O1", "T7": "T7"}

bands_dict = {"delta": "delta", "theta": "theta", "alpha": "alpha", "beta": "beta"}

baseline_dict = {"v0": "v0", "v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4", "v5": "v5", "v6": "v6", "v7": "v7",
    "v8": "v8"}

modalities_dict = {"Equivital": "Equivital", "HRV": "Equivital", "EEG": "EEG", "Centrifuge": "G-Force",
    "Pupil": "Eye Tracking", "Participant": "Demographics"}

channels_bands_dict = {}

for electrode in channels_dict.values():
    for band in bands_dict.values():

        combined_name = f"{electrode}_{band}"

        # search term : displayed name
        channels_bands_dict[f"{electrode}_{band}"] = combined_name


def load_explanations(classifier, file_folder, num_folds=10):

    explanations = []

    for i in range(1, num_folds + 1):

        filename = os.path.join(
            file_folder,
            f"shap_explanation_{classifier}_fold_{i}.pkl"
        )

        with open(filename, "rb") as f:
            exp = pickle.load(f)

        explanations.append(exp)

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

    return combined_explanation


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


def plot_keyword_violin(explanation, keywords):

    grouped_values = []
    grouped_data = []
    grouped_names = []

    for keyword in keywords:

        idx = [
            i for i, name in enumerate(explanation.feature_names)
            if keyword.lower() in name.lower()
        ]

        if len(idx) == 0:
            continue

        shap_concat = explanation.values[:, idx].flatten()

        data_concat = explanation.data[:, idx].flatten()

        grouped_values.append(shap_concat)
        grouped_data.append(data_concat)
        grouped_names.append(keyword)

    min_len = min(len(x) for x in grouped_values)

    grouped_values = np.column_stack([
        x[:min_len] for x in grouped_values
    ])

    grouped_data = np.column_stack([
        x[:min_len] for x in grouped_data
    ])

    grouped_explanation = shap.Explanation(
        values=grouped_values,
        data=grouped_data,
        feature_names=grouped_names
    )

    plt.figure(figsize=(10, 8))

    shap.plots.violin(
        grouped_explanation,
        plot_type="violin",
        show=False
    )

    plt.title(f"Feature Importance - {classifier} Explicit Complete")

    plt.subplots_adjust(left=0.3)

    plt.show()

def plot_keyword_violin_dict(explanation, keyword_map):

    grouped_dict = {}

    for keyword, display_name in keyword_map.items():

        idx = [
            i for i, name in enumerate(explanation.feature_names)
            if keyword.lower() in name.lower()
        ]

        if len(idx) == 0:
            continue

        shap_concat = explanation.values[:, idx].flatten()

        data_concat = explanation.data[:, idx].flatten()

        if display_name not in grouped_dict:

            grouped_dict[display_name] = {
                "values": [],
                "data": []
            }

        grouped_dict[display_name]["values"].append(shap_concat)

        grouped_dict[display_name]["data"].append(data_concat)

    grouped_values = []
    grouped_data = []
    grouped_names = []

    for display_name, content in grouped_dict.items():

        combined_values = np.concatenate(content["values"])

        combined_data = np.concatenate(content["data"])

        grouped_values.append(combined_values)

        grouped_data.append(combined_data)

        grouped_names.append(display_name)

    min_len = min(len(x) for x in grouped_values)

    grouped_values = np.column_stack([
        x[:min_len] for x in grouped_values
    ])

    grouped_data = np.column_stack([
        x[:min_len] for x in grouped_data
    ])

    grouped_explanation = shap.Explanation(
        values=grouped_values,
        data=grouped_data,
        feature_names=grouped_names
    )

    plt.figure(figsize=(10, 8))

    shap.plots.violin(
        grouped_explanation,
        plot_type="violin",
        show=False
    )

    plt.title(f"Feature Importance - {classifier} Explicit Complete")

    plt.subplots_adjust(left=0.3)

    plt.show()


def plot_keyword_violin_dict_new(explanation, keyword_map):
    n_samples = explanation.values.shape[0]

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
            indices = [
                j for j, name in enumerate(explanation.feature_names)
                if kw.lower() in name.lower()
            ]
            matching_indices.update(indices)

        idx_list = list(matching_indices)

        if not idx_list:
            continue

        # Sum the SHAP values
        grouped_values[:, i] = np.sum(explanation.values[:, idx_list], axis=1)

        # Take mean of the feature data
        grouped_data[:, i] = np.mean(explanation.data[:, idx_list], axis=1)

    # 3. Create the new Explanation object
    grouped_explanation = shap.Explanation(
        values=grouped_values,
        data=grouped_data,
        feature_names=unique_display_names
    )

    # 4. Plotting
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

def main():

    explanation = load_explanations(classifier, file_folder)

    plot_original_violin(explanation, classifier)

    plot_keyword_violin_dict_new(explanation, modalities_dict)
    exit()

    plot_keyword_violin_dict_new(explanation, channels_dict)

    plot_keyword_violin_dict_new(explanation, bands_dict)

    plot_keyword_violin_dict_new(explanation, baseline_dict)

    plot_keyword_violin_dict_new(explanation, channels_bands_dict)


if __name__ == "__main__":
    main()