from pathlib import Path
from datetime import datetime
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import shap


# =============================================================================
# SETTINGS
# =============================================================================

CLASSIFIER = "EGB"

FILE_FOLDER = (
    Path("../..")
    / "feature_study"
    / "Explicit Complete"
    / "feature_importance"
    / "shap_all_10CV_3may"
)


# =============================================================================
# DICTIONARY CREATION
# =============================================================================

def create_keyword_dictionaries():
    """Create all keyword dictionaries used for grouped SHAP plots."""

    channels_dict = {
        "F1": "F1", "Fz": "Fz", "F3": "F3", "FC3": "FC3", "FT9": "FT9",
        "FC5": "FC5", "FC1": "FC1", "C3": "C3", "C5": "C5", "TP9": "TP9",
        "CP5": "CP5", "CP1": "CP1", "Pz": "Pz", "P3": "P3", "P7": "P7",
        "P5": "P5", "P1": "P1", "P6": "P6", "P4": "P4", "P8": "P8",
        "TP10": "TP10", "CP6": "CP6", "CP2": "CP2", "Cz": "Cz", "C4": "C4",
        "T8": "T8", "FT10": "FT10", "FC6": "FC6", "FC2": "FC2", "AF4": "AF4",
        "FC4": "FC4", "AFz": "AFz", "O2": "O2", "F4": "F4", "O1": "O1",
        "T7": "T7",
    }

    bands_dict = {
        "delta": "delta",
        "theta": "theta",
        "alpha": "alpha",
        "beta": "beta",
    }

    baseline_dict = {
        "v0": "v0", "v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4",
        "v5": "v5", "v6": "v6", "v7": "v7", "v8": "v8",
    }

    modalities_dict = {
        "Equivital": "Equivital",
        "HRV": "Equivital",
        "EEG": "EEG",
        "Centrifuge": "G-Force",
        "Pupil": "Eye Tracking",
        "Participant": "Demographics",
    }

    channels_bands_dict = {}

    for electrode in channels_dict.values():
        for band in bands_dict.values():
            combined_name = f"{electrode}_{band} "
            raw_channel = f"{electrode} "

            channels_bands_dict[f"{electrode}_{band}"] = combined_name
            channels_dict[electrode] = raw_channel

    raw_processed_dict = channels_bands_dict | channels_dict

    raw_vs_psd_dict = {}

    for electrode in channels_dict.keys():
        raw_vs_psd_dict[f"{electrode} "] = f"{electrode}_Raw"

        for band in bands_dict.keys():
            raw_vs_psd_dict[f"{electrode}_{band} "] = f"{electrode}_PSD"

    return {
        "channels": channels_dict,
        "bands": bands_dict,
        "baseline": baseline_dict,
        "modalities": modalities_dict,
        "channels_bands": channels_bands_dict,
        "raw_processed": raw_processed_dict,
        "raw_vs_psd": raw_vs_psd_dict,
    }


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(output_folder, classifier):
    """Create timestamped log file and configure console/file logging."""

    log_folder = Path(output_folder) / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_folder / f"{classifier}_shap_log_{timestamp}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Log file created: {log_path}")
    return log_path


# =============================================================================
# SHAP LOADING / PROCESSING
# =============================================================================

def load_explanations(classifier, file_folder, num_folds=10):
    """Load fold-level SHAP explanations and combine them into one Explanation."""

    explanations = []

    for fold_idx in range(1, num_folds + 1):
        filename = Path(file_folder) / f"shap_explanation_{classifier}_fold_{fold_idx}.pkl"
        logging.info(f"Loading {filename}")

        try:
            with open(filename, "rb") as file:
                explanation = pickle.load(file)

        except Exception as error:
            logging.info(f"No file found for {classifier} classifier: {error}")
            raise SystemExit(1)

        explanations.append(explanation)

    logging.info("Loading Complete\n")

    combined_explanation = shap.Explanation(
        values=np.concatenate([exp.values for exp in explanations], axis=0),
        base_values=np.concatenate([exp.base_values for exp in explanations], axis=0),
        data=np.concatenate([exp.data for exp in explanations], axis=0),
        feature_names=explanations[0].feature_names,
    )

    if classifier in {"KNN", "RF"}:
        combined_explanation = shap.Explanation(
            values=combined_explanation.values[:, :, 1],
            base_values=combined_explanation.base_values[:, 1],
            data=combined_explanation.data,
            feature_names=combined_explanation.feature_names,
        )

    return combined_explanation, explanations


def get_class_1_explanation(explanation):
    """Return class-1 explanation if explanation has class dimension."""

    if len(explanation.values.shape) == 3:
        class1_explanation = explanation[:, :, 1]
    else:
        class1_explanation = explanation

    class1_explanation.feature_names = explanation.feature_names
    return class1_explanation


# =============================================================================
# MATCHING / SCORING HELPERS
# =============================================================================

def get_keywords_for_display_name(keyword_map, display_name):
    """Return all keywords mapped to a given display name."""

    return [keyword for keyword, value in keyword_map.items() if value == display_name]


def get_matching_feature_indices(feature_names, keywords, match_prefix=False):
    """Find feature indices matching any keyword."""

    matching_indices = set()

    for keyword in keywords:
        keyword_lower = keyword.lower()

        for idx, feature_name in enumerate(feature_names):
            feature_name_lower = feature_name.lower()

            if match_prefix:
                is_match = feature_name_lower.startswith(keyword_lower)
            else:
                is_match = keyword_lower in feature_name_lower

            if is_match:
                matching_indices.add(idx)

    return list(matching_indices)


def score_by_fold(explanations, keyword_map, target_group):
    """Calculate fold-level grouped SHAP scores for a target group."""

    fold_sum_scores = []
    fold_mean_scores = []
    fold_norm_mean_scores = []

    for explanation in explanations:
        values = (
            explanation.values[:, :, 1]
            if len(explanation.values.shape) == 3
            else explanation.values
        )

        feature_names = explanation.feature_names
        keywords = get_keywords_for_display_name(keyword_map, target_group)
        indices = get_matching_feature_indices(feature_names, keywords)
        indices = sorted(indices)

        sum_score = np.sum(np.sum(np.abs(values[:, indices]), axis=1))
        mean_score = np.mean(np.sum(np.abs(values[:, indices]), axis=1))
        norm_mean_score = np.mean(np.abs(values[:, indices]), axis=1)

        fold_sum_scores.append(sum_score)
        fold_mean_scores.append(mean_score)
        fold_norm_mean_scores.append(norm_mean_score)

    return (
        np.array(fold_mean_scores),
        np.array(fold_norm_mean_scores),
        np.array(fold_sum_scores),
    )


# =============================================================================
# PLOTTING
# =============================================================================

def plot_original_violin(explanation, classifier, print_vals=False):
    """Plot the original SHAP violin plot."""

    if print_vals:
        logging.info("--- Feature Importance Sum of Effect Magnitudes ---")

        feature_scores = []

        for idx in range(explanation.shape[1]):
            score = np.sum(np.abs(explanation.values[:, idx]))
            feature_scores.append((explanation.feature_names[idx], score))

        feature_scores = sorted(feature_scores, key=lambda item: item[1], reverse=True)

        for name, score in feature_scores:
            logging.info(f"{name}: {score}")

    plt.figure(figsize=(12, 8))

    shap.plots.violin(
        explanation,
        max_display=20,
        show=False,
    )

    plt.title(f"Feature Importance - {classifier} Explicit Complete")
    plt.tight_layout()
    plt.show()


def build_grouped_explanation(explanation, keyword_map, match_prefix=False):
    """Create a grouped SHAP Explanation object from a keyword map."""

    n_samples = explanation.values.shape[0]
    unique_display_names = list(dict.fromkeys(keyword_map.values()))

    grouped_values = np.zeros((n_samples, len(unique_display_names)))
    grouped_norm_mean_values = np.zeros((n_samples, len(unique_display_names)))
    grouped_mean_values = np.zeros((n_samples, len(unique_display_names)))
    grouped_sum_values = np.zeros((n_samples, len(unique_display_names)))
    grouped_data = np.zeros((n_samples, len(unique_display_names)))

    match_counts = {}

    for group_idx, display_name in enumerate(unique_display_names):
        keywords = get_keywords_for_display_name(keyword_map, display_name)
        indices = get_matching_feature_indices(
            explanation.feature_names,
            keywords,
            match_prefix=match_prefix,
        )

        match_counts[display_name] = len(indices)

        if not indices:
            continue

        grouped_values[:, group_idx] = np.sum(explanation.values[:, indices], axis=1)
        grouped_norm_mean_values[:, group_idx] = np.mean(
            np.abs(explanation.values[:, indices]),
            axis=1,
        )
        grouped_sum_values[:, group_idx] = np.sum(
            np.sum(np.abs(explanation.values[:, indices]), axis=1)
        )
        grouped_mean_values[:, group_idx] = np.mean(
            np.sum(np.abs(explanation.values[:, indices]), axis=1)
        )
        grouped_data[:, group_idx] = np.mean(explanation.data[:, indices], axis=1)

    log_feature_matching_info(
        unique_display_names,
        match_counts,
        grouped_norm_mean_values,
        grouped_mean_values,
        grouped_sum_values,
    )

    if sum(match_counts.values()) == 0:
        logging.info("No matches for keywords\n")
        return None

    matched_indices = [
        idx
        for idx, name in enumerate(unique_display_names)
        if match_counts.get(name, 0) > 0
    ]

    matched_display_names = [
        name
        for name in unique_display_names
        if match_counts.get(name, 0) > 0
    ]

    return shap.Explanation(
        values=grouped_values[:, matched_indices],
        data=grouped_data[:, matched_indices],
        feature_names=matched_display_names,
    )


def log_feature_matching_info(
    unique_display_names,
    match_counts,
    grouped_norm_mean_values,
    grouped_mean_values,
    grouped_sum_values,
):
    """Log feature matching counts and grouped SHAP summary statistics."""

    logging.info("\n--- Feature Matching Info ---")

    for idx, name in enumerate(unique_display_names):
        norm_means = grouped_norm_mean_values[:, idx]
        means = grouped_mean_values[:, idx]
        sums = grouped_sum_values[:, idx]

        if match_counts.get(name, 0) > 0:
            logging.info(f"-- {name} --")
            logging.info(f"Matches: {match_counts.get(name, 0)}")
            logging.info(
                "Mean of Effect Magnitude Normalized by Number of Features -- "
                f"Mean: {norm_means.mean()}, Std: {norm_means.std()}"
            )
            logging.info(
                "Mean of Effect Magnitude -- "
                f"Mean: {means.mean()}, Std: {means.std()}"
            )
            logging.info(
                "Sum of Effect Magnitude -- "
                f"Sum: {sums.mean()}, Std: {sums.std()}"
            )
        else:
            logging.info(f"-- No matches for {name} --")


def plot_keyword_violin_dict(explanation, keyword_map, classifier, match_prefix=False):
    """Create and plot grouped SHAP violin plot based on keyword map."""

    grouped_explanation = build_grouped_explanation(
        explanation,
        keyword_map,
        match_prefix=match_prefix,
    )

    if grouped_explanation is None:
        return

    try:
        plt.figure(figsize=(10, 8))

        shap.plots.bar(
            grouped_explanation,
            show=False,
        )

        plt.title(f"Feature Importance - {classifier} Explicit Complete")
        plt.tight_layout()
        plt.show()

    except Exception as error:
        logging.info(f"Unable to Create Bar Plot:{error}")
        plt.close()

def plot_all_shap(
    subfolder_name,
    fold_num,
    classifier="KNN",
    plot_types=("violin", "beeswarm", "bar"),
    base_dir=None,
    explanation=None,
):
    """Generate SHAP violin, beeswarm, and/or bar plots for one explanation."""

    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]

    results_folder = Path(base_dir) / "feature_study" / subfolder_name / "streamwise"

    class1_explanation = get_class_1_explanation(explanation)

    plot_functions = {
        "violin": shap.plots.violin,
        "beeswarm": shap.plots.beeswarm,
        "bar": shap.plots.bar,
    }

    for plot_type in plot_types:
        if plot_type not in plot_functions:
            logging.info(f"Skipping unknown plot type: {plot_type}")
            continue

        try:
            plt.figure()
            plot_functions[plot_type](class1_explanation, show=False)

            save_path = results_folder / f"shap_{classifier}_{plot_type}_plot_f{fold_num}.png"

            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

            logging.info(f"Saved {plot_type} plot to: {save_path}")

        except Exception as error:
            logging.info(f"Skipping {plot_type} plot due to error: {error}")
            plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    setup_logging(FILE_FOLDER, CLASSIFIER)

    keyword_dicts = create_keyword_dictionaries()

    explanation, fold_explanations = load_explanations(CLASSIFIER, FILE_FOLDER)

    plot_original_violin(explanation, CLASSIFIER, print_vals=False)

    plot_keyword_violin_dict(explanation, keyword_dicts["modalities"], CLASSIFIER)

    plot_keyword_violin_dict(explanation, keyword_dicts["baseline"], CLASSIFIER)

    plot_keyword_violin_dict(explanation, keyword_dicts["channels"], CLASSIFIER, match_prefix=True)

    plot_keyword_violin_dict(explanation, keyword_dicts["bands"], CLASSIFIER)

    plot_keyword_violin_dict(explanation, keyword_dicts["channels_bands"], CLASSIFIER, match_prefix=True)

    plot_keyword_violin_dict(explanation, keyword_dicts["raw_processed"], CLASSIFIER, match_prefix=True)

    plot_keyword_violin_dict(explanation, keyword_dicts["raw_vs_psd"], CLASSIFIER, match_prefix=True)

    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()