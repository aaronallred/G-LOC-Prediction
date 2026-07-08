"""Grouped SHAP bar plotting helpers.

This module is intended to sit next to ``shap_analysis.py`` as:

    src/modes/shap_grouped_plots.py

It does not load models or SHAP explanations. Instead, it takes an existing
``shap.Explanation`` object, groups features by keyword dictionaries, and saves
bar plots for those grouped feature categories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import shap

logger = logging.getLogger(__name__)

ScoreMode = Literal["sum_abs", "mean_abs"]

DEFAULT_GROUPED_PLOT_TYPES: tuple[str, ...] = (
    "modalities",
    "baseline",
    "channels",
    "bands",
    "channels_bands",
    "raw_processed",
    "raw_vs_psd",
)

MATCH_PREFIX_BY_PLOT_TYPE: dict[str, bool] = {
    "modalities": False,
    "baseline": False,
    "channels": True,
    "bands": False,
    "channels_bands": True,
    "raw_processed": True,
    "raw_vs_psd": True,
}

PLOT_TITLE_BY_PLOT_TYPE: dict[str, str] = {
    "modalities": "Feature Group Importance by Modality",
    "baseline": "Feature Group Importance by Baseline Window",
    "channels": "Feature Group Importance by EEG Channel",
    "bands": "Feature Group Importance by EEG Band",
    "channels_bands": "Feature Group Importance by EEG Channel and Band",
    "raw_processed": "Feature Group Importance by Raw/Processed Feature",
    "raw_vs_psd": "Feature Group Importance by Raw vs PSD",
}


def create_keyword_dictionaries() -> dict[str, dict[str, str]]:
    """Create keyword dictionaries used for grouped SHAP bar plots.

    The returned dictionaries map feature-name keywords to display labels.
    For example, multiple keywords can map to the same display label when they
    should be grouped together in one bar.
    """

    electrode_names = [
        "F1", "Fz", "F3", "FC3", "FT9", "FC5", "FC1", "C3", "C5",
        "TP9", "CP5", "CP1", "Pz", "P3", "P7", "P5", "P1", "P6",
        "P4", "P8", "TP10", "CP6", "CP2", "Cz", "C4", "T8", "FT10",
        "FC6", "FC2", "AF4", "FC4", "AFz", "O2", "F4", "O1", "T7",
    ]

    band_names = ["delta", "theta", "alpha", "beta"]

    channels_dict = {electrode: electrode for electrode in electrode_names}
    bands_dict = {band: band for band in band_names}
    baseline_dict = {f"v{idx}": f"v{idx}" for idx in range(9)}

    modalities_dict = {
        "ECG": "Equivital",
        "HR": "Equivital",
        "BR": "Equivital",
        "Temperature": "Equivital",
        "HRV": "Equivital",
        "Equivital": "Equivital",
        "EEG": "EEG",
        "Centrifuge": "G-Force",
        "Pupil": "Eye Tracking",
        "Participant": "Demographics",
    }

    channels_bands_dict = {
        f"{electrode}_{band}": f"{electrode}_{band}"
        for electrode in electrode_names
        for band in band_names
    }

    # This mirrors the old raw_processed plot: individual raw channel groups plus
    # individual channel-band groups. The name is retained for continuity with
    # your existing analysis terminology.
    raw_processed_dict = channels_bands_dict | channels_dict

    # This mirrors the old raw-vs-PSD matching convention. The trailing spaces are
    # intentional because the older feature names appeared to use them to separate
    # raw channel names from subsequent feature descriptors.
    raw_vs_psd_dict: dict[str, str] = {}
    for electrode in electrode_names:
        raw_vs_psd_dict[f"{electrode} "] = f"{electrode}_Raw"
        for band in band_names:
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


def select_class_explanation(explanation, class_index: int = 1):
    """Return a 2D SHAP explanation for plotting.

    Binary/multiclass SHAP explanations often have shape
    ``(n_samples, n_features, n_classes)``. In that case, this function selects
    the configured class. Already-2D explanations are returned unchanged.
    """

    values = np.asarray(explanation.values)

    if values.ndim == 2:
        return explanation

    if values.ndim == 3:
        if class_index < 0 or class_index >= values.shape[2]:
            raise ValueError(
                f"class_index={class_index} is out of bounds for SHAP values "
                f"with {values.shape[2]} classes."
            )

        class_explanation = explanation[:, :, class_index]

        if class_explanation.feature_names is None and explanation.feature_names is not None:
            class_explanation.feature_names = explanation.feature_names

        return class_explanation

    raise ValueError(f"Unexpected SHAP values shape: {values.shape}")


def get_keywords_for_display_name(
    keyword_map: dict[str, str],
    display_name: str,
) -> list[str]:
    """Return all keywords that map to a given display name."""

    return [
        keyword
        for keyword, mapped_display_name in keyword_map.items()
        if mapped_display_name == display_name
    ]


def get_matching_feature_indices(
    feature_names: Sequence[str],
    keywords: Sequence[str],
    match_prefix: bool = False,
) -> list[int]:
    """Find feature indices matching any keyword."""

    matching_indices: set[int] = set()

    for keyword in keywords:
        keyword_lower = keyword.lower()

        for idx, feature_name in enumerate(feature_names):
            feature_name_lower = str(feature_name).lower()

            if match_prefix:
                is_match = feature_name_lower.startswith(keyword_lower)
            else:
                is_match = keyword_lower in feature_name_lower

            if is_match:
                matching_indices.add(idx)

    return sorted(matching_indices)


def build_grouped_shap_explanation(
    explanation,
    keyword_map: dict[str, str],
    match_prefix: bool = False,
    class_index: int = 1,
    score_mode: ScoreMode = "sum_abs",
    log_group_stats: bool = True,
):
    """Create a grouped SHAP Explanation object from a keyword map.

    Parameters
    ----------
    explanation:
        Original SHAP explanation. Can be 2D or 3D.
    keyword_map:
        Mapping from feature-name keyword to grouped display name.
    match_prefix:
        If True, match only when a feature name starts with the keyword. If
        False, match when the keyword appears anywhere in the feature name.
    class_index:
        Class index to plot if the explanation has a class dimension.
    score_mode:
        ``"sum_abs"`` groups features by summing absolute SHAP values across all
        matched features for each sample. This matches your old grouped bar plot.
        ``"mean_abs"`` normalizes by feature count by averaging absolute SHAP
        values across matched features for each sample.
    log_group_stats:
        If True, log match counts and summary statistics for each group.
    """

    if score_mode not in {"sum_abs", "mean_abs"}:
        raise ValueError("score_mode must be either 'sum_abs' or 'mean_abs'.")

    explanation = select_class_explanation(
        explanation=explanation,
        class_index=class_index,
    )

    if explanation.feature_names is None:
        raise ValueError("SHAP explanation must have feature_names before grouped plotting.")

    feature_names = list(explanation.feature_names)
    values = np.asarray(explanation.values)

    if values.ndim != 2:
        raise ValueError(f"Grouped plotting requires 2D SHAP values. Got {values.shape}.")

    if values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP values have {values.shape[1]} features, "
            f"but feature_names has {len(feature_names)}."
        )

    data = getattr(explanation, "data", None)
    data_array = np.asarray(data) if data is not None else None
    has_compatible_data = (
        data_array is not None
        and data_array.ndim == 2
        and data_array.shape[0] == values.shape[0]
        and data_array.shape[1] == values.shape[1]
    )

    unique_display_names = list(dict.fromkeys(keyword_map.values()))

    grouped_value_columns: list[np.ndarray] = []
    grouped_data_columns: list[np.ndarray] = []
    matched_display_names: list[str] = []
    group_stats: dict[str, dict[str, float | int]] = {}

    for display_name in unique_display_names:
        keywords = get_keywords_for_display_name(keyword_map, display_name)
        matching_indices = get_matching_feature_indices(
            feature_names=feature_names,
            keywords=keywords,
            match_prefix=match_prefix,
        )

        if not matching_indices:
            group_stats[display_name] = {"matches": 0}
            continue

        abs_values = np.abs(values[:, matching_indices])
        sum_abs_per_sample = np.sum(abs_values, axis=1)
        mean_abs_per_sample = np.mean(abs_values, axis=1)

        if score_mode == "sum_abs":
            grouped_values = sum_abs_per_sample
        else:
            grouped_values = mean_abs_per_sample

        grouped_value_columns.append(grouped_values)
        matched_display_names.append(display_name)

        if has_compatible_data:
            grouped_data_columns.append(np.mean(data_array[:, matching_indices], axis=1))

        group_stats[display_name] = {
            "matches": len(matching_indices),
            "mean_abs_per_feature_mean": float(np.mean(mean_abs_per_sample)),
            "mean_abs_per_feature_std": float(np.std(mean_abs_per_sample)),
            "sum_abs_per_sample_mean": float(np.mean(sum_abs_per_sample)),
            "sum_abs_per_sample_std": float(np.std(sum_abs_per_sample)),
            "total_abs_sum": float(np.sum(sum_abs_per_sample)),
        }

    if log_group_stats:
        log_grouped_feature_info(group_stats=group_stats, score_mode=score_mode)

    if not grouped_value_columns:
        logger.info("No grouped SHAP matches found for the provided keyword map.")
        return None

    grouped_values_matrix = np.column_stack(grouped_value_columns)

    if has_compatible_data and grouped_data_columns:
        grouped_data_matrix = np.column_stack(grouped_data_columns)
        return shap.Explanation(
            values=grouped_values_matrix,
            data=grouped_data_matrix,
            feature_names=matched_display_names,
        )

    return shap.Explanation(
        values=grouped_values_matrix,
        feature_names=matched_display_names,
    )


def log_grouped_feature_info(
    group_stats: dict[str, dict[str, float | int]],
    score_mode: ScoreMode,
) -> None:
    """Log feature matching counts and grouped SHAP summary statistics."""

    logger.info("--- Grouped SHAP feature matching info | score_mode=%s ---", score_mode)

    for display_name, stats in group_stats.items():
        matches = int(stats.get("matches", 0))

        if matches == 0:
            logger.info("-- No matches for %s --", display_name)
            continue

        logger.info("-- %s --", display_name)
        logger.info("Matches: %d", matches)
        logger.info(
            "Mean abs SHAP per feature/sample -- Mean: %.6f, Std: %.6f",
            stats["mean_abs_per_feature_mean"],
            stats["mean_abs_per_feature_std"],
        )
        logger.info(
            "Sum abs SHAP per sample -- Mean: %.6f, Std: %.6f",
            stats["sum_abs_per_sample_mean"],
            stats["sum_abs_per_sample_std"],
        )
        logger.info("Total abs SHAP: %.6f", stats["total_abs_sum"])


def plot_grouped_shap_bar(
    explanation,
    keyword_map: dict[str, str],
    model_name: str,
    model_type,
    save_path: Path,
    plot_name: str,
    match_prefix: bool = False,
    class_index: int = 1,
    score_mode: ScoreMode = "sum_abs",
    max_display: int = 20,
    plot_width: float = 14,
    plot_height: float = 8,
    left_margin: float = 0.35,
    right_margin: float = 0.95,
    top_margin: float = 0.9,
    bottom_margin: float = 0.12,
    log_group_stats: bool = True,
    bar_value_format: str = "+.4f",
) -> Path | None:
    """Create and save one grouped SHAP bar plot."""

    grouped_explanation = build_grouped_shap_explanation(
        explanation=explanation,
        keyword_map=keyword_map,
        match_prefix=match_prefix,
        class_index=class_index,
        score_mode=score_mode,
        log_group_stats=log_group_stats,
    )

    if grouped_explanation is None:
        logger.info("Skipping grouped SHAP plot with no matches: %s", plot_name)
        return None

    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        plt.figure(figsize=(plot_width, plot_height))

        shap.plots.bar(
            grouped_explanation,
            max_display=max_display,
            show=False,
        )

        fig = plt.gcf()
        fig.set_size_inches(plot_width, plot_height)

        ax = plt.gca()
        title = PLOT_TITLE_BY_PLOT_TYPE.get(
            plot_name,
            f"Feature Group Importance by {plot_name.replace('_', ' ').title()}",
        )
        ax.set_title(
            f"{title} - {model_name} {model_type.get_folder_name()}",
            fontsize=14,
        )

        _update_bar_value_labels(
            ax=ax,
            grouped_explanation=grouped_explanation,
            value_format=bar_value_format,
        )

        plt.subplots_adjust(
            left=left_margin,
            right=right_margin,
            top=top_margin,
            bottom=bottom_margin,
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Saved grouped SHAP bar plot to %s", save_path)
        return save_path

    except Exception as exc:
        logger.warning("Unable to create grouped SHAP bar plot %s: %s", plot_name, exc)
        plt.close()
        return None


def plot_all_grouped_shap_bars(
    explanation,
    model_name: str,
    model_type,
    save_dir: Path,
    fold_num: int | None = None,
    plot_types: Sequence[str] | None = None,
    class_index: int = 1,
    score_mode: ScoreMode = "sum_abs",
    max_display: int = 20,
    plot_width: float = 14,
    plot_height: float = 8,
    left_margin: float = 0.35,
    right_margin: float = 0.95,
    top_margin: float = 0.9,
    bottom_margin: float = 0.12,
    log_group_stats: bool = True,
    bar_value_format: str = "+.4f",
) -> list[Path]:
    """Create all configured grouped SHAP bar plots for one explanation."""

    keyword_dictionaries = create_keyword_dictionaries()
    selected_plot_types = list(plot_types or DEFAULT_GROUPED_PLOT_TYPES)
    saved_paths: list[Path] = []

    for plot_type in selected_plot_types:
        if plot_type not in keyword_dictionaries:
            logger.warning("Skipping unknown grouped SHAP plot type: %s", plot_type)
            continue

        fold_prefix = f"fold_{fold_num}_" if fold_num is not None else ""
        save_path = save_dir / f"{fold_prefix}shap_{plot_type}_{score_mode}_bar.png"

        saved_path = plot_grouped_shap_bar(
            explanation=explanation,
            keyword_map=keyword_dictionaries[plot_type],
            model_name=model_name,
            model_type=model_type,
            save_path=save_path,
            plot_name=plot_type,
            match_prefix=MATCH_PREFIX_BY_PLOT_TYPE.get(plot_type, False),
            class_index=class_index,
            score_mode=score_mode,
            max_display=max_display,
            plot_width=plot_width,
            plot_height=plot_height,
            left_margin=left_margin,
            right_margin=right_margin,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
            log_group_stats=log_group_stats,
            bar_value_format=bar_value_format,
        )

        if saved_path is not None:
            saved_paths.append(saved_path)

    return saved_paths


def _update_bar_value_labels(ax, grouped_explanation, value_format: str) -> None:
    """Update SHAP bar text labels with consistent numeric formatting."""

    if not ax.texts:
        return

    bar_values = np.mean(np.abs(grouped_explanation.values), axis=0)
    ordered_values = bar_values[np.argsort(bar_values)[::-1]]

    # SHAP may display only the top max_display groups, so only update the labels
    # that are actually present on the axes.
    for text, value in zip(ax.texts, ordered_values):
        text.set_text(format(value, value_format))
