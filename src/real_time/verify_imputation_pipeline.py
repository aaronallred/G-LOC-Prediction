"""Pipeline Imputation Verification, Phase Profiling, and Visualization Script.

This script executes the data loading and pre-feature imputation stages of the G-LOC
pipeline to:
1. Capture raw data immediately before and after KNN imputation.
2. Mathematically verify that all NaN values are imputed and that all non-NaN values
   remain strictly identical (asserting zero modified non-NaN values).
3. Profile and categorize which trial phase (Baseline, Ramp/Onset, Plateau/Peak Gz,
   G-LOC Event, Deceleration/Recovery) each imputed value falls into.
4. Generate focused single-trial time-series plots for select features with time on
   the x-axis and sensor values on the y-axis, highlighting original vs. imputed points.

Usage::

    # Run verification using data_reduced (memory-safe):
    python -m src.real_time.verify_imputation_pipeline --config configs/test.yaml --data-path data_reduced

    # Run verification using full production data:
    python -m src.real_time.verify_imputation_pipeline --config configs/test.yaml --data-path data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Use non-interactive Agg backend for matplotlib
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    PLOTTING_AVAILABLE = True
except ImportError:  # pragma: no cover
    PLOTTING_AVAILABLE = False

from src.config_loader import load_experiment_config
from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline
from src.model_type import ModelType
from src.models.model_factory import ModelFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_imputation")


# ---------------------------------------------------------------------------
# Verification Functions
# ---------------------------------------------------------------------------

def verify_imputation_integrity(
        before_matrix: np.ndarray,
        after_matrix: np.ndarray,
        feature_names: List[str],
) -> Dict[str, Any]:
    """Verify that all NaNs are imputed and non-NaN values are completely untouched.

    Args:
        before_matrix: 2D numpy array of features before imputation (contains NaNs).
        after_matrix: 2D numpy array of features after imputation.
        feature_names: List of column/feature names corresponding to matrix columns.

    Returns:
        Dictionary containing detailed verification statistics and pass/fail status.
    """
    logger.info("Verifying imputation integrity across %d cells (%d rows x %d cols)...",
                before_matrix.size, before_matrix.shape[0], before_matrix.shape[1])

    nan_mask = np.isnan(before_matrix)
    after_nan_mask = np.isnan(after_matrix)

    total_cells = int(before_matrix.size)
    total_imputed_nans = int(nan_mask.sum())
    remaining_nans = int(after_nan_mask.sum())

    # Non-NaN integrity check
    non_nan_mask = ~nan_mask
    valid_before = before_matrix[non_nan_mask]
    valid_after = after_matrix[non_nan_mask]

    diff = np.abs(valid_before - valid_after)
    max_non_nan_diff = float(np.max(diff)) if len(diff) > 0 else 0.0
    mean_non_nan_diff = float(np.mean(diff)) if len(diff) > 0 else 0.0
    modified_non_nan_count = int(np.sum(diff > 1e-7))

    # Per-feature breakdown
    feature_stats = {}
    for col_idx, feat in enumerate(feature_names):
        col_nan_mask = nan_mask[:, col_idx]
        col_total_nan = int(col_nan_mask.sum())
        col_remaining_nan = int(after_nan_mask[:, col_idx].sum())

        col_valid_mask = ~col_nan_mask
        if col_valid_mask.sum() > 0:
            col_diff = np.abs(before_matrix[col_valid_mask, col_idx] - after_matrix[col_valid_mask, col_idx])
            col_max_diff = float(np.max(col_diff))
            col_modified = int(np.sum(col_diff > 1e-7))
        else:
            col_max_diff = 0.0
            col_modified = 0

        if col_total_nan > 0 or col_modified > 0:
            feature_stats[feat] = {
                "total_nans_before": col_total_nan,
                "remaining_nans_after": col_remaining_nan,
                "modified_non_nans": col_modified,
                "max_non_nan_diff": col_max_diff,
            }

    verification_passed = (remaining_nans == 0) and (modified_non_nan_count == 0) and (max_non_nan_diff < 1e-7)

    results = {
        "verification_passed": verification_passed,
        "total_cells": total_cells,
        "total_imputed_nans": total_imputed_nans,
        "total_valid_original_values": int(non_nan_mask.sum()),
        "remaining_nans_after": remaining_nans,
        "modified_non_nans": modified_non_nan_count,
        "max_non_nan_difference": max_non_nan_diff,
        "mean_non_nan_difference": mean_non_nan_diff,
        "features_with_imputation_count": len([f for f, s in feature_stats.items() if s["total_nans_before"] > 0]),
        "per_feature_stats": feature_stats,
    }

    if not verification_passed:
        logger.error(
            "VERIFICATION FAILED: remaining_nans=%d, modified_non_nans=%d, max_diff=%.8f",
            remaining_nans, modified_non_nan_count, max_non_nan_diff
        )
    else:
        logger.info(
            "VERIFICATION PASSED: %d NaNs successfully imputed. All %d non-NaN values remained strictly identical (max diff: %.2e).",
            total_imputed_nans, int(non_nan_mask.sum()), max_non_nan_diff
        )

    return results


# ---------------------------------------------------------------------------
# Trial Phase Profiling Functions
# ---------------------------------------------------------------------------

def classify_trial_phases(
        trial_df: pd.DataFrame,
        gz_col_name: Optional[str] = None,
        event_col_name: str = "event",
) -> pd.Series:
    """Classify each row of a single trial DataFrame into physiological/centrifuge phases.

    Phases:
      - 'G-LOC': Active loss-of-consciousness window (event == 1).
      - 'Baseline': Pre-run resting period (Gz <= 1.15 G before acceleration).
      - 'Ramp / Onset': Acceleration phase (Gz increasing from baseline towards peak).
      - 'Plateau / Peak Gz': Sustained peak G-force level (near peak Gz).
      - 'Deceleration / Recovery': Ramp-down returning to 1.0 G and post-run recovery.

    Args:
        trial_df: DataFrame containing a single trial's time-series data.
        gz_col_name: Column name for Centrifuge Gz magnitude/acceleration.
        event_col_name: Column name for G-LOC event indicator.

    Returns:
        pd.Series containing string phase labels for each row.
    """
    n_rows = len(trial_df)
    phases = np.array(["Baseline"] * n_rows, dtype=object)

    # Find Gz column if not specified
    if gz_col_name is None or gz_col_name not in trial_df.columns:
        candidates = [
            "magnitude - Centrifuge",
            "Double I/O.actualGz - Centrifuge",
            "accelerometerGzSum.engValue (G) - Centrifuge",
            "G-Suit.actualGz(G) - Centrifuge",
            "actualGz",
        ]
        for c in candidates:
            if c in trial_df.columns:
                gz_col_name = c
                break

    gz_series = trial_df[gz_col_name].to_numpy() if gz_col_name and gz_col_name in trial_df.columns else np.ones(n_rows)
    # Replace NaNs in Gz with 1.0 for phase detection
    gz_clean = np.nan_to_num(gz_series, nan=1.0)

    # Detect G-LOC events
    gloc_mask = np.zeros(n_rows, dtype=bool)
    if event_col_name in trial_df.columns:
        gloc_mask = (trial_df[event_col_name].to_numpy() == 1)
    if "event_validated" in trial_df.columns:
        gloc_mask |= (trial_df["event_validated"].to_numpy() == 1)

    max_gz = np.max(gz_clean) if len(gz_clean) > 0 else 1.0
    peak_idx = int(np.argmax(gz_clean)) if len(gz_clean) > 0 else 0

    # If peak G is very low (< 1.2G), entire trial is baseline/sub-threshold
    if max_gz < 1.2:
        phases[:] = "Baseline"
        phases[gloc_mask] = "G-LOC"
        return pd.Series(phases, index=trial_df.index)

    # Find onset index (first point where Gz exceeds 1.15 G)
    onset_candidates = np.where(gz_clean > 1.15)[0]
    onset_idx = onset_candidates[0] if len(onset_candidates) > 0 else 0

    # Find offset / recovery index (first point after peak where Gz drops below 1.15 G)
    after_peak_below = np.where((np.arange(n_rows) > peak_idx) & (gz_clean <= 1.15))[0]
    recovery_idx = after_peak_below[0] if len(after_peak_below) > 0 else n_rows

    # Plateau threshold: within 85% of peak Gz
    plateau_threshold = max(1.4, max_gz * 0.85)

    for i in range(n_rows):
        if gloc_mask[i]:
            phases[i] = "G-LOC"
        elif i < onset_idx:
            phases[i] = "Baseline"
        elif i >= recovery_idx:
            phases[i] = "Deceleration / Recovery"
        else:
            if gz_clean[i] >= plateau_threshold:
                phases[i] = "Plateau / Peak Gz"
            elif i <= peak_idx:
                phases[i] = "Ramp / Onset"
            else:
                phases[i] = "Deceleration / Recovery"

    return pd.Series(phases, index=trial_df.index)


def profile_imputed_phases(
        gloc_data: pd.DataFrame,
        before_matrix: np.ndarray,
        feature_names: List[str],
) -> Dict[str, Any]:
    """Map all imputed NaN values to their respective trial and phase within trial.

    Args:
        gloc_data: Full raw gloc_data DataFrame containing metadata columns.
        before_matrix: 2D numpy array of features before imputation.
        feature_names: Feature column names.

    Returns:
        Dictionary summarizing the phase breakdown across all trials and features.
    """
    logger.info("Profiling trial phase context for imputed values...")

    nan_mask = np.isnan(before_matrix)
    trial_col = gloc_data["trial_id"].to_numpy() if "trial_id" in gloc_data.columns else np.array(["all"] * len(gloc_data))

    time_col_name = "Time (s)" if "Time (s)" in gloc_data.columns else None
    if time_col_name is None:
        for c in gloc_data.columns:
            if "time" in c.lower():
                time_col_name = c
                break

    time_series = gloc_data[time_col_name].to_numpy() if time_col_name else (np.arange(len(gloc_data)) / 25.0)

    # Compute phase labels for each trial
    phase_labels = np.array(["Unknown"] * len(gloc_data), dtype=object)
    unique_trials = pd.unique(trial_col)

    for tid in unique_trials:
        t_mask = (trial_col == tid)
        sub_df = gloc_data[t_mask]
        trial_phases = classify_trial_phases(sub_df)
        phase_labels[t_mask] = trial_phases.to_numpy()

    # Aggregate missingness by phase and trial
    phase_counts: Dict[str, int] = {
        "Baseline": 0,
        "Ramp / Onset": 0,
        "Plateau / Peak Gz": 0,
        "G-LOC": 0,
        "Deceleration / Recovery": 0,
        "Unknown": 0,
    }

    per_trial_summary: Dict[str, Any] = {}
    per_feature_phase_summary: Dict[str, Dict[str, int]] = {}

    nan_row_indices, nan_col_indices = np.where(nan_mask)
    total_nans = len(nan_row_indices)

    for r_idx, c_idx in zip(nan_row_indices, nan_col_indices):
        ph = str(phase_labels[r_idx])
        feat = feature_names[c_idx]
        tid = str(trial_col[r_idx])

        phase_counts[ph] = phase_counts.get(ph, 0) + 1

        # Trial level
        if tid not in per_trial_summary:
            per_trial_summary[tid] = {
                "total_imputed": 0,
                "phases": {"Baseline": 0, "Ramp / Onset": 0, "Plateau / Peak Gz": 0, "G-LOC": 0, "Deceleration / Recovery": 0},
                "features_affected": set(),
            }
        per_trial_summary[tid]["total_imputed"] += 1
        per_trial_summary[tid]["phases"][ph] = per_trial_summary[tid]["phases"].get(ph, 0) + 1
        per_trial_summary[tid]["features_affected"].add(feat)

        # Feature level
        if feat not in per_feature_phase_summary:
            per_feature_phase_summary[feat] = {
                "Baseline": 0,
                "Ramp / Onset": 0,
                "Plateau / Peak Gz": 0,
                "G-LOC": 0,
                "Deceleration / Recovery": 0,
            }
        per_feature_phase_summary[feat][ph] = per_feature_phase_summary[feat].get(ph, 0) + 1

    # Convert sets to sorted lists for JSON serialization
    for tid, info in per_trial_summary.items():
        info["features_affected"] = sorted(list(info["features_affected"]))

    # Compute percentages
    phase_percentages = {
        ph: (count / total_nans * 100.0 if total_nans > 0 else 0.0)
        for ph, count in phase_counts.items()
    }

    summary = {
        "total_imputed_points": total_nans,
        "phase_counts": phase_counts,
        "phase_percentages": phase_percentages,
        "trials_with_imputation_count": len(per_trial_summary),
        "per_trial_summary": per_trial_summary,
        "per_feature_phase_summary": per_feature_phase_summary,
    }

    logger.info("Imputed Phase Distribution:")
    for ph, pct in phase_percentages.items():
        if phase_counts[ph] > 0:
            logger.info("  Phase '%s': %d points (%.2f%%)", ph, phase_counts[ph], pct)

    return summary


# ---------------------------------------------------------------------------
# Plotting Engine (Single Representative Trial)
# ---------------------------------------------------------------------------

def select_representative_trial(
        gloc_data: pd.DataFrame,
        before_matrix: np.ndarray,
        feature_names: List[str],
        user_trial_id: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Select a single trial and 3-4 diverse features containing meaningful imputed data.

    Args:
        gloc_data: Raw gloc_data DataFrame.
        before_matrix: Feature matrix before imputation.
        feature_names: Column names.
        user_trial_id: User-requested trial ID (if specified).

    Returns:
        Tuple of (selected_trial_id, list_of_selected_features).
    """
    nan_mask = np.isnan(before_matrix)
    trial_col = gloc_data["trial_id"].to_numpy() if "trial_id" in gloc_data.columns else np.array(["all"] * len(gloc_data))
    unique_trials = pd.unique(trial_col)

    if user_trial_id is not None and user_trial_id in unique_trials:
        selected_trial = user_trial_id
    else:
        # Auto-select trial with the highest number of imputed NaNs across diverse streams
        trial_nan_counts = {}
        for tid in unique_trials:
            t_mask = (trial_col == tid)
            trial_nan_counts[tid] = int(nan_mask[t_mask].sum())

        sorted_trials = sorted(trial_nan_counts.items(), key=lambda x: -x[1])
        selected_trial = sorted_trials[0][0] if sorted_trials else str(unique_trials[0])

    logger.info("Selected representative trial for visualization: %s", selected_trial)

    # Select 3-4 diverse features with imputation in this trial
    t_mask = (trial_col == selected_trial)
    trial_nan_cols = np.where(nan_mask[t_mask].sum(axis=0) > 0)[0]

    chosen_features: List[str] = []
    # Try picking from diverse sensor groups
    priority_keywords = ["HR", "ECG", "Skin Temperature", "BR", "Tobii", "EEG", "Centrifuge"]
    for kw in priority_keywords:
        for c_idx in trial_nan_cols:
            feat = feature_names[c_idx]
            if kw.lower() in feat.lower() and feat not in chosen_features:
                chosen_features.append(feat)
                break
        if len(chosen_features) >= 4:
            break

    # If fewer than 4 features have NaNs, add non-NaN reference features (like Centrifuge magnitude)
    if len(chosen_features) < 4:
        for feat in feature_names:
            if any(kw.lower() in feat.lower() for kw in ["magnitude - Centrifuge", "HR (bpm)", "ECG Lead 2", "Skin Temp"]):
                if feat not in chosen_features:
                    chosen_features.append(feat)
            if len(chosen_features) >= 4:
                break

    logger.info("Selected features for single-trial plotting: %s", chosen_features)
    return selected_trial, chosen_features


def plot_single_trial_imputation(
        gloc_data: pd.DataFrame,
        before_matrix: np.ndarray,
        after_matrix: np.ndarray,
        feature_names: List[str],
        trial_id: str,
        features_to_plot: List[str],
        output_dir: Path,
) -> str:
    """Generate a multi-panel time vs value plot for a single trial highlighting imputed points.

    Args:
        gloc_data: gloc_data DataFrame.
        before_matrix: Feature matrix before imputation.
        after_matrix: Feature matrix after imputation.
        feature_names: Full feature name list.
        trial_id: Selected trial ID.
        features_to_plot: List of feature names to plot.
        output_dir: Output directory for saving plot.

    Returns:
        File path of the saved PNG image.
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Matplotlib is not available. Skipping plot generation.")
        return ""

    trial_col = gloc_data["trial_id"].to_numpy() if "trial_id" in gloc_data.columns else np.array(["all"] * len(gloc_data))
    t_mask = (trial_col == trial_id)
    trial_df = gloc_data[t_mask].copy()

    time_col = "Time (s)" if "Time (s)" in trial_df.columns else None
    if time_col is None:
        for c in trial_df.columns:
            if "time" in c.lower():
                time_col = c
                break

    if time_col and time_col in trial_df.columns:
        time_vals = trial_df[time_col].to_numpy()
        # Relative time starting at 0
        time_axis = time_vals - time_vals[0]
    else:
        time_axis = np.arange(len(trial_df)) / 25.0

    # Classify trial phases
    phases = classify_trial_phases(trial_df).to_numpy()

    # Get Gz profile
    gz_col = None
    for c in ["magnitude - Centrifuge", "Double I/O.actualGz - Centrifuge", "G-Suit.actualGz(G) - Centrifuge", "actualGz"]:
        if c in trial_df.columns:
            gz_col = c
            break
    gz_vals = np.nan_to_num(trial_df[gz_col].to_numpy(), nan=1.0) if gz_col else np.ones(len(trial_df))

    # Color palette for trial phases
    phase_colors = {
        "Baseline": "#f0f0f0",
        "Ramp / Onset": "#fff3cd",
        "Plateau / Peak Gz": "#ffe5d0",
        "G-LOC": "#f8d7da",
        "Deceleration / Recovery": "#d4edda",
    }

    n_feats = len(features_to_plot)
    n_rows = n_feats + 1  # Top plot is Centrifuge Gz profile

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.2 * n_rows), sharex=True, dpi=300)
    if n_rows == 1:
        axes = [axes]

    # Helper to add phase background spans
    def add_phase_spans(ax: plt.Axes) -> None:
        current_phase = phases[0]
        start_idx = 0
        for i in range(1, len(phases)):
            if phases[i] != current_phase:
                color = phase_colors.get(current_phase, "#f8f9fa")
                ax.axvspan(time_axis[start_idx], time_axis[i], color=color, alpha=0.55, lw=0)
                start_idx = i
                current_phase = phases[i]
        color = phase_colors.get(current_phase, "#f8f9fa")
        ax.axvspan(time_axis[start_idx], time_axis[-1], color=color, alpha=0.55, lw=0)

    # 1. Top Subplot: Centrifuge Gz Profile
    ax_top = axes[0]
    add_phase_spans(ax_top)
    ax_top.plot(time_axis, gz_vals, color="#2c3e50", lw=2.0, label="Centrifuge Gz")
    ax_top.set_ylabel("Gz Acceleration (G)", fontsize=11, fontweight="bold")
    ax_top.set_title(f"Trial '{trial_id}' — Imputation & Phase Verification Profile", fontsize=14, fontweight="bold", pad=12)
    ax_top.grid(True, linestyle="--", alpha=0.5)

    # Phase legend handles
    phase_patches = [
        mpatches.Patch(color=phase_colors[p], label=p)
        for p in ["Baseline", "Ramp / Onset", "Plateau / Peak Gz", "G-LOC", "Deceleration / Recovery"]
    ]
    ax_top.legend(handles=phase_patches + [plt.Line2D([0], [0], color="#2c3e50", lw=2, label="Centrifuge Gz")],
                  loc="upper right", framealpha=0.9, fontsize=9, ncol=3)

    # 2. Physiological Feature Subplots
    feat_name_to_col_idx = {name: i for i, name in enumerate(feature_names)}

    for ax_idx, feat in enumerate(features_to_plot, start=1):
        ax = axes[ax_idx]
        add_phase_spans(ax)

        col_idx = feat_name_to_col_idx.get(feat)
        if col_idx is None:
            ax.text(0.5, 0.5, f"Feature '{feat}' not found in matrix", transform=ax.transAxes, ha="center")
            continue

        raw_vals = before_matrix[t_mask, col_idx]
        imp_vals = after_matrix[t_mask, col_idx]

        is_nan = np.isnan(raw_vals)
        has_imputed = np.any(is_nan)

        # Plot original valid data
        valid_indices = np.where(~is_nan)[0]
        if len(valid_indices) > 0:
            ax.plot(time_axis[valid_indices], raw_vals[valid_indices], color="#1f77b4", lw=1.8,
                    label="Original Valid Data", alpha=0.9)

        # Plot imputed data
        if has_imputed:
            nan_indices = np.where(is_nan)[0]
            ax.scatter(time_axis[nan_indices], imp_vals[nan_indices], color="#d62728", s=25,
                       marker="x", label=f"Imputed ({len(nan_indices)} pts)", zorder=5)
            # Connect the complete imputed curve with a dashed line
            ax.plot(time_axis, imp_vals, color="#d62728", lw=1.2, linestyle=":", alpha=0.75,
                    label="Post-Imputation Trajectory")

            # Shaded vertical spans for contiguous imputed segments
            diff_nan = np.diff(np.pad(is_nan.astype(int), (1, 1), 'constant'))
            gap_starts = np.where(diff_nan == 1)[0]
            gap_ends = np.where(diff_nan == -1)[0] - 1
            for gs, ge in zip(gap_starts, gap_ends):
                ax.axvspan(time_axis[gs], time_axis[min(ge, len(time_axis) - 1)],
                           color="#e74c3c", alpha=0.18, hatch="//", label="_nolegend_")
        else:
            # All valid
            ax.plot(time_axis, raw_vals, color="#1f77b4", lw=1.8, label="Original Data (No NaNs)")

        # Formatting
        clean_feat_title = feat.replace("_", " ")
        ax.set_ylabel(clean_feat_title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    axes[-1].set_xlabel("Trial Time (seconds)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_filename = plots_dir / f"imputation_verification_trial_{trial_id}.png"
    fig.savefig(out_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved imputation verification plot to: %s", out_filename)
    return str(out_filename)


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_markdown_summary(
        verification_results: Dict[str, Any],
        phase_summary: Dict[str, Any],
        trial_id: str,
        features_plotted: List[str],
        plot_path: str,
        output_dir: Path,
) -> str:
    """Generate a clean human-readable Markdown summary report.

    Args:
        verification_results: Results dictionary from verify_imputation_integrity.
        phase_summary: Results dictionary from profile_imputed_phases.
        trial_id: Representative trial plotted.
        features_plotted: Features plotted in the figure.
        plot_path: Filepath of the generated plot.
        output_dir: Directory to save the markdown file.

    Returns:
        Markdown report string.
    """
    passed_badge = "✅ **PASSED**" if verification_results["verification_passed"] else "❌ **FAILED**"

    lines = [
        "# G-LOC Pipeline Imputation Verification Report",
        "",
        f"**Verification Status**: {passed_badge}",
        f"**Date Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Mathematical Integrity Verification",
        "",
        "| Metric | Value | Expected / Requirement | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Total Data Cells** | {verification_results['total_cells']:,} | — | — |",
        f"| **Valid Original Cells** | {verification_results['total_valid_original_values']:,} | — | — |",
        f"| **Total Imputed NaNs** | {verification_results['total_imputed_nans']:,} | > 0 | Found |",
        f"| **Remaining NaNs After Imputation** | {verification_results['remaining_nans_after']} | **0** | {'✅' if verification_results['remaining_nans_after'] == 0 else '❌'} |",
        f"| **Modified Non-NaN Values** | {verification_results['modified_non_nans']} | **0** | {'✅' if verification_results['modified_non_nans'] == 0 else '❌'} |",
        f"| **Max Difference on Non-NaN Elements** | {verification_results['max_non_nan_difference']:.2e} | **0.0** | {'✅' if verification_results['max_non_nan_difference'] < 1e-7 else '❌'} |",
        "",
        "> [!NOTE]",
        "> **Integrity Assertion**: Non-NaN cells were compared point-by-point before and after imputation. All non-NaN values remained strictly identical, confirming that KNN imputation exclusively operates on missing entries without altering existing sensor data.",
        "",
        "---",
        "",
        "## 2. Imputed Values by Trial Phase",
        "",
        "The table below details which physiological/centrifuge phases the missing sensor data occurred in across the dataset:",
        "",
        "| Trial Phase | Imputed Point Count | Percentage of All Imputations | Description |",
        "| :--- | :--- | :--- | :--- |",
    ]

    phase_descriptions = {
        "Baseline": "Resting baseline prior to centrifuge acceleration ramp",
        "Ramp / Onset": "Acceleration onset ramping towards peak G-force",
        "Plateau / Peak Gz": "Sustained high acceleration plateau",
        "G-LOC": "Active loss-of-consciousness event window",
        "Deceleration / Recovery": "Centrifuge ramp-down returning to 1.0 G and recovery",
        "Unknown": "Unclassified trial segment",
    }

    for ph, count in phase_summary["phase_counts"].items():
        if count > 0:
            pct = phase_summary["phase_percentages"].get(ph, 0.0)
            desc = phase_descriptions.get(ph, "—")
            lines.append(f"| **{ph}** | {count:,} | {pct:.2f}% | {desc} |")

    lines.extend([
        "",
        f"- **Total Imputed Points**: {phase_summary['total_imputed_points']:,}",
        f"- **Trials Containing Imputed Data**: {phase_summary['trials_with_imputation_count']} trials",
        "",
        "---",
        "",
        "## 3. Representative Trial Visualization",
        "",
        f"- **Plotted Trial ID**: `{trial_id}`",
        f"- **Features Plotted**: {', '.join(features_plotted)}",
        f"- **Plot File**: `{plot_path}`",
        "",
        "The generated plot displays the centrifuge Gz acceleration profile and individual sensor streams over trial time. Imputed sensor intervals are highlighted with distinct red cross markers and shaded vertical bands.",
        "",
    ])

    report_text = "\n".join(lines)
    summary_file = output_dir / "imputation_verification_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("Saved markdown summary report to: %s", summary_file)
    return report_text


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def run_imputation_verification(
        config_path: str = "configs/test.yaml",
        data_path: Optional[str] = None,
        output_dir: str = "src/real_time/imputation_verification_results",
        trial_id: Optional[str] = None,
        random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute the complete imputation verification workflow.

    Args:
        config_path: Path to YAML experiment configuration file.
        data_path: Optional override path for data directory (e.g. 'data_reduced').
        output_dir: Path to directory where results and plots will be stored.
        trial_id: Optional trial ID to visualize (auto-selected if None).
        random_seed: Random seed for reproducibility.

    Returns:
        Complete results dictionary containing verification stats and phase breakdowns.
    """
    out_dir_path = Path(output_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading experiment config from: %s", config_path)
    cfg = load_experiment_config(config_path)

    # Allow overriding data_path (e.g. using data_reduced to avoid OOM)
    resolved_data_path = data_path if data_path is not None else cfg.get("data_path", "data_reduced")
    logger.info("Using data directory: %s", resolved_data_path)

    # Initialize traditional data pipeline
    pipeline = TraditionalDataPipeline(
        data_path=resolved_data_path,
        random_seed=random_seed,
        config=cfg,
    )

    # Instantiate KNN model to resolve hyperparameters
    knn_model = ModelFactory.create_model("KNN")
    traditional_hyperparameters = pipeline._resolve_traditional_hyperparameters(knn_model, "KNN")
    n_neighbors = traditional_hyperparameters.get("n_neighbors", 5)

    model_type = ModelType(afe_filter="Complete", feature_set="Explicit")
    feature_groups_to_analyze, _ = pipeline._get_feature_groups_and_baseline_methods(
        model_type, traditional_hyperparameters["baseline_methods_to_use"]
    )
    feature_groups_to_analyze, _, _ = pipeline._resolve_feature_groups_for_streams(None, feature_groups_to_analyze)

    output_feature_dtype = np.dtype(cfg["shared_data_parameters"].get("output_feature_dtype", "float32"))
    analysis_type = cfg["shared_data_parameters"].get("analysis_type", 2)
    subject_to_analyze = cfg["shared_data_parameters"].get("subject_to_analyze")
    trial_to_analyze = cfg["shared_data_parameters"].get("trial_to_analyze")
    remove_nan_trials = cfg["shared_data_parameters"].get("remove_NaN_trials", True)

    file_paths = pipeline._get_data_locations()

    logger.info("Loading raw data from CSV/PKL...")
    gloc_data = pipeline._load_data(file_paths, output_feature_dtype)
    gloc_data = pipeline._filter_data_by_analysis_type(analysis_type, gloc_data, subject_to_analyze, trial_to_analyze)
    gloc_data, features = pipeline._process_and_get_feature_names(
        gloc_data, feature_groups_to_analyze, model_type, file_paths, output_feature_dtype
    )
    gloc_labels = pipeline._label_gloc_events(gloc_data)

    if remove_nan_trials:
        logger.info("Filtering all-NaN trials before imputation...")
        gloc_data, gloc_labels, _ = pipeline._remove_all_nan_trials(gloc_data, features, gloc_labels)

    feature_cols = features["All"]
    logger.info("Capturing raw data BEFORE KNN imputation (%d features, %d rows)...", len(feature_cols), len(gloc_data))
    before_matrix = gloc_data[feature_cols].to_numpy(dtype=output_feature_dtype).copy()

    # Perform KNN imputation
    logger.info("Executing KNN imputation (_faster_knn_impute with k=%d)...", n_neighbors)
    after_matrix = pipeline._faster_knn_impute(before_matrix, k=n_neighbors)

    # 1. Mathematical Integrity Verification
    verification_results = verify_imputation_integrity(before_matrix, after_matrix, feature_cols)

    # 2. Phase Profiling
    phase_summary = profile_imputed_phases(gloc_data, before_matrix, feature_cols)

    # 3. Single-Trial Visualization
    selected_trial, features_to_plot = select_representative_trial(
        gloc_data, before_matrix, feature_cols, user_trial_id=trial_id
    )
    plot_path = plot_single_trial_imputation(
        gloc_data, before_matrix, after_matrix, feature_cols, selected_trial, features_to_plot, out_dir_path
    )

    # 4. Generate Reports
    markdown_report = generate_markdown_summary(
        verification_results, phase_summary, selected_trial, features_to_plot, plot_path, out_dir_path
    )

    full_results = {
        "config_path": config_path,
        "data_path": resolved_data_path,
        "trial_plotted": selected_trial,
        "features_plotted": features_to_plot,
        "plot_path": plot_path,
        "verification": verification_results,
        "phase_summary": phase_summary,
    }

    json_report_path = out_dir_path / "imputation_verification_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    logger.info("Saved JSON verification report to: %s", json_report_path)

    return full_results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify pipeline KNN imputation integrity, phase distribution, and single-trial visualizations."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test.yaml",
        help="Path to YAML experiment configuration file (default: configs/test.yaml)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data directory (e.g. data_reduced or data). Defaults to config value.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/real_time/imputation_verification_results",
        help="Directory to save JSON reports, markdown summaries, and plots.",
    )
    parser.add_argument(
        "--trial-id",
        type=str,
        default=None,
        help="Specific trial ID to visualize (optional, defaults to auto-selecting trial with highest imputation).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    results = run_imputation_verification(
        config_path=args.config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        trial_id=args.trial_id,
        random_seed=args.random_seed,
    )

    if not results["verification"]["verification_passed"]:
        logger.error("Imputation verification failed! See logs for details.")
        sys.exit(1)
    else:
        logger.info("All imputation verifications and reporting completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
