import os
import glob
import textwrap
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from itertools import combinations
import math
import numpy as np
# import seaborn as sns

from scripts.temporal_functions import get_model_subfolder


# ----------------------------
# Constants
# ----------------------------
STREAM_ORDER = [
    "Equivital-HRV",
    "EEG",
    "Pupil",
    "Centrifuge",
    "Participant"
]


# ----------------------------
# Helpers
# ----------------------------
import os
import glob
import pandas as pd

def get_matching_files(results_folder, classifier, required_stream):
    pattern = os.path.join(results_folder, f"feature_importance_results_{classifier}_*.csv")
    all_files = glob.glob(pattern)

    matches = []
    for file_path in all_files:
        filename = os.path.basename(file_path)
        data_stream = filename.replace(f"feature_importance_results_{classifier}_", "").replace(".csv", "")
        if required_stream in data_stream:
            matches.append((data_stream, file_path))
    return matches


def build_overall_summary(results_folder, classifier, required_stream):
    matching_files = get_matching_files(results_folder, classifier, required_stream)

    all_long = []
    for data_stream, file_path in matching_files:
        df_long = pd.read_csv(file_path)
        df_long["data_stream"] = data_stream
        all_long.append(df_long[["data_stream", "fold", "feature", "importance"]])

    combined = pd.concat(all_long, ignore_index=True)

    overall_summary = combined.groupby("feature")["importance"].agg(
        mean_importance="mean",
        median_importance="median",
        std_importance="std",
        n="count"
    ).sort_values("mean_importance", ascending=False)

    return overall_summary

def build_data_stream(selected_streams):
    """Return data stream string in required canonical order."""
    sorted_streams = [s for s in STREAM_ORDER if s in selected_streams]
    return "-".join(sorted_streams)


def get_stream_color(feature_name):
    """Color features by source stream for rank/raw plots."""
    if "Equivital" in feature_name or "HRV" in feature_name:
        return "blue"
    if "EEG" in feature_name:
        return "red"
    if "Pupil" in feature_name:
        return "purple"
    if "Centrifuge" in feature_name:
        return "orange"
    if "Participant" in feature_name or "participant" in feature_name:
        return "green"
    return "black"


def load_feature_importance_data(file_path):
    """Load long-format CSV and pivot to folds x features."""
    df_long = pd.read_csv(file_path)
    df = df_long.pivot(index="fold", columns="feature", values="importance")
    return df, df_long


def build_summary(df, df_long):
    """Build summary table of rank-based and raw importance metrics."""
    rank_df = df.rank(axis=1, ascending=False)

    # Across-fold std (what you already have)
    std_across_folds = df.std()

    # Within-fold std (average of permutation std across folds)
    std_within_folds = df_long.groupby("feature")["std"].mean()

    summary = pd.DataFrame({
        "mean_rank": rank_df.mean(),
        "std_rank": rank_df.std(),
        "mean_importance": df.mean(),
        "median_importance": df.median(),
        "std_across_folds": std_across_folds,
        "std_within_folds": std_within_folds
    }).sort_values("mean_rank")

    rank_counts = rank_df.apply(lambda col: col.value_counts()).fillna(0)
    return rank_df, summary, rank_counts


def plot_bump_chart(rank_df, subfolder, classifier, data_stream):
    """Optional bump chart showing rank changes across folds."""
    # plt.figure(figsize=(10, 6))
    #
    # for col in rank_df.columns:
    #     plt.plot(rank_df.index, rank_df[col], marker='o', label=col)
    #
    # plt.gca().invert_yaxis()  # rank 1 at top
    # plt.xlabel("Fold")
    # plt.ylabel("Rank")
    # plt.title(f"Feature Rank Changes - {subfolder} {classifier}: {data_stream}")
    # plt.xticks(rank_df.index)
    # plt.grid(axis='x', linestyle='-', color='gray', alpha=0.5)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    # plt.tight_layout()
    # plt.show()
    pass


def plot_rank_stability(summary, subfolder, classifier, data_stream):
    """Scatter plot of mean rank vs rank variability."""
    colors = [get_stream_color(feature) for feature in summary.index]

    legend_items = [
        mpatches.Patch(color="blue", label="Equivital/HRV"),
        mpatches.Patch(color="red", label="EEG"),
        mpatches.Patch(color="purple", label="Pupil"),
        mpatches.Patch(color="orange", label="Centrifuge"),
        mpatches.Patch(color="green", label="Participant"),
    ]

    plt.figure(figsize=(8, 6))
    plt.scatter(summary["mean_rank"], summary["std_rank"], c=colors)

    plt.xlabel("Mean Rank")
    plt.ylabel("Rank Std (stability)")
    plt.title(f"Feature Importance vs Stability\n{classifier}: {data_stream}")
    plt.legend(handles=legend_items, loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_raw_importance(summary, subfolder, classifier, data_stream):
    """Scatter plot of median importance vs importance variability."""
    colors = [get_stream_color(feature) for feature in summary.index]

    legend_items = [
        mpatches.Patch(color="blue", label="Equivital/HRV"),
        mpatches.Patch(color="red", label="EEG"),
        mpatches.Patch(color="purple", label="Pupil"),
        mpatches.Patch(color="orange", label="Centrifuge"),
        mpatches.Patch(color="green", label="Participant"),
    ]

    plt.figure(figsize=(8, 6))
    plt.scatter(summary["median_importance"], summary["std_across_folds"], c=colors)

    plt.xlabel("Median Importance")
    plt.ylabel("Importance Std")
    plt.title(f"Raw Importance vs Variability\n{classifier}: {data_stream}")

    plt.xscale("log")
    plt.xlim(1e-6, 1e-3)
    # plt.xscale('symlog', linthresh=1e-10)

    plt.axvline(x=0, color="black", linestyle="--")
    plt.legend(handles=legend_items, loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_summary_importance(summary, subfolder, results_folder, classifier, data_stream):
    """Main summary plot with mean importance, std, sign coloring, and numbered labels."""
    summary = summary.sort_values(by="mean_importance", ascending=False)
    if classifier == "EGB":
        summary = summary.tail(500)
    median_vals = summary["median_importance"]

    # Color points by sign of median importance
    colors = ["blue" if v > 0 else "black" if v == 0 else "red" for v in median_vals]

    fig = plt.figure(figsize=(8, 6))

    # # Optional: horizontal error bars for positive-median features only
    # plt.errorbar(
    #     summary.loc[pos_mask, "mean_importance"],
    #     summary.loc[pos_mask, "std_importance"],
    #     xerr=summary.loc[pos_mask, "std_importance"],
    #     fmt='none',
    #     ecolor='gray',
    #     alpha=0.5
    # )

    plt.scatter(
        summary["mean_importance"],
        summary["std_across_folds"],
        c=colors
    )

    # Label features that are median-positive and remain positive after subtracting 1 std
    label_mask = (
        (summary["median_importance"] > 0) &
        ((summary["mean_importance"] - summary["std_across_folds"]) > 0)
    )
    labeled_features = summary.index[label_mask]
    feature_labels = {feat: i + 1 for i, feat in enumerate(labeled_features)}

    # Group overlapping labels by rounded coordinates
    groups = defaultdict(list)
    for feat, num in feature_labels.items():
        x = summary.loc[feat, "mean_importance"]
        y = summary.loc[feat, "std_across_folds"]
        key = (round(x, 6), round(y, 6))
        groups[key].append(num)

    x_offset = summary["mean_importance"].max()/75
    y_offset = summary["std_across_folds"].max()/75
    for (x, y), nums in groups.items():
        label = ",".join(map(str, nums))
        plt.text(x + x_offset, y - y_offset, label, fontsize=9, ha="center", va="center")
        # plt.text(x + 5e-6, y - 4e-6, label, fontsize=9, ha="center", va="center")

    # Side legend mapping numbers to full feature names
    wrapped_lines = [
        f"{num}: " + "\n    ".join(textwrap.wrap(feat, width=40))
        for feat, num in feature_labels.items()
    ]
    legend_text = "\n".join(wrapped_lines)
    plt.gcf().text(0.80, 0.5, legend_text, fontsize=8, va="center")

    # Threshold line: mean - std = 0  -> std = mean
    x_min = 0
    x_max = summary["mean_importance"].max()
    x_vals = np.linspace(x_min, x_max, 100)  # smooth range
    y_vals = x_vals  # same slope
    plt.plot(x_vals, y_vals, linestyle="--", color="black", label="mean - std = 0")

    ymin = summary["std_across_folds"].min()
    ymax = summary["std_across_folds"].max() * 1.1
    plt.ylim(ymin, ymax)

    plt.axvline(0, linestyle="--", color="black")
    plt.xlabel("Mean Importance")
    plt.ylabel("Across Fold Std")
    plt.title(f"Importance vs Variability\n{subfolder} - {classifier}: {data_stream}")

    legend_items = [
        mpatches.Patch(color="blue", label="Positive"),
        mpatches.Patch(color="black", label="Zero"),
        mpatches.Patch(color="red", label="Negative"),
    ]
    plt.legend(handles=legend_items, title="Median Importance", loc="upper right")

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()

    plot_path = os.path.join(results_folder, f"figures/feature_importance_plot_{classifier}_{data_stream}.png")
    fig.savefig(plot_path, bbox_inches='tight')
    print(f"Saved feature importance plot to {plot_path}")


def save_summary_table(summary, results_folder, classifier, data_stream):
    """Save full summary table to CSV."""
    table_summary = summary.reset_index().rename(columns={"index": "feature"})
    out_path = os.path.join(results_folder, f"summary/feature_importance_summary_{classifier}_{data_stream}.csv")
    table_summary.to_csv(out_path, index=False)


def main():
    # ----------------------------
    # Settings
    # ----------------------------
    model_type = ["complete", "explicit"]
    classifier = "EGB"

    selected_streams = ["EEG", "Equivital-HRV", "Participant", "Centrifuge", "Pupil"]

    # # Can use the combinations below to iterate through all combinatinos of selected_streams has all data streams selected
    # all_combinations = []
    #
    # for r in range(1, len(selected_streams) + 1):
    #     for combo in combinations(selected_streams, r):
    #         all_combinations.append(combo)

    data_stream = build_data_stream(selected_streams)
    subfolder = get_model_subfolder(model_type)
    results_folder = os.path.join(".", "feature_importance", subfolder, classifier)
    file_path = os.path.join(
        results_folder,
        f"feature_importance_results_{classifier}_{data_stream}.csv"
    )

    os.makedirs(results_folder, exist_ok=True)

    # ----------------------------
    # Load data
    # ----------------------------
    df, df_long = load_feature_importance_data(file_path)

    print(df.head())
    print(df.columns)
    print(df.info())

    # ----------------------------
    # Analysis
    # ----------------------------
    rank_df, summary, rank_counts = build_summary(df, df_long)

    print(summary.to_string())
    # print(len(summary["mean_importance"]))
    # exit()
    # print(rank_counts.to_string())

    # ----------------------------
    # Plots
    # ----------------------------
    # plot_bump_chart(rank_df, subfolder, classifier, data_stream)
    # plot_rank_stability(summary, subfolder, classifier, data_stream)
    # plot_raw_importance(summary, subfolder, classifier, data_stream)
    plot_summary_importance(summary, subfolder, results_folder, classifier, data_stream)

    # overall_summary = build_overall_summary(results_folder, classifier, "EEG")
    # plot_summary_importance(overall_summary, subfolder, classifier, 'All EEG')

    # ----------------------------
    # Save outputs
    # ----------------------------
    save_summary_table(summary, results_folder, classifier, data_stream)
    print(f"Saved outputs to: {results_folder}")

from xgboost import XGBClassifier
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')


if __name__ == "__main__":
    main()