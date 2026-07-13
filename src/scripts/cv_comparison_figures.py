"""Generate presentation figures from cross-validation result folders.

Example:
    python -m src.scripts.cv_comparison_figures \
        --results-root Results/Cross_Validation_Temp/Complete_Explicit \
        --models EGB XGB
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)

MODEL_LABELS = {
    "EGB": "Native SkLearn\nGradient Boosting",
    "XGB": "XGBoost",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "specificity": "Specificity",
    "g_mean": "G-mean",
}

PRIMARY_METRICS = ("f1", "recall", "precision", "specificity", "g_mean", "accuracy")
PALETTE = {
    "EGB": "#31688e",
    "XGB": "#e07b39",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CV comparison figures for SkLearn Gradient Boosting vs XGBoost.",
    )
    parser.add_argument(
        "--results-root",
        default="Results/Cross_Validation_GB_vs_XGB/Complete_Explicit",
        help="Folder containing per-model result folders, for example Results/Cross_Validation/Complete_Explicit.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["EGB", "XGB"],
        help="Model folder names to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save figures. Defaults to <results-root>/presentation_figures.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_summary_rows(results_root: Path, models: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in models:
        summary = read_json(results_root / model / "summary.json")
        if summary is None:
            LOGGER.warning("Missing summary.json for %s", model)
            continue

        for metric in PRIMARY_METRICS:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key not in summary:
                continue
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "mean": float(summary[mean_key]),
                    "std": float(summary.get(std_key, 0.0)),
                    "num_folds": int(summary.get("num_folds", 0)),
                }
            )
    return pd.DataFrame(rows)


def load_fold_rows(results_root: Path, models: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in models:
        model_dir = results_root / model
        for fold_path in sorted(model_dir.glob("fold_*/fold_result.json")):
            fold_result = read_json(fold_path)
            if fold_result is None:
                continue
            metrics = fold_result.get("metrics", {})
            for metric in PRIMARY_METRICS:
                if metric not in metrics:
                    continue
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "fold": int(fold_result.get("fold", fold_path.parent.name.replace("fold_", ""))),
                        "metric": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "value": float(metrics[metric]),
                    }
                )
    return pd.DataFrame(rows)


def load_feature_rows(results_root: Path, models: list[str]) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    rows: list[dict[str, Any]] = []
    feature_sets: dict[str, set[str]] = {}
    for model in models:
        median = read_json(results_root / model / "median_hyperparameters.json")
        if median is None:
            LOGGER.warning("Missing median_hyperparameters.json for %s", model)
            continue
        features = median.get("selected_features", [])
        feature_sets[model] = set(features)
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "selected_feature_count": len(features),
                "median_fold_f1": float(median.get("f1_score", 0.0)),
            }
        )
    return pd.DataFrame(rows), feature_sets


def feature_family(feature_name: str) -> str:
    if " - " in feature_name:
        return feature_name.split(" - ", 1)[1].split("_", 1)[0]
    if "_" in feature_name:
        return feature_name.split("_", 1)[0]
    return "Other"


def load_feature_family_rows(feature_sets: dict[str, set[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, features in feature_sets.items():
        counts = Counter(feature_family(feature) for feature in features)
        for family, count in counts.most_common():
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "feature_family": family,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def save_metric_bar(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        LOGGER.warning("Skipping metric bar plot because no summary rows were found.")
        return

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x_positions = range(len(PRIMARY_METRICS))
    width = 0.36
    models = list(summary_df["model"].drop_duplicates())

    for offset_idx, model in enumerate(models):
        model_df = summary_df[summary_df["model"] == model].set_index("metric")
        means = [model_df.loc[metric, "mean"] if metric in model_df.index else 0 for metric in PRIMARY_METRICS]
        stds = [model_df.loc[metric, "std"] if metric in model_df.index else 0 for metric in PRIMARY_METRICS]
        offset = (offset_idx - (len(models) - 1) / 2) * width
        ax.bar(
            [pos + offset for pos in x_positions],
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=PALETTE.get(model),
            label=MODEL_LABELS.get(model, model).replace("\n", " "),
            edgecolor="#263238",
            linewidth=0.6,
        )

    ax.set_title("Cross-Validation Performance Comparison", fontsize=16, weight="bold")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([METRIC_LABELS[metric] for metric in PRIMARY_METRICS], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(output_dir / "metric_comparison_bar.png", dpi=300)
    fig.savefig(output_dir / "metric_comparison_bar.svg")
    plt.close(fig)


def save_f1_comparison(summary_df: pd.DataFrame, fold_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        LOGGER.warning("Skipping F1 comparison plot because no summary rows were found.")
        return

    f1_summary = summary_df[summary_df["metric"] == "f1"].copy()
    if f1_summary.empty:
        LOGGER.warning("Skipping F1 comparison plot because no F1 summary rows were found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sns.barplot(
        data=f1_summary,
        x="model_label",
        y="mean",
        hue="model",
        palette=PALETTE,
        dodge=False,
        ax=ax,
        legend=False,
        edgecolor="#263238",
        linewidth=0.7,
    )

    for idx, row in f1_summary.reset_index(drop=True).iterrows():
        ax.errorbar(
            x=idx,
            y=row["mean"],
            yerr=row["std"],
            color="#263238",
            capsize=5,
            linewidth=1.4,
            fmt="none",
        )
        ax.text(
            idx,
            min(row["mean"] + row["std"] + 0.025, 1.03),
            f"{row['mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )

    if not fold_df.empty:
        f1_folds = fold_df[fold_df["metric"] == "f1"].copy()
        if not f1_folds.empty:
            sns.stripplot(
                data=f1_folds,
                x="model_label",
                y="value",
                color="#263238",
                alpha=0.65,
                size=5,
                jitter=0.12,
                ax=ax,
            )

    ax.set_title("F1 Score: SkLearn Gradient Boosting vs XGBoost", fontsize=15, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Mean cross-validation F1 score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "f1_gradient_boosting_vs_xgboost.png", dpi=300)
    fig.savefig(output_dir / "f1_gradient_boosting_vs_xgboost.svg")
    plt.close(fig)


def save_fold_distribution(fold_df: pd.DataFrame, output_dir: Path) -> None:
    if fold_df.empty:
        LOGGER.warning("Skipping fold distribution plot because no fold rows were found.")
        return

    plot_df = fold_df[fold_df["metric"].isin(["f1", "recall", "specificity", "g_mean"])].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    sns.boxplot(
        data=plot_df,
        x="metric_label",
        y="value",
        hue="model_label",
        palette=[PALETTE.get(model, "#607d8b") for model in plot_df["model"].drop_duplicates()],
        ax=ax,
        fliersize=0,
    )
    sns.stripplot(
        data=plot_df,
        x="metric_label",
        y="value",
        hue="model_label",
        dodge=True,
        palette=["#263238"] * plot_df["model_label"].nunique(),
        alpha=0.55,
        size=4,
        ax=ax,
        legend=False,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: len(set(labels))], labels[: len(set(labels))], frameon=False, loc="lower right")
    ax.set_title("Fold-Level Score Distribution", fontsize=16, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "fold_score_distribution.png", dpi=300)
    fig.savefig(output_dir / "fold_score_distribution.svg")
    plt.close(fig)


def save_feature_count_plot(feature_df: pd.DataFrame, output_dir: Path) -> None:
    if feature_df.empty:
        LOGGER.warning("Skipping feature count plot because no feature rows were found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sns.barplot(
        data=feature_df,
        x="model_label",
        y="selected_feature_count",
        hue="model",
        palette=PALETTE,
        dodge=False,
        ax=ax,
        legend=False,
    )
    ax.set_title("Median-Fold Selected Feature Count", fontsize=15, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Selected features")
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)
    fig.savefig(output_dir / "selected_feature_count.png", dpi=300)
    fig.savefig(output_dir / "selected_feature_count.svg")
    plt.close(fig)


def save_feature_family_plot(family_df: pd.DataFrame, output_dir: Path) -> None:
    if family_df.empty:
        LOGGER.warning("Skipping feature family plot because no selected feature families were found.")
        return

    top_families = (
        family_df.groupby("feature_family")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    plot_df = family_df[family_df["feature_family"].isin(top_families)].copy()
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    sns.barplot(
        data=plot_df,
        y="feature_family",
        x="count",
        hue="model_label",
        palette=[PALETTE.get(model, "#607d8b") for model in plot_df["model"].drop_duplicates()],
        ax=ax,
    )
    ax.set_title("Top Selected Feature Families", fontsize=15, weight="bold")
    ax.set_xlabel("Median-fold selected feature count")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(output_dir / "selected_feature_families.png", dpi=300)
    fig.savefig(output_dir / "selected_feature_families.svg")
    plt.close(fig)


def save_feature_overlap(feature_sets: dict[str, set[str]], output_dir: Path) -> None:
    if len(feature_sets) < 2:
        LOGGER.warning("Skipping feature overlap plot because fewer than two feature sets were found.")
        return

    models = list(feature_sets)[:2]
    first, second = models
    first_only = len(feature_sets[first] - feature_sets[second])
    overlap = len(feature_sets[first] & feature_sets[second])
    second_only = len(feature_sets[second] - feature_sets[first])

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    labels = [
        f"{MODEL_LABELS.get(first, first).replace(chr(10), ' ')} only",
        "Shared",
        f"{MODEL_LABELS.get(second, second).replace(chr(10), ' ')} only",
    ]
    values = [first_only, overlap, second_only]
    colors = [PALETTE.get(first, "#607d8b"), "#6c757d", PALETTE.get(second, "#607d8b")]
    ax.bar(labels, values, color=colors, edgecolor="#263238", linewidth=0.6)
    ax.set_title("Median-Fold Selected Feature Overlap", fontsize=15, weight="bold")
    ax.set_ylabel("Feature count")
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)
    fig.savefig(output_dir / "selected_feature_overlap.png", dpi=300)
    fig.savefig(output_dir / "selected_feature_overlap.svg")
    plt.close(fig)


def save_hyperparameter_table(results_root: Path, models: list[str], output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model in models:
        median = read_json(results_root / model / "median_hyperparameters.json")
        if median is None:
            continue
        best_params = median.get("best_params", {})
        for parameter, value in best_params.items():
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model).replace("\n", " "),
                    "parameter": parameter,
                    "value": value,
                }
            )

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "median_hyperparameters_long.csv", index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root / "presentation_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary_rows(results_root, args.models)
    fold_df = load_fold_rows(results_root, args.models)
    feature_df, feature_sets = load_feature_rows(results_root, args.models)
    family_df = load_feature_family_rows(feature_sets)

    if not summary_df.empty:
        summary_df.to_csv(output_dir / "metric_summary_long.csv", index=False)
    if not fold_df.empty:
        fold_df.to_csv(output_dir / "fold_metrics_long.csv", index=False)
    if not feature_df.empty:
        feature_df.to_csv(output_dir / "selected_feature_summary.csv", index=False)
    if not family_df.empty:
        family_df.to_csv(output_dir / "selected_feature_families.csv", index=False)
    save_hyperparameter_table(results_root, args.models, output_dir)

    sns.set_theme(style="whitegrid", context="talk")
    save_f1_comparison(summary_df, fold_df, output_dir)

    LOGGER.info("Saved the F1 comparison figure and supporting tables to %s", output_dir)


if __name__ == "__main__":
    main()
