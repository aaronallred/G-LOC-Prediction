"""Create violin plots from legacy 10-fold performance_metrics.csv files.

Example:
    python -m src.scripts.plot_cv_violin \
        --run-dir ModelSave/CV/Complete_Explicit
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_MODELS = ["LogRegTS", "NAM", "LSTM", "Transformer", "TCN"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1-score": "F1 Score",
    "specificity": "Specificity",
    "g mean": "G-mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot saved cross-validation scores as violins.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("ModelSave/CV/Complete_Explicit"),
        help="Directory containing numbered fold folders.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names in the same order as the rows in each CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path (default: <run-dir>/cv_violin_plot.png).",
    )
    return parser.parse_args()


def load_fold_metrics(run_dir: Path, models: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    fold_dirs = sorted(
        (path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )

    for fold_dir in fold_dirs:
        csv_path = fold_dir / "performance_metrics.csv"
        if not csv_path.is_file():
            continue

        frame = pd.read_csv(csv_path)
        if len(frame) < len(models):
            raise ValueError(
                f"{csv_path} has {len(frame)} rows, but {len(models)} model names were supplied."
            )

        # These legacy files were append-only. The final block is the latest run.
        latest = frame.tail(len(models)).reset_index(drop=True)
        for model, (_, row) in zip(models, latest.iterrows()):
            for metric in METRIC_LABELS:
                records.append(
                    {
                        "fold": int(fold_dir.name),
                        "model": model,
                        "metric": metric,
                        "score": float(row[metric]),
                    }
                )

    data = pd.DataFrame(records)
    if data.empty:
        raise FileNotFoundError(f"No fold performance files found beneath {run_dir}")
    return data


def create_plot(data: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    metrics = list(METRIC_LABELS)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True, constrained_layout=True)
    palette = sns.color_palette("Set2", n_colors=data["model"].nunique())

    for ax, metric in zip(axes.flat, metrics):
        subset = data[data["metric"] == metric]
        sns.violinplot(
            data=subset,
            x="model",
            y="score",
            hue="model",
            palette=palette,
            inner="quartile",
            cut=0,
            linewidth=1,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=subset,
            x="model",
            y="score",
            color="#202020",
            size=4,
            jitter=0.10,
            alpha=0.7,
            ax=ax,
        )
        ax.set_title(METRIC_LABELS[metric], weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score" if ax in axes[:, 0] else "")
        ax.set_ylim(0, 1.03)
        ax.tick_params(axis="x", rotation=25, labelsize=10)

    fig.suptitle("10-Fold Cross-Validation Performance", fontsize=22, weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output or args.run_dir / "cv_violin_plot.png"
    data = load_fold_metrics(args.run_dir, args.models)
    create_plot(data, output)
    print(f"Loaded {data['fold'].nunique()} folds for {data['model'].nunique()} models.")
    print(f"Saved {output} and {output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
