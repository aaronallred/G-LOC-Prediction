"""Create an XGBoost-only violin plot from saved fold_result.json files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "specificity": "Specificity",
    "g_mean": "G-mean",
}


def load_results(model_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(model_dir.glob("fold_*/fold_result.json")):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        for metric, label in METRIC_LABELS.items():
            rows.append(
                {
                    "fold": int(result["fold"]),
                    "metric": label,
                    "score": float(result["metrics"][metric]),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No fold_result.json files found under {model_dir}")
    return pd.DataFrame(rows)


def create_plot(data: pd.DataFrame, output: Path) -> None:
    fold_count = data["fold"].nunique()
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    sns.violinplot(
        data=data,
        x="metric",
        y="score",
        hue="metric",
        order=list(METRIC_LABELS.values()),
        palette="Blues",
        inner="quartile",
        cut=0,
        linewidth=1.2,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="metric",
        y="score",
        order=list(METRIC_LABELS.values()),
        color="#17202a",
        size=7,
        jitter=0.06,
        alpha=0.85,
        ax=ax,
    )

    ax.set_title(f"XGBoost Cross-Validation Performance ({fold_count} Folds)", weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("Results/Cross_Validation_XGB/Complete_Explicit/XGB"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.model_dir / "xgboost_violin_plot.png"
    data = load_results(args.model_dir)
    create_plot(data, output)
    print(f"Loaded {data['fold'].nunique()} XGBoost folds.")
    print(f"Saved {output} and {output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
