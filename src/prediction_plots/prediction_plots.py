"""
File: prediction_plots.py (place at src/prediction_plots.py)
Save per-fold GLOC predictions during a run, then plot actual vs. predicted
GLOC over time with the centrifuge G magnitude overlaid, one figure per
subject (trials laid end to end, dashed lines at trial boundaries).
Works for both pipeline configurations:
train on original / evaluate on new file -> every test window is a new-subject window
combined dataset + stratified k-fold -> concatenating all folds gives
out-of-fold predictions for every window
Usage after a run:
python -m src.prediction_plots Results/<save_results_folder>/Complete_Common/KNN/ECG-BR-Temperature-Centrifuge \
--title-suffix "trained on original data, evaluated on new subjects"
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Any, Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
logger = logging.getLogger(__name__)
ACTUAL_COLOR = "black"
PREDICTED_COLOR = "#c8506e"
G_COLOR = "#4682b4"
PREDICTED_OFFSET = 0.04 # draw predicted slightly below actual so both stay visible when they overlap
print("prediction plotting script started", flush=True)
def save_fold_predictions(
output_dir: str | Path,
kfold_id: int,
metadata: dict[str, Any] | None,
y_true: np.ndarray,
y_pred: np.ndarray,
) -> Path | None:
    """Write one CSV of test-window predictions for a fold.
    ``metadata`` is ``pipeline.last_split_metadata`` from the same
    ``pipeline.get_data(...)`` call that produced ``y_true``.
    """
    if metadata is None:
        logger.warning("No split metadata available (advanced pipeline?) - predictions not saved.")
        return None
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    trial_ids = np.asarray(metadata["test_trial_id"]).astype(str)
    if not (len(trial_ids) == len(y_true) == len(y_pred)):
        raise ValueError(
        f"Row mismatch: metadata={len(trial_ids)}, y_true={len(y_true)}, y_pred={len(y_pred)}. "
        "Metadata must come from the same get_data() call as y_test."
        )
    rank = {str(t): i for i, t in enumerate(metadata["trial_order"])}
    lengths = {str(t): int(n) for t, n in metadata["trial_lengths"].items()}
    frame = pd.DataFrame(
    {
    "fold": kfold_id,
    "trial_id": trial_ids,
    "trial_rank": [rank[t] for t in trial_ids],
    "trial_length": [lengths[t] for t in trial_ids],
    "window_idx": np.asarray(metadata["test_window_idx"]).astype(int),
    "g_mean": np.asarray(metadata["test_g_mean"], dtype=float),
    "y_true": y_true.astype(int),
    "y_pred": y_pred.astype(int),
    }
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"predictions_fold_{kfold_id}.csv"
    frame.to_csv(path, index=False)
    return path

def load_predictions(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype={"trial_id": str}) for p in paths]
    return pd.concat(frames, ignore_index=True)

def _subject_of(trial_id: str) -> str:
    return trial_id.split("-")[0]

def _subject_label(subject: str) -> str:
    return str(int(subject)) if subject.isdigit() else subject

def build_subject_timeline(subject_df: pd.DataFrame) -> dict[str, Any]:
    """Lay a subject's trials end to end. Windows not in any test fold stay NaN (gaps)."""
    trials = (
    subject_df[["trial_id", "trial_rank", "trial_length"]]
    .drop_duplicates("trial_id")
    .sort_values("trial_rank")
    )
    actual, predicted, g_mean = [], [], []
    starts, labels, offset = [], [], 0
    for trial_id, length in zip(trials["trial_id"], trials["trial_length"]):
        length = int(length)
        seg = subject_df[subject_df["trial_id"] == trial_id]
        idx = seg["window_idx"].to_numpy(dtype=int)
        keep = idx < length
        a = np.full(length, np.nan)
        p = np.full(length, np.nan)
        g = np.full(length, np.nan)
        a[idx[keep]] = seg["y_true"].to_numpy(dtype=float)[keep]
        p[idx[keep]] = seg["y_pred"].to_numpy(dtype=float)[keep]
        g[idx[keep]] = seg["g_mean"].to_numpy(dtype=float)[keep]
        actual.append(a)
        predicted.append(p)
        g_mean.append(g)
        starts.append(offset)
        labels.append(trial_id)
        offset += length
    return {
    "actual": np.concatenate(actual),
    "predicted": np.concatenate(predicted),
    "g_mean": np.concatenate(g_mean),
    "starts": starts,
    "labels": labels,
    "total": offset,
    }

def plot_subject(subject: str, subject_df: pd.DataFrame, model_name: str, title_suffix: str):
    tl = build_subject_timeline(subject_df)
    x = np.arange(tl["total"])
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(x, tl["actual"], color=ACTUAL_COLOR, lw=1.8, drawstyle="steps-post")
    ax.plot(x, tl["predicted"] - PREDICTED_OFFSET, color=PREDICTED_COLOR, lw=1.5, drawstyle="steps-post")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No GLOC", "GLOC"], fontsize=12)
    ax.set_ylim(-0.2, 1.3)
    ax.set_xlim(-0.03 * tl["total"], 1.03 * tl["total"])
    ax.set_xlabel("Window index (time, ordered by trial)", fontsize=12)
    ax_g = ax.twinx()
    ax_g.plot(x, tl["g_mean"], color=G_COLOR, lw=1.0, alpha=0.8)
    ax_g.set_ylabel("Centrifuge G magnitude (G)", color=G_COLOR, fontsize=12)
    ax_g.tick_params(axis="y", colors=G_COLOR)
    ends = tl["starts"][1:] + [tl["total"]]
    for start, end, label in zip(tl["starts"], ends, tl["labels"]):
        ax.axvline(start, color="gray", ls="--", lw=0.6)
        ax.text((start + end) / 2, 0.93, label, transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=9, family="serif")
    ax.set_title(
    f"{model_name} Subject {_subject_label(subject)}: Actual vs Predicted GLOC Over Time ({title_suffix})",
    fontsize=14,
    )
    handles = [
    Line2D([0], [0], color=ACTUAL_COLOR, lw=1.8, label="Actual GLOC"),
    Line2D([0], [0], color=PREDICTED_COLOR, lw=1.5, label="Predicted GLOC"),
    Line2D([0], [0], color=G_COLOR, lw=1.0, label="Centrifuge G magnitude (mean per prediction window)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig

def plot_all_subjects(
results_dir: str | Path,
model_name: str | None = None,
title_suffix: str = "held-out",
subjects: list[str] | None = None,
out_dir: str | Path | None = None,
show: bool = False,
) -> list[Path]:
    """Plot every subject found in ``predictions_fold_.csv`` under ``results_dir``."""
    results_dir = Path(results_dir)
    paths = sorted(results_dir.glob("predictions_fold_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No predictions_fold_.csv files in {results_dir}")
    model_name = model_name or results_dir.parent.name
    out_dir = Path(out_dir) if out_dir else results_dir / "prediction_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(paths)
    df["subject"] = df["trial_id"].map(_subject_of)
    if subjects:
        wanted = {_subject_label(s) for s in subjects}
        df = df[df["subject"].map(_subject_label).isin(wanted)]
        if df.empty:
            raise ValueError(f"None of the requested subjects {subjects} are in the predictions.")
    saved: list[Path] = []
    for subject, subject_df in df.groupby("subject", sort=False):
        fig = plot_subject(subject, subject_df, model_name, title_suffix)
        path = out_dir / f"{model_name}_subject_{_subject_label(subject)}_actual_vs_predicted.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        saved.append(path)
        if show:
            plt.show()
        plt.close(fig)
    return saved

def main() -> None:
    print("prediction plotting script started", flush=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", help="Folder containing predictions_fold_.csv")
    parser.add_argument("--model", default=None, help="Model name for the title (default: parent folder name)")
    parser.add_argument("--subjects", nargs="*", default=None, help="Only plot these subjects, e.g. 5 165")
    parser.add_argument("--title-suffix", default="held-out")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.show:
        plt.switch_backend("Agg")
    for path in plot_all_subjects(args.results_dir, args.model, args.title_suffix, args.subjects, args.out_dir, args.show):
        print(f"Saved {path}")

if __name__ == "__main__":
    main()