#!/usr/bin/env python3
"""Comprehensive analysis & report generator for `traditional_standardization_metrics.py`.

This script is a pure downstream consumer: it reads the JSON results previously
produced by ``src/real_time/traditional_standardization_metrics.py`` (which
compare train-only vs all-rows -- i.e. fold-aware vs legacy leaky -- z-score
standardization) and:

1.  Walks the per-model / per-fold / per-feature / stratified JSON trees into
    `pandas.DataFrame` tables.
2.  Computes per-model headline statistics, fold-stability, top-offending
    features, and severity-classified leakage verdicts.
3.  Runs the optional scipy Wilcoxon / Kruskal-Wallis statistical tests.
4.  Renders publication-quality PNG charts (matplotlib + seaborn).
5.  Writes tidy CSV export tables alongside the existing raw per-fold JSON.
6.  Emits an aggregated ``analysis.json`` and a Markdown narrative report
    suitable for committing to the repo or pasting into a paper.

The script never invokes the data pipeline itself; all analysis is offline
against the already-saved JSON in ``Results/Traditional_Standardization_Metrics``.

Usage::

    python -m src.real_time.standardization_metrics_analysis
    python -m src.real_time.standardization_metrics_analysis --results-dir Results/Traditional_Standardization_Metrics
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

# Use the non-interactive Agg backend so the script runs headless on servers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Default results directory that ``traditional_standardization_metrics.py``
# writes to (the producer hardcodes ``Results/Traditional_Standardization_Metrics``).
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "Results" / "Traditional_Standardization_Metrics"

# Subdirectory (created under the results dir) that holds all analysis outputs
# (Markdown, JSON, CSVs, PNGs).
DEFAULT_OUTPUT_SUBDIR = "analysis"

# Plotting style settings; published as a sibling script
# (``prediction_latency_analysis.py``) follows the same palette.
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    }
)

# Ordered enumerations used to keep stratified tables in canonical order:
FEATURE_TYPE_ORDER = ["mean", "stddev", "max", "range", "additional"]
BASELINE_METHOD_ORDER = ["v0", "v1", "v2", "v5", "v6", "none"]
S1_S2_ORDER = ["s1", "s2"]

# Fixed colour palette for the two standardization types so that s1 / s2 are
# rendered with the same colour in every plot that uses them as a hue.
S1_S2_PALETTE = {"s1": "#1f77b4", "s2": "#ff7f0e"}  # blue=s1, orange=s2

# Default severity thresholds (expressed in z-score units). The thresholds
# classify how much a single test-row's z-score would shift between the old
# (leaky) and new (fold-aware) standardization. Each threshold compares
# against a per-fold statistic reported in the JSON:
#
#   - ``max_abs_delta`` per fold  -> single worst test row's z-score shift.
#                                    1.0 = one full standard-deviation.
#   - ``mean_abs_delta`` per fold -> mean shift averaged across all test rows
#                                    and all (888) target features in the fold.
#
# A fold is classified by taking the *worst* label its metrics trigger.
DEFAULT_BENIGN_MEAN_ABS = 0.005
DEFAULT_MODERATE_MEAN_ABS = 0.05
DEFAULT_BENIGN_MAX_ABS = 0.1
DEFAULT_MODERATE_MAX_ABS = 1.0


# ---------------------------------------------------------------------------
# Loading & flattening JSON into pandas DataFrames
# ---------------------------------------------------------------------------
def load_summary(results_dir: Path) -> dict[str, Any]:
    """Read the top-level ``summary.json`` produced by the pipeline.

    The file lives at ``<results_dir>/summary.json`` and contains::

        {
          "config":   {num_splits, random_seed, model_type_string,
                       feature_streams, target_substrings},
          "models": [
            {"name": ...,
             "baseline_window": float, "window_size": float, "stride": float,
             "summary": {... across_folds aggregates ...},
             "folds": [per-fold dictionary, ...]
            }, ...
          ]
        }
    """
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"summary.json not found at {summary_path}. "
            "Run `python -m src.real_time.traditional_standardization_metrics` first."
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_per_fold_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """Flatten the per-model ``folds`` list into one row per (model, fold).

    Returned columns (each row = one CV fold, one model)::

        model              : str   -- e.g. "KNN"
        fold_id            : int   -- 0 .. num_splits-1
        n_train_rows       : int   -- rows in train fold (post-NaN-removal)
        n_test_rows        : int   -- rows in test fold (post-NaN-removal)
        n_total_features_cols : int -- width of fully standardized matrix
                                       (== 2 * raw feature count, since s1|s2)
        n_target_features  : int   -- subset of features whose name matches
                                      "Equivital" or "Centrifuge" (the streams
                                      under ablation); the rest are EEG etc.
        mean_abs_delta     : float -- mean |fold-aware - leaky| over all test
                                      rows × target features
        median_abs_delta   : float -- median of the same population
        max_abs_delta      : float -- single worst test-row divergent shift
        std_abs_delta      : float -- population std-dev of |delta|

    Source: ``models[i].folds[j].summary`` block in summary.json.
    """
    records: list[dict[str, Any]] = []
    for model_entry in summary["models"]:
        model_name = model_entry["name"]  # from ModelFactory().create_model(name)
        for fold in model_entry["folds"]:
            s = fold["summary"]  # the per-fold aggregate stats dict
            records.append(
                {
                    "model": model_name,
                    "fold_id": int(fold["fold_id"]),
                    "n_train_rows": int(fold["n_train_rows"]),
                    "n_test_rows": int(fold["n_test_rows"]),
                    "n_total_features_cols": int(fold["n_total_features_cols"]),
                    "n_target_features": int(fold["n_target_features"]),
                    "mean_abs_delta": float(s["mean_abs_delta"]),
                    "median_abs_delta": float(s["median_abs_delta"]),
                    "max_abs_delta": float(s["max_abs_delta"]),
                    "std_abs_delta": float(s["std_abs_delta"]),
                }
            )
    return pd.DataFrame.from_records(records)


def build_per_feature_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """Flatten the per-fold ``per_feature`` dict into one row per
    (model, fold, feature_name).

    Returned columns::

        model           : str   -- e.g. "RF"
        fold_id         : int   -- 0 .. num_splits-1
        feature_name    : str   -- e.g. "HR (bpm) - Equivital_v0_mean_s1"
        mean_abs_delta  : float -- mean |fold-aware - leaky| for this single
                                    feature across all test rows of this fold
        max_abs_delta   : float -- worst single test-row |delta| for this feature
        mean_delta      : float -- signed mean of (fold-aware - leaky); sign
                                    shows *which* standardizer gives larger z
        std_delta       : float -- std-dev of (fold-aware - leaky) over test rows
        s1_or_s2        : str   -- derived from feature_name suffix; "s1" =
                                    per-trial standardization, "s2" = pooled
        feature_type    : str   -- one of {mean, stddev, max, range, additional}
        baseline_method : str   -- one of {v0, v1, v2, v5, v6, none}

    Source: ``models[i].folds[j].per_feature`` in summary.json.
    """
    records: list[dict[str, Any]] = []
    for model_entry in summary["models"]:
        model_name = model_entry["name"]
        for fold in model_entry["folds"]:
            fold_id = int(fold["fold_id"])
            for fname, stats_dict in fold["per_feature"].items():
                records.append(
                    {
                        "model": model_name,
                        "fold_id": fold_id,
                        "feature_name": fname,
                        "mean_abs_delta": float(stats_dict["mean_abs_delta"]),
                        "max_abs_delta": float(stats_dict["max_abs_delta"]),
                        "mean_delta": float(stats_dict["mean_delta"]),
                        "std_delta": float(stats_dict["std_delta"]),
                        "s1_or_s2": stats_dict.get("s1_or_s2", "unknown"),
                        "feature_type": stats_dict.get("feature_type", "unknown"),
                        "baseline_method": stats_dict.get("baseline_method", "none"),
                    }
                )
    return pd.DataFrame.from_records(records)


def build_stratified_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """Flatten the per-fold ``stratified_summary`` dict into one row per
    (model, fold, s1_or_s2, feature_type, baseline_method) bucket.

    For each fold the producer pre-aggregates over all features that share
    a (s1_or_s2 × feature_type × baseline_method) triple, computing the mean
    across those features' per-feature ``mean_abs_delta`` etc. This table
    exposes those bucketed aggregates.

    Returned columns::

        model           : str
        fold_id         : int
        s1_or_s2        : str   -- "s1" or "s2"
        feature_type    : str   -- {mean, stddev, max, range, additional}
        baseline_method : str   -- {v0, v1, v2, v5, v6, none}
        mean_abs_delta  : float -- mean of per-feature mean_abs_delta inside
                                    this bucket (averaged over its features)
        max_abs_delta   : float -- mean of per-feature max_abs_delta inside
                                    this bucket
        mean_delta      : float -- mean of per-feature mean_delta (signed)
        std_delta       : float -- mean of per-feature std_delta
        n_features      : int   -- count of features inside this bucket
                                   (constant across folds for a given model
                                   within a fixed config; varies across models
                                   because different models use different
                                   feature reduction baseline-derivative sets)

    Source: ``models[i].folds[j].stratified_summary`` in summary.json.
    """
    records: list[dict[str, Any]] = []
    for model_entry in summary["models"]:
        model_name = model_entry["name"]
        for fold in model_entry["folds"]:
            fold_id = int(fold["fold_id"])
            strat = fold.get("stratified_summary", {})
            for s1_s2, ft_dict in strat.items():
                for feature_type, bm_dict in ft_dict.items():
                    for baseline_method, stats_dict in bm_dict.items():
                        records.append(
                            {
                                "model": model_name,
                                "fold_id": fold_id,
                                "s1_or_s2": s1_s2,
                                "feature_type": feature_type,
                                "baseline_method": baseline_method,
                                "mean_abs_delta": float(stats_dict["mean_abs_delta"]),
                                "max_abs_delta": float(stats_dict["max_abs_delta"]),
                                "mean_delta": float(stats_dict["mean_delta"]),
                                "std_delta": float(stats_dict["std_delta"]),
                                "n_features": int(stats_dict["n_features"]),
                            }
                        )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Computed aggregates
# ---------------------------------------------------------------------------
def compute_model_statistics(per_fold_df: pd.DataFrame) -> pd.DataFrame:
    """One row per model aggregating the per-fold rows into headline numbers.

    Returned columns::

        model                    : str
        n_folds                  : int   -- how many folds contributed
        mean_abs_delta_mean      : float -- mean across folds of per-fold
                                            ``mean_abs_delta``
        mean_abs_delta_median    : float -- median across folds
        mean_abs_delta_std       : float -- std across folds (fold stability)
        mean_abs_delta_max       : float -- worst fold's mean_abs_delta
        max_abs_delta_mean       : float -- mean across folds of per-fold
                                            ``max_abs_delta``
        max_abs_delta_max        : float -- globally worst single test-row
                                            shift across all folds
        n_test_rows_total        : int   -- sum of per-fold n_test_rows
        n_target_features         : int   -- (constant across folds; taken
                                             from first fold's value)
        n_train_rows_mean        : float -- mean over folds of n_train_rows
    """
    grouped = per_fold_df.groupby("model", sort=False)
    rows: list[dict[str, Any]] = []
    for model_name, g in grouped:
        rows.append(
            {
                "model": model_name,
                "n_folds": int(g.shape[0]),
                "mean_abs_delta_mean": float(g["mean_abs_delta"].mean()),
                "mean_abs_delta_median": float(g["mean_abs_delta"].median()),
                "mean_abs_delta_std": float(g["mean_abs_delta"].std(ddof=0)),
                "mean_abs_delta_max": float(g["mean_abs_delta"].max()),
                "max_abs_delta_mean": float(g["max_abs_delta"].mean()),
                "max_abs_delta_max": float(g["max_abs_delta"].max()),
                "n_test_rows_total": int(g["n_test_rows"].sum()),
                "n_target_features": int(g["n_target_features"].iloc[0]),
                "n_train_rows_mean": float(g["n_train_rows"].mean()),
            }
        )
    return pd.DataFrame(rows)


def identify_top_offenders(per_feature_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Highest-divergence features per model, ranked by ``max_abs_delta``.

    For each model, take every (fold, feature_name) row from ``per_feature_df``
    and keep the ``top_n`` rows whose ``max_abs_delta`` is the greatest.
    The returned DataFrame concatenates the top-N across all models.

    Returned columns (same as the per-feature table, plus ``rank``)::

        model, fold_id, feature_name, mean_abs_delta, max_abs_delta,
        mean_delta, std_delta, s1_or_s2, feature_type, baseline_method, rank
    """
    out_frames: list[pd.DataFrame] = []
    for model_name, g in per_feature_df.groupby("model", sort=False):
        # Default Pandas sort is ascending; we want biggest delta first.
        ranked = g.sort_values("max_abs_delta", ascending=False).head(top_n).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        out_frames.append(ranked)
    return pd.concat(out_frames, ignore_index=True)


def classify_leakage_severity(
        stats_df: pd.DataFrame,
        benign_mean_abs: float,
        moderate_mean_abs: float,
        benign_max_abs: float,
        moderate_max_abs: float,
) -> pd.DataFrame:
    """Apply threshold constants to each model's headline statistics and
    bucket the models into benign / moderate / severe leakage severity.

    The thresholds are interpreted as follows:

      * The "mean" family classifies the *typical* row's z-score shift.
      * The "max" family classifies the *worst single row's* z-score shift.

    A model is classified as ``severe`` if *either* family crosses its
    moderate threshold, ``moderate`` if either crosses the benign threshold
    (but neither is severe), and ``benign`` otherwise.

    Returned columns::

        model                  : str
        mean_abs_delta_mean    : float  -- from stats_df
        max_abs_delta_max      : float  -- from stats_df
        mean_severity          : str    -- {benign, moderate, severe}
        max_severity           : str
        overall_severity       : str    -- worst of the two
        triggers               : str    -- human-readable short string,
                                           e.g. "max>1.0 (1.42)"
    """
    rows: list[dict[str, Any]] = []
    for _, r in stats_df.iterrows():
        mean_val = r["mean_abs_delta_mean"]
        max_val = r["max_abs_delta_max"]
        if mean_val >= moderate_mean_abs:
            mean_sev = "severe"
        elif mean_val >= benign_mean_abs:
            mean_sev = "moderate"
        else:
            mean_sev = "benign"
        if max_val >= moderate_max_abs:
            max_sev = "severe"
        elif max_val >= benign_max_abs:
            max_sev = "moderate"
        else:
            max_sev = "benign"
        severity_rank = {"benign": 0, "moderate": 1, "severe": 2}
        overall = max(mean_sev, max_sev, key=lambda s: severity_rank[s])
        triggers: list[str] = []
        if mean_sev != "benign":
            triggers.append(f"mean>{benign_mean_abs if mean_sev == 'moderate' else moderate_mean_abs} ({mean_val:.4f})")
        if max_sev != "benign":
            triggers.append(f"max>{benign_max_abs if max_sev == 'moderate' else moderate_max_abs} ({max_val:.4f})")
        rows.append(
            {
                "model": r["model"],
                "mean_abs_delta_mean": mean_val,
                "max_abs_delta_max": max_val,
                "mean_severity": mean_sev,
                "max_severity": max_sev,
                "overall_severity": overall,
                "triggers": "; ".join(triggers) if triggers else "none",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests (optional but useful sanity checks)
# ---------------------------------------------------------------------------
def perform_wilcoxon_test(per_fold_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Wilcoxon signed-rank test of H0: per-fold ``mean_abs_delta`` is centered at 0.

    Because ``mean_abs_delta`` is the absolute value of a difference, it is
    strictly positive, so the test is mathematically guaranteed to reject
    H0 -- this function is included only as a sanity check that the values
    are not statistically zero (which would mean fold-aware and leaky
    standardization are identical).
    """
    rows: list[dict[str, Any]] = []
    for model_name, g in per_fold_df.groupby("model", sort=False):
        x = g["mean_abs_delta"].to_numpy()
        if len(x) >= 2 and np.all(x > 0):
            try:
                stat_w, p_value = stats.wilcoxon(x)
            except ValueError:
                # Wilcoxon can fail if the diff array collapses to constant 0.
                stat_w, p_value = float("nan"), float("nan")
        else:
            stat_w, p_value = float("nan"), float("nan")
        rows.append(
            {
                "model": model_name,
                "wilcoxon_stat": float(stat_w) if not np.isnan(stat_w) else None,
                "p_value": float(p_value) if not np.isnan(p_value) else None,
                "reject_h0_at_0_05": bool(p_value < 0.05) if not np.isnan(p_value) else False,
                "interpretation": (
                    "mean_abs_delta is statistically distinguishable from zero"
                    if (not np.isnan(p_value) and p_value < 0.05)
                    else "not distinguishable (insufficient signal or sample)"
                ),
            }
        )
    return rows


def perform_fold_stability_test(per_fold_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Kruskal-Wallis H test across folds for each model.

    For each model we have ``num_splits`` data points (one per fold). The
    Kruskal-Wallis test asks whether these folds are drawn from the same
    distribution -- i.e., whether some folds leak more than others (e.g.,
    if a fold happens to contain trials whose statistics are far from the
    training pool, the leaky standardization diverges more from the
    fold-aware one).
    """
    rows: list[dict[str, Any]] = []
    for model_name, g in per_fold_df.groupby("model", sort=False):
        # We have one point per fold; Kruskal-Wallis requires groupings.
        # Here treat each fold as a group of one and compare across folds;
        # scipy.stats.kruskal(*list_of_samples) is happy with singletons but
        # cannot reject H0. Fall back to descriptive CV instead.
        x = g["mean_abs_delta"].to_numpy()
        cv = float(np.std(x, ddof=0) / np.mean(x)) if np.mean(x) > 0 else 0.0
        rows.append(
            {
                "model": model_name,
                "mean_abs_delta_cv": cv,
                "fold_min": float(np.min(x)),
                "fold_max": float(np.max(x)),
                "fold_span_ratio": float(np.max(x) / np.min(x)) if np.min(x) > 0 else float("inf"),
                "interpretation": (
                    "folds are stable (CV<0.25)"
                    if cv < 0.25
                    else "folds drift (CV>=0.25) -- some folds diverge more than others"
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------
def plot_per_fold_deltas(per_fold_df: pd.DataFrame, out_dir: Path) -> Path:
    """Two-panel bar chart of per-fold mean and max abs deltas.

    Left panel:  bar plot ``mean_abs_delta`` per fold, hue=model.
    Right panel: bar plot ``max_abs_delta`` per fold on log-y, hue=model.

    Returns the saved PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(
        data=per_fold_df,
        x="fold_id",
        y="mean_abs_delta",
        hue="model",
        ax=axes[0],
        errorbar=None,
    )
    axes[0].set_title("Per-fold mean |Δ| (test rows × target features)")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("mean |Δ| (z-score units)")
    axes[0].legend(title="Model")

    sns.barplot(
        data=per_fold_df,
        x="fold_id",
        y="max_abs_delta",
        hue="model",
        ax=axes[1],
        errorbar=None,
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Per-fold max |Δ| (single worst test row)")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("max |Δ|  (log scale, z-score units)")
    axes[1].legend(title="Model")

    fig.suptitle("Per-fold standardization delta: fold-aware vs legacy leaky")
    fig.tight_layout()
    out_path = out_dir / "per_fold_deltas.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_s1_vs_s2_distribution(per_feature_df: pd.DataFrame, out_dir: Path) -> Path:
    """Box plot of per-feature ``mean_abs_delta`` broken down by
    (s1_or_s2 × model); inner strip overlay shows individual features.

    Each box aggregates the 4×10×99 ≈ 3960 per-feature mean_abs_delta values
    (4 feature types × 10 folds × ~99 features per feature_type per fold per
    s1/s2 -- varies slightly based on baseline-method availability).

    Returns the saved PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=per_feature_df,
        x="model",
        y="mean_abs_delta",
        hue="s1_or_s2",
        hue_order=S1_S2_ORDER,
        palette=S1_S2_PALETTE,
        showfliers=False,
        ax=ax,
    )
    ax.set_title("Per-feature mean |Δ| distribution by standardization type")
    ax.set_xlabel("Model")
    ax.set_ylabel("per-feature mean |Δ| (z-score units)")
    ax.legend(title="s1=per-trial, s2=pooled")
    fig.tight_layout()
    out_path = out_dir / "s1_vs_s2_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_feature_type_heatmap(strat_df: pd.DataFrame, out_dir: Path) -> Path:
    """3-panel heatmap (one per model): rows=feature_type, cols=baseline_method,
    cell colour=mean across folds of bucket's ``mean_abs_delta``.
    Separate heatmaps (top): s1, (bottom): s2.

    Returns the saved PNG path.
    """
    models = sorted(strat_df["model"].unique())
    feat_types = [ft for ft in FEATURE_TYPE_ORDER if ft in strat_df["feature_type"].unique()]
    bm_methods = [bm for bm in BASELINE_METHOD_ORDER if bm in strat_df["baseline_method"].unique()]

    fig, axes = plt.subplots(
        len(models),
        2,
        figsize=(10, 3.2 * len(models)),
        sharex=True,
        sharey=True,
    )
    if len(models) == 1:
        axes = axes.reshape(1, -1)

    # Compute a shared colour scale so all panels are comparable.
    vmin = float(strat_df["mean_abs_delta"].min())
    vmax = float(strat_df["mean_abs_delta"].max())

    for i, model_name in enumerate(models):
        for j, s1_s2 in enumerate(S1_S2_ORDER):
            sub = strat_df[
                (strat_df["model"] == model_name)
                & (strat_df["s1_or_s2"] == s1_s2)
                ]
            pivot = (
                sub.groupby(["feature_type", "baseline_method"])["mean_abs_delta"]
                .mean()
                .unstack(level="baseline_method")
                .reindex(index=feat_types, columns=bm_methods)
            )
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".4f",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                ax=axes[i, j],
                cbar=j == 1,
            )
            axes[i, j].set_title(f"{model_name} | {s1_s2}")
            axes[i, j].set_xlabel("Baseline method")
            axes[i, j].set_ylabel("Feature type")
    fig.suptitle("Stratified mean |Δ| by (model × s1/s2 × feature_type × baseline_method)")
    fig.tight_layout()
    out_path = out_dir / "feature_type_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_top_offenders(top_df: pd.DataFrame, out_dir: Path, top_n: int = 10) -> Path:
    """One subplot per model: horizontal bar chart of the top-N worst features
    ranked by ``max_abs_delta``. Colors encode s1 vs s2.

    Returns the saved PNG path.
    """
    models = sorted(top_df["model"].unique())
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 4 * len(models)), sharex=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, models):
        sub = (
            top_df[top_df["model"] == model_name]
            .sort_values("max_abs_delta", ascending=False)
            .head(top_n)
            .copy()
        )
        # Shorten feature names so the y-axis fits without truncation.
        labels = [n if len(n) <= 60 else (n[:57] + "...") for n in sub["feature_name"]]
        sns.barplot(
            data=sub,
            y=labels,
            x="max_abs_delta",
            hue="s1_or_s2",
            hue_order=S1_S2_ORDER,
            palette=S1_S2_PALETTE,
            dodge=False,
            ax=ax,
        )
        ax.set_title(f"{model_name} -- top {len(sub)} features by max |Δ|")
        ax.set_xlabel("max |Δ| (z-score units)")
        ax.set_ylabel("Feature name")
        ax.legend(title="s1/s2", loc="lower right")
    fig.suptitle("Top offending features (single worst row divergence)")
    fig.tight_layout()
    out_path = out_dir / "top_offenders.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fold_distribution(per_fold_df: pd.DataFrame, out_dir: Path) -> Path:
    """Violin+strip plot showing the per-fold distribution of mean_abs_delta
    and max_abs_delta per model.

    Returns the saved PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.violinplot(
        data=per_fold_df,
        x="model",
        y="mean_abs_delta",
        inner="point",
        ax=axes[0],
    )
    axes[0].set_title("Distribution of per-fold mean |Δ|")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("mean |Δ| (z-score units)")

    sns.violinplot(
        data=per_fold_df,
        x="model",
        y="max_abs_delta",
        inner="point",
        ax=axes[1],
    )
    axes[1].set_title("Distribution of per-fold max |Δ|")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("max |Δ| (z-score units)")
    fig.tight_layout()
    out_path = out_dir / "fold_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------
def write_csv_tables(
        per_fold_df: pd.DataFrame,
        strat_df: pd.DataFrame,
        per_feature_df: pd.DataFrame,
        top_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        severity_df: pd.DataFrame,
        out_dir: Path,
) -> list[Path]:
    """Persist tidy CSV exports of all the per-model/fold/feature tables."""
    paths: list[Path] = []
    mapping = {
        "per_fold_summary.csv": per_fold_df,
        "stratified_summary.csv": strat_df,
        "per_feature_summary.csv": per_feature_df,
        "top_offenders.csv": top_df,
        "model_statistics.csv": stats_df,
        "leakage_severity.csv": severity_df,
    }
    for name, df in mapping.items():
        p = out_dir / name
        df.to_csv(p, index=False)
        paths.append(p)
        logger.info("Wrote CSV %s", p)
    return paths


def write_aggregated_json(
        stats_df: pd.DataFrame,
        severity_df: pd.DataFrame,
        wilcoxon: list[dict[str, Any]],
        fold_stability: list[dict[str, Any]],
        top_df: pd.DataFrame,
        config: dict[str, Any],
        out_path: Path,
) -> Path:
    """Compact aggregated JSON containing every number a downstream tool
    (numbered figure, table, etc.) might need without re-parsing raw data.
    """
    aggregated = {
        "config": config,
        "model_statistics": stats_df.to_dict(orient="records"),
        "leakage_severity": severity_df.to_dict(orient="records"),
        "statistical_tests": {
            "wilcoxon_signed_rank": wilcoxon,
            "fold_stability_cv": fold_stability,
        },
        "top_offenders_by_model": {
            model_name: top_df[top_df["model"] == model_name]
            .drop(columns=["model"])
            .to_dict(orient="records")
            for model_name in sorted(top_df["model"].unique())
        },
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregated, handle, indent=2, default=str)
    logger.info("Wrote aggregated JSON %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a pandas DataFrame to a Markdown pipe table (lightweight)."""
    if df.empty:
        return "_(no data)_\n"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def generate_markdown_report(
        config: dict[str, Any],
        per_fold_df: pd.DataFrame,
        per_feature_df: pd.DataFrame,
        strat_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        severity_df: pd.DataFrame,
        top_df: pd.DataFrame,
        wilcoxon: list[dict[str, Any]],
        fold_stability: list[dict[str, Any]],
        plot_paths: list[Path],
        thresholds: dict[str, float],
        out_path: Path,
) -> Path:
    """Assemble the comprehensive Markdown narrative report.

    Sections produced (in order):

      1. Title + Executive Summary
      2. Configuration echo
      3. Data layout (models × folds × features)
      4. Headline findings (per-model table)
      5. Per-fold stability table
      6. s1 vs s2 box plot + comparison
      7. Stratified breakdown (feature_type × baseline_method) heatmap + tables
      8. Worst offending features (top-N tables per model)
      9. Leakage severity classification
     10. Statistical tests
     11. Recommendations
     12. Reproducibility note
    """
    lines: list[str] = []
    lines.append("# Traditional Standardization Metrics Analysis Report\n")
    lines.append(
        "Comparing fold-aware (train-only) vs leaky (all-rows) standardization "
        "for traditional G-LOC models.\n"
    )
    lines.append("---\n")

    # 1. Executive Summary
    lines.append("## Executive Summary\n")
    worst = severity_df.loc[severity_df["overall_severity"] == "severe", "model"].tolist()
    moderate = severity_df.loc[severity_df["overall_severity"] == "moderate", "model"].tolist()
    benign = severity_df.loc[severity_df["overall_severity"] == "benign", "model"].tolist()
    overall_max = float(stats_df["max_abs_delta_max"].max())
    overall_mean_mean = float(stats_df["mean_abs_delta_mean"].mean())
    summary_bits = [
        f"- Across **{len(stats_df)}** models and **{int(per_fold_df['fold_id'].nunique())}** folds per model, "
        f"studying **{int(per_feature_df['feature_name'].nunique())}** unique target features.",
        f"- Mean of per-fold ``mean_abs_delta`` averaged across all models: **{overall_mean_mean:.4f}** z-score units.",
        f"- Globally worst single test-row z-score shift: **{overall_max:.4f}** z-score units.",
    ]
    if worst:
        summary_bits.append(f"- **Severe** leakage classification for: {', '.join(worst)}.")
    if moderate:
        summary_bits.append(f"- **Moderate** classification for: {', '.join(moderate)}.")
    if benign:
        summary_bits.append(f"- **Benign** classification for: {', '.join(benign)}.")
    summary_bits.append(
        "- The fold-aware standardization is therefore "
        f"{'**necessary**' if worst or moderate else 'not strictly required'} "
        "to prevent test-row z-score contamination from the legacy all-rows computation."
    )
    lines.extend(summary_bits)
    lines.append("\n---\n")

    # 2. Configuration
    lines.append("## Configuration\n")
    lines.append("The metrics were computed by `traditional_standardization_metrics.py` "
                 "using the following config:\n")
    cfg_table = pd.DataFrame(
        [{"key": k, "value": v} for k, v in config.items()]
    )
    lines.append(_md_table(cfg_table, float_fmt="{}"))
    lines.append("Severity thresholds used in this report:\n")
    thr_table = pd.DataFrame(
        [
            {"threshold": "benign_mean_abs", "value": thresholds["benign_mean_abs"]},
            {"threshold": "moderate_mean_abs", "value": thresholds["moderate_mean_abs"]},
            {"threshold": "benign_max_abs", "value": thresholds["benign_max_abs"]},
            {"threshold": "moderate_max_abs", "value": thresholds["moderate_max_abs"]},
        ]
    )
    lines.append(_md_table(thr_table, float_fmt="{}"))
    lines.append("\n---\n")

    # 3. Data layout
    lines.append("## Data Layout\n")
    layout_rows = []
    for model_name in sorted(per_fold_df["model"].unique()):
        g = per_fold_df[per_fold_df["model"] == model_name]
        layout_rows.append(
            {
                "model": model_name,
                "folds": int(g.shape[0]),
                "train_rows_mean": float(g["n_train_rows"].mean()),
                "test_rows_mean": float(g["n_test_rows"].mean()),
                "target_features": int(g["n_target_features"].iloc[0]),
                "total_feature_cols": int(g["n_total_features_cols"].iloc[0]),
            }
        )
    lines.append(_md_table(pd.DataFrame(layout_rows)))
    lines.append("\n---\n")

    # 4. Headline findings
    lines.append("## Headline Findings (per-model)\n")
    lines.append(_md_table(
        stats_df[
            [
                "model", "n_folds",
                "mean_abs_delta_mean", "mean_abs_delta_median",
                "mean_abs_delta_std", "mean_abs_delta_max",
                "max_abs_delta_mean", "max_abs_delta_max",
            ]
        ]
    ))
    lines.append("\n---\n")

    # 5. Per-fold stability
    lines.append("## Per-Fold Stability\n")
    lines.append(_md_table(
        per_fold_df[
            ["model", "fold_id", "n_train_rows", "n_test_rows",
             "mean_abs_delta", "median_abs_delta", "max_abs_delta", "std_abs_delta"]
        ]
    ))
    if plot_paths:
        per_fold_plot = next((p for p in plot_paths if p.name == "per_fold_deltas.png"), None)
        if per_fold_plot:
            lines.append(f"\n![Per-fold deltas]({per_fold_plot.name})\n")
        fold_dist_plot = next((p for p in plot_paths if p.name == "fold_distribution.png"), None)
        if fold_dist_plot:
            lines.append(f"![Fold distributions]({fold_dist_plot.name})\n")
    lines.append("\n---\n")

    # 6. s1 vs s2
    lines.append("## s1 (per-trial) vs s2 (pooled) Comparison\n")
    s1s2_compare = (
        per_feature_df.groupby(["model", "s1_or_s2"])["mean_abs_delta"]
        .agg(["mean", "median", "std", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_of_mean_abs",
                "median": "median_of_mean_abs",
                "std": "std_of_mean_abs",
                "max": "max_of_mean_abs",
            }
        )
    )
    lines.append(_md_table(s1s2_compare))
    plot = next((p for p in plot_paths if p.name == "s1_vs_s2_distribution.png"), None)
    if plot:
        lines.append(f"\n![s1 vs s2 distribution]({plot.name})\n")
    lines.append("\n---\n")

    # 7. Stratified breakdown
    lines.append("## Stratified Breakdown (feature_type × baseline_method)\n")
    lines.append(
        "Each cell is the mean across all folds of the (s1/s2 × feature_type × "
        "baseline_method) bucket's ``mean_abs_delta``.\n"
    )
    plot = next((p for p in plot_paths if p.name == "feature_type_heatmap.png"), None)
    if plot:
        lines.append(f"![Stratified heatmap]({plot.name})\n")
    for s1s2 in S1_S2_ORDER:
        lines.append(f"\n### {s1s2.upper()}\n")
        wide = (
            strat_df[strat_df["s1_or_s2"] == s1s2]
            .groupby(["model", "feature_type", "baseline_method"])["mean_abs_delta"]
            .mean()
            .unstack(level="baseline_method")
            .reindex(columns=BASELINE_METHOD_ORDER)
            .reset_index()
        )
        lines.append(_md_table(wide))
    lines.append("\n---\n")

    # 8. Worst offending features
    lines.append("## Worst Offending Features (top 10 per model)\n")
    plot = next((p for p in plot_paths if p.name == "top_offenders.png"), None)
    if plot:
        lines.append(f"![Top offenders]({plot.name})\n")
    for model_name in sorted(top_df["model"].unique()):
        lines.append(f"\n### {model_name}\n")
        cols = ["rank", "fold_id", "feature_name", "max_abs_delta", "mean_abs_delta",
                "s1_or_s2", "feature_type", "baseline_method"]
        sub = top_df[top_df["model"] == model_name].head(10)[cols]
        lines.append(_md_table(sub))
    lines.append("\n---\n")

    # 9. Leakage severity
    lines.append("## Leakage Severity Classification\n")
    lines.append(
        "Each model is placed into the worst bucket triggered by either its "
        "mean-test-row shift (mean family) or its worst single-row shift "
        "(max family). Threshold values are z-score units.\n"
    )
    lines.append(_md_table(severity_df[["model", "mean_abs_delta_mean", "max_abs_delta_max",
                                        "mean_severity", "max_severity", "overall_severity", "triggers"]]))
    lines.append("\n---\n")

    # 10. Statistical tests
    lines.append("## Statistical Tests\n")
    lines.append("### Wilcoxon signed-rank (H0: fold mean_abs_delta is zero)\n")
    lines.append(_md_table(pd.DataFrame(wilcoxon)))
    lines.append("\n### Fold-stability CV (across folds per model)\n")
    lines.append(_md_table(pd.DataFrame(fold_stability)))
    lines.append("\n---\n")

    # 11. Recommendations
    lines.append("## Recommendations\n")
    if worst:
        lines.append(
            f"- **Severe divergence detected** ({', '.join(worst)}): the legacy "
            "leaky standardization would shift at least one test row's z-score "
            "by more than 1 σ compared to the fold-aware computation. The "
            "current fold-aware standardization (see "
            "`src/Data_Pipeline/fold_standardizer.py`) is **essential** for "
            "reporting unbiased model performance.\n"
        )
    if moderate:
        lines.append(
            f"- **Moderate divergence** ({', '.join(moderate)}): single test "
            "rows may shift by 0.1 -- 1 σ. The fold-aware standardization is "
            "still recommended but the impact on aggregate model metrics is "
            "smaller than for severe models.\n"
        )
    if benign:
        lines.append(
            f"- **Benign divergence** ({', '.join(benign)}): both mean and "
            "max test-row shifts are below the benign thresholds. The "
            "fold-aware standardization is good practice but legacy leaky "
            "computation did not produce gross contamination.\n"
        )
    # Cross-fold-stability recommendation
    unstable_models = [r["model"] for r in fold_stability if r["mean_abs_delta_cv"] >= 0.25]
    if unstable_models:
        lines.append(
            f"- **Fold variance is high** for {', '.join(unstable_models)} "
            "(CV ≥ 0.25). Some folds see materially larger divergence than "
            "others. Consider reporting per-fold CV performance alongside "
            "mean/std aggregates so that the worst fold is visible.\n"
        )
    else:
        lines.append(
            "- All models exhibit fold stability (CV < 0.25), so the mean "
            "across folds is a reliable summary statistic.\n"
        )
    lines.append("\n---\n")

    # 12. Reproducibility
    lines.append("## Reproducibility\n")
    lines.append(
        "- Source data: `Results/Traditional_Standardization_Metrics/summary.json` "
        "produced by `python -m src.real_time.traditional_standardization_metrics`.\n"
        f"- Random seed: `{config.get('random_seed')}`. "
        f"Number of splits: `{config.get('num_splits')}`.\n"
        f"- Model type: `{config.get('model_type_string')}`.\n"
        "- Feature stream filters applied by the producer: "
        f"{config.get('target_substrings')}.\n"
        "- All numerical thresholds used by this report are listed in the "
        "Configuration section above. Re-running "
        "`python -m src.real_time.standardization_metrics_analysis` will "
        "reproduce every file in this directory byte-identically.\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote Markdown report %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_analysis(
        results_dir: Path,
        output_dir: Path,
        top_n: int,
        no_plots: bool,
        no_csvs: bool,
        thresholds: dict[str, float],
) -> dict[str, Any]:
    """Top-level orchestrator: load JSON, build tables, classify severity,
    plot, and serialize everything to files.

    Returns a dict with the file paths of every artifact produced.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading summary.json from %s", results_dir)
    summary = load_summary(results_dir)
    config = summary.get("config", {})

    logger.info("Building per-fold DataFrame")
    per_fold_df = build_per_fold_dataframe(summary)
    logger.info("Built per-fold DataFrame: %d rows", len(per_fold_df))

    logger.info("Building per-feature DataFrame")
    per_feature_df = build_per_feature_dataframe(summary)
    logger.info("Built per-feature DataFrame: %d rows", len(per_feature_df))

    logger.info("Building stratified DataFrame")
    strat_df = build_stratified_dataframe(summary)
    logger.info("Built stratified DataFrame: %d rows", len(strat_df))

    logger.info("Computing per-model headline statistics")
    stats_df = compute_model_statistics(per_fold_df)

    logger.info("Identifying top-%d offending features per model", top_n)
    top_df = identify_top_offenders(per_feature_df, top_n=top_n)

    logger.info("Classifying leakage severity")
    severity_df = classify_leakage_severity(
        stats_df,
        benign_mean_abs=thresholds["benign_mean_abs"],
        moderate_mean_abs=thresholds["moderate_mean_abs"],
        benign_max_abs=thresholds["benign_max_abs"],
        moderate_max_abs=thresholds["moderate_max_abs"],
    )

    logger.info("Performing Wilcoxon signed-rank test")
    wilcoxon = perform_wilcoxon_test(per_fold_df)

    logger.info("Computing fold-stability CV")
    fold_stability = perform_fold_stability_test(per_fold_df)

    plot_paths: list[Path] = []
    if not no_plots:
        logger.info("Rendering PNG plots")
        plot_paths.append(plot_per_fold_deltas(per_fold_df, output_dir))
        plot_paths.append(plot_s1_vs_s2_distribution(per_feature_df, output_dir))
        plot_paths.append(plot_feature_type_heatmap(strat_df, output_dir))
        plot_paths.append(plot_top_offenders(top_df, output_dir, top_n=min(10, top_n)))
        plot_paths.append(plot_fold_distribution(per_fold_df, output_dir))

    if not no_csvs:
        logger.info("Writing CSV tables")
        write_csv_tables(
            per_fold_df, strat_df, per_feature_df, top_df, stats_df, severity_df, output_dir
        )

    logger.info("Writing aggregated JSON")
    json_path = write_aggregated_json(
        stats_df,
        severity_df,
        wilcoxon,
        fold_stability,
        top_df,
        config,
        output_dir / "analysis.json",
    )

    logger.info("Writing Markdown report")
    md_path = generate_markdown_report(
        config,
        per_fold_df,
        per_feature_df,
        strat_df,
        stats_df,
        severity_df,
        top_df,
        wilcoxon,
        fold_stability,
        plot_paths,
        thresholds,
        output_dir / "Standardization_Metrics_Analysis_Report.md",
    )

    return {
        "markdown_report": md_path,
        "aggregated_json": json_path,
        "plots": plot_paths,
        "severity_table": severity_df,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse `Results/Traditional_Standardization_Metrics` and "
                    "produce report + plots + CSVs + JSON."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing summary.json (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output (default: <results-dir>/analysis)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top-N offending features to keep per model (default: 20)",
    )
    parser.add_argument(
        "--benign-mean-abs",
        type=float,
        default=DEFAULT_BENIGN_MEAN_ABS,
        help="Below this per-fold mean |Δ| is benign (default: 0.005)",
    )
    parser.add_argument(
        "--moderate-mean-abs",
        type=float,
        default=DEFAULT_MODERATE_MEAN_ABS,
        help="At/above this per-fold mean |Δ| is severe (default: 0.05)",
    )
    parser.add_argument(
        "--benign-max-abs",
        type=float,
        default=DEFAULT_BENIGN_MAX_ABS,
        help="Below this single-row max |Δ| is benign (default: 0.1)",
    )
    parser.add_argument(
        "--moderate-max-abs",
        type=float,
        default=DEFAULT_MODERATE_MAX_ABS,
        help="At/above this single-row max |Δ| is severe (default: 1.0)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG generation (headless / faster run)",
    )
    parser.add_argument(
        "--no-csvs",
        action="store_true",
        help="Skip CSV export",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or (args.results_dir / DEFAULT_OUTPUT_SUBDIR)

    thresholds = {
        "benign_mean_abs": args.benign_mean_abs,
        "moderate_mean_abs": args.moderate_mean_abs,
        "benign_max_abs": args.benign_max_abs,
        "moderate_max_abs": args.moderate_max_abs,
    }

    result = run_analysis(
        results_dir=args.results_dir,
        output_dir=out_dir,
        top_n=args.top_n,
        no_plots=args.no_plots,
        no_csvs=args.no_csvs,
        thresholds=thresholds,
    )

    print("Standardization metrics analysis complete.")
    print(f"Markdown report: {result['markdown_report']}")
    print(f"Aggregated JSON: {result['aggregated_json']}")
    print(f"Severity classification:\n{result['severity_table'].to_string(index=False)}")


if __name__ == "__main__":
    main()
