"""Comprehensive comparison of raw Equivital data vs. pipeline-processed Equivital data.

This script compares the raw Equivital columns from the main G-LOC dataset CSV
against the ~860+ engineered Equivital features produced by the
traditional data pipeline (KNN, Complete/Explicit,
``traditional_feature_selection=\"raw\"``).

It loads both datasets, compares dimensions, feature names, value distributions,
label distributions, and temporal properties, then outputs a JSON report,
human-readable text report, and several plots.

Usage::

    python -m src.real_time.equivital_dataset_differences
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config_loader import load_experiment_config
import yaml

# Optional plotting imports
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:  # pragma: no cover
    PLOTTING_AVAILABLE = False  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_CSV_FILENAME = "all_trials_25_hz_stacked_null_str_filled.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_equivital_columns(csv_path: Path) -> list[str]:
    """Read the CSV header and return only columns containing 'Equivital'."""
    logger.info("Reading header from %s to find Equivital columns...", csv_path)
    header = pd.read_csv(csv_path, nrows=0)
    cols = [c for c in header.columns if "Equivital" in c]
    if not cols:
        raise ValueError(f"No columns with 'Equivital' found in {csv_path}")
    logger.info("Found %d Equivital columns.", len(cols))
    return cols


def _load_raw_equivital_data(project_root: Path) -> pd.DataFrame:
    """Load only the Equivital columns from the main CSV."""
    csv_path = project_root / "data" / RAW_CSV_FILENAME
    equivital_cols = _get_equivital_columns(csv_path)
    raw_df = pd.read_csv(csv_path, usecols=equivital_cols)
    return raw_df


def _extract_baseline_method(feature_name: str) -> str | None:
    """Extract baseline method (v0/v1/v2/v5/v6) from a pipeline feature name."""
    for bm in ("v0", "v1", "v2", "v5", "v6"):
        if f"_{bm}" in feature_name or feature_name.endswith(f"_{bm}"):
            return bm
    return None


def _extract_feature_type(feature_name: str) -> str | None:
    """Extract feature type (mean/stddev/max/range/additional) from a pipeline feature name."""
    for ft in ("mean", "stddev", "max", "range", "additional"):
        if f"_{ft}_" in feature_name or feature_name.endswith(f"_{ft}"):
            return ft
    return None


def _extract_standardization(feature_name: str) -> str | None:
    """Extract standardization suffix (s1/s2) from a pipeline feature name."""
    for s in ("s1", "s2"):
        if feature_name.endswith(f"_{s}"):
            return s
    return None


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _compare_dimensions(raw_df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Compare the dimensions of the raw and pipeline datasets."""
    raw_shape = raw_df.shape
    pipeline_shape = X.shape
    label_shape = y.shape
    return {
        "raw_shape": list(raw_shape),
        "pipeline_shape": list(pipeline_shape),
        "label_shape": list(label_shape),
        "raw_samples": raw_shape[0],
        "raw_features": raw_shape[1],
        "pipeline_samples": pipeline_shape[0],
        "pipeline_features": pipeline_shape[1],
        "label_samples": label_shape[0],
        "shape_comment": (
            f"Raw data: {raw_shape[0]} rows x {raw_shape[1]} columns (time-series). "
            f"Pipeline data: {pipeline_shape[0]} windows x {pipeline_shape[1]} engineered features."
        ),
    }


def _compare_feature_names(raw_cols: list[str], pipeline_features: list[str]) -> dict[str, Any]:
    """Compare the feature names between raw and pipeline datasets."""
    baseline_counts: Counter[str] = Counter()
    feature_type_counts: Counter[str] = Counter()
    standardization_counts: Counter[str] = Counter()

    raw_to_pipeline: dict[str, list[str]] = {rc: [] for rc in raw_cols}
    unmatched_features: list[str] = []

    for pf in pipeline_features:
        bm = _extract_baseline_method(pf)
        if bm:
            baseline_counts[bm] += 1
        ft = _extract_feature_type(pf)
        if ft:
            feature_type_counts[ft] += 1
        s = _extract_standardization(pf)
        if s:
            standardization_counts[s] += 1

        matched = False
        for rc in raw_cols:
            if rc in pf or pf.startswith(rc):
                raw_to_pipeline[rc].append(pf)
                matched = True
                break
        if not matched:
            unmatched_features.append(pf)

    return {
        "raw_columns": raw_cols,
        "pipeline_features": pipeline_features,
        "n_raw_columns": len(raw_cols),
        "n_pipeline_features": len(pipeline_features),
        "baseline_method_counts": dict(baseline_counts),
        "feature_type_counts": dict(feature_type_counts),
        "standardization_counts": dict(standardization_counts),
        "raw_to_pipeline_mapping": {rc: pfs for rc, pfs in raw_to_pipeline.items() if pfs},
        "raw_columns_with_no_pipeline_features": [
            rc for rc, pfs in raw_to_pipeline.items() if not pfs
        ],
        "unmatched_pipeline_features": unmatched_features[:20] if unmatched_features else [],
        "n_unmatched_pipeline_features": len(unmatched_features),
    }


def _compare_value_distributions(raw_df: pd.DataFrame, X: np.ndarray, pipeline_features: list[str]) -> dict[str, Any]:
    """Compare value distributions between raw and pipeline datasets."""
    raw_stats = raw_df.describe().to_dict()
    raw_nan = raw_df.isna().sum().to_dict()

    pipeline_df = pd.DataFrame(X, columns=pipeline_features)
    pipeline_stats = pipeline_df.describe().to_dict()
    pipeline_nan = pipeline_df.isna().sum().to_dict()

    stats_keys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    def _trim_dict(d: dict, limit: int = 20) -> dict:
        """Limit dictionary to first N keys for brevity."""
        return {k: d[k] for i, k in enumerate(d.keys()) if i < limit}

    return {
        "raw_stats": {
            k: {stat: raw_stats[k].get(stat) for stat in stats_keys}
            for k in list(raw_stats.keys())
        },
        "pipeline_stats": {
            k: {stat: pipeline_stats[k].get(stat) for stat in stats_keys}
            for k in list(pipeline_stats.keys())[:20]
        } if pipeline_stats else {},
        "raw_nan_counts": {k: int(v) for k, v in raw_nan.items()},
        "pipeline_nan_counts": {k: int(v) for k, v in list(pipeline_nan.items())[:20]} if pipeline_nan else {},
    }


def _compare_label_distribution(y: np.ndarray) -> dict[str, Any]:
    """Analyze the label distribution of the pipeline dataset."""
    unique, counts = np.unique(y, return_counts=True)
    return {
        "total_windows": len(y),
        "class_counts": {int(u): int(c) for u, c in zip(unique, counts)},
        "class_percentages": {int(u): round(float(c) / len(y) * 100, 2) for u, c in zip(unique, counts)},
        "gloc_event_rate": round(float(np.sum(y == 1)) / len(y) * 100, 2) if len(y) > 0 else 0.0,
    }


def _compare_temporal_info(raw_df: pd.DataFrame) -> dict[str, Any]:
    """Provide temporal information about the raw dataset."""
    first_col = raw_df.columns[0]
    return {
        "raw_columns": list(raw_df.columns),
        "n_rows": len(raw_df),
        "n_columns": len(raw_df.columns),
        "first_column": first_col,
        "note": (
            "Raw dataset is a time-series with ~25 Hz sampling. "
            "Pipeline transforms this into sliding windows (~15s windows, 0.25s stride)."
        ),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _save_json_report(report: dict[str, Any], output_path: Path) -> None:
    """Save the report as a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved JSON report to %s", output_path)


def _save_text_report(report: dict[str, Any], output_path: Path) -> None:
    """Save the report as a human-readable text file."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("EQUIVITAL DATASET COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Dimensions
    lines.append("-" * 40)
    lines.append("1. DIMENSIONS")
    lines.append("-" * 40)
    for k, v in report["dimensions"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Feature names
    lines.append("-" * 40)
    lines.append("2. FEATURE NAMES")
    lines.append("-" * 40)
    fc = report["feature_comparison"]
    lines.append(f"  Raw columns: {fc['n_raw_columns']}")
    lines.append(f"  Pipeline features: {fc['n_pipeline_features']}")
    lines.append("  Baseline method counts:")
    for bm, c in fc.get("baseline_method_counts", {}).items():
        lines.append(f"    {bm}: {c}")
    lines.append("  Feature type counts:")
    for ft, c in fc.get("feature_type_counts", {}).items():
        lines.append(f"    {ft}: {c}")
    lines.append("  Standardization counts:")
    for s, c in fc.get("standardization_counts", {}).items():
        lines.append(f"    {s}: {c}")
    lines.append(f"  Unmatched: {fc.get('n_unmatched_pipeline_features', 'N/A')}")
    lines.append("")

    # Labels
    lines.append("-" * 40)
    lines.append("3. LABEL DISTRIBUTION")
    lines.append("-" * 40)
    for k, v in report["label_distribution"].items():
        if isinstance(v, dict):
            lines.append(f"  {k}:")
            for kk, vv in v.items():
                lines.append(f"    {kk}: {vv}")
        else:
            lines.append(f"  {k}: {v}")
    lines.append("")

    # Temporal
    lines.append("-" * 40)
    lines.append("4. TEMPORAL INFO")
    lines.append("-" * 40)
    for k, v in report["temporal_info"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Value distributions
    lines.append("-" * 40)
    lines.append("5. VALUE DISTRIBUTIONS (Summary)")
    lines.append("-" * 40)
    dist = report["distributions"]
    lines.append(f"  Raw stats: {len(dist.get('raw_stats', {}))} columns.")
    lines.append(f"  Pipeline stats: {len(dist.get('pipeline_stats', {}))} columns.")
    lines.append("  See JSON report for full details.")
    lines.append("")

    lines.append("=" * 80)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved text report to %s", output_path)


# ---------------------------------------------------------------------------
# Plotting (optional — skipped if matplotlib unavailable)
# ---------------------------------------------------------------------------

def _create_plots(report: dict[str, Any], output_dir: Path) -> None:
    """Generate and save comparison plots to the given directory."""
    if not PLOTTING_AVAILABLE:
        logger.warning("matplotlib not available; skipping plot generation.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_counts = report.get("feature_comparison", {}).get("baseline_method_counts", {})
    if baseline_counts:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(baseline_counts.keys(), baseline_counts.values())
        ax.set_title("Equivital Engineered Features by Baseline Method")
        ax.set_xlabel("Baseline Method")
        ax.set_ylabel("Feature Count")
        fig.tight_layout()
        fig.savefig(output_dir / "01_baseline_method_counts.png")
        plt.close(fig)

    feature_type_counts = report.get("feature_comparison", {}).get("feature_type_counts", {})
    if feature_type_counts:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(feature_type_counts.keys(), feature_type_counts.values())
        ax.set_title("Equivital Engineered Features by Feature Type")
        ax.set_xlabel("Feature Type")
        ax.set_ylabel("Feature Count")
        fig.tight_layout()
        fig.savefig(output_dir / "02_feature_type_counts.png")
        plt.close(fig)

    label_counts = report.get("label_distribution", {}).get("class_counts", {})
    if label_counts:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar([str(k) for k in label_counts.keys()], label_counts.values())
        ax.set_title("GLOC Label Distribution (Pipeline Windows)")
        ax.set_xlabel("Label (0=No GLOC, 1=GLOC)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(output_dir / "03_label_distribution.png")
        plt.close(fig)

    logger.info("Saved plots to %s", output_dir)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_comparison(
        config_path: Path | None = None,
        project_root: Path | None = None,
        output_dir: Path | None = None,
) -> dict[str, Any]:
    """Load both datasets, compare, and save reports.

    Parameters
    ----------
    config_path :
        Path to the YAML config. If None, uses ``configs/real_time_equivital.yaml``.
    project_root :
        Path to the project root. If None, inferred from this file's location.
    output_dir :
        Directory to write reports and plots. If None, uses ``src/real_time/``.

    Returns
    -------
    dict[str, Any]
        The complete comparison report.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    if config_path is None:
        config_path = project_root / "configs" / "real_time_equivital.yaml"
    if output_dir is None:
        output_dir = project_root / "src" / "real_time"

    # 1) Load raw data (memory-safe)
    raw_df = _load_raw_equivital_data(project_root)
    raw_cols = list(raw_df.columns)

    # 2) Load pipeline data
    from src.Data_Pipeline.data_pipeline import DataPipeline
    from src.models.model_factory import ModelFactory
    from src.model_type import ModelType

    config = load_experiment_config(config_path)

    pipeline = DataPipeline(config=config)
    model_factory = ModelFactory()
    knn_model = model_factory.create_model("KNN")

    model_type = ModelType(afe_filter="Complete", feature_set="Explicit")
    pipeline.set_model_type(model_type)
    pipeline.set_random_seed(config.get("real_time_equivital", {}).get("random_seed", 42))

    logger.info("Loading pipeline data with traditional_feature_selection='raw' ...")
    X, y, all_features = pipeline.get_data(
        model=knn_model,
        traditional_feature_selection="raw",
        return_feature_names=True,
    )
    equivital_features = [f for f in all_features if "Equivital" in f]
    # Build a name->index map that handles dropped features / out-of-sync lists
    feature_index_map = {name: idx for idx, name in enumerate(all_features) if idx < X.shape[1]}
    equivital_indices = [feature_index_map[f] for f in equivital_features if f in feature_index_map]
    X_eq = X[:, equivital_indices]
    logger.info("Pipeline: %d total features, %d are Equivital.", len(all_features), len(equivital_features))

    # 3) Run comparisons
    dim_comparison = _compare_dimensions(raw_df, X_eq, y)
    feature_comparison = _compare_feature_names(raw_cols, equivital_features)
    label_comparison = _compare_label_distribution(y)
    temporal_comparison = _compare_temporal_info(raw_df)
    dist_comparison = _compare_value_distributions(raw_df, X_eq, equivital_features)

    report = {
        "dimensions": dim_comparison,
        "feature_comparison": feature_comparison,
        "label_distribution": label_comparison,
        "temporal_info": temporal_comparison,
        "distributions": dist_comparison,
    }

    # 4) Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json_report(report, output_dir / "equivital_comparison_report.json")
    _save_text_report(report, output_dir / "equivital_comparison_report.txt")
    _create_plots(report, output_dir / "plots")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Compare raw and pipeline-processed Equivital data.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (defaults to configs/real_time_equivital.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save reports and plots (defaults to src/real_time/).",
    )
    args = parser.parse_args()

    config_path_arg = Path(args.config) if args.config else None
    output_dir_arg = Path(args.output_dir) if args.output_dir else None

    report = run_comparison(config_path=config_path_arg, output_dir=output_dir_arg)
    print("Comparison complete. Report saved to src/real_time/")
