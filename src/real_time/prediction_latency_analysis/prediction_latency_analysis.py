#!/usr/bin/env python3
"""Comprehensive Real-Time Latency Analysis and Report Generator for G-LOC Prediction.

This script scans for `real_time_summary.json` files in `Results/Real_Time_Prediction_Latency`,
conducts robust statistical analysis on latency metrics (mean, tail latencies, real-time deadlines),
generates publication-quality charts, and compiles a Markdown summary report.

All outputs are saved to the current working directory.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
STREAM_RATE_HZ: float = 25.0
LATENCY_BUDGET_MS: float = 1000.0 / STREAM_RATE_HZ  # 40.0 ms budget per sample
RESULTS_DIR_NAME = "/home/jasper_shen/Projects/G-LOC-Prediction/Results/Real_Time_Prediction_Latency"
OUTPUT_DIR = Path(".")

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})


def load_realtime_summaries(results_dir: Path) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Find and load all real_time_summary.json files from the results directory."""
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir.resolve())
        sys.exit(1)

    json_files = list(results_dir.glob("**/real_time_summary.json"))
    if not json_files:
        logger.error("No 'real_time_summary.json' files found in %s", results_dir.resolve())
        sys.exit(1)

    logger.info("Found %d summary JSON file(s). Loading data...", len(json_files))

    raw_summaries: List[Dict[str, Any]] = []
    per_fold_list: List[Dict[str, Any]] = []
    per_sample_list: List[Dict[str, Any]] = []

    for file_path in json_files:
        try:
            with file_path.open("r") as handle:
                data = json.load(handle)
                raw_summaries.append(data)

                model_name = data.get("model", "Unknown")
                model_type = data.get("model_type", "Unknown")
                streams_str = "-".join(data.get("streams", []))

                for fold_info in data.get("per_fold", []):
                    fold_id = fold_info["fold_id"]
                    recomputed = fold_info.get("metrics_recomputed", {})
                    lat_summary = fold_info.get("latency_ms", {})
                    samples = fold_info.get("per_sample_latency_ms", [])

                    # Fold-level record
                    per_fold_record = {
                        "model": model_name,
                        "model_type": model_type,
                        "streams": streams_str,
                        "fold_id": fold_id,
                        "n_test": fold_info.get("n_test", 0),
                        "f1_score": recomputed.get("f1", np.nan),
                        "accuracy": recomputed.get("accuracy", np.nan),
                        "g_mean": recomputed.get("g_mean", np.nan),
                        "lat_mean": lat_summary.get("mean", np.nan),
                        "lat_median": lat_summary.get("median", np.nan),
                        "lat_std": lat_summary.get("std", np.nan),
                        "lat_p95": lat_summary.get("p95", np.nan),
                        "lat_p99": lat_summary.get("p99", np.nan),
                        "lat_max": lat_summary.get("max", np.nan),
                    }
                    per_fold_list.append(per_fold_record)

                    # Sample-level record
                    for sample_idx, lat in enumerate(samples):
                        per_sample_list.append({
                            "model": model_name,
                            "model_type": model_type,
                            "streams": streams_str,
                            "fold_id": fold_id,
                            "sample_idx": sample_idx,
                            "latency_ms": lat,
                        })
        except Exception as exc:
            logger.error("Failed to parse %s: %s", file_path, exc)

    df_folds = pd.DataFrame(per_fold_list)
    df_samples = pd.DataFrame(per_sample_list)

    return raw_summaries, df_folds, df_samples


def compute_model_statistics(df_samples: pd.DataFrame, df_folds: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated latency and performance statistics grouped by Model."""
    stats_list = []

    for model, group in df_samples.groupby("model"):
        lats = group["latency_ms"].values
        fold_group = df_folds[df_folds["model"] == model]

        n_total = len(lats)
        mean_lat = float(np.mean(lats))
        std_lat = float(np.std(lats))
        median_lat = float(np.median(lats))
        p95_lat = float(np.percentile(lats, 95))
        p99_lat = float(np.percentile(lats, 99))
        p99_9_lat = float(np.percentile(lats, 99.9))
        max_lat = float(np.max(lats))
        min_lat = float(np.min(lats))

        # Real-time compliance check (< 40 ms budget)
        violations = int(np.sum(lats >= LATENCY_BUDGET_MS))
        compliance_pct = (1.0 - (violations / n_total)) * 100.0 if n_total > 0 else 0.0

        # Throughput (samples per second)
        throughput_hz = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        # Classification performance averages
        f1_mean = fold_group["f1_score"].mean()
        acc_mean = fold_group["accuracy"].mean()
        gmean_mean = fold_group["g_mean"].mean()

        stats_list.append({
            "Model": model,
            "Total Samples": n_total,
            "Mean (ms)": mean_lat,
            "Std (ms)": std_lat,
            "Median / P50 (ms)": median_lat,
            "P95 (ms)": p95_lat,
            "P99 (ms)": p99_lat,
            "P99.9 (ms)": p99_9_lat,
            "Max (ms)": max_lat,
            "Throughput (samp/s)": throughput_hz,
            "Compliance (<40ms) %": compliance_pct,
            "Violations": violations,
            "F1 Score": f1_mean,
            "Accuracy": acc_mean,
            "G-Mean": gmean_mean,
        })

    return pd.DataFrame(stats_list).sort_values(by="Mean (ms)").reset_index(drop=True)


def plot_latency_distributions(df_samples: pd.DataFrame, output_dir: Path) -> Path:
    """Plot Kernel Density Estimate (KDE) and Boxplot of sample latencies per model."""
    fig, (ax_kde, ax_box) = plt.subplots(1, 2, figsize=(14, 5.5))

    models = df_samples["model"].unique()
    palette = sns.color_palette("muted", n_colors=len(models))

    # KDE Plot
    sns.kdeplot(
        data=df_samples,
        x="latency_ms",
        hue="model",
        common_norm=False,
        fill=True,
        alpha=0.3,
        linewidth=1.5,
        ax=ax_kde,
        palette=palette,
    )
    ax_kde.axvline(
        LATENCY_BUDGET_MS,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"25Hz Limit ({LATENCY_BUDGET_MS:.1f}ms)",
    )
    ax_kde.set_title("Latency Density Distribution per Model")
    ax_kde.set_xlabel("Inference Latency (ms)")
    ax_kde.set_ylabel("Density")
    ax_kde.legend(loc="upper right")

    # Box Plot (Log Scale to handle tail latencies cleanly)
    sns.boxplot(
        data=df_samples,
        x="model",
        y="latency_ms",
        ax=ax_box,
        palette=palette,
        fliersize=2,
    )
    ax_box.axhline(
        LATENCY_BUDGET_MS,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"25Hz Limit ({LATENCY_BUDGET_MS:.1f}ms)",
    )
    ax_box.set_title("Latency Distribution & Outliers (Log Scale)")
    ax_box.set_xlabel("Model")
    ax_box.set_ylabel("Inference Latency (ms)")
    ax_box.set_yscale("log")
    ax_box.legend(loc="upper left")

    plt.tight_layout()
    plot_path = output_dir / "latency_distributions.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved latency distributions plot to %s", plot_path)
    return plot_path


def plot_latency_percentiles(df_stats: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart comparing P50, P95, and P99 tail latencies across models."""
    df_melted = df_stats.melt(
        id_vars=["Model"],
        value_vars=["Median / P50 (ms)", "P95 (ms)", "P99 (ms)"],
        var_name="Percentile",
        value_name="Latency_ms",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df_melted,
        x="Model",
        y="Latency_ms",
        hue="Percentile",
        palette="Blues_d",
        ax=ax,
    )

    ax.axhline(
        LATENCY_BUDGET_MS,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Real-time Budget ({LATENCY_BUDGET_MS:.1f} ms)",
    )

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    ax.set_title("Inference Latency Percentiles (P50, P95, P99) vs Real-time Budget")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Model")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plot_path = output_dir / "latency_percentiles.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved latency percentiles plot to %s", plot_path)
    return plot_path


def plot_latency_cdf(df_samples: pd.DataFrame, output_dir: Path) -> Path:
    """Cumulative Distribution Function (CDF) plot showing real-time deadline adherence."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    models = df_samples["model"].unique()
    palette = sns.color_palette("muted", n_colors=len(models))

    for idx, model in enumerate(models):
        sub = df_samples[df_samples["model"] == model]["latency_ms"].sort_values()
        y = np.arange(1, len(sub) + 1) / len(sub)
        ax.plot(sub, y, label=model, color=palette[idx], linewidth=2.0)

    ax.axvline(
        LATENCY_BUDGET_MS,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"25Hz Budget ({LATENCY_BUDGET_MS:.1f}ms)",
    )
    ax.set_title("Cumulative Distribution Function (CDF) of Latencies")
    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel("Empirical Cumulative Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plot_path = output_dir / "latency_cdf.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved latency CDF plot to %s", plot_path)
    return plot_path


def plot_fold_stability(df_folds: pd.DataFrame, output_dir: Path) -> Path:
    """Plot latency consistency across 10 folds to assess stability."""
    fig, ax = plt.subplots(figsize=(11, 5))

    sns.lineplot(
        data=df_folds,
        x="fold_id",
        y="lat_mean",
        hue="model",
        marker="o",
        linewidth=2,
        ax=ax,
    )

    ax.set_title("Mean Latency Stability Across Cross-Validation Folds")
    ax.set_xlabel("Fold ID")
    ax.set_ylabel("Mean Inference Latency (ms)")
    ax.set_xticks(sorted(df_folds["fold_id"].unique()))
    ax.legend(title="Model", loc="best")

    plt.tight_layout()
    plot_path = output_dir / "fold_stability.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved fold stability plot to %s", plot_path)
    return plot_path


def perform_statistical_tests(df_samples: pd.DataFrame) -> Dict[str, Any]:
    """Perform non-parametric Kruskal-Wallis H-test across models."""
    models = df_samples["model"].unique()
    if len(models) < 2:
        return {"test": "N/A", "reason": "Fewer than 2 models to compare."}

    groups = [df_samples[df_samples["model"] == m]["latency_ms"].values for m in models]
    stat, p_val = stats.kruskal(*groups)

    return {
        "test": "Kruskal-Wallis H-test",
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
    }


def generate_markdown_report(
        df_stats: pd.DataFrame,
        df_folds: pd.DataFrame,
        stat_results: Dict[str, Any],
        output_dir: Path,
) -> Path:
    """Compile analysis findings into a comprehensive Markdown report."""
    report_path = output_dir / "G_LOC_Latency_Analysis_Report.md"

    # Convert dataframe to markdown table
    stats_table_md = df_stats[
        [
            "Model",
            "Mean (ms)",
            "Std (ms)",
            "Median / P50 (ms)",
            "P95 (ms)",
            "P99 (ms)",
            "Max (ms)",
            "Throughput (samp/s)",
            "Compliance (<40ms) %",
            "F1 Score",
        ]
    ].to_markdown(index=False, floatfmt=".4f")

    # Fastest & Slowest
    fastest_model = df_stats.iloc[0]["Model"]
    fastest_latency = df_stats.iloc[0]["Mean (ms)"]
    slowest_model = df_stats.iloc[-1]["Model"]
    slowest_latency = df_stats.iloc[-1]["Mean (ms)"]

    # Real-time verdict
    all_compliant = (df_stats["Compliance (<40ms) %"] == 100.0).all()
    verdict = (
        "**ALL EVALUATED MODELS ARE REAL-TIME CAPABLE**"
        if all_compliant
        else "**SOME MODELS EXCEEDED REAL-TIME DEADLINES**"
    )

    p_val_str = (
        f"{stat_results['p_value']:.4e}"
        if isinstance(stat_results.get("p_value"), float)
        else "N/A"
    )

    report_content = f"""# Real-Time G-LOC Prediction: Latency & Performance Evaluation Report

## Executive Summary
This report evaluates per-sample inference latency for machine learning models streaming **Equivital** (ECG) and **Centrifuge** telemetry data for real-time **G-Induced Loss of Consciousness (G-LOC)** prediction. Data streaming is paced at **{STREAM_RATE_HZ:.1f} Hz**, which establishes a strict per-sample compute deadline of **{LATENCY_BUDGET_MS:.2f} ms**.

### Key Verdict: {verdict}
* **Fastest Model:** `{fastest_model}` with a mean latency of **{fastest_latency:.4f} ms** (Throughput: **{df_stats.iloc[0]['Throughput (samp/s)']:.1f} samples/sec**).
* **Slowest Model:** `{slowest_model}` with a mean latency of **{slowest_latency:.4f} ms** (Throughput: **{df_stats.iloc[-1]['Throughput (samp/s)']:.1f} samples/sec**).
* **Streaming Budget:** {LATENCY_BUDGET_MS:.2f} ms per sample (25 Hz stream rate).

---

## Model Latency & Performance Comparison

{stats_table_md}

---

## Statistical Significance Analysis
To verify whether latency differences between evaluated models are statistically significant, a **{stat_results.get('test', 'N/A')}** was performed on the per-sample inference latencies across all folds:
* **H-Statistic / Test Statistic:** `{stat_results.get('statistic', 'N/A')}`
* **p-value:** `{p_val_str}`
* **Statistically Significant ($\alpha=0.05$):** `{stat_results.get('significant', 'N/A')}`

---

## Visual Diagnostic Plots

### 1. Latency Distribution & Outliers
Density distributions and boxplots showing the spread and scale of sample latencies relative to the 40 ms hard deadline.
![Latency Distributions](latency_distributions.png)

### 2. Tail Latency Analysis (P50, P95, P99)
Comparison of median vs. extreme percentiles to ensure safety margins during peak workload.
![Latency Percentiles](latency_percentiles.png)

### 3. Empirical Cumulative Distribution Function (CDF)
Empirical cumulative probability of predictions completing within time budgets.
![Latency CDF](latency_cdf.png)

### 4. Cross-Validation Stability
Stability of average model latency across 10 fold test sets.
![Fold Stability](fold_stability.png)

---

## Recommendations for Real-Time Deployment
1. **Model Selection Trade-off:**
   * Select models balancing both F1-score and low P99 tail latency.
   * Tail latencies (P95/P99) are critical in physiological monitoring to prevent frame dropping or buffer buildup during sudden dynamic maneuvers.
2. **Buffer Safety Margin:**
   * Even the highest tail latencies observed should comfortably sit under the **{LATENCY_BUDGET_MS:.1f} ms** threshold to allow headroom for hardware interrupt jitter, telemetry LSL serialization overhead, and display rendering.
"""

    with report_path.open("w") as f:
        f.write(report_content)

    logger.info("Saved analysis report to %s", report_path.resolve())
    return report_path


def main() -> None:
    """Main execution entry point."""
    logger.info("Starting G-LOC Prediction Real-Time Latency Data Analysis...")

    results_dir = Path(RESULTS_DIR_NAME)

    # 1. Load Data
    raw_summaries, df_folds, df_samples = load_realtime_summaries(results_dir)

    if df_samples.empty:
        logger.error("No sample data extracted from JSON files. Exiting.")
        sys.exit(1)

    # 2. Compute Statistics
    logger.info("Computing latency statistics and percentile summaries...")
    df_stats = compute_model_statistics(df_samples, df_folds)

    # Print summary table to stdout
    print("\n" + "=" * 80)
    print("REAL-TIME G-LOC INFERENCE LATENCY SUMMARY")
    print("=" * 80)
    print(df_stats.to_string(index=False))
    print("=" * 80 + "\n")

    # 3. Perform Statistical Tests
    stat_results = perform_statistical_tests(df_samples)

    # 4. Generate Visualizations
    logger.info("Generating publication-quality charts...")
    plot_latency_distributions(df_samples, OUTPUT_DIR)
    plot_latency_percentiles(df_stats, OUTPUT_DIR)
    plot_latency_cdf(df_samples, OUTPUT_DIR)
    plot_fold_stability(df_folds, OUTPUT_DIR)

    # 5. Generate Markdown Report
    logger.info("Compiling final analysis markdown report...")
    report_file = generate_markdown_report(df_stats, df_folds, stat_results, OUTPUT_DIR)

    logger.info("Analysis complete! All artifacts saved successfully in '%s'.", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
