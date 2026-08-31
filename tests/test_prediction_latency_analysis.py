"""Tests for prediction latency analysis and reporting pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.real_time.prediction_latency_analysis.prediction_latency_analysis import (
    compute_comprehensive_statistics,
    generate_markdown_report,
    load_realtime_summaries,
    perform_statistical_tests,
    plot_latency_cdf,
    plot_latency_component_breakdown,
    plot_latency_distributions,
    plot_latency_tail_percentiles,
    plot_latency_time_series,
)


def _create_mock_results(tmp_path: Path) -> Path:
    results_dir = tmp_path / "Results_Latency"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Decomposed model 1 (EGB)
    egb_dir = results_dir / "Complete_Explicit" / "EGB" / "ECG-HR-BR-Temperature-Centrifuge"
    egb_dir.mkdir(parents=True, exist_ok=True)
    egb_data = {
        "model": "EGB",
        "model_type": "Complete_Explicit",
        "streams": ["ECG", "HR", "BR", "Temperature", "Centrifuge"],
        "trial_id": "01-01",
        "n_raw_samples": 1000,
        "n_predictions": 100,
        "use_real_time_sleep": False,
        "data_processing_latency_ms": {"mean": 1.7, "std": 0.04, "median": 1.7, "p95": 1.74},
        "inference_latency_ms": {"mean": 0.4, "std": 0.05, "median": 0.4, "p95": 0.42},
        "total_latency_ms": {"mean": 2.1, "std": 0.06, "median": 2.1, "p95": 2.15},
        "per_prediction_data_proc_latency_ms": [1.7 + 0.01 * (i % 5) for i in range(100)],
        "per_prediction_inference_latency_ms": [0.4 + 0.01 * (i % 3) for i in range(100)],
        "per_prediction_total_latency_ms": [2.1 + 0.02 * (i % 5) for i in range(100)],
    }
    with open(egb_dir / "real_time_summary.json", "w") as f:
        json.dump(egb_data, f)

    # 2. Decomposed model 2 (RF)
    rf_dir = results_dir / "Complete_Explicit" / "RF" / "ECG-HR-BR-Temperature-Centrifuge"
    rf_dir.mkdir(parents=True, exist_ok=True)
    rf_data = {
        "model": "RF",
        "model_type": "Complete_Explicit",
        "streams": ["ECG", "HR", "BR", "Temperature", "Centrifuge"],
        "trial_id": "01-01",
        "n_raw_samples": 1000,
        "n_predictions": 100,
        "use_real_time_sleep": False,
        "data_processing_latency_ms": {"mean": 1.2, "std": 0.03, "median": 1.2, "p95": 1.25},
        "inference_latency_ms": {"mean": 0.8, "std": 0.05, "median": 0.8, "p95": 0.86},
        "total_latency_ms": {"mean": 2.0, "std": 0.06, "median": 2.0, "p95": 2.08},
        "per_prediction_data_proc_latency_ms": [1.2 + 0.01 * (i % 4) for i in range(100)],
        "per_prediction_inference_latency_ms": [0.8 + 0.01 * (i % 3) for i in range(100)],
        "per_prediction_total_latency_ms": [2.0 + 0.02 * (i % 4) for i in range(100)],
    }
    with open(rf_dir / "real_time_summary.json", "w") as f:
        json.dump(rf_data, f)

    return results_dir


def test_load_realtime_summaries(tmp_path: Path):
    results_dir = _create_mock_results(tmp_path)
    raw_summaries, df_summaries, df_samples = load_realtime_summaries(results_dir)

    assert len(raw_summaries) == 2
    assert len(df_summaries) == 2
    assert len(df_samples) == 200
    assert "data_proc_ms" in df_samples.columns
    assert "inference_ms" in df_samples.columns
    assert "total_ms" in df_samples.columns
    assert set(df_samples["model"].unique()) == {"EGB", "RF"}


def test_compute_comprehensive_statistics(tmp_path: Path):
    results_dir = _create_mock_results(tmp_path)
    _, _, df_samples = load_realtime_summaries(results_dir)

    df_stats = compute_comprehensive_statistics(df_samples, deadline_ms=250.0)

    assert len(df_stats) == 2
    assert "Mean Total (ms)" in df_stats.columns
    assert "P95 Total (ms)" in df_stats.columns
    assert "Throughput (preds/sec)" in df_stats.columns
    assert "Data Proc (%)" in df_stats.columns
    assert "Inference (%)" in df_stats.columns
    assert "Compliance (<250ms) %" in df_stats.columns

    # Verify percentages sum close to 100%
    for _, row in df_stats.iterrows():
        proc_pct = row["Data Proc (%)"]
        infer_pct = row["Inference (%)"]
        assert np.isclose(proc_pct + infer_pct, 100.0, atol=1.0)
        assert row["Compliance (<250ms) %"] == 100.0


def test_perform_statistical_tests(tmp_path: Path):
    results_dir = _create_mock_results(tmp_path)
    _, _, df_samples = load_realtime_summaries(results_dir)

    stat_results = perform_statistical_tests(df_samples)

    assert stat_results["test"] == "Kruskal-Wallis H-test"
    assert "statistic" in stat_results
    assert "p_value" in stat_results
    assert len(stat_results["pairwise_mann_whitney"]) == 1


def test_end_to_end_analysis_generation(tmp_path: Path):
    results_dir = _create_mock_results(tmp_path)
    output_dir = tmp_path / "Output_Analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_summaries, df_summaries, df_samples = load_realtime_summaries(results_dir)
    df_stats = compute_comprehensive_statistics(df_samples, deadline_ms=250.0)
    stat_results = perform_statistical_tests(df_samples)

    p1 = plot_latency_component_breakdown(df_stats, output_dir, show_deadlines=True, deadline_ms=250.0)
    p2 = plot_latency_distributions(df_samples, output_dir, show_deadlines=True, deadline_ms=250.0)
    p3 = plot_latency_tail_percentiles(df_stats, output_dir, show_deadlines=True, deadline_ms=250.0)
    p4 = plot_latency_cdf(df_samples, output_dir, show_deadlines=True, deadline_ms=250.0)
    p5 = plot_latency_time_series(df_samples, output_dir)
    report = generate_markdown_report(df_stats, stat_results, output_dir, show_deadlines=True, deadline_ms=250.0)

    for p in [p1, p2, p3, p4, p5, report]:
        assert p.exists(), f"Expected artifact {p} was not created"
    assert report.read_text().startswith("# Real-Time G-LOC Prediction")
