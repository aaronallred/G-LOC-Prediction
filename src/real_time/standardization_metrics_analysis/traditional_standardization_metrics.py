"""Compare train-only vs all-rows (legacy leaky) standardization metrics.

For each traditional model in ``configs/real_time_sensor_ablation.yaml``, this
script extracts the raw feature matrix that ``_standardize_raw`` consumes, re-apply both
standardization styles, and saves per-feature delta statistics for test rows.

Usage::

    python -m src.real_time.traditional_standardization_metrics
    python -m src.real_time.traditional_standardization_metrics --config configs/real_time_sensor_ablation.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.Data_Pipeline.data_pipeline import DataPipeline, TraditionalDataPipeline
from src.Data_Pipeline.fold_standardizer import GlobalStandardizer, TrialAwareStandardizer
from src.models.model_factory import ModelFactory
from src.model_type import ModelType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = "configs/real_time_sensor_ablation.yaml"
OUTPUT_DIR_NAME = "Results/Traditional_Standardization_Metrics"
EXCLUDE_FEATURE_NAMES = {"AFE_indicator_windowed"}
TARGET_SUBSTRINGS = ("Equivital", "Centrifuge")

# ---------------------------------------------------------------------------
# monkey-patch machinery
# ---------------------------------------------------------------------------
_CapturedStandardization: dict[int, dict[str, np.ndarray]] = {}
_CapturedFeatures: dict[int, dict[str, Any]] = {}
_CURRENT_FOLD: int = -1

_OriginalStandardizeRaw = TraditionalDataPipeline._standardize_raw
_OriginalFeatureGeneration = TraditionalDataPipeline._feature_generation


def _capturing_standardize_raw(self, x_raw, trial_id_per_row, train_mask):
    global _CURRENT_FOLD, _CapturedStandardization
    _CapturedStandardization[_CURRENT_FOLD] = {
        "X_raw": np.asarray(x_raw, dtype=np.float64).copy(),
        "trial_id_per_row": np.array(trial_id_per_row, copy=True),
        "train_mask": np.array(train_mask, dtype=bool, copy=True),
    }
    return _OriginalStandardizeRaw(self, x_raw, trial_id_per_row, train_mask)


def _capturing_feature_generation(self, *args, **kwargs):
    global _CURRENT_FOLD, _CapturedFeatures
    result = _OriginalFeatureGeneration(self, *args, **kwargs)
    _CapturedFeatures[_CURRENT_FOLD] = {
        "y_gloc_labels": np.array(result[0], copy=True),
        "x_feature_matrix": np.array(result[1], copy=True),
        "all_features": list(result[2]),
        "trial_id_per_row": np.array(result[3], copy=True),
    }
    return result

# apply the monkey-patches (after both capturing functions are defined)
TraditionalDataPipeline._standardize_raw = _capturing_standardize_raw
TraditionalDataPipeline._feature_generation = _capturing_feature_generation

# ---------------------------------------------------------------------------
# feature-name helpers
# ---------------------------------------------------------------------------
def _extract_baseline_method(feature_name: str) -> str | None:
    for bm in ("v0", "v1", "v2", "v5", "v6"):
        if f"_{bm}" in feature_name or feature_name.endswith(f"_{bm}"):
            return bm
    return None


def _extract_feature_type(feature_name: str) -> str | None:
    for ft in ("mean", "stddev", "max", "range", "additional"):
        if f"_{ft}_" in feature_name or feature_name.endswith(f"_{ft}"):
            return ft
    return None


def _extract_standardization(feature_name: str) -> str | None:
    for s in ("s1", "s2"):
        if feature_name.endswith(f"_{s}"):
            return s
    return None


def _is_target_feature(name: str) -> bool:
    if name in EXCLUDE_FEATURE_NAMES:
        return False
    return any(ts in name for ts in TARGET_SUBSTRINGS)


# ---------------------------------------------------------------------------
# re-standardization
# ---------------------------------------------------------------------------
def _standardize_train_only(
    X_raw: np.ndarray, trial_id: np.ndarray, train_mask: np.ndarray
) -> np.ndarray:
    s1 = TrialAwareStandardizer().fit(X_raw, trial_id, train_mask).transform(X_raw, trial_id)
    s2 = GlobalStandardizer().fit(X_raw[train_mask]).transform(X_raw)
    return np.hstack([s1, s2])


def _standardize_all_rows(X_raw: np.ndarray, trial_id: np.ndarray) -> np.ndarray:
    all_true = np.ones(X_raw.shape[0], dtype=bool)
    s1 = TrialAwareStandardizer().fit(X_raw, trial_id, all_true).transform(X_raw, trial_id)
    s2 = GlobalStandardizer().fit(X_raw).transform(X_raw)
    return np.hstack([s1, s2])


# ---------------------------------------------------------------------------
# per-feature deltas
# ---------------------------------------------------------------------------
def _per_feature_statistics(
    train_only_test: np.ndarray,
    all_rows_test: np.ndarray,
    feature_names: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, dict[str, float]]]]]:
    deltas = train_only_test - all_rows_test
    per_feature: dict[str, dict[str, float]] = {}
    stratified_totals: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = {}

    for j, fname in enumerate(feature_names):
        col_deltas = deltas[:, j]
        stats: dict[str, float] = {
            "mean_abs_delta": float(np.mean(np.abs(col_deltas))),
            "max_abs_delta": float(np.max(np.abs(col_deltas))),
            "mean_delta": float(np.mean(col_deltas)),
            "std_delta": float(np.std(col_deltas)),
            "s1_or_s2": _extract_standardization(fname) or "unknown",
            "feature_type": _extract_feature_type(fname) or "unknown",
            "baseline_method": _extract_baseline_method(fname) or "none",
        }
        per_feature[fname] = stats

        strat_s1_s2 = stats["s1_or_s2"]
        strat_ft = stats["feature_type"]
        strat_bm = stats["baseline_method"]
        stratified_totals.setdefault(strat_s1_s2, {}).setdefault(strat_ft, {}).setdefault(strat_bm, []).append(stats)

    # aggregate stratified totals into mean across features in each bucket
    stratified_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for s1s2, ft_dict in stratified_totals.items():
        for ft, bm_dict in ft_dict.items():
            for bm, stat_list in bm_dict.items():
                keys = ["mean_abs_delta", "max_abs_delta", "mean_delta", "std_delta"]
                aggregated = {
                    k: float(np.mean([s[k] for s in stat_list]))
                    for k in keys
                }
                aggregated["n_features"] = len(stat_list)
                stratified_summary.setdefault(s1s2, {}).setdefault(ft, {})[bm] = aggregated

    return per_feature, stratified_summary


# ---------------------------------------------------------------------------
# per-fold analysis
# ---------------------------------------------------------------------------
def _analyse_fold(
    X_raw: np.ndarray,
    trial_id: np.ndarray,
    train_mask: np.ndarray,
    all_features: list[str],
) -> dict[str, Any]:
    std_train_only = _standardize_train_only(X_raw, trial_id, train_mask)
    std_all_rows = _standardize_all_rows(X_raw, trial_id)
    test_mask = ~train_mask
    n_test_rows = int(test_mask.sum())

    target_idx = [i for i, n in enumerate(all_features) if _is_target_feature(n)]
    target_features = [all_features[i] for i in target_idx]
    n_target = len(target_features)

    per_feature, stratified = _per_feature_statistics(
        std_train_only[test_mask][:, target_idx],
        std_all_rows[test_mask][:, target_idx],
        target_features,
    )

    abs_deltas = [
        abs(std_train_only[test_mask, i] - std_all_rows[test_mask, i])
        for i in target_idx
    ]
    all_flat = np.concatenate(abs_deltas) if abs_deltas else np.zeros(0)
    summary = {
        "mean_abs_delta": float(np.mean(all_flat)) if len(all_flat) else 0.0,
        "median_abs_delta": float(np.median(all_flat)) if len(all_flat) else 0.0,
        "max_abs_delta": float(np.max(all_flat)) if len(all_flat) else 0.0,
        "std_abs_delta": float(np.std(all_flat)) if len(all_flat) else 0.0,
    }

    return {
        "n_train_rows": int(train_mask.sum()),
        "n_test_rows": n_test_rows,
        "n_total_features_cols": int(std_train_only.shape[1]),
        "n_target_features": n_target,
        "per_feature": per_feature,
        "stratified_summary": stratified,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# model-level aggregation
# ---------------------------------------------------------------------------
def _aggregate_model_summary(folds: list[dict[str, Any]]) -> dict[str, float]:
    mean_abs = [f["summary"]["mean_abs_delta"] for f in folds]
    median_abs = [f["summary"]["median_abs_delta"] for f in folds]
    max_abs = [f["summary"]["max_abs_delta"] for f in folds]
    return {
        "mean_abs_delta_across_folds": float(np.mean(mean_abs)),
        "median_abs_delta_across_folds": float(np.median(median_abs)),
        "max_abs_delta_across_folds": float(np.max(max_abs)),
    }


# ---------------------------------------------------------------------------
# text report writers
# ---------------------------------------------------------------------------
def _save_text_per_fold(fold_report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("FOLD REPORT")
    lines.append("=" * 80)
    lines.append(f"  n_train_rows: {fold_report['n_train_rows']}")
    lines.append(f"  n_test_rows:  {fold_report['n_test_rows']}")
    lines.append(f"  n_total_features_cols: {fold_report['n_total_features_cols']}")
    lines.append(f"  n_target_features:     {fold_report['n_target_features']}")
    lines.append("")
    s = fold_report["summary"]
    lines.append(f"  mean_abs_delta:   {s['mean_abs_delta']:.6f}")
    lines.append(f"  median_abs_delta: {s['median_abs_delta']:.6f}")
    lines.append(f"  max_abs_delta:    {s['max_abs_delta']:.6f}")
    lines.append(f"  std_abs_delta:    {s['std_abs_delta']:.6f}")
    lines.append("")

    lines.append("-" * 40)
    lines.append("STRATIFIED SUMMARY (s1/s2 x feature_type x baseline_method)")
    lines.append("-" * 40)
    for s1s2 in ("s1", "s2"):
        ft_dict = fold_report.get("stratified_summary", {}).get(s1s2, {})
        if not ft_dict:
            continue
        for ft in ("mean", "stddev", "max", "range", "additional"):
            bm_dict = ft_dict.get(ft, {})
            if not bm_dict:
                continue
            for bm in ("v0", "v1", "v2", "v5", "v6", "none"):
                stats = bm_dict.get(bm)
                if stats is None:
                    continue
                lines.append(f"  {s1s2:>3s} | {ft:>10s} | {bm:>4s} -> "
                             f"mean_abs_delta={stats['mean_abs_delta']:.6f}  "
                             f"max_abs_delta={stats['max_abs_delta']:.6f}  "
                             f"n_features={stats['n_features']}")
    lines.append("")

    lines.append("-" * 40)
    lines.append("PER-FEATURE DELTA STATISTICS (test rows, train_only - all_rows)")
    lines.append("-" * 40)
    for fname, stats_outer in fold_report.get("per_feature", {}).items():
        lines.append(f"  {fname}")
        lines.append(f"    mean_abs_delta={stats_outer['mean_abs_delta']:.6f}  "
                     f"max_abs_delta={stats_outer['max_abs_delta']:.6f}  "
                     f"mean_delta={stats_outer['mean_delta']:.6f}  "
                     f"std_delta={stats_outer['std_delta']:.6f}  "
                     f"s1s2={stats_outer.get('s1_or_s2','?')}  "
                     f"ft={stats_outer.get('feature_type','?')}  "
                     f"bm={stats_outer.get('baseline_method','?')}")
    lines.append("=" * 80)
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_model_text_summary(
    model_name: str,
    summary: dict[str, float],
    model_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"MODEL SUMMARY: {model_name}")
    lines.append("=" * 80)
    lines.append(f"  mean_abs_delta_across_folds:  {summary['mean_abs_delta_across_folds']:.6f}")
    lines.append(f"  median_abs_delta_across_folds: {summary['median_abs_delta_across_folds']:.6f}")
    lines.append(f"  max_abs_delta_across_folds:    {summary['max_abs_delta_across_folds']:.6f}")
    lines.append("=" * 80)
    (model_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _save_text_overall(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("TRADITIONAL STANDARDIZATION METRICS COMPARISON")
    lines.append("Train-Only (current) vs All-Rows (legacy leaky) Standardization")
    lines.append("=" * 80)
    lines.append("")

    cfg = report["config"]
    lines.append("-" * 40)
    lines.append("CONFIG")
    lines.append("-" * 40)
    for k, v in cfg.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("-" * 40)
    lines.append("PER-MODEL SUMMARY")
    lines.append("-" * 40)
    header = f"{'Model':>6s} | {'mean_abs_delta':>14s} | {'median_abs_delta':>16s} | {'max_abs_delta':>13s}"
    lines.append(header)
    lines.append("-" * len(header))
    for mdata in report["models"]:
        name = mdata["name"]
        s = mdata["summary"]
        lines.append(f"{name:>6s} | {s['mean_abs_delta_across_folds']:14.6f} | "
                     f"{s['median_abs_delta_across_folds']:16.6f} | "
                     f"{s['max_abs_delta_across_folds']:13.6f}")
    lines.append("")

    lines.append("-" * 40)
    lines.append("PER-FOLD TABLE (one row per model×fold)")
    lines.append("-" * 40)
    header2 = f"{'Model':>6s} | {'Fold':>4s} | {'#train':>6s} | {'#test':>5s} | {'n_target':>8s} | {'mean_abs':>10s} | {'max_abs':>10s}"
    lines.append(header2)
    lines.append("-" * len(header2))
    for mdata in report["models"]:
        name = mdata["name"]
        for f in mdata["folds"]:
            s = f["summary"]
            lines.append(f"{name:>6s} | {f['fold_id']:4d} | {f['n_train_rows']:6d} | "
                         f"{f['n_test_rows']:5d} | {f['n_target_features']:8d} | "
                         f"{s['mean_abs_delta']:10.6f} | {s['max_abs_delta']:10.6f}")
    lines.append("")

    lines.append("=" * 80)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved JSON to %s", path)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


def run_standardization_comparison(
    config_path: Path | None = None,
    project_root: Path | None = None,
    output_dir: Path | None = None,
    data_path: Path | None = None,
) -> dict[str, Any]:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    if config_path is None:
        config_path = project_root / DEFAULT_CONFIG
    if output_dir is None:
        output_dir = project_root / OUTPUT_DIR_NAME

    from src.config_loader import load_experiment_config

    config = load_experiment_config(config_path)
    if data_path is not None:
        config["data_path"] = str(data_path)
    training_cfg = config["sensor_ablation"]["training"]
    num_splits: int = training_cfg["num_splits"]
    random_seed: int = training_cfg["random_seed"]
    model_names: list[str] = training_cfg["models"]
    model_type: ModelType = training_cfg["model_type"]
    feature_streams: list[str] = training_cfg["streams"][0]

    pipeline = DataPipeline(config=config)
    pipeline.set_model_type(model_type)
    pipeline.set_random_seed(random_seed)
    factory = ModelFactory()

    out_root = output_dir / model_type.get_folder_name()
    out_root.mkdir(parents=True, exist_ok=True)

    overall_report: dict[str, Any] = {
        "config": {
            "num_splits": num_splits,
            "random_seed": random_seed,
            "model_type_string": str(model_type),
            "feature_streams": feature_streams,
            "target_substrings": list(TARGET_SUBSTRINGS),
        },
        "models": [],
    }

    global _CURRENT_FOLD

    try:
        for model_name in model_names:
            model = factory.create_model(model_name)
            model_out = out_root / model.name
            model_out.mkdir(parents=True, exist_ok=True)

            fold_reports: list[dict[str, Any]] = []

            for fold_id in range(num_splits):
                fold_json_path = model_out / f"fold_{fold_id}" / "fold_result.json"
                if fold_json_path.exists():
                    logger.info(
                        "Skipping fold %d for model %s (already exists at %s)",
                        fold_id, model.name, fold_json_path,
                    )
                    try:
                        existing_report = json.loads(fold_json_path.read_text())
                        fold_reports.append(existing_report)
                        fold_txt_path = fold_json_path.with_name("fold_result.txt")
                        if not fold_txt_path.exists():
                            _save_text_per_fold(existing_report, fold_txt_path)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Could not load existing fold_result.json for fold %d; will recompute",
                            fold_id,
                        )
                    continue
                _CURRENT_FOLD = fold_id

                try:
                    _ = pipeline.get_data(
                        model=model,
                        kfold_id=fold_id,
                        num_splits=num_splits,
                        feature_streams=feature_streams,
                        traditional_feature_selection="raw",
                        return_feature_names=True,
                    )
                except Exception:
                    logger.error(
                        "get_data failed for model=%s fold=%d",
                        model_name,
                        fold_id,
                        exc_info=True,
                    )
                    continue

                cap_std = _CapturedStandardization.pop(fold_id, None)
                cap_feat = _CapturedFeatures.pop(fold_id, None)
                if cap_std is None or cap_feat is None:
                    logger.error(
                        "Missing captured data for model=%s fold=%d. "
                        "std=%s feat=%s",
                        model_name,
                        fold_id,
                        cap_std is not None,
                        cap_feat is not None,
                    )
                    continue

                fold_report = _analyse_fold(
                    cap_std["X_raw"],
                    cap_std["trial_id_per_row"],
                    cap_std["train_mask"],
                    cap_feat["all_features"],
                )
                fold_report["fold_id"] = fold_id

                fold_dir = model_out / f"fold_{fold_id}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                _save_json(fold_report, fold_dir / "fold_result.json")
                _save_text_per_fold(fold_report, fold_dir / "fold_result.txt")

                fold_reports.append(fold_report)

            model_summary = _aggregate_model_summary(fold_reports)
            model_entry: dict[str, Any] = {
                "name": model.name,
                "baseline_window": model.data_pipeline_hyperparameters.get("baseline_window"),
                "window_size": model.data_pipeline_hyperparameters.get("window_size"),
                "stride": model.data_pipeline_hyperparameters.get("stride"),
                "summary": model_summary,
                "folds": fold_reports,
            }
            overall_report["models"].append(model_entry)

            _save_json({"summary": model_summary}, model_out / "summary.json")
            _save_model_text_summary(model.name, model_summary, model_out)

    finally:
        TraditionalDataPipeline._standardize_raw = _OriginalStandardizeRaw
        TraditionalDataPipeline._feature_generation = _OriginalFeatureGeneration

    _save_json(overall_report, out_root.parent / "summary.json")
    _save_text_overall(overall_report, out_root.parent / "summary.txt")

    return overall_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compare train-only vs all-rows standardization metrics for traditional models."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output reports",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override the data_path from the YAML config (e.g., 'data_reduced' to save memory)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    data_path = Path(args.data_path) if args.data_path else None

    report = run_standardization_comparison(
        config_path=config_path,
        output_dir=output_dir,
        data_path=data_path,
    )
    print("Standardization metrics comparison complete.")
    print(f"Results saved to {OUTPUT_DIR_NAME}/")