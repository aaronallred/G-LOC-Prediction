"""Rank the top offending features by standardization delta per model.

Loads the per-fold ``delta_data.npz`` / ``trial_data.npz`` / ``fold_metadata.json``
files written by ``traditional_standardization_metrics.py`` and reports, for each
model, the ``top_n`` (fold, feature) entries with the largest |delta| across all
subjects, trials, and folds.

Both s1 (per-trial) and s2 (global per-column) deltas are pooled into one
ranked list per model; s1 entries carry the subject/trial of the single worst
row, while s2 entries are global and show ``--``. Ranking uses the μ shift
(``delta_s1_mean`` / ``delta_s2_mean``); the σ shift at the same location is
reported as a secondary column.

Only three files per fold are required::

    delta_data.npz        delta_s1_mean/std (n_rows, n_cols), delta_s2_mean/std (n_cols,)
    trial_data.npz        subject, trial (n_rows,)
    fold_metadata.json    raw_feature_names, model_name, fold_id

Folds missing any of these are skipped with a warning, so a partial download
set works fine.

Usage::

    python -m src.real_time.standardization_metrics_analysis.top_deltas_analysis \
        --root Results/Traditional_Standardization_Metrics
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "Results/Traditional_Standardization_Metrics"
REQUIRED_NPZ = {
    "delta_data.npz": ("delta_s1_mean", "delta_s1_std", "delta_s2_mean", "delta_s2_std"),
    "trial_data.npz": ("subject", "trial"),
}


def _load_fold(fold_dir: Path) -> dict | None:
    """Load a fold's deltas + row ids + feature names, or ``None`` if incomplete."""
    missing = []
    arrays = {}
    for filename, keys in REQUIRED_NPZ.items():
        path = fold_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        with np.load(path) as zf:
            for key in keys:
                if key not in zf:
                    missing.append(f"{filename}::{key}")
                else:
                    arrays[key] = zf[key]

    meta_path = fold_dir / "fold_metadata.json"
    if not meta_path.exists():
        missing.append("fold_metadata.json")
    else:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    if missing:
        logger.warning("Skipping %s: missing %s", fold_dir, ", ".join(sorted(set(missing))))
        return None

    feature_names = list(meta["raw_feature_names"])
    n_cols = arrays["delta_s1_mean"].shape[1]
    if len(feature_names) != n_cols:
        logger.warning(
            "Skipping %s: %d raw_feature_names but %d delta columns",
            fold_dir, len(feature_names), n_cols,
        )
        return None

    return {
        "model_name": str(meta["model_name"]),
        "fold_id": int(meta["fold_id"]),
        "feature_names": feature_names,
        "delta_s1_mean": arrays["delta_s1_mean"],
        "delta_s1_std": arrays["delta_s1_std"],
        "delta_s2_mean": arrays["delta_s2_mean"],
        "delta_s2_std": arrays["delta_s2_std"],
        "subject": arrays["subject"].astype(str),
        "trial": arrays["trial"].astype(str),
    }


def _fold_entries(fold_data: dict) -> list[dict]:
    """Per-feature worst-row entries for one fold (s1 + s2 pooled)."""
    n_cols = len(fold_data["feature_names"])
    entries: list[dict] = []
    for j in range(n_cols):
        abs_mean = np.abs(fold_data["delta_s1_mean"][:, j])
        argmax = int(np.argmax(abs_mean))
        entries.append(
            {
                "feature": fold_data["feature_names"][j],
                "s1_or_s2": "s1",
                "abs_delta": float(abs_mean[argmax]),
                "delta_mean": float(fold_data["delta_s1_mean"][argmax, j]),
                "delta_std": float(fold_data["delta_s1_std"][argmax, j]),
                "subject": fold_data["subject"][argmax],
                "trial": fold_data["trial"][argmax],
                "fold_id": fold_data["fold_id"],
            }
        )
        entries.append(
            {
                "feature": fold_data["feature_names"][j],
                "s1_or_s2": "s2",
                "abs_delta": float(abs(fold_data["delta_s2_mean"][j])),
                "delta_mean": float(fold_data["delta_s2_mean"][j]),
                "delta_std": float(fold_data["delta_s2_std"][j]),
                "subject": "--",
                "trial": "--",
                "fold_id": fold_data["fold_id"],
            }
        )
    return entries


def _collect_entries(root: Path) -> dict[str, list[dict]]:
    """Group per-fold worst-row entries by model name, walking the output tree."""
    by_model: dict[str, list[dict]] = {}
    for meta_path in sorted(root.rglob("fold_metadata.json")):
        fold_data = _load_fold(meta_path.parent)
        if fold_data is None:
            continue
        by_model.setdefault(fold_data["model_name"], []).extend(_fold_entries(fold_data))
    return by_model


def run_top_deltas(root: str | Path = DEFAULT_ROOT, top_n: int = 10, output_path: str | Path | None = None) -> dict[str, list[dict]]:
    """Return and print the top-``top_n`` |delta| (fold, feature) entries per model."""
    root = Path(root)
    by_model = _collect_entries(root)
    if not by_model:
        logger.warning("No complete folds found under %s. Required per fold: %s",
                       root, ", ".join(REQUIRED_NPZ) + ", fold_metadata.json")

    results: dict[str, list[dict]] = {}
    for model_name in sorted(by_model):
        ranked = sorted(by_model[model_name], key=lambda e: e["abs_delta"], reverse=True)[:top_n]
        results[model_name] = ranked
        print(f"\n{model_name} — top {len(ranked)} by |delta mean|")
        print("Rank | Feature | s1/s2 | |dmean| | dmean | dstd | Subject | Trial | Fold")
        for rank, e in enumerate(ranked, start=1):
            print(
                f"{rank:<5}| {e['feature']} | {e['s1_or_s2']} | "
                f"{e['abs_delta']:.4f} | {e['delta_mean']:.4f} | {e['delta_std']:.4f} | "
                f"{e['subject']} | {e['trial']} | {e['fold_id']}"
            )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=float)
        logger.info("Wrote top deltas to %s", out)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Output tree root (default: %(default)s)")
    parser.add_argument("--top-n", type=int, default=10, help="Entries per model (default: %(default)s)")
    parser.add_argument("--output", default=None, help="Optional JSON path for the results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_top_deltas(args.root, top_n=args.top_n, output_path=args.output)


if __name__ == "__main__":
    main()