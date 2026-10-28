"""
Plot new-format k-fold CV results

Example terminal command:
python -m src.scripts.plot_cv_results --results Results/XGB_CV/noAFE_Explicit Results/XGB_CV_ridge2/noAFE_Explicit --labels lasso ridge

labels applied to plots
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from matplotlib import pyplot as plt

from src.scripts.GLOC_visualization_traditional import (
	bridget_recall_specificity_balanced_accuracy,
	plot_bayes_tuning,
	plot_cross_val_sp,
	plot_f1_bar_with_ci,
)


def _model_dirs(results_dir: Path) -> List[Path]:
	"""Model subfolders of a results folder, sorted by name."""
	return sorted(p for p in results_dir.iterdir() if p.is_dir())


def _display_name(model_name: str, label: Optional[str]) -> str:
	"""Model name shown in plots; tagged with the run label when comparing several results folders."""
	return model_name if label is None else f"{model_name} ({label})"


def load_cv_results(results_dirs: List[Path], labels: Optional[List[str]]) -> Dict[str, pd.DataFrame]:
	"""Return {fold_name: DataFrame indexed by model display name, one column per metric} across all folders."""
	rows: Dict[str, Dict[str, dict]] = {}
	for results_dir, label in zip(results_dirs, labels or [None] * len(results_dirs)):
		found_any = False
		for model_dir in _model_dirs(results_dir):
			display_name = _display_name(model_dir.name, label)
			for fold_dir in sorted(model_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1])):
				fold_result_path = fold_dir / "fold_result.json"
				if not fold_result_path.exists():
					continue
				metrics = json.loads(fold_result_path.read_text())["metrics"]
				# Legacy plots expect 'f1-score' instead of 'f1'
				metrics["f1-score"] = metrics.pop("f1")
				if display_name in rows.get(fold_dir.name, {}):
					raise ValueError(f"Duplicate model name '{display_name}'. Use --labels to tell the runs apart.")
				rows.setdefault(fold_dir.name, {})[display_name] = metrics
				found_any = True

		if not found_any:
			raise FileNotFoundError(f"No fold_*/fold_result.json files found under {results_dir}")

	return {fold: pd.DataFrame.from_dict(models, orient="index") for fold, models in rows.items()}


def plot_median_fold_bayes_tuning(results_dirs: List[Path], labels: Optional[List[str]]) -> None:
	"""Plot BayesSearchCV convergence for each model's median-F1 fold in every results folder."""
	for results_dir, label in zip(results_dirs, labels or [None] * len(results_dirs)):
		for model_dir in _model_dirs(results_dir):
			median_path = model_dir / "median_hyperparameters.json"
			if not median_path.exists():
				continue
			fold_id = json.loads(median_path.read_text())["fold_id"]
			model_path = model_dir / f"fold_{fold_id}" / "model.pkl"
			if not model_path.exists():
				continue
			plot_bayes_tuning(joblib.load(model_path))
			plt.gcf().axes[0].set_title(
				f"{_display_name(model_dir.name, label)} (median fold {fold_id}): BayesSearchCV Best Score Over Iterations"
			)


def save_figures_instead_of_showing(save_dir: Path) -> None:
	"""Make plt.show() save every open figure to save_dir instead of opening windows."""
	save_dir.mkdir(parents=True, exist_ok=True)
	counter = itertools.count(1)

	def _save_and_close(*args, **kwargs):
		for fig_num in plt.get_fignums():
			fig = plt.figure(fig_num)
			fig.savefig(save_dir / f"figure_{next(counter):02d}.png", dpi=150, bbox_inches="tight")
			plt.close(fig)

	plt.show = _save_and_close


def _resolve_labels(results_dirs: List[Path], labels: Optional[List[str]]) -> Optional[List[str]]:
	"""No labels for a single folder; otherwise the given labels or each run's folder name."""
	if labels is not None:
		if len(labels) != len(results_dirs):
			raise ValueError(f"Got {len(labels)} --labels for {len(results_dirs)} --results folders; counts must match.")
		return labels
	if len(results_dirs) == 1:
		return None
	return [results_dir.resolve().parent.name for results_dir in results_dirs]


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--results", type=Path, nargs="+", required=True,
						help="One or more folders containing one subfolder per model, "
							 "e.g. Results/XGB_CV_lasso/Complete_Explicit Results/XGB_CV_ridge/Complete_Explicit")
	parser.add_argument("--labels", nargs="+", default=None,
						help="Optional label per --results folder (same order), used as '<model> (<label>)' in plots")
	parser.add_argument("--save-dir", type=Path, default=None,
						help="Save figures as PNGs here instead of opening plot windows")
	parser.add_argument("--no-bayes", action="store_true",
						help="Skip the BayesSearchCV convergence plots")
	args = parser.parse_args()

	labels = _resolve_labels(args.results, args.labels)

	if args.save_dir is not None:
		save_figures_instead_of_showing(args.save_dir)

	data_dict = load_cv_results(args.results, labels)
	model_names = sorted({name for df in data_dict.values() for name in df.index})
	print(f"Loaded {len(data_dict)} folds for models: {model_names}")

	plot_cross_val_sp(data_dict)
	plot_f1_bar_with_ci(data_dict)
	bridget_recall_specificity_balanced_accuracy(data_dict)

	if not args.no_bayes:
		plot_median_fold_bayes_tuning(args.results, labels)
		plt.show()


if __name__ == "__main__":
	main()