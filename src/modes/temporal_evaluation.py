import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.advanced_experiment_utils import baseline_down_select, get_advanced_predictions_and_targets
from src.model_type import ModelType
from src.models.base import BaseModel
from src.models.model_factory import ModelFactory
from src.modes.sensor_ablation import _evaluate_model, _save_run_config
from src.traditional_experiment_utils import apply_imbalance, get_hyperparameters_from_json


def run_temporal_evaluation_training(
		config: Dict[str, Any],
		pipeline: DataPipeline,
		model_factory: ModelFactory,
		project_root_path: Path,
) -> None:
	"""Evaluate each configured model across all horizons and write temporal_summary.json per model."""
	training_config = config["temporal_evaluation"]["training"]
	model_names: List[str] = training_config["models"]
	model_type: ModelType = training_config["model_type"]
	horizons: List[float] = _resolve_horizons(training_config["horizons"])
	num_splits: int = training_config["num_splits"]
	random_seed: int = training_config["random_seed"]
	class_weight = training_config.get("class_weight")
	save_models = bool(training_config.get("save_models", False))
	data_rate = config["traditional_data_parameters"]["data_rate"]

	median_hyperparameters_folder = Path(project_root_path / training_config["median_hyperparameters_folder"])
	results_root = Path(project_root_path / training_config["save_results_folder"]) / model_type.get_folder_name()
	results_root.mkdir(parents=True, exist_ok=True)

	config_path = _save_run_config(config, results_root)
	logging.info("Saved run config to %s", config_path)

	if config.get("traditional_data_parameters", {}).get("backstep", 0):
		logging.warning("traditional_data_parameters.backstep shifts the reference labels used to stratify the fold "
						"split; set it to 0 for temporal evaluation. Horizons=%s", horizons)
	if config.get("advanced_data_parameters", {}).get("horizon", 0):
		logging.warning("advanced_data_parameters.horizon is ignored in temporal evaluation; using horizons=%s",
						horizons)

	pipeline.set_random_seed(random_seed)
	pipeline.set_model_type(model_type)

	horizon_samples = {horizon: int(horizon * data_rate) for horizon in horizons}

	for model_name in model_names:
		proto_model = model_factory.create_model(model_name)
		output_dir = results_root / proto_model.name
		output_dir.mkdir(parents=True, exist_ok=True)

		best_params, selected_features, median_fold_id, median_f1 = get_hyperparameters_from_json(
			median_hyperparameters_folder, model_type, proto_model.name
		)
		logging.info("Temporal evaluation for %s | median fold %s (F1=%.4f) | best_params=%s",
					 proto_model.name, median_fold_id, median_f1, dict(best_params))

		pipeline_hyperparameters = proto_model.data_pipeline_hyperparameters
		summary_metadata = {
			"model": proto_model.name,
			"model_type": model_type.get_folder_name(),
			"pipeline": "traditional" if proto_model.is_traditional_model else "advanced",
			"horizons": horizons,
			"data_rate": data_rate,
			"horizon_samples": {_format_horizon(h): n for h, n in horizon_samples.items()},
			"num_splits": num_splits,
			"random_seed": random_seed,
			"imbalance_type": pipeline_hyperparameters.get("imbalance_type"),
			"class_weight": class_weight,
			"window_size": pipeline_hyperparameters.get("window_size"),
			"hyperparameters_source": {
				"median_hyperparameters_folder": str(median_hyperparameters_folder),
				"fold_id": median_fold_id,
				"f1_score": median_f1,
				"best_params": dict(best_params),
			},
		}

		_run_horizons(
			proto_model, model_factory, pipeline, dict(best_params), list(selected_features), horizon_samples,
			num_splits, random_seed, class_weight, save_models, output_dir, summary_metadata,
		)

	logging.info("Temporal evaluation training complete. Results saved to %s", results_root)


def run_temporal_evaluation_review(
		config: Dict[str, Any],
		project_root_path: Path,
) -> None:
	"""Plot metrics vs horizon from saved temporal_summary.json files."""
	review_config = config["temporal_evaluation"]["review"]
	model_type: ModelType = review_config["model_type"]
	model_names: List[str] = review_config["models"]
	error_type = review_config.get("error_type", "std")
	show_plots = bool(review_config.get("show_plots", True))
	results_root = Path(project_root_path / review_config["save_results_folder"]) / model_type.get_folder_name()

	if len(model_names) == 0:
		raise ValueError("No models specified for temporal evaluation review. Update temporal_evaluation.review.models.")

	summaries: Dict[str, Dict[str, Any]] = {}
	for model_name in model_names:
		summary_path = results_root / model_name / "temporal_summary.json"
		if not summary_path.exists():
			raise FileNotFoundError(f"Expected temporal summary not found: {summary_path}")
		with open(summary_path, "r") as f:
			summaries[model_name] = json.load(f)

		plot_path = results_root / model_name / "metrics_vs_horizon.png"
		_plot_metrics_vs_horizon(summaries[model_name], error_type, plot_path)
		logging.info("Saved %s metrics-vs-horizon plot to %s", model_name, plot_path)

	if len(summaries) > 1:
		plot_path = results_root / "f1_vs_horizon_all_models.png"
		_plot_f1_vs_horizon_all_models(summaries, error_type, plot_path)
		logging.info("Saved F1-vs-horizon comparison plot to %s", plot_path)

	if show_plots:
		plt.show()
	else:
		plt.close("all")


def _run_horizons(
		proto_model: BaseModel,
		model_factory: ModelFactory,
		pipeline: DataPipeline,
		best_params: Dict[str, Any],
		selected_features: List[str],
		horizon_samples: Dict[float, int],
		num_splits: int,
		random_seed: int,
		class_weight,
		save_models: bool,
		output_dir: Path,
		summary_metadata: Dict[str, Any],
) -> None:
	"""
	Load each fold once, train and evaluate every horizon, then rewrite the summaries with folds so far.
	"""
	horizons = list(horizon_samples)
	fold_results_by_horizon: Dict[float, List[Dict[str, Any]]] = {horizon: [] for horizon in horizons}
	run_fold = _run_traditional_fold if proto_model.is_traditional_model else _run_advanced_fold

	for kfold_id in range(num_splits):
		load_start = time.perf_counter()
		get_data_kwargs: Dict[str, Any] = {
			"model": proto_model,
			"kfold_id": kfold_id,
			"num_splits": num_splits,
			"horizons": horizon_samples,
		}
		if proto_model.is_traditional_model:
			get_data_kwargs.update(traditional_feature_selection="raw", return_feature_names=True)

		X_train, X_test, y_train_by_horizon, y_test_by_horizon, features = pipeline.get_data(**get_data_kwargs)

		if proto_model.is_traditional_model:
			X_train, X_test, features = _keep_selected_features(X_train, X_test, features, selected_features)

		logging.info("Loaded fold %d for %s: X_train %s, X_test %s (%.1f s)",
					 kfold_id, proto_model.name, X_train.shape, X_test.shape, time.perf_counter() - load_start)

		for horizon in horizons:
			fold_result = run_fold(
				proto_model, model_factory, best_params, X_train, X_test, y_train_by_horizon[horizon],
				y_test_by_horizon[horizon], features, horizon, kfold_id, random_seed, class_weight, save_models,
				output_dir / f"horizon_{_format_horizon(horizon)}",
			)
			fold_results_by_horizon[horizon].append(fold_result)

		_write_summaries(output_dir, summary_metadata, fold_results_by_horizon, len(features), kfold_id + 1)

def _write_summaries(
		output_dir: Path,
		summary_metadata: Dict[str, Any],
		fold_results_by_horizon: Dict[float, List[Dict[str, Any]]],
		num_features: int,
		num_folds_completed: int,
) -> Path:
	"""Rewrite/update each horizon's summary.json and temporal_summary.json from the folds completed."""
	results: Dict[str, Dict[str, Any]] = {}
	for horizon, fold_results in fold_results_by_horizon.items():
		horizon_key = _format_horizon(horizon)
		summary = _summarize_horizon(fold_results)
		horizon_dir = output_dir / f"horizon_{horizon_key}"
		horizon_dir.mkdir(parents=True, exist_ok=True)
		with open(horizon_dir / "summary.json", "w") as f:
			json.dump(summary, f, indent=4)
		results[horizon_key] = summary

	complete = num_folds_completed == summary_metadata["num_splits"]
	temporal_summary = {
		**summary_metadata,
		"num_folds_completed": num_folds_completed,
		"complete": complete,
		"num_features": num_features,
		"results": results,
	}

	temporal_summary_path = output_dir / "temporal_summary.json"
	with open(temporal_summary_path, "w") as f:
		json.dump(temporal_summary, f, indent=4)

	logging.info("Updated temporal summary after %d/%d folds: %s",
				 num_folds_completed, summary_metadata["num_splits"], temporal_summary_path)
	if complete:
		for horizon_key, summary in results.items():
			logging.info("%s horizon=%s s: F1 %.4f +/- %.4f",
						 summary_metadata["model"], horizon_key, summary["f1_mean"], summary["f1_std"])
	return temporal_summary_path

def _keep_selected_features(
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        selected_features: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """ Subset raw-feature matrices to the CV-selected features that exist in this run"""
    feature_index = {name: i for i, name in enumerate(feature_names)}
    kept = [name for name in selected_features if name in feature_index]
    missing = [name for name in selected_features if name not in feature_index]

    if not kept:
        raise ValueError(
            f"None of the {len(selected_features)} CV-selected features are in this run's feature set. "
            "Check that the temporal config matches the CV run's data settings."
        )
    if missing:
        logging.warning(
            "Skipping %d of %d CV-selected features not in this run's feature set (e.g. %s); using %d.",
            len(missing), len(selected_features), missing[:3], len(kept),
        )

    columns = [feature_index[name] for name in kept]
    return X_train[:, columns], X_test[:, columns], kept


def _run_traditional_fold(
		proto_model: BaseModel,
		model_factory: ModelFactory,
		best_params: Dict[str, Any],
		X_train: np.ndarray,
		X_test: np.ndarray,
		y_train: np.ndarray,
		y_test: np.ndarray,
		features: List[str],
		horizon: float,
		kfold_id: int,
		random_seed: int,
		class_weight,
		save_models: bool,
		horizon_dir: Path,
) -> Dict[str, Any]:
	"""Train and evaluate one traditional fold with fixed hyperparameters and the model's imbalance_type."""
	imbalance_type = proto_model.data_pipeline_hyperparameters["imbalance_type"]
	y_train = np.asarray(y_train).astype(int)
	y_test = np.asarray(y_test).astype(int)
	if int(y_train.sum()) == 0:
		raise ValueError(f"Horizon {horizon} s, fold {kfold_id}: no positive windows in the training split.")

	n_train = len(y_train)
	X_train, y_train = apply_imbalance(imbalance_type, X_train, y_train, random_seed)

	fold_model = model_factory.create_model(proto_model.name, model_hyperparameters=dict(best_params))
	model_parameters = fold_model.get_model_parameters()
	if class_weight is not None and "class_weight" in model_parameters:
		fold_model.set_model_parameters({"class_weight": class_weight})
	if "random_state" in model_parameters:
		fold_model.set_model_parameters({"random_state": random_seed})

	fold_model.train(X_train, y_train)
	fold_metrics = _evaluate_model(y_test, fold_model.predict(X_test))

	fold_result = {
		"fold": kfold_id,
		"horizon": horizon,
		"metrics": {key: float(value) for key, value in fold_metrics.items()},
		"n_train": n_train,
		"n_train_resampled": len(y_train),
		"n_test": len(y_test),
		"n_test_positive": int(np.sum(y_test)),
	}
	_save_fold(fold_result, fold_model, horizon_dir / f"fold_{kfold_id}", save_models, ".pkl")
	return fold_result


def _run_advanced_fold(
		proto_model: BaseModel,
		model_factory: ModelFactory,
		best_params: Dict[str, Any],
		X_train: np.ndarray,
		X_test: np.ndarray,
		y_train: np.ndarray,
		y_test: np.ndarray,
		features: List[str],
		horizon: float,
		kfold_id: int,
		random_seed: int,
		class_weight,
		save_models: bool,
		horizon_dir: Path,
) -> Dict[str, Any]:
	"""Train and evaluate one advanced fold with fixed hyperparameters"""
	fold_model = model_factory.create_model(proto_model.name, model_hyperparameters=dict(best_params))
	fold_model.all_features = features
	if "random_state" in fold_model.get_model_parameters():
		fold_model.set_model_parameters({"random_state": random_seed})

	fold_model.train(X_train, y_train, class_weight=class_weight)

	X_test_ds, _ = baseline_down_select(X_test, fold_model.all_features, fold_model.best_params["baseline_method"])
	actual_labels, predicted_labels = get_advanced_predictions_and_targets(
		model=fold_model,
		X=X_test_ds,
		y=y_test,
		sequence_length=fold_model.best_params["sequence_length"],
		step_size=10,
		batch_size=fold_model.best_params["batch_size"],
	)
	fold_metrics = _evaluate_model(actual_labels, predicted_labels)

	fold_result = {
		"fold": kfold_id,
		"horizon": horizon,
		"metrics": {key: float(value) for key, value in fold_metrics.items()},
		"n_train": len(y_train),
		"n_test": len(actual_labels),
		"n_test_positive": int(np.sum(actual_labels)),
	}
	_save_fold(fold_result, fold_model, horizon_dir / f"fold_{kfold_id}", save_models, ".pt")
	return fold_result


def _save_fold(fold_result: Dict[str, Any], fold_model: BaseModel, fold_dir: Path, save_models: bool,
			   model_extension: str) -> None:
	"""Write fold_result.json (and the model when requested) and log the fold metrics."""
	fold_dir.mkdir(parents=True, exist_ok=True)
	with open(fold_dir / "fold_result.json", "w") as f:
		json.dump(fold_result, f, indent=4)
	if save_models:
		fold_model.save_model(str(fold_dir / f"model{model_extension}"))
	logging.info("horizon=%s s fold=%d: %s",
				 _format_horizon(fold_result["horizon"]), fold_result["fold"], fold_result["metrics"])


def _summarize_horizon(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Mean/std of each metric across folds (CV summary.json field names) plus per-fold values for plotting."""
	summary: Dict[str, Any] = {}
	fold_metrics: Dict[str, List[float]] = {}
	for key in ("accuracy", "precision", "recall", "f1", "specificity", "g_mean"):
		values = [fold_result["metrics"][key] for fold_result in fold_results]
		summary[f"{key}_mean"] = float(np.mean(values))
		summary[f"{key}_std"] = float(np.std(values))
		fold_metrics[key] = values

	summary["num_folds"] = len(fold_results)
	summary["n_positive"] = int(sum(fold_result["n_test_positive"] for fold_result in fold_results))
	summary["n_total"] = int(sum(fold_result["n_test"] for fold_result in fold_results))
	summary["fold_metrics"] = fold_metrics
	return summary


def _metric_band(temporal_summary: Dict[str, Any], metric: str, error_type: str
				 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Mean and shaded band (1 std or min–max across folds) of one metric at every horizon."""
	per_horizon = [
		np.asarray(temporal_summary["results"][_format_horizon(h)]["fold_metrics"][metric], dtype=float)
		for h in temporal_summary["horizons"]
	]
	mean = np.array([values.mean() for values in per_horizon])
	if error_type == "std":
		std = np.array([values.std() for values in per_horizon])
		return mean, mean - std, mean + std
	if error_type == "range":
		return mean, np.array([values.min() for values in per_horizon]), np.array([values.max() for values in per_horizon])
	raise ValueError(f"Unsupported error_type '{error_type}'. Supported: 'std', 'range'.")


def _plot_metrics_vs_horizon(temporal_summary: Dict[str, Any], error_type: str, save_path: Path) -> None:
	"""grid of every metric vs horizon for one model, with the window size marked for traditional models."""
	horizons = temporal_summary["horizons"]
	window_size = temporal_summary.get("window_size")
	band_label = "±1 std" if error_type == "std" else "Range (min–max)"

	fig, axes = plt.subplots(2, 3, figsize=(18, 9))
	for ax, metric in zip(axes.flatten(), ("accuracy", "precision", "recall", "f1", "specificity", "g_mean")):
		mean, lower, upper = _metric_band(temporal_summary, metric, error_type)
		ax.plot(horizons, mean, marker="o", label="Mean")
		ax.fill_between(horizons, lower, upper, color="gray", alpha=0.3, label=band_label)
		if window_size is not None:
			ax.axvline(x=window_size, color="black", linestyle="--", linewidth=1.5, label="Window size")
		ax.set_title(f"{metric} vs horizon")
		ax.set_xlabel("Horizon (s)")
		ax.set_ylabel(metric)
		ax.set_ylim(0, 1.05)
		ax.grid(True)
		ax.legend()

	fig.suptitle(f"Metrics vs horizon — {temporal_summary['model']} {temporal_summary['model_type']}",
				 fontsize=16, fontweight="bold")
	fig.tight_layout()
	fig.savefig(save_path, dpi=150, bbox_inches="tight")


def _plot_f1_vs_horizon_all_models(summaries: Dict[str, Dict[str, Any]], error_type: str, save_path: Path) -> None:
	"""Overlay F1 vs horizon for several models."""
	fig, ax = plt.subplots(figsize=(10, 6))
	for model_name, temporal_summary in summaries.items():
		horizons = temporal_summary["horizons"]
		mean, lower, upper = _metric_band(temporal_summary, "f1", error_type)
		line, = ax.plot(horizons, mean, marker="o", label=model_name)
		ax.fill_between(horizons, lower, upper, color=line.get_color(), alpha=0.15)

	model_type = next(iter(summaries.values()))["model_type"]
	ax.set_title(f"F1 vs horizon — {model_type} ({'±1 std' if error_type == 'std' else 'min–max'} shaded)")
	ax.set_xlabel("Horizon (s)")
	ax.set_ylabel("F1")
	ax.set_ylim(0, 1.05)
	ax.grid(True)
	ax.legend()
	fig.tight_layout()
	fig.savefig(save_path, dpi=150, bbox_inches="tight")


def _format_horizon(horizon: float) -> str:
	"""Stable string form for folder names and JSON keys (0 -> '0', 2.5 -> '2.5')."""
	return f"{horizon:g}"

def _resolve_horizons(horizons_config: Any) -> List[float]:
	"""Accept an explicit list of horizons (s) or {start, stop, step} with stop included."""
	if isinstance(horizons_config, dict):
		start = float(horizons_config["start"])
		stop = float(horizons_config["stop"])
		step = float(horizons_config["step"])
		if step <= 0 or stop < start:
			raise ValueError(f"Invalid horizons range {horizons_config}: need step > 0 and stop >= start.")
		count = int(round((stop - start) / step)) + 1
		return [round(start + i * step, 6) for i in range(count)]
	return [float(horizon) for horizon in horizons_config]