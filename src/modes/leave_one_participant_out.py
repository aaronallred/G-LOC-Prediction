"""Leave-One-Participant-Out (LOPO) cross-validation mode for G-LOC.

For each subject, holds that subject out entirely as the test set, trains on
the remaining subjects, and evaluates. This is a *grouped* CV strategy at the
subject level (as opposed to the trial-level grouping used by the existing
`cross_validation` mode's StratifiedGroupKFold) -- no row belonging to a given
subject is ever split across train and test.

"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.Data_Pipeline.data_pipeline import DataPipeline
from src.models.model_factory import ModelFactory
from src.modes.cross_validation import (
    _aggregate_cv_results,
    _build_fold_result,
    _evaluate_model,
    _run_traditional_hpo,
    _smote_resampling,
)

logger = logging.getLogger(__name__)


def _cache_subject_data(
        pipeline: DataPipeline,
        config: Dict[str, Any],
        model: Any,
        subjects: List[int],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Fetch each subject's (X, y) individually via the traditional pipeline's
    'cache' feature-selection mode, so every subject comes back with the same
    fixed, pre-selected columns and can be safely concatenated later.

    Temporarily mutates config['shared_data_parameters']['subject_to_analyze']
    and ['analysis_type'] for each call (this is DataPipeline's documented
    interface for single-subject pulls), restoring the original values
    afterward regardless of success or failure.
    """
    shared_params = config["shared_data_parameters"]
    original_subject = shared_params.get("subject_to_analyze")
    original_analysis_type = shared_params.get("analysis_type")

    subject_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    try:
        for subject in subjects:
            shared_params["subject_to_analyze"] = subject
            shared_params["analysis_type"] = 1  # all trials, one subject

            logger.info(f"  Fetching data for subject {subject}...")
            X_subj, y_subj = pipeline.get_data(
                model=model,
                traditional_feature_selection="cache",
            )
            subject_cache[subject] = (X_subj, y_subj)
            logger.info(f"    -> X shape {X_subj.shape}, y shape {y_subj.shape}")
    finally:
        shared_params["subject_to_analyze"] = original_subject
        shared_params["analysis_type"] = original_analysis_type

    return subject_cache


def _run_lopo_fold(
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        held_out_subject: int,
        random_seed: int,
        class_weight: Optional[str],
) -> Tuple[Dict[str, Any], Any]:
    """Run one LOPO fold: SMOTE-resample the training set, run BayesSearchCV
    HPO (reusing cross_validation's traditional HPO routine), and evaluate on
    the held-out subject.

    Returns
    -------
    tuple
        (fold_result dict, fitted BayesSearchCV searcher)
    """
    logger.info(
        f"  Running LOPO fold for held-out subject {held_out_subject} "
        f"(train rows={len(X_train)}, test rows={len(X_test)})"
    )

    X_train_res, y_train_res = _smote_resampling(
        X_train=X_train,
        y_train=y_train,
        random_seed=random_seed,
    )

    hpo_result, search = _run_traditional_hpo(
        model, X_train_res, y_train_res, random_seed, class_weight
    )

    preds = search.predict(X_test)
    fold_metrics = _evaluate_model(y_test, preds)

    fold_result = _build_fold_result(
        fold_idx=held_out_subject,
        metrics=fold_metrics,
        n_train=len(X_train_res),
        n_val=len(X_test),
        best_params=hpo_result["best_params"],
    )
    fold_result["held_out_subject"] = held_out_subject

    return fold_result, search


def run_leave_one_participant_out(
        config: Dict[str, Any],
        pipeline: DataPipeline,
        model_factory: ModelFactory,
        project_root_path: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run LOPO cross-validation for a list of traditional models.

    Mirrors the structure and result-saving conventions of
    `cross_validation.run_cross_validation`, but splits by subject rather
    than by k-fold, and saves under `Results/{...}/{model_type}/{model_name}/
    subject_{N}/` instead of `fold_{N}/`.

    Args:
        config: Loaded YAML experiment configuration mapping. Must contain a
            `leave_one_participant_out` section, and (due to a pipeline quirk
            documented at the top of this module) a
            `sensor_ablation.training.median_hyperparameters_folder` value.
        pipeline: DataPipeline instance for data loading
        model_factory: ModelFactory instance for creating models
        project_root_path: Root path of the project, used for saving results

    Returns:
        Dict mapping model names to lists of per-subject fold result dicts.
    """
    lopo_config = config["leave_one_participant_out"]
    models = lopo_config["models"]
    model_type = lopo_config["model_type"]
    random_seed = lopo_config["random_seed"]
    class_weight = lopo_config.get("class_weight")
    results_path = Path(project_root_path / lopo_config["save_results_folder"])
    subjects = lopo_config.get("subjects") or list(range(1, 14))  # 1-13, per README

    model_type_folder = model_type.get_folder_name()
    results_root_with_type = results_path / model_type_folder

    logger.info(
        "Starting LOPO cross-validation over %s subjects and %s model(s)",
        len(subjects),
        len(models),
    )

    pipeline.set_random_seed(random_seed)
    pipeline.set_model_type(model_type)

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for model_name in models:
        model = model_factory.create_model(model_name)

        if not model.is_traditional_model:
            logger.warning(
                f"Skipping '{model.name}': leave_one_participant_out currently "
                "supports traditional models only. AdvancedDataPipeline has no "
                "subject-level (unsplit) data-pull mode -- see this module's "
                "docstring for details."
            )
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"LOPO cross-validating model: {model.name}")
        logger.info(f"{'=' * 60}")

        subject_cache = _cache_subject_data(pipeline, config, model, subjects)

        model_results_dir = results_root_with_type / model.name
        fold_results: List[Dict[str, Any]] = []

        for held_out_subject in subjects:
            if held_out_subject not in subject_cache:
                logger.warning(f"  No cached data for subject {held_out_subject}, skipping.")
                continue

            X_test, y_test = subject_cache[held_out_subject]
            train_subjects = [s for s in subjects if s != held_out_subject and s in subject_cache]

            if not train_subjects:
                logger.warning(
                    f"  No training subjects available when holding out {held_out_subject}, skipping."
                )
                continue

            X_train = np.concatenate([subject_cache[s][0] for s in train_subjects], axis=0)
            y_train = np.concatenate([subject_cache[s][1] for s in train_subjects], axis=0)

            fold_result, search = _run_lopo_fold(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                held_out_subject=held_out_subject,
                random_seed=random_seed,
                class_weight=class_weight,
            )

            subject_dir = model_results_dir / f"subject_{held_out_subject}"
            subject_dir.mkdir(parents=True, exist_ok=True)

            model_path = subject_dir / "model.pkl"
            with open(model_path, "wb") as f:
                joblib.dump(search, f)
            logger.info(f"  Saved held-out-subject {held_out_subject} BayesSearchCV object to {model_path}")

            fold_result_path = subject_dir / "fold_result.json"
            with open(fold_result_path, "w") as f:
                json.dump(fold_result, f, indent=4)
            logger.info(f"  Saved subject {held_out_subject} metrics to {fold_result_path}")

            fold_results.append(fold_result)

        if not fold_results:
            logger.warning(f"No LOPO folds completed for model {model.name}; skipping summary.")
            continue

        # Reuses cross_validation's aggregation helper -- it only assumes each
        # fold_result has a "metrics" dict with the standard metric keys, which
        # `_build_fold_result` guarantees regardless of the split strategy.
        aggregated, median_fold_idx, median_f1 = _aggregate_cv_results(fold_results)
        summary_path = model_results_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(aggregated, f, indent=4)
        logger.info(f"Saved aggregated LOPO summary to {summary_path}")
        logger.info(f"Model {model.name} LOPO results: {aggregated}")

        all_results[model.name] = fold_results

    logger.info(f"\nLOPO cross-validation complete. Results saved to {results_root_with_type}")
    return all_results