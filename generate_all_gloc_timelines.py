"""
generate_all_gloc_timelines.py

Plots actual vs predicted GLOC over time for every subject that has a finished
LOPO fold, with trial boundaries and the centrifuge G-level overlaid.
One PNG per subject.

Before you run it
  - Finish a LOPO run for the model. This script reads the saved model for each
    subject from:
        Results/LOPO_CV/<model_type>/<model>/subject_<N>/model.pkl
  - Use the same config file (same data paths and feature settings) as that run.

How to run (from the repo root)
    python generate_all_gloc_timelines.py --config configs/<your_config>.yaml --model RF

    # optional: plot only some subjects (the default is all of them)
    python generate_all_gloc_timelines.py --config configs/<your_config>.yaml --model RF --subjects 1,3,7

Options
    --config      Experiment config (required).
    --model       Model folder name, e.g. KNN or RF (default: KNN).
    --subjects    Comma-separated subjects, e.g. 1,3,7 (default: every subject with a saved model).
    --output-dir  Where to save the PNGs (default: gloc_timeline_plots_<model>/).

What it does, per subject
  1. Re-fetches the subject's held-out data with the same pipeline LOPO used.
  2. Loads the already-trained model (no retraining) and predicts.
  3. Plots the result and saves gloc_timeline_subject_<N>.png.
A subject is skipped, with a warning, if its saved model no longer matches the
current feature list (for example, median_hyperparameters.json changed after
that fold was trained). The rest of the batch still runs.

How to read a plot
  - Black line: actual GLOC. Red line: predicted GLOC, drawn slightly lower so
    the two stay visible when they overlap.
  - Blue line (right axis): mean centrifuge Gz over each prediction window.
    A gap means no Gz reading was recorded there; it does not mean 0 G.
  - Dashed vertical lines: trial boundaries, labelled with the trial id.
  - The blue overlay is switched off for a subject if its windows cannot be
    lined up exactly with the predictions (a warning says so).
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config_loader import load_experiment_config
from src.models.model_factory import ModelFactory

import src.Data_Pipeline.data_pipeline as dp_module
from src.Data_Pipeline.data_pipeline import DataPipeline, TraditionalDataPipeline

_last_instance = None


class TrialLevelTraditionalDataPipeline(TraditionalDataPipeline):
    """Traditional pipeline that also records how many prediction windows each trial has."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_number_windows = None
        global _last_instance
        _last_instance = self

    def _unpack_dict(self, *args, **kwargs):
        # The third positional argument is number_windows ({trial_id: n_windows}).
        if len(args) >= 3:
            self._last_number_windows = dict(args[2])
        return super()._unpack_dict(*args, **kwargs)


# Make DataPipeline use the subclass above.
dp_module.TraditionalDataPipeline = TrialLevelTraditionalDataPipeline

GLEVEL_COLUMN = "Double I/O.actualGz - Centrifuge"
TIME_COLUMN = "Time (s)"


def fetch_raw_glevel_with_time_by_trial(config, subject_str: str) -> dict:
    """Return {trial_id: (times, gz_values)}: the raw Gz signal and its Time (s) for each trial.

    NaNs are kept on purpose. Part of a trial can have no Gz reading at all, and
    dropping those rows would stretch the real G profile across the whole trial
    and misalign it with the predictions."""
    from src.Data_Pipeline.data_pipeline import TraditionalDataPipeline as _RawPipeline

    shared = config["shared_data_parameters"]
    pipeline = _RawPipeline(data_path=config["data_path"], random_seed=42, config=config)
    file_paths = pipeline._get_data_locations()
    gloc_data = pipeline._load_data(file_paths, shared["output_feature_dtype"])
    gloc_data = pipeline._filter_data_by_analysis_type(1, gloc_data, subject_to_analyze=subject_str)

    if GLEVEL_COLUMN not in gloc_data.columns:
        centrifuge_cols = [c for c in gloc_data.columns if "entrifuge" in c.lower()]
        gz_candidates = [c for c in centrifuge_cols if "gz" in c.lower()]
        print(f"  WARNING: '{GLEVEL_COLUMN}' not found in raw data.")
        if gz_candidates:
            print(f"  Possible Gz-axis alternatives found: {gz_candidates}")
        return {}

    if TIME_COLUMN not in gloc_data.columns:
        print(f"  WARNING: '{TIME_COLUMN}' not found in raw data -- cannot align G-level to real time.")
        return {}

    result = {}
    for trial_id, trial_df in gloc_data.groupby("trial_id", sort=False):
        times = trial_df[TIME_COLUMN].to_numpy(dtype=np.float64)
        gz = trial_df[GLEVEL_COLUMN].to_numpy(dtype=np.float64)
        result[trial_id] = (times, gz)
    return result


def compute_window_aligned_glevel(times: np.ndarray, gz: np.ndarray, time_start: float,
                                   stride: float, window_size: float, n_windows: int,
                                   no_data_fill_value: float = np.nan) -> np.ndarray:
    """Mean Gz inside each of the model's own sliding windows.

    Window j covers [time_start + j*stride, time_start + j*stride + window_size),
    the same formula the pipeline uses to build its windows.

    A window with no Gz readings stays NaN, so it shows as a gap in the plot.
    "No reading" is not known to mean "0 G". Do not change no_data_fill_value
    without first confirming what this sensor reads while the centrifuge is
    inactive."""
    result = np.full(n_windows, no_data_fill_value)
    valid_mask = ~np.isnan(gz)
    valid_times = times[valid_mask]
    valid_gz = gz[valid_mask]

    if len(valid_times) == 0:
        return result  # no Gz data anywhere in this trial

    time_iteration = time_start
    for j in range(n_windows):
        in_window = (time_iteration <= valid_times) & (valid_times < time_iteration + window_size)
        if in_window.any():
            result[j] = valid_gz[in_window].mean()
        time_iteration += stride

    return result


def plot_one_subject(config, model_type, model_type_folder, model_name, subject_int, output_dir):
    """Fetch, predict and plot one subject. Returns True if the PNG was saved."""
    subject_str = f"{subject_int:02d}"
    shared = config["shared_data_parameters"]
    shared["subject_to_analyze"] = subject_str
    shared["analysis_type"] = 1

    model = ModelFactory.create_model(model_name)
    pipeline = DataPipeline(config=config)
    pipeline.set_random_seed(42)
    pipeline.set_model_type(model_type)

    print(f"\n--- Subject {subject_int} ---")

    # 1. Held-out data and per-trial window counts
    print(f"Fetching held-out data...")
    try:
        X_test, y_test = pipeline.get_data(model=model, traditional_feature_selection="cache")
    except Exception as e:
        print(f"  SKIPPED: failed to fetch data ({e})")
        return False

    inst = _last_instance
    if inst is None or inst._last_number_windows is None:
        print("  SKIPPED: failed to capture trial boundaries.")
        return False
    number_windows = dict(inst._last_number_windows)

    # 2. Saved model and predictions
    model_path = Path("Results") / "LOPO_CV" / model_type_folder / model_name / f"subject_{subject_int}" / "model.pkl"
    if not model_path.is_file():
        print(f"  SKIPPED: no saved model at {model_path} (fold not completed).")
        return False

    print(f"Loading trained model from {model_path}...")
    search = joblib.load(model_path)

    print("Generating predictions...")
    try:
        y_pred = search.predict(X_test)
    except ValueError as e:
        print(f"  SKIPPED: prediction failed ({e})")
        print("  This usually means the feature set has changed since this fold trained")
        print("  (e.g. median_hyperparameters.json was modified afterward).")
        return False

    total_from_trials = sum(number_windows.values())
    if total_from_trials != len(y_test):
        print(
            f"  WARNING: trial boundary row count ({total_from_trials}) does not match "
            f"y_test length ({len(y_test)}). Trial markers may be misaligned."
        )

    # 3. G-level overlay, binned into the same windows as the predictions
    print("Fetching raw G-level data for overlay (window-exact alignment)...")
    backend_pipeline = pipeline._build_backend(model)
    hyperparams = backend_pipeline._resolve_traditional_hyperparameters(model, model_name)
    window_size = hyperparams["window_size"]
    stride = hyperparams["stride"]
    traditional_config = config["traditional_data_parameters"]
    offset = traditional_config["offset"]
    time_start = traditional_config["time_start"]
    print(f"  Using model's real windowing: window_size={window_size}, stride={stride}, time_start={time_start}")

    glevel_by_trial_raw = fetch_raw_glevel_with_time_by_trial(config, f"{subject_int:02d}")
    glevel_binned = []
    glevel_ok = bool(glevel_by_trial_raw)
    if glevel_by_trial_raw:
        for trial_id, n_rows in number_windows.items():
            trial_time_gz = glevel_by_trial_raw.get(trial_id)
            if trial_time_gz is None:
                print(f"  WARNING: no raw G-level found for trial {trial_id} -- filling with NaN.")
                glevel_binned.append(np.full(n_rows, np.nan))
                continue

            times, gz = trial_time_gz
            time_end = times.max()
            # Safety check: the window count we expect must equal the real prediction count.
            expected_n_windows = int(((time_end - offset) // stride) - (window_size // stride - 1))
            if expected_n_windows != n_rows:
                print(f"  WARNING: window-count mismatch for trial {trial_id} "
                      f"(replicated formula gives {expected_n_windows}, actual prediction rows = {n_rows}). "
                      f"G-level alignment cannot be verified for this subject -- disabling G-level overlay entirely.")
                glevel_ok = False
                break

            glevel_binned.append(compute_window_aligned_glevel(times, gz, time_start, stride, window_size, n_rows))

        glevel_binned = np.concatenate(glevel_binned) if (glevel_binned and glevel_ok) else np.array([])
    else:
        glevel_binned = np.array([])

    if not glevel_ok:
        glevel_binned = np.array([])

    # 4. Plot
    fig, ax = plt.subplots(figsize=(16, 4))
    x = np.arange(len(y_test))
    y_test_arr = np.asarray(y_test).astype(float)
    y_pred_arr = np.asarray(y_pred).astype(float)

    ax.step(x, y_test_arr, where="post", label="Actual GLOC", color="black", linewidth=1.5)
    ax.step(x, y_pred_arr - 0.05, where="post", label="Predicted GLOC", color="crimson", linewidth=1.2, alpha=0.8)

    ax2 = None
    if len(glevel_binned) == len(x) and not np.all(np.isnan(glevel_binned)):
        ax2 = ax.twinx()
        ax2.plot(x, glevel_binned, label="Centrifuge Gz (mean per prediction window)", color="#1f77b4", linewidth=1.0, alpha=0.7, zorder=1)
        ax2.set_ylabel("Centrifuge Gz (G)", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")
    elif len(glevel_binned) != len(x):
        print(f"  WARNING: G-level array length ({len(glevel_binned)}) does not match prediction length ({len(x)}) -- skipping G-level overlay.")

    # Trial boundaries and labels
    start = 0
    for trial_id, n_rows in number_windows.items():
        ax.axvline(start, color="gray", linestyle="--", linewidth=0.5)
        ax.text(start + n_rows / 2, 1.15, str(trial_id), ha="center", fontsize=8)
        start += n_rows

    ax.set_ylim(-0.3, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No GLOC", "GLOC"])
    ax.set_xlabel("Row index (time, ordered by trial)")
    ax.set_title(f"{model_name} Subject {subject_int}: Actual vs Predicted GLOC Over Time (LOPO held-out)")
    lines1, labels1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
                  bbox_to_anchor=(0.0, -0.18), framealpha=0.9, ncol=3, fontsize=9)
    else:
        ax.legend(lines1, labels1, loc="upper left",
                  bbox_to_anchor=(0.0, -0.18), framealpha=0.9, ncol=2, fontsize=9)

    # Saving can fail briefly on Windows-mounted folders (antivirus/OneDrive file locks), so retry.
    output_path = output_dir / f"gloc_timeline_subject_{subject_int}.png"
    save_succeeded = False
    for attempt in range(1, 4):
        try:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            save_succeeded = True
            break
        except OSError as e:
            print(f"  WARNING: save attempt {attempt}/3 failed ({e}). "
                  f"This is usually a transient file lock (antivirus/OneDrive) on Windows-mounted paths. Retrying...")
            time.sleep(2)
    if not save_succeeded:
        print(f"  SKIPPED: could not save plot for subject {subject_int} after 3 attempts.")
        plt.close(fig)
        return False
    plt.close(fig)

    correct = int((y_pred_arr == y_test_arr).sum())
    false_positives = int(((y_pred_arr == 1) & (y_test_arr == 0)).sum())
    false_negatives = int(((y_pred_arr == 0) & (y_test_arr == 1)).sum())

    print(f"  Saved: {output_path}")
    print(f"  Accuracy: {correct}/{len(y_test)} ({100 * correct / len(y_test):.2f}%) | "
          f"FP: {false_positives} | FN: {false_negatives}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch-generate GLOC timeline plots for all completed LOPO subjects.")
    parser.add_argument("--config", required=True, help="Path to an existing experiment config")
    parser.add_argument("--model", default="KNN", help="Model alias matching the LOPO results folder, e.g. KNN or RF")
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated subject numbers to plot (e.g. '1,3,7'). Default: auto-detect all subjects with a saved model.",
    )
    parser.add_argument("--output-dir", default=None, help="Folder to save PNGs into (default: gloc_timeline_plots_{model}/)")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    model_type = config["cross_validation"]["model_type"]
    model_type_folder = model_type.get_folder_name()

    output_dir_name = args.output_dir if args.output_dir is not None else f"gloc_timeline_plots_{args.model}"
    output_dir = Path(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.subjects:
        subjects = [int(s) for s in args.subjects.split(",")]
    else:
        # Auto-detect: every subject_N folder under Results/LOPO_CV/{model_type}/{model}/ with a saved model.pkl
        lopo_dir = Path("Results") / "LOPO_CV" / model_type_folder / args.model
        subjects = []
        if lopo_dir.is_dir():
            for entry in sorted(lopo_dir.iterdir()):
                if entry.is_dir() and entry.name.startswith("subject_") and (entry / "model.pkl").is_file():
                    subjects.append(int(entry.name.replace("subject_", "")))
        if not subjects:
            print(f"No completed subjects found under {lopo_dir}. Nothing to plot.")
            return

    print(f"Generating timeline plots for {len(subjects)} subject(s): {subjects}")

    succeeded = []
    skipped = []
    for subject_int in subjects:
        ok = plot_one_subject(config, model_type, model_type_folder, args.model, subject_int, output_dir)
        (succeeded if ok else skipped).append(subject_int)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Succeeded: {succeeded}")
    print(f"Skipped:   {skipped}")
    print(f"\nPlots saved to: {output_dir}/")


if __name__ == "__main__":
    main()