# G-LOC-Prediction

This repository contains the current G-LOC prediction pipeline for feature extraction, sensor ablation experiments, and traditional model evaluation.

## What lives here

- `src/main.py` is the main entry point.
- `GLOC_experiment_config.yaml` controls which models, data settings, and sensor ablation modes run.
- `data/` contains the input CSV files and supporting datasets used by the pipeline.
- `Results/` is where sensor-ablation outputs and plots are written.
- `ModelSave/` contains saved model artifacts when the code paths produce them.

## Environment

The project is configured for the Conda environment in `environment.yml`. That environment includes the RAPIDS and cuML stack used by the current pipeline.

Typical setup:

```bash
conda env create -f environment.yml
conda activate gloc
```

## Running the pipeline

Run the main module from the repository root:

```bash
python -m src.main
```

To use a different YAML file:

```bash
python -m src.main --config /path/to/your_config.yaml
```

If `--config` is omitted, the code uses the root-level `GLOC_experiment_config.yaml`.

## YAML config overview

The config file is a YAML mapping with these top-level sections:

- `models`: list of model names to instantiate.
- `model_type`: two-item list describing the AFE filter and feature set.
- `random_seed`: integer seed used for splitting and model execution.
- `data_path`: path to the dataset directory.
- `shared_data_parameters`: settings shared by the advanced and traditional pipelines.
- `advanced_data_parameters`: settings used by the advanced-classifier path.
- `traditional_data_parameters`: settings used by the traditional-classifier path.
- `sensor_ablation`: training and review settings for stream ablation experiments.

### `models`

Use one or more of the supported model aliases below:

- `EGB` or `Extreme Gradient Boosting`
- `KNN` or `K Nearest Neighbors`
- `RF` or `Random Forest`
- `LDA` or `Linear Discriminant Analysis`
- `LogReg` or `Logistic Regression`
- `SVM` or `Support Vector Machine`
- `Trans` or `Transformer`

### `model_type`

This must be a two-item list:

```yaml
model_type:
  - Complete
  - Explicit
```

The first item is the AFE filter and must be `Complete` or `noAFE`. The second item is the feature set and must be `Explicit` or `Implicit`.

The folder name for results is derived from this value. For example, `Complete` + `Explicit` writes to `Results/Sensor_Ablation/Complete_Explicit/`.

### `shared_data_parameters`

These values are passed into the data pipeline and control how the shared dataset is prepared.

- `subject_to_analyze`: subject ID or `null` for all subjects.
- `trial_to_analyze`: trial ID or `null` for all trials.
- `analysis_type`: analysis mode selector used by the pipeline.
- `remove_NaN_trials`: whether to discard trials containing NaNs.
- `impute_file_name`: filename used when saving or loading imputed data.
- `save_impute`: save imputed data to disk.
- `load_impute`: load imputed data from disk if present.
- `should_impute`: run KNN imputation.
- `output_feature_dtype`: output dtype such as `float32` or `float64`.

Current defaults in the root config:

```yaml
shared_data_parameters:
  subject_to_analyze: null
  trial_to_analyze: null
  analysis_type: 2
  remove_NaN_trials: true
  impute_file_name: imputed_data.pkl
  save_impute: false
  load_impute: false
  should_impute: true
  output_feature_dtype: float32
```

### `advanced_data_parameters`

These affect the advanced-classifier workflow.

- `num_splits`: number of cross-validation folds.
- `kfold_ID`: fold index used by the pipeline.
- `n_neighbors`: neighbor count for KNN imputation.
- `baseline_window`: baseline window in seconds.

### `traditional_data_parameters`

These settings are used by the traditional pipeline.

- `backstep`: look-back window in seconds.
- `data_rate`: sampling rate in Hz.
- `offset`: time offset in seconds.
- `time_start`: starting time in seconds.

### `sensor_ablation.training`

Enable this section to run stream ablation experiments and generate new cached F1 results.

```yaml
sensor_ablation:
  training:
    enabled: true
    streams:
      - [EEG]
      - [Pupil]
      - [EEG, Pupil]
```

Each item in `streams` is a group of stream labels. The code validates stream names against the supported label set, so typos are rejected early.

The current root config enables a larger training sweep over ECG, HR, BR, Temperature, EEG, Pupil, Centrifuge, and Participant combinations.

### `sensor_ablation.review`

Enable this section to reload previously saved F1 arrays and replot them without rerunning model training.

```yaml
sensor_ablation:
  review:
    enabled: true
    models:
      - KNN
      - RF
    stream_group:
      - EEG
    sort_streams_by_median: false
```

Rules enforced by the parser:

- `models` must be a non-empty list when review is enabled.
- `stream_group` must be a non-empty list when review is enabled and `sort_streams_by_median` is `false`.
- The stream group is matched as a set, so order does not matter when loading cached results.

When `sort_streams_by_median: true`, the review path mirrors legacy preference 11 behavior:

- Loads all cached streams for the selected models.
- Ranks streams by combined median F1 score across selected models.
- Uses the matrix-style violin plot layout.
- Applies hard-coded label replacements (not configurable in YAML):
  - `ECG-HR-BR-Temperature` -> `Equivital`
  - `Participant` -> `Demographics`
  - `Centrifuge` -> `G Force`

In the current root config, review is disabled.

### `feature_space_review` (Legacy preference 9)

Enable this section to inspect overlap of selected features across classifiers.

```yaml
feature_space_review:
  enabled: true
  models:
    - KNN
    - EGB
    - RF
```

This is the YAML-driven equivalent of legacy preference 9.

### `hyperparameter_save` (Legacy preference 10)

Enable this section to save median-fold hyperparameters and selected features to JSON for configured models.

```yaml
hyperparameter_save:
  enabled: true
  models:
    - KNN
```

This is the YAML-driven equivalent of legacy preference 10.

### `cross_validation` (New unified CV runner)

Enable this section to run cross-validation with automatic model-type detection and results aggregation.

```yaml
cross_validation:
  enabled: true
  num_splits: 5
  kfold_ID: 0
  classifiers:
    - KNN
    - RF
    - EGB
  save_results_folder: Results/CrossValidation
  random_seed: 42
  class_weight: balanced
  support_deep_learning: false
  impute_handling: {}
```

Key features:
- **Automatic model detection**: Detects traditional (legacy `classify_traditional` contract), advanced (modern `train/evaluate` interface), or DL-adapted models.
- **Unified fold management**: Uses `stratified_kfold_split` for deterministic, reproducible cross-validation splits.
- **Legacy compatibility**: Preserves traditional model metrics filenames (`metrics_fold_<i>.pkl`) and advanced model nested structure (`fold_<i>/metrics.pkl`).
- **Deep-learning support**: Via an adapter pattern (see `src/models/dl_adapter.py`); users can subclass `DLModelAdapter` to integrate PyTorch, TensorFlow, or other frameworks without heavy core dependencies.
- **Result aggregation**: Computes mean and std of metrics across folds; saves to `summary.json` and per-fold `metrics.pkl`.

Example CV run:
```yaml
cross_validation:
  enabled: true
  num_splits: 10
  classifiers: [KNN, RF, EGB, LogReg]
  random_seed: 42
```

### Preference mapping summary

- Legacy preference 9 -> `feature_space_review.enabled: true`
- Legacy preference 10 -> `hyperparameter_save.enabled: true`
- Legacy preference 11 -> `sensor_ablation.review.enabled: true` with `sort_streams_by_median: true`

## Testing

The repository includes pytest coverage for config parsing, the sensor ablation flow, and pipeline parity checks.

Run the test suite with:

```bash
pytest
```

## Notes

- `GLOC_experiment_config.yaml` must be valid YAML, not JSON.
- `models`, `model_type`, `random_seed`, and `data_path` are required.
- The main entry point installs cuML acceleration after config parsing, so the environment needs the RAPIDS dependencies from `environment.yml`.
