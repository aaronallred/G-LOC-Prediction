# G-LOC-Prediction

This repository contains the current G-LOC prediction pipeline for normal, temporal, and sensor ablation studies.

## What lives here

- `src/main.py` is the main entry point for each of the different pipeline modes.
- `configs/' is a directory containing YAML configuration files for each modes.
    - `configs/master.yaml` is the master configuration file that has configurations for all modes and data settings.
- `data/` contains the input CSV files and supporting datasets used by the pipeline. Any folder can serve as a data
  source.
- `Results/` is where cross-validation results and sensor ablation results are stored. Any folder can serve as a results
  destination.

## Setting up the Environment

The project is configured for the using Conda to setup the environment. That environment includes the cuML (
GPU-enabled sklearn model training) stack used by the current pipeline.

I recommend using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main). Conda helps manage package
dependencies and versions
since some developers may make changes that break other packages.

Which environment YAML file to use depends on your system.

### CPU-Only Environment

This section applies if you do not have a NVIDIA GPU on your system. With this setup, model training will be very slow
but still possible.

I wouldn't recommend performing any model training here, and instead to use a system with a NVIDIA
GPU to perform model training to utilize the GPU packages. However, this environment can be used for general
development.

Run the following to setup the environment:

```bash
conda env create -f environment-cpu.yaml
conda activate gloc
```

### NVIDIA GPU on Windows

This section applies if you have a NVIDIA GPU, but your development environment is primarily on Windows. The developers
of the cuML package have restricted the download to Linux environments, so it can't be fully utilized yet. Regardless,
this still enables PyTorch GPU acceleration for training deep learning models faster.

First, open up a terminal (like Windows PowerShell) and run the command:

```bash
nvidia-smi
```

and look at the top left of the output. It should say something about "CUDA Version".

Then, go into the `environment-gpu-windows.yaml` file and look for the line

```yaml
      - --extra-index-url https://download.pytorch.org/whl/cu132
```

If the CUDA version is anything before 13.0, then change the "132" at the end to "126".\
If the CUDA version is either 13.0 or 13.1, then remove this line.\
If the CUDA version is 13.2+, then leave this line as-is.

Make sure to save the file and run the following to setup the environment:

```bash
conda env create -f environment-gpu-windows.yaml
conda activate gloc
```

### NVIDIA GPU on Linux (or WSL) (**Recommended**)

This section applied if you have both a NVIDIA GPU and your development environment is on Linux. For Windows users, you
can use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).

First, open up a terminal (like Windows PowerShell) and run the command:

```bash
nvidia-smi
```

and look at the top left of the output. It should say something about "CUDA Version".

Then, go into the `environment-gpu-linux.yaml` file and look at the line:

```yaml
  - cuda-version>=13.0,<=13.2
```

If the CUDA version is less than 12.2, then look into updating your NVIDIA drivers.\
If the CUDA version is between 12.2 and 12.9, then replace this line with the following: `cuda-version>=12.2,<=12.9`\
If the CUDA version is 13.0+, then leave this line as-is.

```yaml
      - --extra-index-url https://download.pytorch.org/whl/cu132
```

If the CUDA version is less than 12.6, then look into updating your NVIDIA drivers.\
If the CUDA version is between 12.6 and 12.9, then change the "132" at the end to "126".\
If the CUDA version is either 13.0 or 13.1, then remove this line.\
If the CUDA version is 13.2+, then leave this line as-is.

Make sure to save the file and run the following to setup the environment:

```bash
conda env create -f environment-gpu-linux.yaml
conda activate gloc
```

## Running the Pipeline

Run the main module from the repository root (G-LOC-Prediction/) with a specific YAML configuration file:

```bash
python -m src.main --config /configs/your_config.yaml
```

## YAML Configuration

The GLOC pipeline is controlled entirely by the YAML configuration files. The config uses a **mode-based architecture**
where only sections with `enabled: true` execute.

### Configuration Structure

The config file has the following top-level sections:

- **Root parameters**: `data_path` (required by all modes)
- **Shared parameters**: Data preprocessing settings used by all modes for the data pipeline
- **Advanced parameters**: KNN imputation settings for advanced data pipeline
- **Traditional parameters**: Timing and rate settings for traditional data pipeline
- **Mode sections**: Each execution mode has its own enabled/disabled section

### Root Parameters

#### `data_path`

**Purpose**: Absolute path to the directory containing input CSV files and datasets.

**Available inputs**: Any valid file system path.

**Example**:

```yaml
data_path: /home/gloc/G-LOC-Prediction/data
```

**Constraints**:

- Required by all modes
- Directory must exist and contain expected dataset files

### Shared Data Parameters

These parameters control data preprocessing and are used by all modes. Configure under `shared_data_parameters`:

#### `subject_to_analyze`

**Purpose**: Filter data to a specific subject ID (1-13).

**Available inputs**:

- Subject ID (integer or string) to analyze only that subject.
- `null` to analyze all subjects

**Example**:

```yaml
subject_to_analyze: null  # Analyze all subjects
```

#### `trial_to_analyze`

**Purpose**: Filter data to a specific trial ID (1-6).

**Available inputs**:

- Trial ID to analyze only that trial
- `null` to analyze all trials

**Example**:

```yaml
trial_to_analyze: null  # Analyze all trials
```

#### `analysis_type`

**Purpose**: Selects the analysis mode used by the data pipeline. Complements the `subject_to_analyze` and
`trial_to_analyze` filters.

**Available inputs**: Integer from 0 - 2 (e.g., `2` for default mode).

0: Analyze one trial from one subject (`subject_to_analyze` and `trial_to_analyze` must be set)\
1: Analyze all trials from one subject (`subject_to_analyze` must be set)\
2: Analyze all trials from all subjects

**Example**:

```yaml
analysis_type: 2
```

#### `remove_NaN_trials`

**Purpose**: Whether to discard trials containing NaN values before processing.

**Available inputs**: `true` or `false`

**Example**:

```yaml
remove_NaN_trials: true  # Remove trials with NaNs
```

#### `impute_file_name`

**Purpose**: Filename for saving/loading imputed data from previous runs. Be careful with this since using a different
model, model type, data parameters, etc. will result in different imputed data and loading in an incorrect imputed data
file may result in data leakage.

**Available inputs**: Any valid filename string.

**Example**:

```yaml
impute_file_name: imputed_data.pkl
```

#### `save_impute`

**Purpose**: Whether to save imputed data after running the KNN imputation.

**Available inputs**: `true` or `false`

**Example**:

```yaml
save_impute: false  # Don't save imputation cache
```

#### `load_impute`

**Purpose**: Whether to load imputed data from a previous run, but there must be a saved imputed data file from the
previous run.

**Available inputs**: `true` or `false`

**Example**:

```yaml
load_impute: false  # Don't load imputation cache
```

#### `impute_phase`

**Purpose**: Control when imputation is performed.

**Available inputs**:

`none`: Don't perform imputation\
`pre_feature`: Perform imputation before feature extraction\
`post_feature_remove_rows`: Perform imputation after feature extraction and remove rows with NaN values\
`post_feature_knn`: Perform imputation after feature extraction

**Example**:

```yaml
impute_phase: pre_feature  # Perform KNN imputation on raw data before feature extraction
```

#### `output_feature_dtype`

**Purpose**: NumPy dtype for output feature arrays.

**Available inputs**: `float32`, `float64`, or other valid NumPy dtype strings.

**Example**:

```yaml
output_feature_dtype: float32  # Use 32-bit floating point
```

### Advanced Data Parameters

These parameters control the advanced pipeline behavior for how the data should be processed for the deep learners.
Configure under `advanced_data_parameters`:

#### `n_neighbors`

**Purpose**: Number of nearest neighbors to use for KNN imputation.

**Available inputs**: Positive integer.

**Example**:

```yaml
n_neighbors: 4
```

#### `baseline_window`

**Purpose**: Baseline window duration in seconds for feature extraction. This probably doesn't need to change.

**Available inputs**: Positive float (seconds).

**Example**:

```yaml
baseline_window: 32.5
```

#### `horizon`

**Purpose**: Temporal forecasting horizon in samples. Shifts GLOC labels earlier so the model predicts GLOC events
`horizon` samples into the future (0 = no shift, baseline). Applied per-trial after the train/test split to avoid data
leakage.

**Available inputs**: Non-negative integer (samples).

**Example**:

```yaml
horizon: 0  # No forecasting shift
```

### Traditional Data Parameters

These parameters control timing and sampling for the traditional pipeline. Configure under
`traditional_data_parameters`:

#### `backstep`

**Purpose**: Look-back window in seconds for traditional feature extraction. Probably doesn't need to change.

**Available inputs**: Non-negative float (seconds).

**Example**:

```yaml
backstep: 0
```

#### `data_rate`

**Purpose**: Sampling rate in Hz (samples per second). Probably doesn't need to change.

**Available inputs**: Positive integer.

**Example**:

```yaml
data_rate: 25  # 25 Hz sampling
```

#### `offset`

**Purpose**: Time offset in seconds for data alignment. This is the parameter to change to perform the temporal
experiments (to offset the GLOC label) instead of the standard experiments.

**Available inputs**: Non-negative float (seconds).

**Example**:

```yaml
offset: 0
```

#### `time_start`

**Purpose**: Starting time in seconds for analysis window. Probably doesn't need to change.

**Available inputs**: Non-negative float (seconds).

**Example**:

```yaml
time_start: 0
```

### Mode: Cross-Validation

Run systematic k-fold cross-validation with automatic model-type detection and metric aggregation.

**Purpose**: Perform hyperparameter optimization for each fold and extract median-fold hyperparameters over all folds.

**Section**: `cross_validation`

#### `enabled`

**Purpose**: Whether to run cross-validation mode.

**Available inputs**: `true` or `false`

**Example**:

```yaml
enabled: true
```

#### `models`

**Purpose**: Models to cross-validate.

**Available inputs**: List of model aliases. Available models:

Traditional (sklearn):

- `EGB` (Extreme Gradient Boosting)
- `KNN` (K Nearest Neighbors)
- `RF` (Random Forest)
- `LDA` (Linear Discriminant Analysis)
- `LogReg` (Logistic Regression)
- `SVM` (Support Vector Machine)

Advanced (PyTorch):

- `LSTM` (Long Short-Term Memory)
- `TCN` (Temporal Convolutional Network)
- `Trans` (Transformer)
- `LogRegTS` (Time-Series Logistic Regression)
- `NAM` (Neural Additive Model)

**Example**:

```yaml
models: [ KNN, RF ]
```

**Constraints**: Must be non-empty when enabled.

#### `model_type`

**Purpose**: Feature extraction configuration for this CV run.

**Available inputs**: Two-item list `[afe_filter, feature_set]`.\
`afe_filter` can be "Complete" (include rows with or without AFE) or "noAFE" (include rows with only no AFE).\
`feature_set` can be "Explicit" (all streams) or "Implicit" (only passive and unprocessed sensor streams).

**Example**:

```yaml
model_type: [ Complete, Explicit ]
```

#### `random_seed`

**Purpose**: Seed for reproducible k-fold splits.

**Available inputs**: Positive integer.

**Example**:

```yaml
random_seed: 42
```

#### `num_splits`

**Purpose**: Number of folds for k-fold cross-validation. Probably shouldn't change since all of our studies have used
10-fold CV.

**Available inputs**: Positive integer.

**Example**:

```yaml
num_splits: 10
```

#### `save_results_folder`

**Purpose**: Base directory for saving cross-validation results.

**Available inputs**: Valid file system path.

**Example**:

```yaml
save_results_folder: Results/Cross_Validation
```

**Output structure**: Results are saved to `{save_results_folder}/{model_type}/{model_name}/` with metrics and
hyperparameters.

#### `class_weight`

**Purpose**: How to handle class imbalance in model training.

**Available inputs**:

- `null` - Don't adjust class weights
- `balanced` - Adjust weights inversely to class frequency so that the class label representation are more balanced.

**Example**:

```yaml
class_weight: null
```

#### `advanced_hpo`

**Purpose**: Hyperparameter optimization settings for advanced (PyTorch) models. Required when any advanced model is in
the `models` list. Ignored for traditional-only runs.

**Available inputs**: Sub-section with the following fields:

##### `use_sampler`

**Purpose**: Whether to use a weighted sampler to address class imbalance during Optuna trial training and final model
training.

**Available inputs**: `true` or `false`

**Example**:

```yaml
use_sampler: true
```

##### `final_early_stop`

**Purpose**: Whether the final trained model (after HPO) should use early stopping on a held-out validation split. When
`true`, 20% of the training data is held out for validation and training stops when the validation metric stops
improving. When `false`, the model trains for a fixed number of epochs using all training data.

**Available inputs**: `true` or `false`

**Example**:

```yaml
final_early_stop: false
```

##### `objective_var`

**Purpose**: Optimization metric used by Optuna to evaluate each trial. Case-insensitive.

**Available inputs**: `F1` or `Acc`

**Example**:

```yaml
objective_var: F1
```

##### `trials`

**Purpose**: Number of Optuna HPO trials to run per cross-validation fold. Each trial samples a hyperparameter
configuration, trains a candidate model, and evaluates it on a validation split. Set to `0` to disable HPO entirely (
models train with default hyperparameters).

**Available inputs**: Non-negative integer.

**Example**:

```yaml
trials: 100
```

**Constraints**:

- Required when any advanced model is in the `models` list. A missing `advanced_hpo` section will raise a `KeyError` at
  runtime.
- Optuna-level parameters (sampler type, pruner settings, timeout) are hardcoded defaults and not exposed in the YAML
  config.

### Mode: Sensor Ablation Training

Enable and configure stream ablation experiments with sensor ablation training mode.

**Purpose**: Train models on different combinations of sensor streams and save their performance.

**Section**: `sensor_ablation.training`

#### `enabled`

**Purpose**: Whether to run sensor ablation training.

**Available inputs**: `true` or `false`

**Example**:

```yaml
sensor_ablation:
  training:
    enabled: true
```

#### `save_results_folder`

**Purpose**: Directory where sensor ablation F1 scores are saved during training.

**Available inputs**: Relative or absolute path to directory.

**Example**:

```yaml
save_results_folder: Results/Sensor_Ablation
```

#### `save_models_folder`

**Purpose**: Directory where trained fold models are saved for later use, including SHAP analysis.

**Available inputs**: Relative or absolute path to a directory.

**Example**:

```yaml
save_models_folder: ModelSave/Sensor_Ablation
```

**Expected output structure**:

```text
{save_models_folder}/{model_type}/{model_name}/{stream_group}/fold_0.pkl
{save_models_folder}/{model_type}/{model_name}/{stream_group}/fold_1.pkl
...
```

#### `median_hyperparameters_folders`

**Purpose**: Directory where median hyperparameters are saved from cross validation.

**Available inputs**: Relative or absolute path to directory.

**Example**:

```yaml
median_hyperparameters_folders: Results/Cross_Validation
```

**Note**: The `model_type` subfolder is automatically appended to this path (e.g.,
`Results/Sensor_Ablation/Complete_Explicit/`). Stream-specific F1 scores are saved as JSON files organized by model name
within this folder.

#### `models`

**Purpose**: Models to train during sensor ablation.

**Available inputs**: List of model aliases:

- `EGB` (Extreme Gradient Boosting)
- `KNN` (K Nearest Neighbors)
- `RF` (Random Forest)
- `LDA` (Linear Discriminant Analysis)
- `LogReg` (Logistic Regression)
- `SVM` (Support Vector Machine)

**Example**:

```yaml
models: [ KNN, RF ]
```

#### `model_type`

**Purpose**: Feature extraction and selection configuration for this mode.

**Available inputs**: Two-item list `[afe_filter, feature_set]`
`afe_filter` can be "Complete" (include rows with or without AFE) or "noAFE" (include rows with only no AFE).\
`feature_set` can be "Explicit" (all streams) or "Implicit" (only passive and unprocessed sensor streams).

**Example**:

```yaml
model_type: [ Complete, Explicit ]
```

#### `random_seed`

**Purpose**: Random seed for reproducible k-fold splits and model training.

**Available inputs**: Positive integer.

**Example**:

```yaml
random_seed: 42
```

#### `num_splits`

**Purpose**: Number of folds for k-fold cross-validation. Probably shouldn't change since all of our studies have used
10-fold CV.

**Available inputs**: Positive integer.

**Example**:

```yaml
num_splits: 10  # 10-fold cross-validation
```

#### `streams`

**Purpose**: List of sensor stream combinations to evaluate.

**Available inputs**: List of lists, where each inner list contains stream names:

- Single streams: `[ECG]`, `[EEG]`, `[Pupil]`, `[Centrifuge]`, `[Participant]`, `[HR]`, `[BR]`, `[Temperature]`
- Combined streams: `[ECG, HR, BR, Temperature]`, `[EEG, Pupil]`, etc.

**Example**:

```yaml
streams:
  - [ ECG, HR, BR, Temperature ]
  - [ EEG ]
  - [ Pupil ]
  - [ ECG, HR, BR, Temperature, EEG ]
```

**Constraints**: Each stream name is validated against the supported label set. Typos are rejected at runtime.

### Mode: Sensor Ablation Review

Plot previously saved sensor ablation F1 results without retraining.

**Purpose**: Visualize results from sensor ablation training runs. The following sensor stream combinations are renamed:

- `ECG-HR-BR-Temperature` → `Equivital`
- `Participant` → `Demographics`
- `Centrifuge` → `G Force`

**Section**: `sensor_ablation.review`

#### `enabled`

**Purpose**: Whether to reload and replot saved F1 results.

**Available inputs**: `true` or `false`

**Example**:

```yaml
sensor_ablation:
  review:
    enabled: true
```

#### `save_results_folder`

**Purpose**: Directory where sensor ablation results are loaded from during review.

**Available inputs**: Relative or absolute path to directory.

**Example**:

```yaml
save_results_folder: Results/Sensor_Ablation
```

#### `models`

**Purpose**: Models whose cached results to load and visualize.

**Available inputs**: List of model aliases (same as training).

**Example**:

```yaml
models: [ KNN, RF ]
```

**Constraints**: Must be non-empty when review is enabled. Must match model results saved during training.

#### `model_type`

**Purpose**: Feature extraction configuration for locating cached results.

**Available inputs**: Two-item list `[afe_filter, feature_set]` (same format as training).

**Example**:

```yaml
model_type: [ Complete, Explicit ]
```

#### `stream_groups`

**Purpose**: Stream combination to filter and display.

**Available inputs**: List of stream names to match.

**Example**:

```yaml
stream_groups: [ EEG, Pupil ]
```

#### `sort_streams_by_median`

**Purpose**: Whether to automatically sort streams by their median F1 score.

**Available inputs**: `true` or `false`

**Example**:

```yaml
sort_streams_by_median: false
```

**Behavior**:

- When `false`: Displays streams in the order they they are specified in the YAML config file.
- When `true`: Loads saved results for selected sensor streams for selected models then sorts by median F1 score.

**Note**: Must point to the same location used by sensor ablation training so that the plots can be loaded. The
`model_type` subfolder is automatically appended to this path. Review will fail if the specified directory does not
contain results from a prior training run.

### Mode: Feature Space Review

Inspect overlap of selected features across trained models. For traditional classifiers only.

**Purpose**: Analyze which features each model selected and identify shared vs. unique features.

**Section**: `feature_space_review`

#### `enabled`

**Purpose**: Whether to run feature space overlap analysis.

**Available inputs**: `true` or `false`

**Example**:

```yaml
feature_space_review:
  enabled: true
```

#### `models`

**Purpose**: Models whose feature selections to compare.

**Available inputs**: List of model aliases (2-4 models recommended for visualization).

**Example**:

```yaml
models: [ KNN, RF ]
```

**Constraints**: Must be non-empty when enabled. Visualizations support up to ~4+ models (Venn diagrams for ≤3, UpSet
plots for ≥4).

#### `model_type`

**Purpose**: Feature extraction configuration for locating saved model hyperparameters.

**Available inputs**: Two-item list `[afe_filter, feature_set]`.

**Example**:

```yaml
model_type: [ Complete, Explicit ]
```

### Mode: SHAP Analysis

Generate SHAP explanations from saved fold models.

**Purpose**: Load trained fold models from a previous sensor ablation run, recreate the matching data splits, generate
SHAP explanations, save those explanation objects, and optionally create SHAP plots.

**Section**: `shap_analysis`

**Standard use case**: Use this mode after sensor ablation training has already saved fold models.

#### `enabled`

**Purpose**: Whether to run SHAP analysis.

**Available inputs**: `true` or `false`

**Example**:

```yaml
shap_analysis:
  enabled: true
```

#### `plot_saved_only`

**Purpose**: Controls whether SHAP explanations are generated or only loaded for plotting.

**Available inputs**:

- `false` — generate SHAP explanations from saved models, save the explanations, and create plots.
- `true` — skip SHAP generation and only plot previously saved SHAP explanations.

**Example for SHAP generation**:

```yaml
plot_saved_only: false
```

#### `saved_models_folder`

**Purpose**: Directory containing trained fold models from sensor ablation training.

**Available inputs**: Relative or absolute path to a directory.

**Example**:

```yaml
saved_models_folder: ModelSave/Sensor_Ablation
```

**Expected input structure**:

```text
{saved_models_folder}/{model_type}/{model_name}/{stream_group}/fold_0.pkl
{saved_models_folder}/{model_type}/{model_name}/{stream_group}/fold_1.pkl
...
```

#### `save_results_folder`

**Purpose**: Directory where SHAP explanation objects are saved.

**Available inputs**: Relative or absolute path to a directory.

**Example**:

```yaml
save_results_folder: Results/SHAP_Analysis
```

**Expected output structure**:

```text
{save_results_folder}/{model_type}/{model_name}/{stream_group}/fold_0_shap_explanation.pkl
{save_results_folder}/{model_type}/{model_name}/{stream_group}/fold_1_shap_explanation.pkl
...
```

#### `save_plots_folder`

**Purpose**: Directory where SHAP plots are saved.

**Available inputs**: Relative or absolute path to a directory.

**Example**:

```yaml
save_plots_folder: Results/SHAP_Plots
```

#### `model_type`

**Purpose**: Feature extraction configuration used to locate saved models and recreate matching data.

**Available inputs**: Two-item list `[afe_filter, feature_set]`.

**Example**:

```yaml
model_type: !ModelType [ Complete, Explicit ]
```

#### `models`

**Purpose**: Models to explain.

**Available inputs**: List of model aliases with saved fold models.

**Example**:

```yaml
models: [ RF, EGB ]
```

#### `streams`

**Purpose**: Stream groups to explain.

**Available inputs**: List of stream-group lists. These should match the stream groups used when the models were
trained.

**Example**:

```yaml
streams:
  - [ ECG, EEG, Centrifuge, Participant, Pupil ]
```

#### `random_seed`

**Purpose**: Random seed used to recreate the same k-fold splits used during model training.

**Available inputs**: Positive integer.

**Example**:

```yaml
random_seed: 42
```

#### `num_splits`

**Purpose**: Number of folds to recreate and explain.

**Available inputs**: Positive integer.

**Example**:

```yaml
num_splits: 10
```

**Important**: This should match the number of folds used during the original sensor ablation training run.

#### `manual_ablation`

**Purpose**: Controls whether SHAP recreates the data using cached selected features or raw stream-specific features.

**Available inputs**:

- `false` — use cached selected features, matching the standard sensor ablation workflow.
- `true` — use raw stream-specific features.

**Example**:

```yaml
manual_ablation: false
```

**Important**: This should match the setting used when the explained models were trained.

#### `nsamples_train`

**Purpose**: Number of training samples to use for SHAP background/reference data when sampling is needed.

**Available inputs**: Positive integer.

**Example**:

```yaml
nsamples_train: 100
```

#### `nsamples_test`

**Purpose**: Number of test samples to explain when sampling is needed.

**Available inputs**: Positive integer.

**Example**:

```yaml
nsamples_test: 50
```

#### `overwrite`

**Purpose**: Whether to overwrite existing saved SHAP explanation files.

**Available inputs**: `true` or `false`

**Example**:

```yaml
overwrite: false
```

#### `max_display`

**Purpose**: Maximum number of features or feature groups to show in SHAP plots.

**Available inputs**: Positive integer.

**Example**:

```yaml
max_display: 20
```

#### `class_index`

**Purpose**: Class index to plot when the SHAP explanation contains a class dimension.

**Available inputs**: Non-negative integer.

**Example**:

```yaml
class_index: 1
```

For binary classification, `1` usually corresponds to the positive/G-LOC class.

#### `print_vals`

**Purpose**: Whether to print/log SHAP feature values while plotting.

**Available inputs**: `true` or `false`

**Example**:

```yaml
print_vals: true
```

#### Violin plot layout settings

**Purpose**: Control figure size and margins for SHAP violin plots.

**Available inputs**: Positive numeric values.

**Example**:

```yaml
violin_plot_width: 26
violin_plot_height: 10
violin_left_margin: 0.36
violin_right_margin: 0.96
```

#### Example SHAP generation config

```yaml
shap_analysis:
  enabled: true
  plot_saved_only: false

  saved_models_folder: ModelSave/Sensor_Ablation
  save_results_folder: Results/SHAP_Analysis
  save_plots_folder: Results/SHAP_Plots

  model_type: !ModelType [ Complete, Explicit ]
  models: [ RF, EGB ]
  streams:
    - [ ECG, EEG, Centrifuge, Participant, Pupil ]

  random_seed: 42
  num_splits: 10
  manual_ablation: false

  nsamples_train: 100
  nsamples_test: 50
  overwrite: false

  max_display: 20
  class_index: 1
  print_vals: true

  violin_plot_width: 26
  violin_plot_height: 10
  violin_left_margin: 0.36
  violin_right_margin: 0.96
```

---

### Mode: SHAP Plotting

Plot previously saved SHAP explanations without regenerating them.

**Purpose**: Load saved SHAP explanation objects from `save_results_folder` and create plots in `save_plots_folder`.
This is useful when SHAP generation has already been completed and you only want to adjust or regenerate visualizations.

**Section**: `shap_analysis`

**Important**: SHAP plotting uses the same top-level YAML section as SHAP generation. The difference is that
`plot_saved_only` is set to `true`.

#### `enabled`

**Purpose**: Whether to run SHAP plotting.

**Available inputs**: `true` or `false`

**Example**:

```yaml
shap_analysis:
  enabled: true
```

#### `plot_saved_only`

**Purpose**: Skip SHAP generation and only plot saved SHAP explanations.

**Available inputs**: `true`

**Example**:

```yaml
plot_saved_only: true
```

#### `save_results_folder`

**Purpose**: Directory containing saved SHAP explanation objects.

**Available inputs**: Relative or absolute path to a directory.

**Example**:

```yaml
save_results_folder: Results/SHAP_Analysis
```

#### `save_plots_folder`

**Purpose**: Directory where SHAP plots are saved.

**Available inputs**: Relative or absolute path to a directory.

**Example**:

```yaml
save_plots_folder: Results/SHAP_Plots
```

#### `model_type`

**Purpose**: Feature extraction configuration used to locate saved SHAP explanations and name the plot outputs.

**Available inputs**: Two-item list `[afe_filter, feature_set]`.

**Example**:

```yaml
model_type: !ModelType [ Complete, Explicit ]
```

#### `models`

**Purpose**: Models whose saved SHAP explanations should be plotted.

**Available inputs**: List of model aliases.

**Example**:

```yaml
models: [ RF ]
```

#### `streams`

**Purpose**: Stream groups whose saved SHAP explanations should be plotted.

**Available inputs**: List of stream-group lists.

**Example**:

```yaml
streams:
  - [ ECG, EEG, Centrifuge, Participant, Pupil ]
```

#### `num_splits`

**Purpose**: Number of saved fold explanations to load.

**Available inputs**: Positive integer.

**Example**:

```yaml
num_splits: 10
```

#### `plot_scope`

**Purpose**: Controls whether plots are generated per fold, across all folds, or both.

**Available inputs**:

- `individual` — create plots for each fold separately
- `all` — combine saved fold explanations and create one overall plot
- `both` — create both individual-fold plots and combined plots

**Example**:

```yaml
plot_scope: all
```

#### `max_display`

**Purpose**: Maximum number of features or feature groups to show in SHAP plots.

**Available inputs**: Positive integer.

**Example**:

```yaml
max_display: 20
```

#### `class_index`

**Purpose**: Class index to plot when the saved SHAP explanation contains a class dimension.

**Available inputs**: Non-negative integer.

**Example**:

```yaml
class_index: 1
```

For binary classification, `1` usually corresponds to the positive/G-LOC class.

#### `print_vals`

**Purpose**: Whether to print/log SHAP feature values while plotting.

**Available inputs**: `true` or `false`

**Example**:

```yaml
print_vals: true
```

#### Violin plot layout settings

**Purpose**: Control figure size and margins for SHAP violin plots.

**Available inputs**: Positive numeric values.

**Example**:

```yaml
violin_plot_width: 26
violin_plot_height: 10
violin_left_margin: 0.36
violin_right_margin: 0.96
```

#### `grouped_bar_plots`

**Purpose**: Whether to create grouped SHAP bar plots in addition to standard SHAP plots.

**Available inputs**: Sub-section with `enabled: true` or `enabled: false`.

**Example**:

```yaml
grouped_bar_plots:
  enabled: true
```

Grouped SHAP bar plots aggregate individual features into interpretable feature groups such as:

- modalities
- baseline windows
- EEG channels
- EEG bands
- EEG channel-band combinations
- raw versus processed features
- raw versus PSD features

Grouped bar plots can use either summed absolute SHAP values or mean absolute SHAP values if that option is exposed in
the plotting configuration. Summed absolute SHAP values show total contribution by group, while mean absolute SHAP
values normalize by the number of features in each group.

#### Example SHAP plot-only config

```yaml
shap_analysis:
  enabled: true
  plot_saved_only: true

  save_results_folder: Results/SHAP_Analysis
  save_plots_folder: Results/SHAP_Plots

  model_type: !ModelType [ Complete, Explicit ]
  models: [ RF ]
  streams:
    - [ ECG, EEG, Centrifuge, Participant, Pupil ]
  num_splits: 10

  max_display: 20
  class_index: 1
  print_vals: true
  plot_scope: all

  violin_plot_width: 26
  violin_plot_height: 10
  violin_left_margin: 0.36
  violin_right_margin: 0.96

  grouped_bar_plots:
    enabled: true
```

### Complete Example

Here is a complete minimal configuration:

```yaml
data_path: /home/gloc/G-LOC-Prediction/data/

shared_data_parameters:
  subject_to_analyze: null
  trial_to_analyze: null
  analysis_type: 2
  remove_NaN_trials: true
  impute_file_name: imputed_data.pkl
  save_impute: false
  load_impute: false
  impute_phase: pre_feature
  output_feature_dtype: float32
advanced_data_parameters:
  n_neighbors: 4
  baseline_window: 32.5
  horizon: 0
traditional_data_parameters:
  backstep: 0
  data_rate: 25
  offset: 0
  time_start: 0

cross_validation:
  enabled: true
  # Mode-specific parameters
  models: [ KNN, EGB ]
  model_type: !ModelType [ Complete, Explicit ]
  random_seed: 42
  num_splits: 10
  save_results_folder: Results/Cross_Validation
  class_weight: null
  advanced_hpo:
    use_sampler: true
    final_early_stop: false
    objective_var: F1
    trials: 100

sensor_ablation:
  training:
    enabled: true
    save_results_folder: Results/Sensor_Ablation
    median_hyperparameters_folder: Results/Cross_Validation
    # Mode-specific parameters
    models: [ KNN, EGB ]
    model_type: !ModelType [ Complete, Explicit ]
    random_seed: 42
    num_splits: 10
    streams:
      - [ EEG ]
      - [ EEG, Pupil ]
      - [ EEG, Pupil, Participant ]
  review:
    enabled: True
    save_results_folder: Results/Sensor_Ablation
    # Mode-specific parameters for review
    models: [ KNN, EGB ]
    model_type: !ModelType [ Complete, Explicit ]
    stream_groups:
      - [ EEG ]
      - [ EEG, Pupil ]
      - [ EEG, Pupil, Participant ]
    sort_streams_by_median: true

feature_space_review:
  enabled: true
  # Mode-specific parameters
  models: [ KNN, EGB ]
  model_type: !ModelType [ Complete, Explicit ]
  median_hyperparameters_folder: Results/Cross_Validation
```

### Mode Execution

Only modes with `enabled: true` will execute. When you run:

```bash
python -m src.main --config configs/your_config.yaml
```

The pipeline checks each mode's `enabled` flag and runs only those with `enabled: true`. This allows you to configure
multiple modes but selectively enable/disable them without editing the entire config file.

### Notes

- The config file must be valid YAML (not JSON).
- `data_path` is required by all modes.
- Mode-specific parameters (`models`, `model_type`, `random_seed`) are only used by their corresponding modes and must
  be configured within each mode's section.
- The pipeline installs cuML acceleration after config parsing, so RAPIDS dependencies from `environment.yaml` are
  required for GPU acceleration. If no GPU is available or cuML fails to import, then the package will revert to using
  the CPU.
