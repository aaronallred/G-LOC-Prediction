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

## YAML Configuration

The GLOC pipeline is controlled entirely by the `GLOC_experiment_config.yaml` configuration file. The config uses a **mode-based architecture** where only sections with `enabled: true` execute.

### Configuration Structure

The config file has the following top-level sections:

- **Root parameters**: `data_path` (required by all modes)
- **Shared parameters**: Data preprocessing settings used by all modes
- **Advanced parameters**: KNN imputation settings for advanced pipeline
- **Traditional parameters**: Timing and rate settings for traditional pipeline
- **Mode sections**: Each execution mode has its own enabled/disabled section

### Root Parameters

#### `data_path`

**Purpose**: Path to the directory containing input CSV files and datasets.

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

**Purpose**: Filter data to a specific subject ID.

**Available inputs**: 
- Subject ID (integer or string) to analyze only that subject
- `null` to analyze all subjects

**Example**:
```yaml
subject_to_analyze: null  # Analyze all subjects
```

#### `trial_to_analyze`

**Purpose**: Filter data to a specific trial ID.

**Available inputs**:
- Trial ID to analyze only that trial
- `null` to analyze all trials

**Example**:
```yaml
trial_to_analyze: null  # Analyze all trials
```

#### `analysis_type`

**Purpose**: Selects the analysis mode used by the data pipeline.

**Available inputs**: Integer analysis mode selector (e.g., `2` for default mode).

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

**Purpose**: Filename for saving/loading imputed data cache.

**Available inputs**: Any valid filename string.

**Example**:
```yaml
impute_file_name: imputed_data.pkl
```

#### `save_impute`

**Purpose**: Whether to save imputed data to disk.

**Available inputs**: `true` or `false`

**Example**:
```yaml
save_impute: false  # Don't save imputation cache
```

#### `load_impute`

**Purpose**: Whether to load imputed data from disk if a cache exists.

**Available inputs**: `true` or `false`

**Example**:
```yaml
load_impute: false  # Don't load imputation cache
```

#### `should_impute`

**Purpose**: Whether to perform KNN imputation on missing values.

**Available inputs**: `true` or `false`

**Example**:
```yaml
should_impute: true  # Perform KNN imputation
```

#### `output_feature_dtype`

**Purpose**: NumPy dtype for output feature arrays.

**Available inputs**: `float32`, `float64`, or other valid NumPy dtype strings.

**Example**:
```yaml
output_feature_dtype: float32  # Use 32-bit floating point
```

### Advanced Data Parameters

These parameters control the advanced (cuML/RAPIDS) pipeline behavior. Configure under `advanced_data_parameters`:

#### `n_neighbors`

**Purpose**: Number of nearest neighbors to use for KNN imputation.

**Available inputs**: Positive integer.

**Example**:
```yaml
n_neighbors: 4
```

#### `baseline_window`

**Purpose**: Baseline window duration in seconds for feature extraction.

**Available inputs**: Positive float (seconds).

**Example**:
```yaml
baseline_window: 32.5
```

### Traditional Data Parameters

These parameters control timing and sampling for the traditional pipeline. Configure under `traditional_data_parameters`:

#### `backstep`

**Purpose**: Look-back window in seconds for traditional feature extraction.

**Available inputs**: Non-negative float (seconds).

**Example**:
```yaml
backstep: 0
```

#### `data_rate`

**Purpose**: Sampling rate in Hz (samples per second).

**Available inputs**: Positive integer.

**Example**:
```yaml
data_rate: 25  # 25 Hz sampling
```

#### `offset`

**Purpose**: Time offset in seconds for data alignment.

**Available inputs**: Non-negative float (seconds).

**Example**:
```yaml
offset: 0
```

#### `time_start`

**Purpose**: Starting time in seconds for analysis window.

**Available inputs**: Non-negative float (seconds).

**Example**:
```yaml
time_start: 0
```

### Mode: Sensor Ablation Training

Enable and configure stream ablation experiments with sensor ablation training mode.

**Purpose**: Train models on different combinations of sensor streams to evaluate stream importance via F1 score changes.

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

#### `models` (Mode-specific)

**Purpose**: Models to train during sensor ablation.

**Available inputs**: List of model aliases:
- `EGB` or `Extreme Gradient Boosting`
- `KNN` or `K Nearest Neighbors`
- `RF` or `Random Forest`
- `LDA` or `Linear Discriminant Analysis`
- `LogReg` or `Logistic Regression`
- `SVM` or `Support Vector Machine`
- `Trans` or `Transformer`

**Example**:
```yaml
models:
  - EGB
  - KNN
  - RF
```

#### `model_type` (Mode-specific)

**Purpose**: Feature extraction and selection configuration for this mode.

**Available inputs**: Two-item list `[afe_filter, feature_set]`
- First item (AFE filter): `Complete` or `noAFE`
- Second item (Feature set): `Explicit` or `Implicit`

**Example**:
```yaml
model_type:
  - Complete
  - Explicit
```

This determines the results folder name (e.g., `Results/Sensor_Ablation/Complete_Explicit/`).

#### `random_seed` (Mode-specific)

**Purpose**: Random seed for reproducible k-fold splits and model training.

**Available inputs**: Positive integer.

**Example**:
```yaml
random_seed: 42
```

#### `num_splits`

**Purpose**: Number of folds for k-fold cross-validation.

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
  - [ECG, HR, BR, Temperature]
  - [EEG]
  - [Pupil]
  - [ECG, HR, BR, Temperature, EEG]
```

**Constraints**: Each stream name is validated against the supported label set. Typos are rejected at runtime.

#### `save_results_folder`

**Purpose**: Directory where sensor ablation F1 scores and plots are saved during training.

**Available inputs**: Relative or absolute path to directory.

**Example**:
```yaml
save_results_folder: Results/Sensor_Ablation
```

**Default**: `Results/Sensor_Ablation` (if not specified)

**Note**: The `model_type` subfolder is automatically appended to this path (e.g., `Results/Sensor_Ablation/Complete_Explicit/`). Stream-specific F1 scores are saved as pickle files organized by model name within this folder.

### Mode: Sensor Ablation Review

Re-plot previously saved sensor ablation F1 results without retraining.

**Purpose**: Visualize and re-analyze cached results from sensor ablation training runs.

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

#### `models` (Mode-specific)

**Purpose**: Models whose cached results to load and visualize.

**Available inputs**: List of model aliases (same as training).

**Example**:
```yaml
models:
  - KNN
  - RF
```

**Constraints**: Must be non-empty when review is enabled. Must match model results saved during training.

#### `model_type` (Mode-specific)

**Purpose**: Feature extraction configuration for locating cached results.

**Available inputs**: Two-item list `[afe_filter, feature_set]` (same format as training).

**Example**:
```yaml
model_type:
  - Complete
  - Explicit
```

#### `stream_group`

**Purpose**: Stream combination to filter and display when `sort_streams_by_median: false`.

**Available inputs**: List of stream names to match.

**Example**:
```yaml
stream_group:
  - EEG
```

**Constraints**: Must be non-empty when `sort_streams_by_median: false`.

#### `sort_streams_by_median`

**Purpose**: Whether to automatically rank streams by combined median F1 score (legacy preference 11 behavior).

**Available inputs**: `true` or `false`

**Example**:
```yaml
sort_streams_by_median: false
```

**Behavior**:
- When `false`: Displays only the streams matching `stream_group` (set-based matching, order-independent)
- When `true`: Loads all cached streams for selected models, ranks by median F1, applies label replacements:
  - `ECG-HR-BR-Temperature` → `Equivital`
  - `Participant` → `Demographics`
  - `Centrifuge` → `G Force`

#### `save_results_folder`

**Purpose**: Directory where sensor ablation results are loaded from during review.

**Available inputs**: Relative or absolute path to directory.

**Example**:
```yaml
save_results_folder: Results/Sensor_Ablation
```

**Default**: `Results/Sensor_Ablation` (if not specified)

**Note**: Must point to the same location used by sensor ablation training so that the plots can be found and reloaded. The `model_type` subfolder is automatically appended to this path. Review will fail if the specified directory does not contain results from a prior training run.

### Mode: Feature Space Review

Inspect overlap of selected features across trained models.

**Purpose**: Analyze which features each model selected and identify shared vs. unique features (legacy preference 9 equivalent).

**Section**: `feature_space_review`

#### `enabled`

**Purpose**: Whether to run feature space overlap analysis.

**Available inputs**: `true` or `false`

**Example**:
```yaml
feature_space_review:
  enabled: true
```

#### `models` (Mode-specific)

**Purpose**: Models whose feature selections to compare.

**Available inputs**: List of model aliases (2-4 models recommended for visualization).

**Example**:
```yaml
models:
  - KNN
  - EGB
  - RF
```

**Constraints**: Must be non-empty when enabled. Visualizations support up to ~4+ models (Venn diagrams for ≤3, UpSet plots for ≥4).

#### `model_type` (Mode-specific)

**Purpose**: Feature extraction configuration for locating saved model hyperparameters.

**Available inputs**: Two-item list `[afe_filter, feature_set]`.

**Example**:
```yaml
model_type:
  - Complete
  - Explicit
```

### Mode: Cross-Validation

Run systematic k-fold cross-validation with automatic model-type detection and metric aggregation.

**Purpose**: Evaluate model generalization, aggregate metrics across folds, and extract median-fold hyperparameters.

**Section**: `cross_validation`

#### `enabled`

**Purpose**: Whether to run cross-validation mode.

**Available inputs**: `true` or `false`

**Example**:
```yaml
cross_validation:
  enabled: true
```

#### `models` (Mode-specific)

**Purpose**: Models to cross-validate.

**Available inputs**: List of model aliases.

**Example**:
```yaml
models:
  - KNN
  - RF
```

**Constraints**: Must be non-empty when enabled.

#### `model_type` (Mode-specific)

**Purpose**: Feature extraction configuration for this CV run.

**Available inputs**: Two-item list `[afe_filter, feature_set]`.

**Example**:
```yaml
model_type: [Complete, Explicit]
```

#### `random_seed` (Mode-specific)

**Purpose**: Seed for reproducible k-fold splits.

**Available inputs**: Positive integer.

**Example**:
```yaml
random_seed: 42
```

#### `num_splits`

**Purpose**: Number of folds for k-fold cross-validation.

**Available inputs**: Positive integer (typically 5, 10, or higher).

**Example**:
```yaml
num_splits: 5
```

#### `save_results_folder`

**Purpose**: Base directory for saving cross-validation results.

**Available inputs**: Valid file system path.

**Example**:
```yaml
save_results_folder: Results/CrossValidation
```

**Output structure**: Results are saved to `{save_results_folder}/{model_name}/` with metrics and optional hyperparameters.

#### `class_weight`

**Purpose**: How to handle class imbalance in model training.

**Available inputs**:
- `null` - treat all classes equally
- `balanced` - adjust weights inversely to class frequency
- Custom weights (framework-dependent)

**Example**:
```yaml
class_weight: null
```

#### `support_deep_learning`

**Purpose**: Whether to enable deep-learning model support via adapter pattern.

**Available inputs**: `true` or `false`

**Example**:
```yaml
support_deep_learning: false
```

**Note**: Requires custom `DLModelAdapter` subclass for non-traditional frameworks (PyTorch, TensorFlow, etc.).

#### `save_median_hyperparameters`

**Purpose**: Whether to extract and save the median-fold model's hyperparameters.

**Available inputs**: `true` or `false`

**Example**:
```yaml
save_median_hyperparameters: true
```

**Behavior**:
- When `true` (default): Automatically identifies the fold with median F1 score and extracts its trained model's hyperparameters to `median_hyperparameters.json`
- When `false`: Only aggregates metrics

#### `impute_handling`

**Purpose**: Framework-specific imputation configuration (typically empty for standard usage).

**Available inputs**: Dictionary (key-value pairs) or empty object.

**Example**:
```yaml
impute_handling: {}  # Use default imputation
```

### Complete Example

Here is a complete minimal configuration:

```yaml
data_path: /home/gloc/G-LOC-Prediction/data

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

advanced_data_parameters:
  n_neighbors: 4
  baseline_window: 32.5

traditional_data_parameters:
  backstep: 0
  data_rate: 25
  offset: 0
  time_start: 0

sensor_ablation:
  training:
    enabled: false
    models:
      - EGB
      - KNN
      - RF
    model_type:
      - Complete
      - Explicit
    random_seed: 42
    num_splits: 10
    streams:
      - [ECG, HR, BR, Temperature]
      - [EEG]
      - [Pupil]

  review:
    enabled: false
    models:
      - KNN
      - RF
    model_type:
      - Complete
      - Explicit
    stream_group:
      - EEG
    sort_streams_by_median: false

feature_space_review:
  enabled: false
  models:
    - KNN
    - EGB
    - RF
  model_type:
    - Complete
    - Explicit

cross_validation:
  enabled: true
  models:
    - KNN
    - RF
  model_type: [Complete, Explicit]
  random_seed: 42
  num_splits: 5
  save_results_folder: Results/CrossValidation
  class_weight: null
  support_deep_learning: false
  save_median_hyperparameters: true
  impute_handling: {}
```

### Mode Execution

Only modes with `enabled: true` will execute. When you run:

```bash
python -m src.main --config GLOC_experiment_config.yaml
```

The pipeline checks each mode's `enabled` flag and runs only those with `enabled: true`. This allows you to configure multiple modes but selectively enable/disable them without editing the entire config file.

### Notes

- The config file must be valid YAML (not JSON).
- `data_path` is required by all modes.
- Mode-specific parameters (`models`, `model_type`, `random_seed`) are only used by their corresponding modes and must be configured within each mode's section.
- The pipeline installs cuML acceleration after config parsing, so RAPIDS dependencies from `environment.yml` are required for GPU acceleration.

## Testing

The repository includes pytest coverage for config parsing, the sensor ablation flow, and pipeline parity checks.

Run the test suite with:

```bash
pytest
```
