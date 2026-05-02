# Copilot instructions for G-LOC-Prediction

This file captures repository-specific guidance for Copilot-style assistants.

## Build / run / test
- Create environment: `conda env create -f environment.yml` and `conda activate gloc` (environment.yml contains RAPIDS/cuML).
- Run pipeline: `python -m src.main` (use `--config /path/to/config.yaml` to override root config).
- Run tests: `pytest` (pytest.ini sets PYTHONPATH=src and log settings).
- Run a single test: `pytest tests/<test_file>.py::test_name` or use `-k <expr>`.

> No linter is preconfigured in the repo (no Makefile, tox, or CI linter workflows found). Add project linter/formatter if desired.

## High-level architecture
- Entry point: `src/main.py` — parses YAML config, enables cuML when available, and dispatches one or more "modes".
- Modes: implemented under `src/modes/` and include sensor ablation training/review, feature-space review, hyperparameter save, and cross-validation. Each mode is toggled by the YAML config.
- Pipeline core: `src/Data_Pipeline` holds shared dataset preparation and feature extraction. `src/traditional_experiment_utils.py` contains plotting and evaluation helpers for the traditional path.
- Models: `src/models/` contains BaseModel implementations (RF, EGB, KNN, LDA, LogReg, SVM, Transformer). The config parser maps model alias names to these classes.
- Results: outputs and plots are written to `Results/`; saved models go to `ModelSave/`.
- Config: `GLOC_experiment_config.yaml` controls everything (required keys: `models`, `model_type`, `random_seed`, `data_path`).

## Key conventions and repository specifics
- `model_type` must be a two-item list: `[AFE_filter, feature_set]` (e.g., `['Complete','Explicit']`). The results folder name is derived from this pair (e.g., `Results/Sensor_Ablation/Complete_Explicit/`).
- Model name aliases are supported (e.g., `EGB` == `Extreme Gradient Boosting`, `RF` == `Random Forest`, `KNN` == `K Nearest Neighbors`). See `src/GLOC_experiment_config_parser.py` mapping.
- Config getters: code uses `GLOCExperimentConfigParser` extensively; many helpers are available (e.g., `get_sensor_ablation_enabled()`, `get_feature_space_review_enabled()`, `get_cross_validation_num_splits()`). Prefer using these getters rather than indexing raw config keys.
- cuML: `src/main.py` attempts to enable cuML acceleration at runtime; the code runs on CPU if RAPIDS/cuML are not present.
- Tests: `pytest.ini` sets `pythonpath = src` so tests run from repo root. There is an `integration` marker for long/data-heavy tests.
- Sensor stream aliasing: `STREAM_LABEL_ALIASES` in `src/main.py` contains human-readable replacements for some stream sets (e.g., `ECG-HR-BR-Temperature` -> `Equivital`).
- Running: always run commands from the repository root so relative paths resolve correctly.
- Cross-validation: `src/modes/cross_validation.py` provides a unified CV runner that auto-detects model types (traditional, advanced, DL-adapted) and saves results to `Results/CrossValidation/<model_name>/`. Configure via `cross_validation:` section in YAML. Legacy models use `metrics_fold_<i>.pkl` naming; advanced models use `fold_<i>/metrics.pkl`.
- Deep-learning adapters: Custom DL models can be integrated via `DLModelAdapter` subclass in `src/models/dl_adapter.py`. Adapters translate between numpy arrays and framework-specific tensors without adding heavy deps to core.

## Where to look for more details
- `README.md` contains an expanded YAML config overview and usage examples.
- `src/GLOC_experiment_config_parser.py` shows accepted config keys and aliases.
- `src/models/` shows model-specific hyperparameter handling and inference interfaces.

