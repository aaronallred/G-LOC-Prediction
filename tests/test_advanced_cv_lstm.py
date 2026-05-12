import numpy as np
from pathlib import Path
from src.modes.cross_validation import run_cross_validation
from src.model_type import ModelType
from src.models.lstm_model import LSTMModel

class FakeConfig:
    def get_cross_validation_save_median_hyperparameters(self):
        return False

    def get_advanced_hpo_settings(self):
        """Return mock advanced HPO settings for testing LSTM models."""
        return {
            "use_sampler": True,
            "final_early_stop": False,
            "metric": "f1",
            "n_trials": 3,
            "train_fraction": 0.8,
            "timeout": None,
            "sampler_seed": 42,
            "pruner_startup_trials": 3,
            "pruner_warmup_steps": 0,
        }

class FakePipeline:
    def __init__(self, n_samples=120, n_features=8, num_splits=3, seed=1):
        self.n_samples = n_samples
        self.n_features = n_features
        self.num_splits = num_splits
        rng = np.random.RandomState(seed)
        # Simulate sequence features flattened per-sample; LSTM will treat as seq_len=1
        X = rng.randn(n_samples, n_features).astype(np.float32)
        weights = rng.randn(n_features)
        logits = X.dot(weights)
        y = (logits > np.median(logits)).astype(int)
        self.X = X
        self.y = y

    def set_random_seed(self, seed: int):
        self.seed = seed

    def set_model_type(self, model_type):
        self.model_type = model_type

    def get_data(self, model=None, kfold_id: int = 0, num_splits: int = None):
        idx = np.arange(self.n_samples)
        val_mask = (idx % self.num_splits) == kfold_id
        train_mask = ~val_mask
        X_train = self.X[train_mask]
        X_val = self.X[val_mask]
        y_train = self.y[train_mask]
        y_val = self.y[val_mask]
        features = [f"f{i}" for i in range(self.n_features)]
        return X_train, X_val, y_train, y_val, features


def test_lstm_cv(tmp_path: Path):
    model = LSTMModel(config={})
    fake_pipeline = FakePipeline(n_samples=90, n_features=6, num_splits=3)
    results = run_cross_validation(
        config=FakeConfig(),
        pipeline=fake_pipeline,
        models=[model],
        num_splits=3,
        random_seed=42,
        class_weight=None,
        support_deep_learning=True,
        save_models=True,
        results_root=tmp_path / "Results",
        model_type=ModelType("noAFE", "Implicit"),
    )

    assert model.get_name() in results
    assert len(results[model.get_name()]) == 3
    for fold in results[model.get_name()]:
        assert "metrics" in fold and "f1" in fold["metrics"]
