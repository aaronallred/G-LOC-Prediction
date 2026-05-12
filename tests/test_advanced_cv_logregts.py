import tempfile
import numpy as np
from pathlib import Path

from src.modes.cross_validation import run_cross_validation
from src.model_type import ModelType
from src.models.logistic_regression_ts import LogRegTS


class FakeConfig:
    def get_cross_validation_save_median_hyperparameters(self):
        return False

    def get_advanced_hpo_settings(self):
        """Return mock advanced HPO settings for testing LogRegTS models."""
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
    def __init__(self, n_samples=120, n_features=8, num_splits=3, seed=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.num_splits = num_splits
        rng = np.random.RandomState(seed)
        X = rng.randn(n_samples, n_features).astype(np.float32)
        # create a simple linearly separable target
        weights = rng.randn(n_features)
        logits = X.dot(weights)
        y = (logits > np.median(logits)).astype(int)
        self.X = X
        self.y = y
        self.seed = seed

    def set_random_seed(self, seed: int):
        self.seed = seed

    def set_model_type(self, model_type):
        self.model_type = model_type

    def get_data(self, model=None, kfold_id: int = 0, num_splits: int = None):
        # simple deterministic split by modulo
        idx = np.arange(self.n_samples)
        val_mask = (idx % self.num_splits) == kfold_id
        train_mask = ~val_mask
        X_train = self.X[train_mask]
        X_val = self.X[val_mask]
        y_train = self.y[train_mask]
        y_val = self.y[val_mask]
        features = [f"f{i}" for i in range(self.n_features)]
        return X_train, X_val, y_train, y_val, features


def test_logregts_cv(tmp_path: Path):
    model = LogRegTS(config={})
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

    # Expect one key for model
    assert isinstance(results, dict)
    model_name = model.get_name()
    assert model_name in results
    folds = results[model_name]
    assert len(folds) == 3
    for fold in folds:
        assert "metrics" in fold
        assert "f1" in fold["metrics"]
