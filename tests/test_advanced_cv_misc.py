import numpy as np
from pathlib import Path
from src.modes.cross_validation import run_cross_validation
from src.model_type import ModelType
from src.models.nam_model import NAMModel
from src.models.tcn_model import TCNModel
from src.models.lstm_model import LSTMModel
from src.models.transformer import TransformerModel

class FakeConfig:
    def get_cross_validation_save_median_hyperparameters(self):
        return False

class FakePipeline:
    def __init__(self, n_samples=120, n_features=8, num_splits=3, seed=7):
        self.n_samples = n_samples
        self.n_features = n_features
        self.num_splits = num_splits
        rng = np.random.RandomState(seed)
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

    def get_data(self, model=None, kfold_id: int = 0):
        idx = np.arange(self.n_samples)
        val_mask = (idx % self.num_splits) == kfold_id
        train_mask = ~val_mask
        X_train = self.X[train_mask]
        X_val = self.X[val_mask]
        y_train = self.y[train_mask]
        y_val = self.y[val_mask]
        features = [f"f{i}" for i in range(self.n_features)]
        return X_train, X_val, y_train, y_val, features


def _run_model_cv(model, tmp_path: Path):
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


def test_nam_tcn_lstm_trans(tmp_path: Path):
    for cls in [NAMModel, TCNModel, LSTMModel, TransformerModel]:
        model = cls(config={})
        _run_model_cv(model, tmp_path)
