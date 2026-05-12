import os
import pickle
from typing import Any, Dict, Optional

import joblib
import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.base import BaseModel
from src.models.sequence_window_utils import create_trial_windows, summarize_windows_mean, max_trial_sequence_length

try:
    from pygam import LogisticGAM, s
except Exception:  # pragma: no cover - handled when instantiated.
    LogisticGAM = None
    s = None


def _require_pygam() -> None:
    if LogisticGAM is None or s is None:
        raise ImportError("GAMModel requires 'pygam'. Install it in the gloc environment before using this model.")


class GAMModel(BaseModel):
    """Logistic GAM advanced classifier aligned with legacy GAM_supporting.py behavior."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        _require_pygam()
        self.is_traditional = False
        self.model: Optional[LogisticGAM] = None

    def tune(self, X, y, groups=None) -> None:
        return None

    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] | None = None) -> None:
        _require_pygam()
        params = dict(params or {})
        sequence_length = int(params.get("sequence_length", 10))
        stride = float(params.get("stride", 0.5))
        step_size = int(params.get("step_size", max(1, round(sequence_length * stride))))
        lam = float(params.get("lam", 1.0))
        n_splines = int(params.get("n_splines", 10))
        random_seed = int(params.get("random_seed", 42))

        np.random.seed(random_seed)
        X_windows, y_windows = create_trial_windows(
            X=X,
            y=y,
            window_size=sequence_length,
            step_size=step_size,
            end_label=True,
        )
        X_flat = summarize_windows_mean(X_windows)
        y_flat = y_windows.reshape(-1).astype(int)

        n_features = int(X_flat.shape[1])
        terms = s(0, n_splines=n_splines)
        for feature_idx in range(1, n_features):
            terms += s(feature_idx, n_splines=n_splines)

        self.model = LogisticGAM(terms, lam=lam)
        self.model.fit(X_flat, y_flat)

        self.best_params = {
            "sequence_length": sequence_length,
            "stride": stride,
            "step_size": step_size,
            "lam": lam,
            "n_splines": n_splines,
            "random_seed": random_seed,
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            return {}

        sequence_length = int(self.best_params.get("sequence_length", 10))
        step_size = int(self.best_params.get("step_size", max(1, sequence_length // 2)))
        X_windows, y_windows = create_trial_windows(
            X=X,
            y=y,
            window_size=sequence_length,
            step_size=step_size,
            end_label=True,
        )
        X_flat = summarize_windows_mean(X_windows)
        y_true = y_windows.reshape(-1).astype(int)
        y_pred = np.round(self.model.predict(X_flat)).astype(int).reshape(-1)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "g_mean": float(geometric_mean_score(y_true, y_pred)),
        }
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        metadata = {"best_params": self.best_params, "split_info": self.split_info, "config": self.config}
        joblib.dump(metadata, os.path.join(path, "gam_metadata.pkl"))
        if self.model is not None:
            with open(os.path.join(path, "gam_model.pkl"), "wb") as f:
                pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        metadata_path = os.path.join(path, "gam_metadata.pkl")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.best_params = metadata.get("best_params", {})
            self.split_info = metadata.get("split_info", {})
        model_path = os.path.join(path, "gam_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([])
        sequence_length = int(self.best_params.get("sequence_length", 10))
        step_size = int(self.best_params.get("step_size", max(1, sequence_length // 2)))
        X_windows, _ = create_trial_windows(
            X=X,
            y=np.zeros(X.shape[0], dtype=np.float32),
            window_size=sequence_length,
            step_size=step_size,
            end_label=True,
        )
        X_flat = summarize_windows_mean(X_windows)
        return np.round(self.model.predict(X_flat)).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([])
        sequence_length = int(self.best_params.get("sequence_length", 10))
        step_size = int(self.best_params.get("step_size", max(1, sequence_length // 2)))
        X_windows, _ = create_trial_windows(
            X=X,
            y=np.zeros(X.shape[0], dtype=np.float32),
            window_size=sequence_length,
            step_size=step_size,
            end_label=True,
        )
        X_flat = summarize_windows_mean(X_windows)
        probs = np.asarray(self.model.predict_proba(X_flat), dtype=np.float32).reshape(-1)
        return np.vstack([1.0 - probs, probs]).T

    def hpo_defaults(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "n_trials": 5,
            "timeout": None,
            "metric": "f1",
            "train_fraction": 0.8,
            "sampler_seed": None,
            "pruner_startup_trials": 3,
            "pruner_warmup_steps": 0,
        }

    def build_hpo_search_space(self, trial, X_train: np.ndarray, random_seed: int) -> Dict[str, Any]:
        max_seq = max_trial_sequence_length(X_train)
        window_size = int(trial.suggest_int("window_size", 1, max(1, max_seq)))
        stride = int(trial.suggest_int("stride", 1, max(1, window_size)))
        return {
            "window_size": window_size,
            "stride": stride,
            "random_seed": int(random_seed),
        }

    def get_name(self) -> str:
        return "GAM"
