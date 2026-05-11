import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import torch
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel
from src.models.sequence_window_utils import create_trial_windows

try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO
    from pyro.nn import PyroModule
    from pyro.optim import Adam
except Exception:  # pragma: no cover - handled at runtime when model is used.
    pyro = None
    dist = None
    SVI = None
    Trace_ELBO = None
    Adam = None
    PyroModule = torch.nn.Module


def _require_pyro() -> None:
    if pyro is None:
        raise ImportError(
            "DGLMModel requires 'pyro-ppl'. Install it in the gloc environment before using this model."
        )


class _DGLMCore(PyroModule):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.coeff_scale = 1.0

    def model(self, x, y=None):
        batch_size, seq_len, input_dim = x.shape
        with pyro.plate("batch", batch_size):
            probs = []
            for t in range(seq_len):
                beta_t = pyro.sample(
                    f"beta_{t}",
                    dist.Normal(
                        torch.zeros(input_dim, device=x.device),
                        self.coeff_scale * torch.ones(input_dim, device=x.device),
                    ).to_event(1),
                )
                logit = (x[:, t, :] * beta_t).sum(-1)
                p = torch.sigmoid(logit)
                probs.append(p)
                if y is not None:
                    pyro.sample(f"obs_{t}", dist.Bernoulli(p), obs=y[:, t])
        return probs

    def guide(self, x, y=None):
        batch_size, seq_len, input_dim = x.shape
        with pyro.plate("batch", batch_size):
            for t in range(seq_len):
                mu_q = pyro.param(f"mu_q_{t}", torch.randn(input_dim, device=x.device))
                sigma_q = pyro.param(
                    f"sigma_q_{t}",
                    torch.ones(input_dim, device=x.device),
                    constraint=dist.constraints.positive,
                )
                pyro.sample(f"beta_{t}", dist.Normal(mu_q, sigma_q).to_event(1))

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            probs = self.model(x)
        return torch.stack(probs, dim=1)


class DGLMModel(BaseModel):
    """Dynamic GLM advanced classifier aligned with legacy DGLM_supporting.py behavior."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        _require_pyro()
        self.is_traditional = False
        self.model: Optional[_DGLMCore] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tune(self, X, y, groups=None) -> None:
        return None

    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] | None = None) -> None:
        _require_pyro()
        params = dict(params or {})
        sequence_length = int(params.get("sequence_length", 25))
        stride = float(params.get("stride", 0.5))
        step_size = int(params.get("step_size", max(1, round(sequence_length * stride))))
        batch_size = int(params.get("batch_size", 64))
        lr = float(params.get("lr", 1e-3))
        num_epochs = int(params.get("num_epochs", 10))
        threshold = float(params.get("threshold", 0.5))
        coeff_scale = float(params.get("coeff_scale", 1.0))
        random_seed = int(params.get("random_seed", 42))

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        pyro.set_rng_seed(random_seed)
        pyro.clear_param_store()

        X_windows, y_windows = create_trial_windows(
            X=X,
            y=y,
            window_size=sequence_length,
            step_size=step_size,
            end_label=False,
        )
        input_dim = int(X_windows.shape[2])
        self.model = _DGLMCore(input_dim=input_dim).to(self.device)
        self.model.coeff_scale = coeff_scale

        train_dataset = TensorDataset(
            torch.tensor(X_windows, dtype=torch.float32),
            torch.tensor(y_windows, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        svi = SVI(self.model.model, self.model.guide, Adam({"lr": lr}), loss=Trace_ELBO())
        for _ in range(num_epochs):
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                svi.step(xb, yb)

        self.best_params = {
            "input_dim": input_dim,
            "sequence_length": sequence_length,
            "stride": stride,
            "step_size": step_size,
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "threshold": threshold,
            "coeff_scale": coeff_scale,
            "random_seed": random_seed,
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            return {}

        sequence_length = int(self.best_params.get("sequence_length", 25))
        step_size = int(self.best_params.get("step_size", max(1, sequence_length // 2)))
        threshold = float(self.best_params.get("threshold", 0.5))

        X_windows, y_windows = create_trial_windows(
            X=X,
            y=y,
            window_size=sequence_length,
            step_size=step_size,
            end_label=False,
        )
        if y_windows.ndim == 1:
            y_windows = y_windows.reshape(-1, 1)

        test_dataset = TensorDataset(
            torch.tensor(X_windows, dtype=torch.float32),
            torch.tensor(y_windows, dtype=torch.float32),
        )
        test_loader = DataLoader(test_dataset, batch_size=int(self.best_params.get("batch_size", 64)), shuffle=False)

        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for xb, yb in test_loader:
            xb = xb.to(self.device)
            probs = self.model.predict(xb).detach().cpu().numpy()
            preds = (probs > threshold).astype(int)
            all_preds.append(preds.reshape(-1))
            all_labels.append(yb.numpy().reshape(-1).astype(int))

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0)

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
        _require_pyro()
        os.makedirs(path, exist_ok=True)
        metadata = {"best_params": self.best_params, "split_info": self.split_info, "config": self.config}
        joblib.dump(metadata, os.path.join(path, "dglm_metadata.pkl"))
        if self.model is not None:
            pyro.get_param_store().save(os.path.join(path, "dglm_param_store.pt"))

    def load(self, path: str) -> None:
        _require_pyro()
        metadata_path = os.path.join(path, "dglm_metadata.pkl")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.best_params = metadata.get("best_params", {})
            self.split_info = metadata.get("split_info", {})

        input_dim = self.best_params.get("input_dim")
        if input_dim is None:
            return
        coeff_scale = float(self.best_params.get("coeff_scale", 1.0))
        self.model = _DGLMCore(input_dim=int(input_dim)).to(self.device)
        self.model.coeff_scale = coeff_scale
        param_store_path = os.path.join(path, "dglm_param_store.pt")
        if os.path.exists(param_store_path):
            pyro.clear_param_store()
            pyro.get_param_store().load(param_store_path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([])
        sequence_length = int(self.best_params.get("sequence_length", 25))
        step_size = int(self.best_params.get("step_size", max(1, sequence_length // 2)))
        threshold = float(self.best_params.get("threshold", 0.5))
        X_windows, _ = create_trial_windows(
            X=X,
            y=np.zeros(X.shape[0], dtype=np.float32),
            window_size=sequence_length,
            step_size=step_size,
            end_label=False,
        )
        x_tensor = torch.tensor(X_windows, dtype=torch.float32, device=self.device)
        probs = self.model.predict(x_tensor).detach().cpu().numpy()
        return (probs > threshold).astype(int).reshape(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([])
        sequence_length = int(self.best_params.get("sequence_length", 25))
        step_size = int(self.best_params.get("step_size", max(1, sequence_length // 2)))
        X_windows, _ = create_trial_windows(
            X=X,
            y=np.zeros(X.shape[0], dtype=np.float32),
            window_size=sequence_length,
            step_size=step_size,
            end_label=False,
        )
        x_tensor = torch.tensor(X_windows, dtype=torch.float32, device=self.device)
        probs = self.model.predict(x_tensor).detach().cpu().numpy().reshape(-1)
        return np.vstack([1.0 - probs, probs]).T

    def get_name(self) -> str:
        return "DGLM"
