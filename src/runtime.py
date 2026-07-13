"""Runtime hardware selection shared by training and hyperparameter tuning."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import torch

logger = logging.getLogger(__name__)

_configured_device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)


def configure_compute(runtime_config: Mapping[str, Any] | None = None) -> torch.device:
    """Resolve and store the requested compute device.

    ``device: auto`` prefers NVIDIA CUDA, then Apple MPS, then CPU. An explicit
    device request fails clearly when that accelerator is unavailable.
    """
    global _configured_device

    runtime_config = runtime_config or {}
    requested = str(runtime_config.get("device", "auto")).lower()
    require_gpu = bool(runtime_config.get("require_gpu", False))

    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("runtime.device must be one of: auto, cuda, mps, cpu")

    cuda_available = torch.cuda.is_available()
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    if requested == "auto":
        selected = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    else:
        selected = requested

    if selected == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access an NVIDIA GPU. "
            "Check the NVIDIA driver, CUDA-enabled PyTorch installation, and nvidia-smi."
        )
    if selected == "mps" and not mps_available:
        raise RuntimeError("MPS was requested, but it is unavailable to PyTorch.")
    if require_gpu and selected == "cpu":
        raise RuntimeError(
            "runtime.require_gpu is true, but no supported GPU is available."
        )

    _configured_device = torch.device(selected)
    if selected == "cuda":
        logger.info(
            "Compute device: CUDA (%s); CUDA runtime %s; %s GPU(s) visible.",
            torch.cuda.get_device_name(0),
            torch.version.cuda,
            torch.cuda.device_count(),
        )
    else:
        logger.info("Compute device: %s", selected.upper())
    return _configured_device


def get_compute_device() -> torch.device:
    """Return the device selected by :func:`configure_compute`."""
    return _configured_device


def xgboost_device() -> str:
    """Translate the selected torch device into an XGBoost device value."""
    return "cuda" if _configured_device.type == "cuda" else "cpu"


def synchronize_accelerator() -> None:
    """Wait for queued accelerator work so elapsed-time measurements are exact."""
    if _configured_device.type == "cuda":
        torch.cuda.synchronize(_configured_device)
    elif (
        _configured_device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "synchronize")
    ):
        torch.mps.synchronize()


def runtime_summary() -> dict[str, Any]:
    """Return JSON-serializable hardware information for timing reports."""
    summary: dict[str, Any] = {
        "device": _configured_device.type,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if _configured_device.type == "cuda":
        summary.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "visible_gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
            }
        )
    return summary
