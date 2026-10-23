"""Training checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import random

import numpy as np
import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_loss: float,
    config: dict[str, Any],
    extra_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "config": config,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "extra_state": extra_state or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
