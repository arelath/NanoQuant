"""Small, consistent safetensors read helpers for research infrastructure."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open


def load_tensors(
    path: str | Path,
    keys: Iterable[str],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load requested tensors through CPU and place them on ``device``."""

    requested = tuple(keys)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("safetensors keys must be non-empty and unique")
    destination = torch.device(device)
    result: dict[str, torch.Tensor] = {}
    with safe_open(Path(path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = tuple(key for key in requested if key not in available)
        if missing:
            raise KeyError(f"safetensors keys are missing: {missing}")
        for key in requested:
            value = handle.get_tensor(key)
            result[key] = value if destination.type == "cpu" else value.to(destination)
    return result


__all__ = ["load_tensors"]
