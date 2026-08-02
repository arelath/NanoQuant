"""Fail-closed deterministic PyTorch execution controls."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


@contextmanager
def deterministic_torch_execution(seed: int, device: str) -> Iterator[None]:
    """Seed one run and reject CUDA kernels without deterministic support."""

    configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured_workspace not in {None, _CUBLAS_WORKSPACE_CONFIG}:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 for deterministic execution"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG

    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cuda_devices: list[int] = []
    if device.startswith("cuda"):
        requested = torch.device(device)
        cuda_devices = [
            torch.cuda.current_device() if requested.index is None else requested.index
        ]
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


__all__ = ["deterministic_torch_execution"]
