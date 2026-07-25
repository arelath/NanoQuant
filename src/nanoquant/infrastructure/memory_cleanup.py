"""Reliable explicit Python and CUDA allocator cleanup."""

from __future__ import annotations

import gc
from collections.abc import Iterator
from contextlib import contextmanager

import torch


def release_memory(
    device: str | torch.device,
    *,
    synchronize: bool = False,
) -> None:
    """Release unreachable Python objects and unused CUDA allocator blocks."""

    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if synchronize:
            torch.cuda.synchronize(torch.device(device))


@contextmanager
def explicit_memory_cleanup(
    device: str | torch.device,
    *,
    synchronize: bool = False,
) -> Iterator[None]:
    """Always perform explicit cleanup after a bounded memory-heavy operation."""

    try:
        yield
    finally:
        release_memory(device, synchronize=synchronize)


@contextmanager
def gpu_memory_scope(
    device: str | torch.device,
    *,
    synchronize: bool = True,
) -> Iterator[None]:
    """Bound a VRAM-heavy unit and release allocator state on every exit path."""

    try:
        yield
    finally:
        release_memory(device, synchronize=synchronize)


__all__ = ["explicit_memory_cleanup", "gpu_memory_scope", "release_memory"]
