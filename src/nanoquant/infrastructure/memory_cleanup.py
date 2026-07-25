"""Reliable explicit Python and CUDA allocator cleanup."""

from __future__ import annotations

import gc
from collections.abc import Iterator
from contextlib import contextmanager

import torch


def release_memory(device: str | torch.device) -> None:
    """Release unreachable Python objects and unused CUDA allocator blocks."""

    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def explicit_memory_cleanup(device: str | torch.device) -> Iterator[None]:
    """Always perform explicit cleanup after a bounded memory-heavy operation."""

    try:
        yield
    finally:
        release_memory(device)


__all__ = ["explicit_memory_cleanup", "release_memory"]
