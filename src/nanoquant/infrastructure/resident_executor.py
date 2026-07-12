"""Resident device executor with scoped tensor leases and reusable buffers."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch


class Cancellation:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise InterruptedError("run was cancelled")


class ResidentExecutor:
    def __init__(self) -> None:
        self._buffers: dict[tuple[str, tuple[int, ...], torch.dtype, str], torch.Tensor] = {}

    @contextmanager
    def device_scope(self, device: str) -> Iterator[None]:
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device scope requested without CUDA")
            with torch.cuda.device(device):
                yield
        else:
            yield

    @contextmanager
    def tensor_lease(self, value: torch.Tensor, device: str) -> Iterator[torch.Tensor]:
        leased = value if str(value.device) == device else value.to(device)
        try:
            yield leased
        finally:
            if leased is not value:
                del leased

    def buffer(self, key: str, shape: tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
        identity = (key, shape, dtype, device)
        if identity not in self._buffers:
            self._buffers[identity] = torch.empty(shape, dtype=dtype, device=device)
        return self._buffers[identity]

    def release(self) -> None:
        self._buffers.clear()
