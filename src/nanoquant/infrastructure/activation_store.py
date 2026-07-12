"""CUDA, pinned-host, and pageable-host activation stores."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


class MemoryActivationStore:
    def __init__(self, kind: str, device: str | None = None) -> None:
        if kind not in {"cuda", "pinned_ram", "ram"}:
            raise ValueError(f"unsupported in-memory activation tier: {kind}")
        if kind == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA activation storage requested without CUDA")
        self.kind = kind
        self.device = device or ("cuda:0" if kind == "cuda" else "cpu")
        self._values: dict[str, torch.Tensor] = {}

    def put(self, key: str, value: torch.Tensor) -> None:
        if not key or key in self._values:
            raise ValueError("activation key must be non-empty and unique")
        if self.kind == "cuda":
            stored = value.detach().to(self.device).clone()
        else:
            stored = value.detach().to("cpu").clone()
            if self.kind == "pinned_ram":
                stored = stored.pin_memory()
        self._values[key] = stored

    @contextmanager
    def read(self, key: str, device: str | None = None) -> Iterator[torch.Tensor]:
        try:
            value = self._values[key]
        except KeyError as exc:
            raise KeyError(f"activation is not stored: {key}") from exc
        target = (
            value if device is None or str(value.device) == device else value.to(device, non_blocking=value.is_pinned())
        )
        yield target

    def remove(self, key: str) -> None:
        del self._values[key]

    def clear(self) -> None:
        self._values.clear()
