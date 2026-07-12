from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

import torch


class ActivationStore(Protocol):
    kind: str

    def put(self, key: str, value: torch.Tensor) -> None: ...
    def read(self, key: str, device: str | None = None) -> AbstractContextManager[torch.Tensor]: ...
    def remove(self, key: str) -> None: ...
    def clear(self) -> None: ...
