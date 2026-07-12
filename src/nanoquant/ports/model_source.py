from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

from nanoquant.domain.models import CheckpointInventory, SourceTensor


class ModelSource(Protocol):
    source: str
    revision: str

    def inventory(self) -> CheckpointInventory: ...
    def read_tensor(self, tensor: SourceTensor, device: str = "cpu") -> AbstractContextManager[object]: ...
