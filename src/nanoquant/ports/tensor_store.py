from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

import torch

from nanoquant.domain.models import TensorRef


class TensorStore(Protocol):
    def put(self, artifact_type: str, tensors: dict[str, torch.Tensor]) -> dict[str, TensorRef]: ...
    def read(self, reference: TensorRef, device: str = "cpu") -> AbstractContextManager[torch.Tensor]: ...
