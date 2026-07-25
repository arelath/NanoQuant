"""Structural typing for the Hugging Face model surface used by NanoQuant."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class HuggingFaceConfig(Protocol):
    use_cache: bool
    hidden_size: int
    final_logit_softcapping: float | None


class HuggingFaceModelOutput(Protocol):
    logits: torch.Tensor


class HuggingFaceModel(Protocol):
    config: HuggingFaceConfig

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        use_cache: bool = False,
        **kwargs: object,
    ) -> HuggingFaceModelOutput: ...

    def get_input_embeddings(self) -> nn.Module: ...


__all__ = [
    "HuggingFaceConfig",
    "HuggingFaceModel",
    "HuggingFaceModelOutput",
]
