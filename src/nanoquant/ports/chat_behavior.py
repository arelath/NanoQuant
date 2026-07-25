"""Model-family chat rendering contract for switchable reasoning behavior."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Protocol

import torch

from nanoquant.config.schema import ReasoningMode


class TokenRole(IntEnum):
    PADDING = 0
    RAW = 1
    PROMPT = 2
    REASONING = 3
    ANSWER = 4
    DELIMITER = 5


class ReasoningModeId(IntEnum):
    PADDING = 0
    RAW = 1
    THINKING = 2
    NON_THINKING = 3


@dataclass(frozen=True, slots=True)
class RenderedBehaviorRecord:
    input_ids: tuple[int, ...]
    token_role_ids: tuple[int, ...]
    reasoning_mode_ids: tuple[int, ...]
    distillation_target_mask: tuple[bool, ...]
    distillation_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.input_ids),
            len(self.token_role_ids),
            len(self.reasoning_mode_ids),
            len(self.distillation_target_mask),
            len(self.distillation_weights),
        }
        if len(lengths) != 1 or not self.input_ids:
            raise ValueError("rendered behavior record tensors must be non-empty and aligned")


class ChatBehaviorPort(Protocol):
    family: str

    @property
    def supported_modes(self) -> tuple[ReasoningMode, ...]: ...

    def policy_identity(self, tokenizer: Any) -> str: ...

    def render_generation_prompt(
        self,
        tokenizer: Any,
        messages: list[dict[str, object]],
        mode: ReasoningMode,
    ) -> tuple[int, ...]: ...

    def render_completed(
        self,
        tokenizer: Any,
        messages: list[dict[str, object]],
        mode: ReasoningMode,
        *,
        assistant_target_weight: float,
        prompt_target_weight: float,
    ) -> RenderedBehaviorRecord: ...


def tensor_sha256(*values: torch.Tensor) -> str:
    """Return a stable content hash for one or more CPU-compatible tensors."""

    import hashlib

    digest = hashlib.sha256()
    for value in values:
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


__all__ = [
    "ChatBehaviorPort",
    "ReasoningModeId",
    "RenderedBehaviorRecord",
    "TokenRole",
    "tensor_sha256",
]
