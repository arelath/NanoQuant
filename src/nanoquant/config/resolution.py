"""Immutable source and local resource resolution."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import torch

from .schema import ActivationStoreKind, ExecutorKind, MemoryPolicyMode, RunConfig


class RevisionResolver(Protocol):
    def resolve(self, source: str, revision: str | None) -> str: ...


def resolve_config(config: RunConfig, revisions: RevisionResolver) -> RunConfig:
    source_revision = revisions.resolve(config.model.source, config.model.revision)
    tokenizer_source = config.model.tokenizer_source or config.model.source
    tokenizer_revision = revisions.resolve(tokenizer_source, config.model.tokenizer_revision)
    executor = config.runtime.executor
    adaptive = config.runtime.memory_policy.mode is MemoryPolicyMode.ADAPTIVE
    if executor is ExecutorKind.AUTO and not adaptive:
        executor = ExecutorKind.RESIDENT if torch.cuda.is_available() else ExecutorKind.CPU_OFFLOAD
    activation_kind = config.runtime.activations.kind
    if activation_kind is ActivationStoreKind.AUTO and not adaptive:
        activation_kind = ActivationStoreKind.CUDA if executor is ExecutorKind.RESIDENT else ActivationStoreKind.RAM
    return replace(
        config,
        model=replace(
            config.model,
            revision=source_revision,
            tokenizer_source=tokenizer_source,
            tokenizer_revision=tokenizer_revision,
        ),
        runtime=replace(
            config.runtime, executor=executor, activations=replace(config.runtime.activations, kind=activation_kind)
        ),
    )
