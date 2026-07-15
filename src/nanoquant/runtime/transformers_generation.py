"""Optional Hugging Face model-shell adapter for the runtime generation engine."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from nanoquant.runtime.backend import WorkloadKind
from nanoquant.runtime.generation import GenerationError, GenerationStep
from nanoquant.runtime.torch_model import execution_workload

WorkloadContext = Callable[[WorkloadKind], AbstractContextManager[None]]
CacheFactory = Callable[[int, int, torch.device, torch.dtype], object]


def _model_dtype(model: nn.Module) -> torch.dtype:
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise GenerationError("NQ-GEN-MODEL model shell contains no parameter from which to infer cache dtype")
    return parameter.dtype


@dataclass(slots=True)
class TransformersGenerationModel:
    """Adapt a cache-capable Transformers causal LM to ``GenerationModel``."""

    model: nn.Module
    cache_factory: CacheFactory
    workload_context: WorkloadContext = execution_workload

    def forward_step(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        cache: object | None,
        max_cache_length: int,
        workload: WorkloadKind,
        deterministic: bool,
    ) -> GenerationStep:
        if not deterministic:
            raise GenerationError("NQ-GEN-MODE deterministic Transformers execution was not requested")
        if cache is None:
            if workload != "prefill":
                raise GenerationError("NQ-GEN-CACHE decode cannot initialize a new cache")
            cache = self.cache_factory(
                input_ids.shape[0],
                max_cache_length,
                input_ids.device,
                _model_dtype(self.model),
            )
        context = self.workload_context(workload)
        with context:
            output = cast(Any, self.model)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                logits_to_keep=1,
                return_dict=True,
            )
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise GenerationError("NQ-GEN-LOGITS Transformers model returned no logits tensor")
        returned_cache = getattr(output, "past_key_values", None)
        return GenerationStep(logits, returned_cache if returned_cache is not None else cache)


def hybrid_cache_factory(
    config: object,
    dtype_override: torch.dtype | None = None,
) -> CacheFactory:
    """Create a total-length-bounded Transformers HybridCache factory."""

    if dtype_override is not None and not dtype_override.is_floating_point:
        raise ValueError("runtime HybridCache dtype override must be floating point")

    def create(
        batch_size: int,
        maximum_cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> object:
        from transformers import HybridCache

        cache_type = cast(Any, HybridCache)

        class PromotingHybridCache(cache_type):  # type: ignore[misc, valid-type]
            def update(
                self,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                layer_idx: int,
                cache_kwargs: dict[str, Any] | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                keys, values = super().update(
                    key_states,
                    value_states,
                    layer_idx,
                    cache_kwargs,
                )
                return keys.to(key_states.dtype), values.to(value_states.dtype)

        selected_type = cache_type if dtype_override is None or dtype_override == dtype else PromotingHybridCache
        return selected_type(
            cast(Any, config),
            max_batch_size=batch_size,
            max_cache_len=maximum_cache_length,
            device=device,
            dtype=dtype if dtype_override is None else dtype_override,
        )

    return create
