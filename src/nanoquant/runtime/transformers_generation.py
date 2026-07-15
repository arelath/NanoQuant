"""Optional Hugging Face model-shell adapter for the runtime generation engine."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import nn

from nanoquant.runtime.backend import WorkloadKind
from nanoquant.runtime.generation import GenerationError, GenerationStep
from nanoquant.runtime.torch_model import execution_workload

WorkloadContext = Callable[[WorkloadKind], AbstractContextManager[None]]
CacheFactory = Callable[[int, int, torch.device, torch.dtype], object]


@dataclass(slots=True)
class _CapturedDecodeGraph:
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    logits: torch.Tensor


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
    cuda_graph_decode: bool = False
    cuda_graph_capture_count: int = field(default=0, init=False)
    cuda_graph_replay_count: int = field(default=0, init=False)
    cuda_graph_fallback_count: int = field(default=0, init=False)
    _cuda_graph_cache: object | None = field(default=None, init=False, repr=False)
    _cuda_graph_cache_key: tuple[int, int, torch.device, torch.dtype] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cuda_graph_pool: Any = field(default=None, init=False, repr=False)
    _cuda_graphs: dict[tuple[int, ...], _CapturedDecodeGraph] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def _cache_for_prefill(
        self,
        batch_size: int,
        max_cache_length: int,
        device: torch.device,
    ) -> object:
        dtype = _model_dtype(self.model)
        key = (batch_size, max_cache_length, device, dtype)
        if self.cuda_graph_decode and device.type == "cuda" and self._cuda_graph_cache_key == key:
            reset = getattr(self._cuda_graph_cache, "reset", None)
            if callable(reset):
                reset()
                assert self._cuda_graph_cache is not None
                return self._cuda_graph_cache
        cache = self.cache_factory(batch_size, max_cache_length, device, dtype)
        if self.cuda_graph_decode and device.type == "cuda":
            self._cuda_graph_cache = cache
            self._cuda_graph_cache_key = key
            self._cuda_graph_pool = None
            self._cuda_graphs.clear()
        return cache

    def _run_model(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        cache: object,
        workload: WorkloadKind,
    ) -> GenerationStep:
        prepare_cache = getattr(cache, "_nanoquant_prepare_for_forward", None)
        if callable(prepare_cache):
            prepare_cache(
                attention_mask.shape[-1] - input_ids.shape[-1],
                input_ids.shape[-1],
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

    def _supports_cuda_graph_decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        cache: object,
    ) -> bool:
        key_cache = getattr(cache, "key_cache", None)
        value_cache = getattr(cache, "value_cache", None)
        position_start = attention_mask.shape[-1] - input_ids.shape[-1]
        return (
            self.cuda_graph_decode
            and input_ids.device.type == "cuda"
            and input_ids.shape == (1, 1)
            and attention_mask.shape[0] == 1
            and position_ids.shape == (1, 1)
            and cache_position.shape == (1,)
            and cache is self._cuda_graph_cache
            and isinstance(key_cache, list)
            and isinstance(value_cache, list)
            and bool(key_cache)
            and len(value_cache) == len(key_cache)
            and all(position_start + 1 < item.shape[2] for item in key_cache)
        )

    def _cuda_graph_forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        cache: object,
    ) -> GenerationStep:
        key = (
            *input_ids.shape,
            *attention_mask.shape,
            *position_ids.shape,
            *cache_position.shape,
        )
        captured = self._cuda_graphs.get(key)
        if captured is not None:
            captured.input_ids.copy_(input_ids)
            captured.attention_mask.copy_(attention_mask)
            captured.position_ids.copy_(position_ids)
            captured.cache_position.copy_(cache_position)
            captured.graph.replay()
            self.cuda_graph_replay_count += 1
            return GenerationStep(captured.logits, cache)

        static_input_ids = input_ids.clone()
        static_attention_mask = attention_mask.clone()
        static_position_ids = position_ids.clone()
        static_cache_position = cache_position.clone()
        # The eager pass compiles every shape-specialized Triton kernel before
        # capture. Repeating a pre-rollover prefix write is idempotent.
        warm_step = self._run_model(
            input_ids=static_input_ids,
            attention_mask=static_attention_mask,
            position_ids=static_position_ids,
            cache_position=static_cache_position,
            cache=cache,
            workload="decode",
        )
        typed_cache = cast(Any, cache)
        cache_tensors = (*typed_cache.key_cache, *typed_cache.value_cache)
        cache_snapshot = tuple(item.clone() for item in cache_tensors)
        torch.cuda.synchronize(input_ids.device)
        if self._cuda_graph_pool is None:
            self._cuda_graph_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._cuda_graph_pool):
            step = self._run_model(
                input_ids=static_input_ids,
                attention_mask=static_attention_mask,
                position_ids=static_position_ids,
                cache_position=static_cache_position,
                cache=cache,
                workload="decode",
            )
        captured = _CapturedDecodeGraph(
            graph,
            static_input_ids,
            static_attention_mask,
            static_position_ids,
            static_cache_position,
            step.logits,
        )
        self._cuda_graphs[key] = captured
        for destination, source in zip(cache_tensors, cache_snapshot, strict=True):
            destination.copy_(source)
        self.cuda_graph_capture_count += 1
        return GenerationStep(warm_step.logits, cache)

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
            cache = self._cache_for_prefill(
                input_ids.shape[0],
                max_cache_length,
                input_ids.device,
            )
        if workload == "decode":
            if self._supports_cuda_graph_decode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                cache=cache,
            ):
                return self._cuda_graph_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    cache=cache,
                )
            if self.cuda_graph_decode:
                self.cuda_graph_fallback_count += 1
        return self._run_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            cache=cache,
            workload=workload,
        )


def hybrid_cache_factory(
    config: object,
    dtype_override: torch.dtype | None = None,
    *,
    fast_sliding_prefix: bool = True,
    fused_cache_prefix: bool = True,
) -> CacheFactory:
    """Create a total-length-bounded Transformers HybridCache factory."""

    if dtype_override is not None and not dtype_override.is_floating_point:
        raise ValueError("runtime HybridCache dtype override must be floating point")
    maximum_positions = getattr(config, "max_position_embeddings", None)
    if type(maximum_positions) is not int or maximum_positions <= 0:
        raise ValueError("runtime HybridCache requires a positive max_position_embeddings")

    def create(
        batch_size: int,
        maximum_cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> object:
        if maximum_cache_length > maximum_positions:
            raise GenerationError(
                "NQ-GEN-CONTEXT requested cache length "
                f"{maximum_cache_length} exceeds model limit {maximum_positions}"
            )
        from transformers import HybridCache

        cache_type = cast(Any, HybridCache)

        class PreparedHybridCache(cache_type):  # type: ignore[misc, valid-type]
            _nanoquant_position_start: int | None = None
            _nanoquant_token_count: int | None = None
            nanoquant_fast_sliding_update_count: int = 0
            nanoquant_fused_cache_update_count: int = 0
            nanoquant_chunked_sliding_update_count: int = 0

            def _nanoquant_prepare_for_forward(
                self,
                position_start: int,
                token_count: int,
            ) -> None:
                if position_start < 0 or token_count <= 0:
                    raise ValueError("runtime cache position contract is invalid")
                self._nanoquant_position_start = position_start
                self._nanoquant_token_count = token_count

            def update(
                self,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                layer_idx: int,
                cache_kwargs: dict[str, Any] | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                kwargs = {} if cache_kwargs is None else cache_kwargs
                cache_position = kwargs.get("cache_position")
                sliding_window = kwargs.get("sliding_window")
                token_count = key_states.shape[-2]
                position_start = self._nanoquant_position_start
                prepared_token_count = self._nanoquant_token_count
                k_out = self.key_cache[layer_idx]
                v_out = self.value_cache[layer_idx]
                prefix_is_pre_rollover = (
                    isinstance(cache_position, torch.Tensor)
                    and cache_position.numel() == token_count
                    and prepared_token_count == token_count
                    and position_start is not None
                    and position_start + token_count < k_out.shape[2]
                    and k_out.device == key_states.device
                    and v_out.device == value_states.device
                )
                if (
                    sliding_window
                    and token_count > 1
                    and position_start is not None
                    and position_start > 0
                    and prepared_token_count == token_count
                ):
                    window = k_out.shape[2]
                    previous_count = min(position_start, window - 1)
                    if position_start < window:
                        previous_keys = k_out[:, :, :previous_count]
                        previous_values = v_out[:, :, :previous_count]
                    else:
                        previous_keys = k_out[:, :, -previous_count:]
                        previous_values = v_out[:, :, -previous_count:]
                    combined_keys = torch.cat(
                        (previous_keys.to(key_states.dtype), key_states),
                        dim=2,
                    )
                    combined_values = torch.cat(
                        (previous_values.to(value_states.dtype), value_states),
                        dim=2,
                    )
                    stored_keys = combined_keys[:, :, -window:].to(k_out.dtype)
                    stored_values = combined_values[:, :, -window:].to(v_out.dtype)
                    k_out.zero_()
                    v_out.zero_()
                    k_out[:, :, : stored_keys.shape[2]].copy_(stored_keys)
                    v_out[:, :, : stored_values.shape[2]].copy_(stored_values)
                    self.nanoquant_chunked_sliding_update_count += 1
                    return combined_keys, combined_values
                if (
                    fused_cache_prefix
                    and prefix_is_pre_rollover
                    and key_states.dtype == torch.float32
                    and value_states.dtype == torch.float32
                    and k_out.dtype == torch.float16
                    and v_out.dtype == torch.float16
                    and key_states.device.type == "cuda"
                    and key_states.is_contiguous()
                    and value_states.is_contiguous()
                    and k_out.is_contiguous()
                    and v_out.is_contiguous()
                ):
                    from nanoquant.runtime.cuda_kernels import launch_cache_prefix_update

                    assert position_start is not None
                    keys, values = launch_cache_prefix_update(
                        key_states,
                        value_states,
                        k_out,
                        v_out,
                        position_start,
                    )
                    self.nanoquant_fused_cache_update_count += 1
                    return keys, values
                if (
                    fast_sliding_prefix
                    and sliding_window
                    and prefix_is_pre_rollover
                ):
                    stored_keys = key_states.to(k_out.dtype)
                    stored_values = value_states.to(v_out.dtype)
                    k_out[:, :, cache_position] = stored_keys
                    v_out[:, :, cache_position] = stored_values
                    self.nanoquant_fast_sliding_update_count += 1
                    return k_out.to(key_states.dtype), v_out.to(value_states.dtype)
                keys, values = super().update(
                    key_states,
                    value_states,
                    layer_idx,
                    kwargs,
                )
                return keys.to(key_states.dtype), values.to(value_states.dtype)

        # Multi-token sliding updates require the chronological correction even
        # when optional prefix fast paths and dtype promotion are disabled.
        return PreparedHybridCache(
            cast(Any, config),
            max_batch_size=batch_size,
            max_cache_len=maximum_cache_length,
            device=device,
            dtype=dtype if dtype_override is None else dtype_override,
        )

    return create
