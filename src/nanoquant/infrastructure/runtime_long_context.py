"""Runtime adapter and Gemma plan for long-context evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from nanoquant.application.long_context_evaluation import (
    LongContextCase,
    LongContextGenerate,
    LongContextGeneration,
    LongContextProtocol,
)
from nanoquant.runtime.generation import (
    GenerationModel,
    GenerationRequest,
    batch_prompts,
    generate,
)


def gemma3_hybrid_long_context_protocol(
    config: object,
    *,
    prefill_chunk_size: int | None = None,
    maximum_unexpected_fallbacks: int = 0,
) -> LongContextProtocol:
    model_type = getattr(config, "model_type", None)
    cache_implementation = getattr(config, "cache_implementation", None)
    maximum = getattr(config, "max_position_embeddings", None)
    sliding = getattr(config, "sliding_window", None)
    interval = getattr(config, "sliding_window_pattern", None)
    if model_type != "gemma3_text" or cache_implementation != "hybrid":
        raise ValueError("long-context runtime plan requires Gemma3 text with HybridCache")
    if any(type(value) is not int or value <= 0 for value in (maximum, sliding, interval)):
        raise ValueError("Gemma3 long-context configuration is incomplete")
    assert isinstance(maximum, int) and isinstance(sliding, int) and isinstance(interval, int)
    chunk = sliding if prefill_chunk_size is None else prefill_chunk_size
    if type(chunk) is not int or chunk <= 0 or chunk > sliding:
        raise ValueError("Gemma3 prefill chunks must be positive and no larger than the sliding window")
    return LongContextProtocol(
        "gemma3-hybrid-cache",
        "1",
        maximum,
        sliding,
        interval,
        chunk,
        maximum_unexpected_fallbacks,
    )


def make_runtime_long_context_generator(
    model: GenerationModel,
    *,
    device: torch.device | str,
    eos_token_ids: tuple[int, ...],
    pad_token_id: int,
    stop_token_sequences: tuple[tuple[int, ...], ...] = (),
    stopping_check_interval: int = 1,
    profile_device_memory: bool = False,
    fallback_counter: Callable[[], int] | None = None,
) -> LongContextGenerate:
    selected_device = torch.device(device)
    if profile_device_memory and selected_device.type != "cuda":
        raise ValueError("long-context device-memory profiling requires CUDA")
    if fallback_counter is None:
        typed_model = cast(Any, model)

        def fallback_counter() -> int:
            return int(getattr(typed_model, "cuda_graph_fallback_count", 0))

    def evaluate(protocol: LongContextProtocol, case: LongContextCase) -> LongContextGeneration:
        tokens, mask = batch_prompts(
            (case.prompt_token_ids,),
            pad_token_id=pad_token_id,
            device=selected_device,
        )
        before_fallbacks = fallback_counter()
        if profile_device_memory:
            torch.cuda.reset_peak_memory_stats(selected_device)
        result = generate(
            GenerationRequest(
                tokens,
                mask,
                len(case.expected_token_ids),
                eos_token_ids,
                pad_token_id,
                stop_token_sequences=stop_token_sequences,
                stopping_check_interval=stopping_check_interval,
                prefill_chunk_size=protocol.prefill_chunk_size,
            ),
            model,
        )
        length = result.lengths[0]
        peak = torch.cuda.max_memory_allocated(selected_device) if profile_device_memory else None
        return LongContextGeneration(
            tuple(int(token) for token in result.token_ids[0, :length].tolist()),
            result.stop_reasons[0],
            result.prefill_forward_count,
            result.decode_forward_count,
            result.maximum_cache_length,
            fallback_counter() - before_fallbacks,
            peak,
        )

    return evaluate
