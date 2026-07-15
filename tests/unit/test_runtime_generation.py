from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, HybridCache

from nanoquant.runtime.generation import (
    GenerationError,
    GenerationRequest,
    GenerationStep,
    SamplingConfig,
    batch_prompts,
    generate,
)
from nanoquant.runtime.torch_model import bind_short_sliding_masks
from nanoquant.runtime.transformers_generation import (
    TransformersGenerationModel,
    hybrid_cache_factory,
)


@dataclass
class ScheduledModel:
    scheduled_tokens: tuple[tuple[int, ...], ...]
    calls: list[dict[str, object]] = field(default_factory=list)

    def forward_step(self, **kwargs: object) -> GenerationStep:
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        call_index = len(self.calls)
        selected = self.scheduled_tokens[call_index]
        vocabulary = max(max(tokens) for tokens in self.scheduled_tokens) + 2
        logits = torch.zeros((input_ids.shape[0], 1, vocabulary), device=input_ids.device)
        for row, token in enumerate(selected):
            logits[row, 0, token] = 10.0
        self.calls.append(
            {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }
        )
        return GenerationStep(logits, ("cache", call_index))


def _request(*, max_new_tokens: int = 4) -> GenerationRequest:
    return GenerationRequest(
        input_ids=torch.tensor([[0, 11, 12], [21, 22, 23]]),
        attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.bool),
        max_new_tokens=max_new_tokens,
        eos_token_ids=(2,),
        pad_token_id=0,
    )


def test_prompt_batching_left_pads_ragged_tokens() -> None:
    tokens, mask = batch_prompts(((11, 12), (21, 22, 23)), pad_token_id=0)

    assert torch.equal(tokens, torch.tensor([[0, 11, 12], [21, 22, 23]]))
    assert torch.equal(
        mask,
        torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.bool),
    )


@pytest.mark.parametrize("prompts", [(), ((1,), ()), ((1, -2),)])
def test_prompt_batching_rejects_invalid_prompts(
    prompts: tuple[tuple[int, ...], ...],
) -> None:
    with pytest.raises(GenerationError):
        batch_prompts(prompts, pad_token_id=0)


def test_generation_passes_explicit_prefill_and_decode_metadata() -> None:
    model = ScheduledModel(((4, 5), (6, 2), (2, 8)))
    result = generate(_request(), model)

    assert result.lengths == (3, 2)
    assert result.stop_reasons == ("eos", "eos")
    assert result.prefill_forward_count == 1
    assert result.decode_forward_count == 2
    assert result.maximum_cache_length == 7
    assert torch.equal(result.token_ids, torch.tensor([[4, 6, 2, 0], [5, 2, 0, 0]]))

    prefill, first_decode, second_decode = model.calls
    assert prefill["workload"] == "prefill"
    assert torch.equal(prefill["position_ids"], torch.tensor([[0, 1, 2], [0, 1, 2]]))
    assert torch.equal(prefill["cache_position"], torch.tensor([0, 1, 2]))
    assert prefill["max_cache_length"] == 7
    assert first_decode["workload"] == "decode"
    assert torch.equal(first_decode["input_ids"], torch.tensor([[4], [5]]))
    assert torch.equal(first_decode["position_ids"], torch.tensor([[3], [3]]))
    assert torch.equal(first_decode["cache_position"], torch.tensor([3]))
    assert torch.equal(
        first_decode["attention_mask"],
        torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool),
    )
    assert torch.equal(second_decode["input_ids"], torch.tensor([[6], [0]]))
    assert torch.equal(second_decode["position_ids"], torch.tensor([[4], [0]]))
    assert torch.equal(second_decode["cache_position"], torch.tensor([4]))
    assert torch.equal(
        second_decode["attention_mask"],
        torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.bool),
    )


def test_chunked_prefill_reuses_one_cache_and_preserves_final_prompt_logits() -> None:
    model = ScheduledModel(((9, 9), (4, 5), (6, 7)))
    request = replace(_request(max_new_tokens=2), prefill_chunk_size=2, eos_token_ids=(99,))

    result = generate(request, model)

    assert torch.equal(result.token_ids, torch.tensor([[4, 6], [5, 7]]))
    assert result.prefill_forward_count == 2
    assert result.decode_forward_count == 1
    first_prefill, second_prefill, decode = model.calls
    assert torch.equal(first_prefill["input_ids"], torch.tensor([[0, 11], [21, 22]]))
    assert torch.equal(first_prefill["attention_mask"], torch.tensor([[0, 1], [1, 1]], dtype=torch.bool))
    assert torch.equal(first_prefill["cache_position"], torch.tensor([0, 1]))
    assert first_prefill["cache"] is None
    assert torch.equal(second_prefill["input_ids"], torch.tensor([[12], [23]]))
    assert torch.equal(
        second_prefill["attention_mask"],
        torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.bool),
    )
    assert torch.equal(second_prefill["cache_position"], torch.tensor([2]))
    assert second_prefill["cache"] == ("cache", 0)
    assert decode["cache"] == ("cache", 1)


def test_generation_stops_at_limit_and_is_repeatable() -> None:
    first_model = ScheduledModel(((4, 5), (6, 7)))
    second_model = ScheduledModel(((4, 5), (6, 7)))
    request = _request(max_new_tokens=2)
    first = generate(request, first_model)
    second = generate(request, second_model)

    assert torch.equal(first.token_ids, torch.tensor([[4, 6], [5, 7]]))
    assert first.lengths == (2, 2)
    assert first.stop_reasons == ("max_new_tokens", "max_new_tokens")
    assert first.decode_forward_count == 1
    assert first.stopping_sync_count == 1
    assert first.terminal_sync_count == 1
    assert torch.equal(first.token_ids, second.token_ids)
    assert first.lengths == second.lengths
    assert first.stop_reasons == second.stop_reasons


def test_generation_stops_without_decode_when_every_row_emits_eos() -> None:
    model = ScheduledModel(((2, 2),))
    result = generate(_request(max_new_tokens=5), model)
    assert result.lengths == (1, 1)
    assert result.stop_reasons == ("eos", "eos")
    assert result.decode_forward_count == 0
    assert len(model.calls) == 1


def test_generation_stops_on_configured_token_sequence() -> None:
    model = ScheduledModel(((4, 7), (9, 8), (2, 6)))
    request = replace(
        _request(max_new_tokens=5),
        stop_token_sequences=((7, 8),),
    )
    result = generate(request, model)

    assert torch.equal(result.token_ids, torch.tensor([[4, 9, 2, 0, 0], [7, 8, 0, 0, 0]]))
    assert result.lengths == (3, 2)
    assert result.stop_reasons == ("eos", "stop_sequence")
    assert result.decode_forward_count == 2


def test_seeded_device_sampling_replays_and_honors_top_k() -> None:
    class SamplingModel:
        def forward_step(self, **kwargs: object) -> GenerationStep:
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, torch.Tensor)
            logits = torch.tensor(
                (0.0, 1.0, 2.0, 3.0, 4.0),
                device=input_ids.device,
            ).repeat(input_ids.shape[0], 1).unsqueeze(1)
            return GenerationStep(logits, object())

    request = replace(
        _request(max_new_tokens=4),
        eos_token_ids=(99,),
        sampling=SamplingConfig(
            mode="sample",
            temperature=0.75,
            top_k=2,
            top_p=0.9,
            seed=1234,
        ),
    )
    first = generate(request, SamplingModel())
    second = generate(request, SamplingModel())

    assert torch.equal(first.token_ids, second.token_ids)
    assert set(first.token_ids.flatten().tolist()) <= {3, 4}
    assert first.stopping_sync_count == 3
    assert first.terminal_sync_count == 1

    nucleus_only = generate(
        replace(
            request,
            max_new_tokens=2,
            sampling=SamplingConfig(mode="sample", top_p=0.01, seed=1234),
        ),
        SamplingModel(),
    )
    assert torch.equal(nucleus_only.token_ids, torch.full((2, 2), 4))


def test_stopping_checks_have_an_explicit_batching_interval() -> None:
    model = ScheduledModel(((2, 2), (4, 5)))
    result = generate(
        replace(_request(max_new_tokens=4), stopping_check_interval=2),
        model,
    )

    assert result.lengths == (1, 1)
    assert result.decode_forward_count == 1
    assert result.stopping_sync_count == 1
    assert torch.equal(result.token_ids, torch.tensor([[2, 0, 0, 0], [2, 0, 0, 0]]))


@pytest.mark.parametrize(
    ("generation_request", "code"),
    [
        (
            GenerationRequest(torch.ones(2, dtype=torch.long), torch.ones(2), 1, (2,), 0),
            "NQ-GEN-SHAPE",
        ),
        (
            GenerationRequest(
                torch.tensor([[1, 0]]), torch.tensor([[1, 0]]), 1, (2,), 0
            ),
            "NQ-GEN-PADDING",
        ),
        (
            GenerationRequest(
                torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), 1, (2,), 0, False
            ),
            "NQ-GEN-MODE",
        ),
        (
            GenerationRequest(
                torch.tensor([[1]]),
                torch.tensor([[1]]),
                1,
                (2,),
                0,
                prefill_chunk_size=0,
            ),
            "NQ-GEN-PREFILL",
        ),
    ],
)
def test_generation_rejects_invalid_requests(
    generation_request: GenerationRequest, code: str
) -> None:
    with pytest.raises(GenerationError, match=code):
        generate(generation_request, ScheduledModel(((2,),)))


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "sample"},
        {"mode": "sample", "seed": 1, "temperature": 0},
        {"mode": "sample", "seed": 1, "top_k": 0},
        {"mode": "sample", "seed": 1, "top_p": 0},
        {"mode": "greedy", "seed": 1},
    ],
)
def test_sampling_configuration_rejects_ambiguous_or_invalid_settings(
    settings: dict[str, object],
) -> None:
    with pytest.raises(GenerationError, match="NQ-GEN-SAMPLING"):
        SamplingConfig(**settings)  # type: ignore[arg-type]


def test_generation_rejects_full_vocabulary_prompt_logits() -> None:
    class BadModel(ScheduledModel):
        def forward_step(self, **kwargs: object) -> GenerationStep:
            return GenerationStep(torch.zeros((2, 3, 4)), object())

    with pytest.raises(GenerationError, match="NQ-GEN-LOGITS"):
        generate(_request(), BadModel(((2, 2),)))


def test_transformers_adapter_allocates_total_bound_and_requests_one_logit_row() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.calls: list[dict[str, object]] = []

        def forward(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, torch.Tensor)
            logits = torch.zeros((input_ids.shape[0], 1, 8))
            logits[:, :, 2] = 1
            return SimpleNamespace(logits=logits, past_key_values=kwargs["past_key_values"])

    cache_calls: list[tuple[int, int, torch.device, torch.dtype]] = []
    cache = object()

    def cache_factory(
        batch: int, length: int, device: torch.device, dtype: torch.dtype
    ) -> object:
        cache_calls.append((batch, length, device, dtype))
        return cache

    shell = Model()
    # This shell has no prepared linears, so use a no-op workload context.
    adapter = TransformersGenerationModel(shell, cache_factory, lambda kind: nullcontext())
    result = generate(_request(max_new_tokens=3), adapter)

    assert result.decode_forward_count == 0
    assert cache_calls == [(2, 6, torch.device("cpu"), torch.float32)]
    assert len(shell.calls) == 1
    call = shell.calls[0]
    assert call["logits_to_keep"] == 1
    assert call["past_key_values"] is cache
    assert torch.equal(call["position_ids"], torch.tensor([[0, 1, 2], [0, 1, 2]]))


def test_hybrid_cache_factory_can_store_lower_precision_and_promote_attention_views() -> None:
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=8,
        sliding_window_pattern=2,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )

    cache = hybrid_cache_factory(config, torch.float16, fused_cache_prefix=True)(
        2,
        12,
        torch.device("cpu"),
        torch.float32,
    )

    assert isinstance(cache, HybridCache)
    assert all(value.dtype == torch.float16 for value in cache.key_cache)
    model = Gemma3ForCausalLM(config).eval()
    tokens, mask = batch_prompts(((2, 4, 5),), pad_token_id=0)
    result = generate(
        GenerationRequest(tokens, mask, 3, (1000,), 0),
        TransformersGenerationModel(
            model,
            hybrid_cache_factory(config, torch.float16, fused_cache_prefix=True),
            lambda kind: nullcontext(),
        ),
    )
    assert result.lengths == (3,)
    with pytest.raises(ValueError, match="floating point"):
        hybrid_cache_factory(config, torch.int64)
    with pytest.raises(GenerationError, match="NQ-GEN-CONTEXT"):
        hybrid_cache_factory(config)(1, 65, torch.device("cpu"), torch.float32)


def test_generation_runs_against_transformers_gemma3_hybrid_cache() -> None:
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=8,
        sliding_window_pattern=2,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=False,
    )
    model = Gemma3ForCausalLM(config).eval()
    tokens, mask = batch_prompts(((2, 4), (2, 5, 6)), pad_token_id=0)
    created_caches: list[object] = []
    factory = hybrid_cache_factory(
        config,
        fast_sliding_prefix=False,
        fused_cache_prefix=False,
    )

    def capture_cache(
        batch_size: int,
        maximum_cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> object:
        cache = factory(batch_size, maximum_cache_length, device, dtype)
        created_caches.append(cache)
        return cache

    adapter = TransformersGenerationModel(model, capture_cache, lambda kind: nullcontext())

    result = generate(GenerationRequest(tokens, mask, 3, (1000,), 0), adapter)

    assert result.token_ids.shape == (2, 3)
    assert result.lengths == (3, 3)
    assert result.maximum_cache_length == 6
    assert result.prefill_forward_count == 1
    assert result.decode_forward_count == 2


def test_long_cached_gemma_generation_matches_transformers_reference() -> None:
    torch.manual_seed(20260715)
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=16,
        sliding_window_pattern=2,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=False,
    )
    model = Gemma3ForCausalLM(config).eval()
    tokens, mask = batch_prompts(((2, 4), (2, 5, 6)), pad_token_id=0)
    created_caches: list[object] = []
    factory = hybrid_cache_factory(config, fast_sliding_prefix=True)

    def capture_cache(
        batch_size: int,
        maximum_cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> object:
        cache = factory(batch_size, maximum_cache_length, device, dtype)
        created_caches.append(cache)
        return cache

    adapter = TransformersGenerationModel(model, capture_cache, lambda kind: nullcontext())
    max_new_tokens = 32
    cached = generate(
        GenerationRequest(
            tokens,
            mask,
            max_new_tokens,
            (1000,),
            0,
            stopping_check_interval=8,
        ),
        adapter,
    )

    with torch.inference_mode():
        reference = model.generate(
            input_ids=tokens,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=0,
            eos_token_id=1000,
            use_cache=True,
            cache_implementation="hybrid",
        )[:, tokens.shape[1] :]

    assert torch.equal(cached.token_ids, reference)
    assert cached.decode_forward_count == max_new_tokens - 1
    assert cached.stopping_sync_count == 3
    assert len(created_caches) == 1
    cache = created_caches[0]
    assert isinstance(cache, HybridCache)
    assert cache.nanoquant_fast_sliding_update_count > 0
    assert cache.get_max_cache_shape() == tokens.shape[1] + max_new_tokens
    assert {tuple(value.shape) for value in cache.key_cache} == {
        (2, 2, 16, 8),
        (2, 2, tokens.shape[1] + max_new_tokens, 8),
    }
    assert [tuple(value.shape) for value in cache.value_cache] == [
        tuple(value.shape) for value in cache.key_cache
    ]


def test_chunked_prefill_reaches_model_context_ceiling_with_reference_parity() -> None:
    torch.manual_seed(20260715)
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=16,
        sliding_window_pattern=2,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = Gemma3ForCausalLM(config).eval()
    assert bind_short_sliding_masks(model) == 1
    prompt = (2, *(3 + index % 60 for index in range(95)))
    tokens, mask = batch_prompts((prompt,), pad_token_id=0)
    max_new_tokens = 32
    created_caches: list[object] = []
    factory = hybrid_cache_factory(
        config,
        fast_sliding_prefix=False,
        fused_cache_prefix=False,
    )

    def capture_cache(
        batch_size: int,
        maximum_cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> object:
        cache = factory(batch_size, maximum_cache_length, device, dtype)
        created_caches.append(cache)
        return cache

    adapter = TransformersGenerationModel(model, capture_cache, lambda kind: nullcontext())

    chunked = generate(
        GenerationRequest(
            tokens,
            mask,
            max_new_tokens,
            (1000,),
            0,
            prefill_chunk_size=16,
        ),
        adapter,
    )
    with torch.inference_mode():
        reference = model.generate(
            input_ids=tokens,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=0,
            eos_token_id=1000,
            use_cache=True,
            cache_implementation="hybrid",
        )[:, tokens.shape[1] :]

    assert torch.equal(chunked.token_ids, reference)
    assert chunked.maximum_cache_length == config.max_position_embeddings
    assert chunked.prefill_forward_count == 6
    assert chunked.decode_forward_count == max_new_tokens - 1
    assert len(created_caches) == 1
    assert created_caches[0].nanoquant_chunked_sliding_update_count > 0
