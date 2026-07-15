from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest
import torch

from nanoquant.application.evaluation import EvaluatorRegistry
from nanoquant.application.long_context_evaluation import (
    LongContextCase,
    LongContextEvaluationRequest,
    LongContextGeneration,
    LongContextProtocol,
    evaluate_long_context,
    register_long_context_evaluator,
)
from nanoquant.infrastructure.runtime_long_context import (
    gemma3_hybrid_long_context_protocol,
    make_runtime_long_context_generator,
)
from nanoquant.runtime.generation import GenerationStep


def _protocol() -> LongContextProtocol:
    return LongContextProtocol("fixture-gemma", "1", 128, 16, 2, 16)


def _case() -> LongContextCase:
    return LongContextCase(
        "sliding-and-global-boundary",
        "1",
        tuple(range(33)),
        (40, 41, 42),
        "max_new_tokens",
    )


def _exact(_protocol_value: LongContextProtocol, case: LongContextCase) -> LongContextGeneration:
    return LongContextGeneration(
        case.expected_token_ids,
        case.expected_stop_reason,
        3,
        2,
        len(case.prompt_token_ids) + len(case.expected_token_ids),
        peak_device_bytes=1234,
    )


def test_long_context_evaluator_requires_boundaries_and_preserves_runtime_metadata() -> None:
    protocol = _protocol()
    case = _case()

    result = evaluate_long_context(LongContextEvaluationRequest(protocol, (case,)), _exact)

    assert result.passed
    assert result.case_count == result.exact_case_count == result.passed_case_count == 1
    assert result.maximum_prompt_tokens == 33
    assert result.maximum_total_tokens == 36
    assert result.peak_device_bytes == 1234
    assert result.cases[0].crossed_sliding_window
    assert result.cases[0].expected_prefill_forward_count == 3
    assert result.cases[0].first_mismatch_index is None


def test_long_context_evaluator_reports_token_stop_cache_chunk_and_fallback_failures() -> None:
    def changed(_protocol_value: LongContextProtocol, _case_value: LongContextCase) -> LongContextGeneration:
        return LongContextGeneration((40, 99), "eos", 2, 1, 35, 1)

    result = evaluate_long_context(LongContextEvaluationRequest(_protocol(), (_case(),)), changed)

    assert not result.passed
    assert result.exact_case_count == result.passed_case_count == 0
    assert result.total_unexpected_fallbacks == 1
    case = result.cases[0]
    assert case.first_mismatch_index == 1
    assert not case.stop_reason_match
    assert case.expected_cache_length == 36 and case.observed_cache_length == 35
    assert case.expected_prefill_forward_count == 3 and case.observed_prefill_forward_count == 2


def test_long_context_evaluator_rejects_short_over_limit_and_duplicate_cases() -> None:
    protocol = _protocol()
    short = replace(_case(), prompt_token_ids=tuple(range(16)))
    with pytest.raises(ValueError, match="sliding window"):
        evaluate_long_context(LongContextEvaluationRequest(protocol, (short,)), _exact)
    over_limit = replace(_case(), prompt_token_ids=tuple(range(126)))
    with pytest.raises(ValueError, match="limit is 128"):
        evaluate_long_context(LongContextEvaluationRequest(protocol, (over_limit,)), _exact)
    with pytest.raises(ValueError, match="unique"):
        evaluate_long_context(LongContextEvaluationRequest(protocol, (_case(), _case())), _exact)


def test_long_context_registry_requires_the_exact_protocol() -> None:
    registry = EvaluatorRegistry()
    protocol = _protocol()
    register_long_context_evaluator(registry, protocol, _exact)
    request = LongContextEvaluationRequest(protocol, (_case(),))

    result = registry.evaluate(protocol.evaluator_spec.name, protocol.evaluator_spec.version, request)

    assert result == evaluate_long_context(request, _exact)
    assert registry.specifications_for_tier("full", cumulative=False) == (protocol.evaluator_spec,)
    with pytest.raises(ValueError, match="does not match"):
        registry.evaluate(
            protocol.evaluator_spec.name,
            protocol.evaluator_spec.version,
            replace(request, protocol=replace(protocol, prefill_chunk_size=8)),
        )


@dataclass
class ScheduledLongContextModel:
    scheduled_tokens: tuple[int, ...]
    calls: list[dict[str, object]] = field(default_factory=list)

    def forward_step(self, **kwargs: object) -> GenerationStep:
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        token = self.scheduled_tokens[len(self.calls)]
        logits = torch.zeros((1, 1, 128))
        logits[0, 0, token] = 10
        self.calls.append(kwargs)
        return GenerationStep(logits, ("cache", len(self.calls)))


def test_gemma_protocol_and_runtime_adapter_execute_all_prefill_chunks() -> None:
    config = SimpleNamespace(
        model_type="gemma3_text",
        cache_implementation="hybrid",
        max_position_embeddings=128,
        sliding_window=16,
        sliding_window_pattern=2,
    )
    protocol = gemma3_hybrid_long_context_protocol(config)
    model = ScheduledLongContextModel((9, 9, 40, 41, 42))
    generate = make_runtime_long_context_generator(
        model,
        device="cpu",
        eos_token_ids=(127,),
        pad_token_id=0,
    )

    result = evaluate_long_context(LongContextEvaluationRequest(protocol, (_case(),)), generate)

    assert result.passed
    assert [call["workload"] for call in model.calls] == [
        "prefill",
        "prefill",
        "prefill",
        "decode",
        "decode",
    ]
    prefill_inputs = [call["input_ids"] for call in model.calls[:3]]
    assert all(isinstance(value, torch.Tensor) for value in prefill_inputs)
    assert [tuple(value.shape) for value in prefill_inputs if isinstance(value, torch.Tensor)] == [
        (1, 16),
        (1, 16),
        (1, 1),
    ]
