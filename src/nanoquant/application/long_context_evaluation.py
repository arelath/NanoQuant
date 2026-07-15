"""Versioned long-context generation parity evaluation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

from nanoquant.application.evaluation import EvaluatorRegistry, EvaluatorSpec
from nanoquant.config.codec import canonical_json


def _hash(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LongContextProtocol:
    name: str
    version: str
    maximum_context_length: int
    sliding_window: int
    global_attention_interval: int
    prefill_chunk_size: int
    maximum_unexpected_fallbacks: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("long-context protocol name and version are required")
        values = (
            self.maximum_context_length,
            self.sliding_window,
            self.global_attention_interval,
            self.prefill_chunk_size,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("long-context protocol dimensions must be positive integers")
        if self.sliding_window > self.maximum_context_length:
            raise ValueError("long-context sliding window exceeds the model context limit")
        if self.prefill_chunk_size > self.maximum_context_length:
            raise ValueError("long-context prefill chunk exceeds the model context limit")
        if type(self.maximum_unexpected_fallbacks) is not int or self.maximum_unexpected_fallbacks < 0:
            raise ValueError("long-context fallback allowance must be a non-negative integer")

    @property
    def semantic_key(self) -> str:
        return _hash(self)

    @property
    def evaluator_spec(self) -> EvaluatorSpec:
        return EvaluatorSpec(
            f"long-context-{self.name}",
            self.version,
            "full",
            (("protocol_semantic_key", self.semantic_key),),
        )


@dataclass(frozen=True, slots=True)
class LongContextCase:
    name: str
    version: str
    prompt_token_ids: tuple[int, ...]
    expected_token_ids: tuple[int, ...]
    expected_stop_reason: str

    def __post_init__(self) -> None:
        if not self.name or not self.version or not self.expected_stop_reason:
            raise ValueError("long-context case identity and stop reason are required")
        if not self.prompt_token_ids or not self.expected_token_ids:
            raise ValueError("long-context cases require prompt and expected tokens")
        if any(token < 0 for token in (*self.prompt_token_ids, *self.expected_token_ids)):
            raise ValueError("long-context token IDs must be non-negative")

    @property
    def semantic_key(self) -> str:
        return _hash(self)


@dataclass(frozen=True, slots=True)
class LongContextGeneration:
    token_ids: tuple[int, ...]
    stop_reason: str
    prefill_forward_count: int
    decode_forward_count: int
    maximum_cache_length: int
    unexpected_fallback_count: int = 0
    peak_device_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.stop_reason or any(token < 0 for token in self.token_ids):
            raise ValueError("long-context generation output is invalid")
        counts = (
            self.prefill_forward_count,
            self.decode_forward_count,
            self.maximum_cache_length,
            self.unexpected_fallback_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("long-context generation counts must be non-negative integers")
        if self.prefill_forward_count == 0 or self.maximum_cache_length == 0:
            raise ValueError("long-context generation requires prefill and cache metadata")
        if self.peak_device_bytes is not None and (
            type(self.peak_device_bytes) is not int or self.peak_device_bytes < 0
        ):
            raise ValueError("long-context peak device bytes must be non-negative when present")


LongContextGenerate = Callable[[LongContextProtocol, LongContextCase], LongContextGeneration]


@dataclass(frozen=True, slots=True)
class LongContextEvaluationRequest:
    protocol: LongContextProtocol
    cases: tuple[LongContextCase, ...]
    maximum_cases: int | None = None


@dataclass(frozen=True, slots=True)
class LongContextCaseResult:
    case_name: str
    case_key: str
    prompt_token_count: int
    expected_token_count: int
    crossed_sliding_window: bool
    expected_prefill_forward_count: int
    observed_prefill_forward_count: int
    observed_decode_forward_count: int
    expected_cache_length: int
    observed_cache_length: int
    exact_tokens: bool
    stop_reason_match: bool
    first_mismatch_index: int | None
    unexpected_fallback_count: int
    peak_device_bytes: int | None
    passed: bool


@dataclass(frozen=True, slots=True)
class LongContextEvaluationResult:
    protocol_key: str
    case_count: int
    exact_case_count: int
    passed_case_count: int
    maximum_prompt_tokens: int
    maximum_total_tokens: int
    total_unexpected_fallbacks: int
    peak_device_bytes: int | None
    passed: bool
    cases: tuple[LongContextCaseResult, ...]


def _first_mismatch(expected: tuple[int, ...], observed: tuple[int, ...]) -> int | None:
    for index, (left, right) in enumerate(zip(expected, observed, strict=False)):
        if left != right:
            return index
    return None if len(expected) == len(observed) else min(len(expected), len(observed))


def evaluate_long_context(
    request: LongContextEvaluationRequest,
    generate: LongContextGenerate,
) -> LongContextEvaluationResult:
    if not request.cases:
        raise ValueError("long-context evaluation requires cases")
    if request.maximum_cases is not None and (
        type(request.maximum_cases) is not int or request.maximum_cases <= 0
    ):
        raise ValueError("long-context maximum cases must be a positive integer")
    cases = request.cases[: request.maximum_cases]
    names = [case.name for case in cases]
    keys = [case.semantic_key for case in cases]
    if len(names) != len(set(names)) or len(keys) != len(set(keys)):
        raise ValueError("long-context cases must have unique names and semantic identities")
    if not any(len(case.prompt_token_ids) > request.protocol.sliding_window for case in cases):
        raise ValueError("long-context evaluation must cross the configured sliding window")
    if not any(len(case.prompt_token_ids) > request.protocol.prefill_chunk_size for case in cases):
        raise ValueError("long-context evaluation must exercise multiple prefill chunks")

    results: list[LongContextCaseResult] = []
    for case in cases:
        prompt_tokens = len(case.prompt_token_ids)
        expected_tokens = len(case.expected_token_ids)
        total_tokens = prompt_tokens + expected_tokens
        if total_tokens > request.protocol.maximum_context_length:
            raise ValueError(
                f"long-context case {case.name} requires {total_tokens} tokens; "
                f"limit is {request.protocol.maximum_context_length}"
            )
        observed = generate(request.protocol, case)
        expected_prefills = math.ceil(prompt_tokens / request.protocol.prefill_chunk_size)
        expected_decodes = expected_tokens - 1
        exact = observed.token_ids == case.expected_token_ids
        stop_match = observed.stop_reason == case.expected_stop_reason
        cache_match = observed.maximum_cache_length == total_tokens
        prefill_match = observed.prefill_forward_count == expected_prefills
        decode_match = observed.decode_forward_count == expected_decodes
        fallback_match = observed.unexpected_fallback_count <= request.protocol.maximum_unexpected_fallbacks
        passed = exact and stop_match and cache_match and prefill_match and decode_match and fallback_match
        results.append(
            LongContextCaseResult(
                case.name,
                case.semantic_key,
                prompt_tokens,
                expected_tokens,
                prompt_tokens > request.protocol.sliding_window,
                expected_prefills,
                observed.prefill_forward_count,
                observed.decode_forward_count,
                total_tokens,
                observed.maximum_cache_length,
                exact,
                stop_match,
                _first_mismatch(case.expected_token_ids, observed.token_ids),
                observed.unexpected_fallback_count,
                observed.peak_device_bytes,
                passed,
            )
        )
    case_results = tuple(results)
    peaks = tuple(result.peak_device_bytes for result in case_results if result.peak_device_bytes is not None)
    return LongContextEvaluationResult(
        request.protocol.semantic_key,
        len(case_results),
        sum(result.exact_tokens for result in case_results),
        sum(result.passed for result in case_results),
        max(result.prompt_token_count for result in case_results),
        max(result.prompt_token_count + result.expected_token_count for result in case_results),
        sum(result.unexpected_fallback_count for result in case_results),
        max(peaks) if peaks else None,
        all(result.passed for result in case_results),
        case_results,
    )


def register_long_context_evaluator(
    registry: EvaluatorRegistry,
    protocol: LongContextProtocol,
    generate: LongContextGenerate,
) -> None:
    def evaluate(request: object) -> object:
        if not isinstance(request, LongContextEvaluationRequest) or request.protocol != protocol:
            raise ValueError("long-context request does not match the registered protocol")
        return evaluate_long_context(request, generate)

    registry.register(protocol.evaluator_spec, evaluate)
