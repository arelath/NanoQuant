"""Versioned evaluators and token-accurate causal NLL/perplexity."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from nanoquant.config.codec import canonical_json

LogitsFunction = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]


@dataclass(frozen=True, slots=True)
class EvaluatorSpec:
    name: str
    version: str
    tier: str
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.version or not self.tier:
            raise ValueError("evaluator name, version, and tier are required")
        names = [name for name, _value in self.parameters]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("evaluator parameters must have unique non-empty names")
        canonical_json(tuple(sorted(self.parameters, key=lambda item: item[0])))

    @property
    def semantic_key(self) -> str:
        identity = (self.name, self.version, self.tier, tuple(sorted(self.parameters, key=lambda item: item[0])))
        return "sha256:" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CausalEvaluationRequest:
    token_ids: torch.Tensor
    attention_mask: torch.Tensor | None = None
    max_length: int = 2048
    stride: int = 2048
    prepend_bos_token_id: int | None = None
    append_eos_token_id: int | None = None
    batch_size: int = 1
    maximum_samples: int | None = None


@dataclass(frozen=True, slots=True)
class CausalEvaluationResult:
    total_negative_log_likelihood: float
    mean_negative_log_likelihood: float
    perplexity: float
    token_count: int
    window_count: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class PackedArtifactStructureRequest:
    packed_artifact: Path


@dataclass(frozen=True, slots=True)
class PackedArtifactStructureResult:
    packed_artifact: Path
    descriptor_sha256: str
    block_count: int
    layer_count: int
    tensor_count: int
    weight_bytes: int
    physical_bytes: int
    hashes_and_headers_verified: bool


@dataclass(frozen=True, slots=True)
class PackedReferenceParityRequest:
    logical_artifact: Path
    packed_artifact: Path
    absolute_tolerance: float = 0.0


@dataclass(frozen=True, slots=True)
class PackedReferenceParityEvaluationResult:
    logical_artifact: Path
    packed_artifact: Path
    layer_count: int
    output_elements: int
    maximum_absolute_error: float
    maximum_error_layer: str
    passed: bool


@dataclass(frozen=True, slots=True)
class PairedComparisonRequest:
    candidate_values: tuple[float, ...]
    baseline_values: tuple[float, ...]
    direction: str
    minimum_meaningful_delta: float
    confidence_level: float = 0.95
    bootstrap_samples: int = 2_000
    seed: int = 0


@dataclass(frozen=True, slots=True)
class PairedComparisonResult:
    sample_count: int
    candidate_mean: float
    baseline_mean: float
    raw_delta: float
    improvement_delta: float
    confidence_interval: tuple[float, float]
    candidate_standard_deviation: float
    baseline_standard_deviation: float
    paired_standard_deviation: float
    minimum_meaningful_delta: float
    confidence_level: float
    bootstrap_samples: int
    seed: int
    outcome: str


@dataclass(frozen=True, slots=True)
class GenerationRegressionRequest:
    case_name: str
    case_version: str
    prompt_token_ids: tuple[int, ...]
    expected_token_ids: tuple[int, ...]
    expected_stop_reason: str
    observed_token_ids: tuple[tuple[int, ...], ...]
    observed_stop_reasons: tuple[str, ...]
    minimum_generated_tokens: int = 1
    maximum_repeated_token_run: int = 16


@dataclass(frozen=True, slots=True)
class GenerationRegressionResult:
    case_key: str
    run_count: int
    generated_token_count: int
    expected_token_sha256: str
    observed_token_sha256: tuple[str, ...]
    deterministic: bool
    exact_match: bool
    stop_reason_match: bool
    first_mismatch_index: int | None
    longest_repeated_token_run: int
    passed: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RepresentationMetrics:
    source_parameter_count: int
    core_bits: int
    effective_core_bpw: float
    logical_artifact_bytes: int
    deployable_artifact_bytes: int
    artifact_bpw: float

    @classmethod
    def build(
        cls,
        source_parameter_count: int,
        core_bits: int,
        logical_artifact_bytes: int,
        deployable_artifact_bytes: int,
    ) -> RepresentationMetrics:
        values = (
            source_parameter_count,
            core_bits,
            logical_artifact_bytes,
            deployable_artifact_bytes,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("representation counts and byte sizes must be non-negative integers")
        if source_parameter_count == 0:
            raise ValueError("representation source parameter count must be positive")
        return cls(
            source_parameter_count,
            core_bits,
            core_bits / source_parameter_count,
            logical_artifact_bytes,
            deployable_artifact_bytes,
            deployable_artifact_bytes * 8 / source_parameter_count,
        )


@dataclass(frozen=True, slots=True)
class MemoryMetrics:
    quantization_peak_device_bytes: int
    quantization_peak_host_bytes: int
    quantization_temporary_disk_bytes: int
    runtime_peak_device_bytes: int
    runtime_peak_host_bytes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (
                self.quantization_peak_device_bytes,
                self.quantization_peak_host_bytes,
                self.quantization_temporary_disk_bytes,
                self.runtime_peak_device_bytes,
                self.runtime_peak_host_bytes,
            )
        ):
            raise ValueError("memory metrics must be non-negative integer byte counts")


@dataclass(frozen=True, slots=True)
class QuantizationCostMetrics:
    calibration_seconds: float
    factorization_seconds: float
    local_tuning_seconds: float
    global_tuning_seconds: float
    packing_seconds: float
    evaluation_seconds: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0
            for value in (
                self.calibration_seconds,
                self.factorization_seconds,
                self.local_tuning_seconds,
                self.global_tuning_seconds,
                self.packing_seconds,
                self.evaluation_seconds,
            )
        ):
            raise ValueError("quantization cost metrics must be finite and non-negative")

    @property
    def accounted_seconds(self) -> float:
        return (
            self.calibration_seconds
            + self.factorization_seconds
            + self.local_tuning_seconds
            + self.global_tuning_seconds
            + self.packing_seconds
            + self.evaluation_seconds
        )


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    time_to_first_token_seconds: float
    prefill_tokens_per_second: float
    inter_token_latency_seconds: float
    decode_tokens_per_second: float
    fallback_count: int

    def __post_init__(self) -> None:
        rates = (
            self.time_to_first_token_seconds,
            self.prefill_tokens_per_second,
            self.inter_token_latency_seconds,
            self.decode_tokens_per_second,
        )
        if any(not math.isfinite(value) or value < 0 for value in rates):
            raise ValueError("runtime metrics must be finite and non-negative")
        if type(self.fallback_count) is not int or self.fallback_count < 0:
            raise ValueError("runtime fallback count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class EvaluationDimensions:
    representation: RepresentationMetrics
    memory: MemoryMetrics
    quantization_cost: QuantizationCostMetrics
    runtime: RuntimeMetrics


Evaluator = Callable[[object], object]


@dataclass(frozen=True, slots=True)
class EvaluationPartition:
    name: str
    version: str
    content_hash: str
    item_hashes: tuple[str, ...]

    @classmethod
    def build(cls, name: str, version: str, items: tuple[object, ...]) -> EvaluationPartition:
        if not name or not version or not items:
            raise ValueError("evaluation partition name, version, and items are required")
        item_hashes = tuple(
            "sha256:" + hashlib.sha256(canonical_json(item).encode()).hexdigest() for item in items
        )
        if len(item_hashes) != len(set(item_hashes)):
            raise ValueError(f"evaluation partition contains duplicate items: {name}")
        content_hash = "sha256:" + hashlib.sha256(canonical_json(item_hashes).encode()).hexdigest()
        return cls(name, version, content_hash, item_hashes)


@dataclass(frozen=True, slots=True)
class EvaluationPartitions:
    calibration: EvaluationPartition
    quick_decision: EvaluationPartition
    final_evaluation: EvaluationPartition

    def __post_init__(self) -> None:
        values = (self.calibration, self.quick_decision, self.final_evaluation)
        names = [value.name for value in values]
        if len(names) != len(set(names)):
            raise ValueError("evaluation partition names must be unique")
        for index, left in enumerate(values):
            for right in values[index + 1 :]:
                overlap = set(left.item_hashes) & set(right.item_hashes)
                if overlap:
                    raise ValueError(
                        f"evaluation partitions overlap: {left.name} and {right.name} share {len(overlap)} items"
                    )


@dataclass(frozen=True, slots=True)
class GateRule:
    metric: str
    minimum: float | None = None
    maximum: float | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.metric or (self.minimum is None and self.maximum is None):
            raise ValueError("gate rules require a metric and at least one bound")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("gate rule minimum exceeds maximum")


@dataclass(frozen=True, slots=True)
class GatePolicy:
    name: str
    version: str
    rules: tuple[GateRule, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.version or not self.rules:
            raise ValueError("gate policy name, version, and rules are required")
        metrics = [rule.metric for rule in self.rules]
        if len(metrics) != len(set(metrics)):
            raise ValueError("gate policy metrics must be unique")

    @property
    def semantic_key(self) -> str:
        return "sha256:" + hashlib.sha256(canonical_json(self).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class GateDecision:
    outcome: str
    policy_key: str
    reasons: tuple[str, ...]


def evaluate_gate(policy: GatePolicy, metrics: tuple[tuple[str, float], ...]) -> GateDecision:
    observed = dict(metrics)
    if len(observed) != len(metrics):
        raise ValueError("gate observations contain duplicate metrics")
    inconclusive = []
    rejected = []
    for rule in policy.rules:
        value = observed.get(rule.metric)
        if value is None:
            if rule.required:
                inconclusive.append(f"missing required metric: {rule.metric}")
            continue
        if not math.isfinite(value):
            inconclusive.append(f"non-finite metric: {rule.metric}")
            continue
        if rule.minimum is not None and value < rule.minimum:
            rejected.append(f"{rule.metric}={value} is below {rule.minimum}")
        if rule.maximum is not None and value > rule.maximum:
            rejected.append(f"{rule.metric}={value} exceeds {rule.maximum}")
    if inconclusive:
        return GateDecision("inconclusive", policy.semantic_key, tuple(inconclusive))
    if rejected:
        return GateDecision("rejection", policy.semantic_key, tuple(rejected))
    return GateDecision("promotion", policy.semantic_key, ())


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sample_standard_deviation(values: tuple[float, ...]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def compare_paired(request: PairedComparisonRequest) -> PairedComparisonResult:
    candidate = request.candidate_values
    baseline = request.baseline_values
    if not candidate or len(candidate) != len(baseline):
        raise ValueError("paired comparison requires equally sized non-empty samples")
    if request.direction not in {"minimize", "maximize"}:
        raise ValueError("paired comparison direction must be minimize or maximize")
    if not all(math.isfinite(value) for value in (*candidate, *baseline)):
        raise ValueError("paired comparison samples must be finite")
    if (
        not math.isfinite(request.minimum_meaningful_delta)
        or request.minimum_meaningful_delta < 0
    ):
        raise ValueError("minimum meaningful delta must be finite and non-negative")
    if not math.isfinite(request.confidence_level) or not 0 < request.confidence_level < 1:
        raise ValueError("paired comparison confidence level must be between zero and one")
    if request.bootstrap_samples <= 0:
        raise ValueError("paired comparison bootstrap sample count must be positive")

    sample_count = len(candidate)
    candidate_mean = statistics.fmean(candidate)
    baseline_mean = statistics.fmean(baseline)
    raw_delta = candidate_mean - baseline_mean
    sign = 1.0 if request.direction == "maximize" else -1.0
    improvements = tuple(sign * (left - right) for left, right in zip(candidate, baseline, strict=True))
    improvement_delta = statistics.fmean(improvements)
    generator = random.Random(request.seed)
    bootstrap = [
        sum(improvements[generator.randrange(sample_count)] for _ in range(sample_count))
        / sample_count
        for _ in range(request.bootstrap_samples)
    ]
    tail = (1.0 - request.confidence_level) / 2.0
    interval = (_percentile(bootstrap, tail), _percentile(bootstrap, 1.0 - tail))
    threshold = request.minimum_meaningful_delta
    if interval[0] > threshold:
        outcome = "meaningful-improvement"
    elif interval[1] < -threshold:
        outcome = "meaningful-regression"
    elif interval[0] >= -threshold and interval[1] <= threshold:
        outcome = "no-meaningful-difference"
    else:
        outcome = "inconclusive"
    return PairedComparisonResult(
        sample_count,
        candidate_mean,
        baseline_mean,
        raw_delta,
        improvement_delta,
        interval,
        _sample_standard_deviation(candidate),
        _sample_standard_deviation(baseline),
        _sample_standard_deviation(improvements),
        threshold,
        request.confidence_level,
        request.bootstrap_samples,
        request.seed,
        outcome,
    )


def _token_hash(tokens: tuple[int, ...]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(tokens).encode()).hexdigest()


def _first_token_mismatch(expected: tuple[int, ...], observed: tuple[int, ...]) -> int | None:
    for index, (left, right) in enumerate(zip(expected, observed, strict=False)):
        if left != right:
            return index
    if len(expected) != len(observed):
        return min(len(expected), len(observed))
    return None


def _longest_repeated_run(tokens: tuple[int, ...]) -> int:
    longest = 0
    current = 0
    previous: int | None = None
    for token in tokens:
        if token == previous:
            current += 1
        else:
            current = 1
            previous = token
        longest = max(longest, current)
    return longest


def evaluate_generation_regression(request: object) -> GenerationRegressionResult:
    if not isinstance(request, GenerationRegressionRequest):
        raise TypeError("generation regression evaluator requires GenerationRegressionRequest")
    if not request.case_name or not request.case_version or not request.expected_stop_reason:
        raise ValueError("generation regression case identity and expected stop reason are required")
    if not request.prompt_token_ids or not request.expected_token_ids or not request.observed_token_ids:
        raise ValueError("generation regression requires prompt, expected, and observed token sequences")
    if len(request.observed_token_ids) != len(request.observed_stop_reasons):
        raise ValueError("generation regression runs and stop reasons differ in count")
    all_tokens = (
        *request.prompt_token_ids,
        *request.expected_token_ids,
        *(token for run in request.observed_token_ids for token in run),
    )
    if any(type(token) is not int or token < 0 for token in all_tokens):
        raise ValueError("generation regression token IDs must be non-negative integers")
    if request.minimum_generated_tokens <= 0 or request.maximum_repeated_token_run <= 0:
        raise ValueError("generation regression sanity limits must be positive")

    first = request.observed_token_ids[0]
    deterministic = all(run == first for run in request.observed_token_ids[1:]) and all(
        reason == request.observed_stop_reasons[0]
        for reason in request.observed_stop_reasons[1:]
    )
    exact_match = all(run == request.expected_token_ids for run in request.observed_token_ids)
    stop_reason_match = all(
        reason == request.expected_stop_reason for reason in request.observed_stop_reasons
    )
    repeated = max((_longest_repeated_run(run) for run in request.observed_token_ids), default=0)
    warnings = []
    if not deterministic:
        warnings.append("generation runs are not deterministic")
    if not exact_match:
        warnings.append("generated tokens differ from the pinned regression output")
    if not stop_reason_match:
        warnings.append("generation stop reason differs from the pinned regression output")
    if any(len(run) < request.minimum_generated_tokens for run in request.observed_token_ids):
        warnings.append("generated output is shorter than the sanity minimum")
    if repeated > request.maximum_repeated_token_run:
        warnings.append("generated output exceeds the repeated-token sanity limit")

    case_payload = (
        request.case_name,
        request.case_version,
        request.prompt_token_ids,
        request.expected_token_ids,
        request.expected_stop_reason,
        request.minimum_generated_tokens,
        request.maximum_repeated_token_run,
    )
    return GenerationRegressionResult(
        "sha256:" + hashlib.sha256(canonical_json(case_payload).encode()).hexdigest(),
        len(request.observed_token_ids),
        len(first),
        _token_hash(request.expected_token_ids),
        tuple(_token_hash(run) for run in request.observed_token_ids),
        deterministic,
        exact_match,
        stop_reason_match,
        _first_token_mismatch(request.expected_token_ids, first),
        repeated,
        not warnings,
        tuple(warnings),
    )


class EvaluatorRegistry:
    TIERS = ("smoke", "quick", "standard", "full")

    def __init__(self) -> None:
        self._evaluators: dict[tuple[str, str], tuple[EvaluatorSpec, Evaluator]] = {}

    def register(self, specification: EvaluatorSpec, evaluator: Evaluator) -> None:
        if not specification.name or not specification.version:
            raise ValueError("evaluator name and version are required")
        if specification.tier not in self.TIERS:
            raise ValueError(f"unsupported evaluator tier: {specification.tier}")
        key = (specification.name, specification.version)
        if key in self._evaluators:
            raise ValueError(f"evaluator is already registered: {key}")
        self._evaluators[key] = (specification, evaluator)

    def specification(self, name: str, version: str) -> EvaluatorSpec:
        try:
            return self._evaluators[(name, version)][0]
        except KeyError as exc:
            raise KeyError(f"evaluator is not registered: {(name, version)}") from exc

    def evaluate(self, name: str, version: str, request: object) -> object:
        try:
            evaluator = self._evaluators[(name, version)][1]
        except KeyError as exc:
            raise KeyError(f"evaluator is not registered: {(name, version)}") from exc
        return evaluator(request)

    def specifications_for_tier(self, tier: str, *, cumulative: bool = True) -> tuple[EvaluatorSpec, ...]:
        if tier not in self.TIERS:
            raise ValueError(f"unsupported evaluator tier: {tier}")
        maximum = self.TIERS.index(tier)
        allowed = set(self.TIERS[: maximum + 1] if cumulative else (tier,))
        specifications = [
            specification
            for specification, _evaluator in self._evaluators.values()
            if specification.tier in allowed
        ]
        return tuple(sorted(specifications, key=lambda spec: (self.TIERS.index(spec.tier), spec.name, spec.version)))

    def evaluate_tier(
        self,
        tier: str,
        requests: dict[tuple[str, str], object],
        *,
        cumulative: bool = True,
    ) -> tuple[tuple[EvaluatorSpec, object], ...]:
        specifications = self.specifications_for_tier(tier, cumulative=cumulative)
        missing = [(spec.name, spec.version) for spec in specifications if (spec.name, spec.version) not in requests]
        if missing:
            raise ValueError(f"evaluation tier requests are missing: {missing}")
        return tuple(
            (spec, self.evaluate(spec.name, spec.version, requests[(spec.name, spec.version)]))
            for spec in specifications
        )


def register_generation_smoke_evaluator(registry: EvaluatorRegistry) -> None:
    registry.register(
        EvaluatorSpec(
            "deterministic-generation-regression",
            "1",
            "smoke",
            (("token_match", "exact"), ("stop_reason_match", "exact")),
        ),
        evaluate_generation_regression,
    )


def model_logits(model: nn.Module) -> LogitsFunction:
    def evaluate(tokens: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        with torch.no_grad():
            output = cast(Any, model)(
                input_ids=tokens,
                attention_mask=attention_mask,
                use_cache=False,
            )
        return cast(torch.Tensor, output.logits)

    return evaluate


def _valid_sequence(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prepend_bos_token_id: int | None,
    append_eos_token_id: int | None,
) -> torch.Tensor:
    if attention_mask is None:
        sequence = token_ids
    else:
        mask = attention_mask.bool()
        if mask.shape != token_ids.shape:
            raise ValueError("attention mask shape does not match token IDs")
        if mask.numel() > 1 and torch.any((~mask[:-1]) & mask[1:]):
            raise ValueError("attention mask must represent contiguous right padding")
        sequence = token_ids[mask]
    additions = []
    if prepend_bos_token_id is not None:
        additions.append(sequence.new_tensor([prepend_bos_token_id]))
    additions.append(sequence)
    if append_eos_token_id is not None:
        additions.append(sequence.new_tensor([append_eos_token_id]))
    return torch.cat(additions)


def evaluate_causal_nll(request: CausalEvaluationRequest, logits: LogitsFunction) -> CausalEvaluationResult:
    tokens = request.token_ids
    if tokens.ndim != 2 or tokens.shape[0] == 0:
        raise ValueError("causal evaluation token IDs must be a non-empty rank-2 tensor")
    if request.max_length < 2 or request.stride <= 0 or request.stride > request.max_length:
        raise ValueError("causal evaluation requires max_length >= 2 and stride in [1, max_length]")
    if request.batch_size <= 0:
        raise ValueError("causal evaluation batch size must be positive")
    if request.maximum_samples is not None:
        if type(request.maximum_samples) is not int or request.maximum_samples <= 0:
            raise ValueError("causal evaluation maximum samples must be a positive integer")
        tokens = tokens[: request.maximum_samples]
    if request.attention_mask is not None and request.attention_mask.shape != request.token_ids.shape:
        raise ValueError("attention mask shape does not match token IDs")
    attention_mask = None if request.attention_mask is None else request.attention_mask[: tokens.shape[0]]
    windows: list[tuple[torch.Tensor, torch.Tensor]] = []
    for row in range(tokens.shape[0]):
        mask = None if attention_mask is None else attention_mask[row]
        sequence = _valid_sequence(
            tokens[row],
            mask,
            request.prepend_bos_token_id,
            request.append_eos_token_id,
        )
        if sequence.numel() < 2:
            continue
        previous_end = 0
        for begin in range(0, sequence.numel() - 1, request.stride):
            end = min(begin + request.max_length, sequence.numel())
            window = sequence[begin:end].unsqueeze(0)
            first_scored_global = max(previous_end, begin + 1)
            labels = window.clone()
            labels[:, : first_scored_global - begin] = -100
            windows.append((window, labels))
            previous_end = end
            if end == sequence.numel():
                break
    total_nll = 0.0
    token_count = 0
    for start in range(0, len(windows), request.batch_size):
        selected = windows[start : start + request.batch_size]
        length = max(window.shape[1] for window, _labels in selected)
        batch = tokens.new_zeros((len(selected), length))
        mask = tokens.new_zeros((len(selected), length))
        batch_labels = tokens.new_full((len(selected), length), -100)
        for index, (window, labels) in enumerate(selected):
            width = window.shape[1]
            batch[index, :width] = window[0]
            mask[index, :width] = 1
            batch_labels[index, :width] = labels[0]
        prediction = logits(batch, mask)
        if prediction.ndim != 3 or prediction.shape[:2] != batch.shape:
            raise ValueError("causal evaluator logits have an invalid shape")
        shifted_prediction = prediction[:, :-1].float().reshape(-1, prediction.shape[-1])
        shifted_labels = batch_labels[:, 1:].reshape(-1)
        count = int((shifted_labels != -100).sum())
        if count:
            loss = torch.nn.functional.cross_entropy(
                shifted_prediction,
                shifted_labels,
                ignore_index=-100,
                reduction="sum",
            )
            total_nll += float(loss)
            token_count += count
    if token_count == 0:
        raise ValueError("causal evaluation has no target tokens")
    mean = total_nll / token_count
    return CausalEvaluationResult(total_nll, mean, math.exp(mean), token_count, len(windows), tokens.shape[0])


def reduce_causal_evaluation_results(
    results: tuple[CausalEvaluationResult, ...],
) -> CausalEvaluationResult:
    """Reduce disjoint evaluator shards using token-weighted sufficient statistics."""

    if not results:
        raise ValueError("causal evaluation reduction requires at least one shard")
    if any(
        not math.isfinite(result.total_negative_log_likelihood)
        or result.total_negative_log_likelihood < 0
        or result.token_count <= 0
        or result.window_count <= 0
        or result.sample_count <= 0
        for result in results
    ):
        raise ValueError("causal evaluation reduction contains an invalid shard")
    total_nll = math.fsum(result.total_negative_log_likelihood for result in results)
    token_count = sum(result.token_count for result in results)
    mean = total_nll / token_count
    return CausalEvaluationResult(
        total_nll,
        mean,
        math.exp(mean),
        token_count,
        sum(result.window_count for result in results),
        sum(result.sample_count for result in results),
    )
