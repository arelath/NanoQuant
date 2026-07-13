"""Versioned evaluators and token-accurate causal NLL/perplexity."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
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

    @property
    def semantic_key(self) -> str:
        return "sha256:" + hashlib.sha256(canonical_json(self).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CausalEvaluationRequest:
    token_ids: torch.Tensor
    attention_mask: torch.Tensor | None = None
    max_length: int = 2048
    stride: int = 2048
    prepend_bos_token_id: int | None = None
    append_eos_token_id: int | None = None
    batch_size: int = 1


@dataclass(frozen=True, slots=True)
class CausalEvaluationResult:
    total_negative_log_likelihood: float
    mean_negative_log_likelihood: float
    perplexity: float
    token_count: int
    window_count: int


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
    if request.attention_mask is not None and request.attention_mask.shape != tokens.shape:
        raise ValueError("attention mask shape does not match token IDs")
    windows: list[tuple[torch.Tensor, torch.Tensor]] = []
    for row in range(tokens.shape[0]):
        mask = None if request.attention_mask is None else request.attention_mask[row]
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
    return CausalEvaluationResult(total_nll, mean, math.exp(mean), token_count, len(windows))
