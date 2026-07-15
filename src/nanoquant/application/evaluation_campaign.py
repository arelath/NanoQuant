"""Ordered layer-replay-to-full evaluation campaigns."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

from nanoquant.application.evaluation import (
    EvaluatorRegistry,
    EvaluatorSpec,
    GateDecision,
    GatePolicy,
    evaluate_gate,
)
from nanoquant.config.codec import canonical_json, to_dict

CampaignMetricExtractor = Callable[[EvaluatorSpec, object], tuple[tuple[str, float], ...]]


def _semantic_key(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(to_dict(value)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LayerReplayEvidence:
    name: str
    version: str
    artifact_id: str
    metrics: tuple[tuple[str, float], ...]
    passed: bool

    def __post_init__(self) -> None:
        if not self.name or not self.version or not self.artifact_id:
            raise ValueError("layer replay identity and artifact are required")
        names = [name for name, _value in self.metrics]
        if not self.metrics or any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("layer replay metrics must be non-empty and unique")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for _name, value in self.metrics
        ):
            raise ValueError("layer replay metrics must be finite numbers")
        if type(self.passed) is not bool:
            raise ValueError("layer replay pass state must be boolean")
        canonical_json(self.metrics)

    @property
    def semantic_key(self) -> str:
        return _semantic_key(self)


@dataclass(frozen=True, slots=True)
class EvaluationTierPlan:
    tier: str
    requests: tuple[tuple[str, str, object], ...]
    policy: GatePolicy

    def __post_init__(self) -> None:
        if self.tier not in ("quick", "standard", "full"):
            raise ValueError(f"campaign tier is unsupported: {self.tier}")
        keys = [(name, version) for name, version, _request in self.requests]
        if not keys or any(not name or not version for name, version in keys):
            raise ValueError("campaign tier requests require evaluator identities")
        if len(keys) != len(set(keys)):
            raise ValueError("campaign tier evaluator requests must be unique")


@dataclass(frozen=True, slots=True)
class CampaignEvaluatorResult:
    name: str
    version: str
    tier: str
    specification_key: str
    result_key: str
    result: object


@dataclass(frozen=True, slots=True)
class EvaluationTierResult:
    tier: str
    evaluators: tuple[CampaignEvaluatorResult, ...]
    metrics: tuple[tuple[str, float], ...]
    decision: GateDecision


@dataclass(frozen=True, slots=True)
class EvaluationCampaignResult:
    candidate: str
    baseline: str
    layer_replay: LayerReplayEvidence
    tiers: tuple[EvaluationTierResult, ...]
    completed_tier: str
    outcome: str
    recommended_next_action: str

    @property
    def passed(self) -> bool:
        return self.outcome == "full-promotion"

    @property
    def semantic_key(self) -> str:
        return _semantic_key(self)


def _recommend(tier: str, decision: GateDecision) -> str:
    if decision.outcome == "rejection":
        return f"Diagnose the recorded {tier}-tier regression before starting another evaluation tier."
    if decision.outcome == "inconclusive":
        return f"Resolve the missing or non-finite {tier}-tier evidence before promotion."
    next_tier = {"quick": "standard", "standard": "full"}.get(tier)
    return (
        "Candidate passed the complete evaluation campaign; proceed to migration/release qualification."
        if next_tier is None
        else f"Run the {next_tier} tier on the same immutable candidate."
    )


def run_evaluation_campaign(
    *,
    candidate: str,
    baseline: str,
    layer_replay: LayerReplayEvidence,
    registry: EvaluatorRegistry,
    plans: tuple[EvaluationTierPlan, ...],
    extract_metrics: CampaignMetricExtractor,
) -> EvaluationCampaignResult:
    """Execute exact tier-local evaluators and stop at the first failed promotion."""

    if not candidate or not baseline:
        raise ValueError("evaluation campaign candidate and baseline are required")
    if tuple(plan.tier for plan in plans) != ("quick", "standard", "full"):
        raise ValueError("evaluation campaign plans must be ordered quick, standard, full")
    if not layer_replay.passed:
        return EvaluationCampaignResult(
            candidate,
            baseline,
            layer_replay,
            (),
            "layer-replay",
            "layer-replay-rejection",
            "Diagnose the captured layer replay before starting quick evaluation.",
        )

    tier_results: list[EvaluationTierResult] = []
    for plan in plans:
        requests = {(name, version): request for name, version, request in plan.requests}
        selected = registry.specifications_for_tier(plan.tier, cumulative=False)
        requested_keys = set(requests)
        selected_keys = {(spec.name, spec.version) for spec in selected}
        if requested_keys != selected_keys:
            raise ValueError(
                f"campaign {plan.tier} requests differ from registered tier evaluators: "
                f"missing={sorted(selected_keys - requested_keys)}, "
                f"unexpected={sorted(requested_keys - selected_keys)}"
            )
        evaluated = registry.evaluate_tier(plan.tier, requests, cumulative=False)
        outcomes: list[CampaignEvaluatorResult] = []
        metrics: list[tuple[str, float]] = []
        for specification, result in evaluated:
            payload = to_dict(result)
            outcomes.append(
                CampaignEvaluatorResult(
                    specification.name,
                    specification.version,
                    specification.tier,
                    specification.semantic_key,
                    _semantic_key(payload),
                    payload,
                )
            )
            metrics.extend(extract_metrics(specification, result))
        metric_names = [name for name, _value in metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError(f"campaign {plan.tier} evaluators produced duplicate metrics")
        decision = evaluate_gate(plan.policy, tuple(metrics))
        tier_result = EvaluationTierResult(
            plan.tier,
            tuple(outcomes),
            tuple(metrics),
            decision,
        )
        tier_results.append(tier_result)
        if decision.outcome != "promotion":
            return EvaluationCampaignResult(
                candidate,
                baseline,
                layer_replay,
                tuple(tier_results),
                plan.tier,
                f"{plan.tier}-{decision.outcome}",
                _recommend(plan.tier, decision),
            )

    return EvaluationCampaignResult(
        candidate,
        baseline,
        layer_replay,
        tuple(tier_results),
        "full",
        "full-promotion",
        _recommend("full", tier_results[-1].decision),
    )
