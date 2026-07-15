from __future__ import annotations

from dataclasses import dataclass

import pytest

from nanoquant.application.evaluation import EvaluatorRegistry, EvaluatorSpec, GatePolicy, GateRule
from nanoquant.application.evaluation_campaign import (
    EvaluationTierPlan,
    LayerReplayEvidence,
    run_evaluation_campaign,
)


@dataclass(frozen=True)
class _Result:
    metric: str
    value: float


def _campaign(values: dict[str, float]):
    registry = EvaluatorRegistry()
    calls: list[str] = []
    plans = []
    for tier in ("quick", "standard", "full"):
        specification = EvaluatorSpec(f"{tier}-fixture", "1", tier)

        def evaluate(request: object, *, selected: str = tier) -> _Result:
            assert request == selected
            calls.append(selected)
            return _Result(f"{selected}_score", values[selected])

        registry.register(specification, evaluate)
        plans.append(
            EvaluationTierPlan(
                tier,
                ((specification.name, specification.version, tier),),
                GatePolicy(f"{tier}-gate", "1", (GateRule(f"{tier}_score", minimum=0.5),)),
            )
        )
    replay = LayerReplayEvidence("captured-layer", "1", "sha256:layer", (("loss", 0.1),), True)

    result = run_evaluation_campaign(
        candidate="candidate",
        baseline="baseline",
        layer_replay=replay,
        registry=registry,
        plans=tuple(plans),
        extract_metrics=lambda _spec, result: ((result.metric, result.value),),
    )
    return result, calls


def test_campaign_progresses_from_layer_replay_through_full() -> None:
    result, calls = _campaign({"quick": 0.8, "standard": 0.7, "full": 0.6})

    assert result.passed
    assert result.outcome == "full-promotion"
    assert result.completed_tier == "full"
    assert calls == ["quick", "standard", "full"]
    assert [tier.decision.outcome for tier in result.tiers] == ["promotion"] * 3
    assert len(result.semantic_key) == 71
    assert result.tiers[0].evaluators[0].result == {"metric": "quick_score", "value": 0.8}


def test_campaign_stops_before_later_tiers_on_rejection() -> None:
    result, calls = _campaign({"quick": 0.8, "standard": 0.4, "full": 0.9})

    assert not result.passed
    assert result.outcome == "standard-rejection"
    assert result.completed_tier == "standard"
    assert calls == ["quick", "standard"]
    assert "standard-tier regression" in result.recommended_next_action


def test_campaign_rejects_layer_replay_without_running_evaluators() -> None:
    registry = EvaluatorRegistry()
    plans = tuple(
        EvaluationTierPlan(
            tier,
            ((f"{tier}-fixture", "1", None),),
            GatePolicy(tier, "1", (GateRule("score", minimum=0),)),
        )
        for tier in ("quick", "standard", "full")
    )
    result = run_evaluation_campaign(
        candidate="candidate",
        baseline="baseline",
        layer_replay=LayerReplayEvidence(
            "captured-layer", "1", "sha256:layer", (("loss", 9.0),), False
        ),
        registry=registry,
        plans=plans,
        extract_metrics=lambda _spec, _result: (),
    )

    assert result.outcome == "layer-replay-rejection"
    assert result.tiers == ()


def test_campaign_requires_ordered_complete_plans_and_exact_tier_requests() -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        LayerReplayEvidence("captured-layer", "1", "sha256:layer", (("loss", float("nan")),), True)
    with pytest.raises(ValueError, match="ordered quick, standard, full"):
        run_evaluation_campaign(
            candidate="candidate",
            baseline="baseline",
            layer_replay=LayerReplayEvidence(
                "captured-layer", "1", "sha256:layer", (("loss", 0.1),), True
            ),
            registry=EvaluatorRegistry(),
            plans=(),
            extract_metrics=lambda _spec, _result: (),
        )

    registry = EvaluatorRegistry()
    for tier in ("quick", "standard", "full"):
        registry.register(EvaluatorSpec(f"{tier}-registered", "1", tier), lambda request: request)
    plans = tuple(
        EvaluationTierPlan(
            tier,
            ((f"{tier}-different", "1", tier),),
            GatePolicy(tier, "1", (GateRule("score", minimum=0),)),
        )
        for tier in ("quick", "standard", "full")
    )
    with pytest.raises(ValueError, match="requests differ"):
        run_evaluation_campaign(
            candidate="candidate",
            baseline="baseline",
            layer_replay=LayerReplayEvidence(
                "captured-layer", "1", "sha256:layer", (("loss", 0.1),), True
            ),
            registry=registry,
            plans=plans,
            extract_metrics=lambda _spec, _result: (),
        )
