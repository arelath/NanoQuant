from dataclasses import replace

from nanoquant.application.fork import plan_fork
from nanoquant.config.schema import ModelConfig, RunConfig


def test_factorization_fork_reuses_upstream_and_invalidates_downstream() -> None:
    parent = RunConfig(ModelConfig("fixture", revision="rev", tokenizer_revision="tok"))
    candidate = replace(parent, factorization=replace(parent.factorization, implementation="new-factorizer"))
    plan = plan_fork("parent", parent, candidate)
    assert plan.fork_from_stage == "quantize"
    assert "factorization.implementation" in plan.changed_paths
    actions = {decision.stage: decision.action for decision in plan.stages}
    assert actions["plan"] == "reuse" and actions["quantize"] == "invalidate" and actions["evaluate"] == "invalidate"


def test_presentation_only_fork_reuses_computation() -> None:
    parent = RunConfig(ModelConfig("fixture"))
    candidate = replace(parent, intent=replace(parent.intent, purpose="new explanation"))
    plan = plan_fork("parent", parent, candidate)
    assert plan.fork_from_stage == "report"
    assert all(decision.action == "reuse" for decision in plan.stages[:-1])


def test_model_change_invalidates_every_stage() -> None:
    parent = RunConfig(ModelConfig("one"))
    candidate = replace(parent, model=replace(parent.model, source="two"))
    plan = plan_fork("parent", parent, candidate)
    assert plan.fork_from_stage == "resolve-source"
    assert all(decision.action == "invalidate" for decision in plan.stages)
