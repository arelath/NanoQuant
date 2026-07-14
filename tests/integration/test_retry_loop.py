import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nanoquant.application.quantization_stages import FactorizationAttemptStage
from nanoquant.application.retry_loop import run_factorization_attempts
from nanoquant.application.stages import StageContext
from nanoquant.config.schema import ADMMConfig
from nanoquant.domain.models import (
    ArtifactRef,
    BitCost,
    BlockId,
    LayerId,
    LayerPlan,
    ObjectiveSpec,
    OutlierPlan,
    RetryPolicy,
    SourceTensor,
    TensorId,
    TensorSpec,
)
from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.runs import BudgetState
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.events import JsonlEventSink
from nanoquant.infrastructure.resident_executor import Cancellation, ResidentExecutor
from nanoquant.infrastructure.tensor_store import LocalTensorStore


def _fixture(tmp_path: Path) -> tuple[LayerPlan, object, object, StageContext]:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    tensors = LocalTensorStore(artifacts)
    values = tensors.put(
        "retry-fixture",
        {
            "weight": torch.tensor([[1.0, -2.0], [2.0, 1.0]]),
            "input": torch.ones(2),
            "output": torch.ones(2),
        },
    )
    layer = LayerId(BlockId(0), "linear")
    artifact = ArtifactRef("calibration", "sha256-" + "0" * 64, 1)
    objective = ObjectiveSpec(
        1,
        layer,
        "diagonal",
        values["input"],
        values["output"],
        None,
        0.01,
        "target_weighted_norm_squared",
        None,
        artifact,
    )
    source = SourceTensor(
        TensorId(layer, "weight"), "linear.weight", "fixture", TensorSpec((2, 2), "float32"), "source-hash"
    )
    plan = LayerPlan(
        1,
        layer,
        source,
        1,
        1,
        2,
        objective,
        OutlierPlan("none", 0, "float16", True),
        RetryPolicy(2, 1.0, 0.0, None, 2, 1000),
        factor_bit_cost(2, 2, 1),
    )
    context = StageContext(
        "run", ResidentExecutor(), artifacts, tensors, JsonlEventSink(tmp_path / "events.jsonl", "run"), Cancellation()
    )
    return plan, values["weight"], values["weight"], context


def test_retry_loop_commits_once_and_updates_budget_after_acceptance(tmp_path: Path) -> None:
    plan, source, residual, context = _fixture(tmp_path)
    plan = replace(plan, estimated_cost=plan.estimated_cost + BitCost(outlier_value_bits=13))
    commits = []
    initial = BudgetState(1000, 0, 0)
    accepted = run_factorization_attempts(
        plan,
        source,
        residual,
        3,
        "config",
        initial,
        context,
        lambda result, attempts: commits.append((result, attempts)),
        FactorizationAttemptStage(ADMMConfig(outer_iterations=2, inner_iterations=1)),
    )
    assert len(accepted.attempts) == 2
    assert sum(attempt.accepted for attempt in accepted.attempts) == 1
    assert len(commits) == 1
    assert accepted.budget.accepted_bits > 0
    selected = next(attempt for attempt in accepted.attempts if attempt.accepted)
    base_factor_cost = factor_bit_cost(2, 2, plan.rank)
    assert accepted.extra_retry_bits == max(0, selected.bit_cost.total - base_factor_cost.total)
    assert accepted.actual_bit_cost.outlier_value_bits == 13
    assert accepted.budget.accepted_bits == accepted.actual_bit_cost.total
    assert initial == BudgetState(1000, 0, 0)
    assert accepted.result.convergence.iterations_completed <= 2
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    decisions = [event["fields"] for event in events if event["name"] == "factorization.retry_decision"]
    assert len(decisions) == 2
    assert {
        "rank",
        "weighted_error",
        "raw_error",
        "weighted_threshold",
        "raw_threshold",
        "retry_score",
        "attempt_bits",
        "available_extra_bits",
        "retry_bits_spent",
        "action",
    } <= decisions[0].keys()


def test_failed_accepted_layer_commit_does_not_mutate_budget(tmp_path: Path) -> None:
    plan, source, residual, context = _fixture(tmp_path)
    initial = BudgetState(1000, 0, 0)

    def fail_commit(result: object, attempts: object) -> None:
        raise OSError("commit failed")

    with pytest.raises(IOError, match="commit failed"):
        run_factorization_attempts(
            plan,
            source,
            residual,
            3,
            "config",
            initial,
            context,
            fail_commit,
            FactorizationAttemptStage(ADMMConfig(outer_iterations=2, inner_iterations=1)),
        )
    assert initial.retry_bits_spent == 0 and initial.accepted_bits == 0
