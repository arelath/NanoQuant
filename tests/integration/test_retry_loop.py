import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from nanoquant.application.quantization_stages import (
    FactorizationAttemptStage,
    OutlierSelectionStage,
    ScaleFitStage,
)
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
from nanoquant.domain.seeds import logical_seed
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.events import JsonlEventSink
from nanoquant.infrastructure.resident_executor import Cancellation, ResidentExecutor
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    _commit_product_codebook_option_evidence,
    _load_product_codebook_option_evidence,
    _product_codebook_execution_hash,
    _product_codebook_screen_reuse_reason,
    _ProductCodebookOptionEvidence,
    _ProductCodebookProbeCandidate,
    _read_product_codebook_option_evidence,
    _run_resident_factorization_attempts,
)


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


def test_custom_factor_cost_replaces_the_same_custom_baseline(tmp_path: Path) -> None:
    plan, source, residual, context = _fixture(tmp_path)
    custom = lambda rank: BitCost(binary_factor_bits=rank * 10)  # noqa: E731
    plan = replace(
        plan,
        estimated_cost=custom(plan.rank) + BitCost(outlier_value_bits=13),
    )

    accepted = run_factorization_attempts(
        plan,
        source,
        residual,
        3,
        "custom-cost",
        BudgetState(1000, 0, 0),
        context,
        lambda _result, _attempts: None,
        FactorizationAttemptStage(ADMMConfig(outer_iterations=2, inner_iterations=1)),
        factor_cost=custom,
    )

    selected = next(attempt for attempt in accepted.attempts if attempt.accepted)
    assert accepted.actual_bit_cost == selected.bit_cost + BitCost(outlier_value_bits=13)


def test_full_rank_retry_adds_and_accounts_for_an_outlier_column(tmp_path: Path) -> None:
    plan, source, _residual, context = _fixture(tmp_path)
    plan = replace(
        plan,
        outliers=OutlierPlan("residual", 0, "int8", False),
        retry=RetryPolicy(2, 1.0, 0.0, None, 1, 1_000, 1),
    )
    request = ResidentQuantizationRequest(
        tmp_path / "snapshot",
        tmp_path / "run",
        "fixture/model",
        "revision",
        ((1, 2),),
        device="cpu",
        admm=ADMMConfig(outer_iterations=2, inner_iterations=1),
    )

    accepted, outliers, _fitted = _run_resident_factorization_attempts(
        plan,
        source,
        request,
        BudgetState(1_000, 0, 0),
        context,
        "config",
        FactorizationAttemptStage(request.admm),
        OutlierSelectionStage(
            residual_probe_iterations=1,
            residual_probe_inner_iterations=1,
        ),
        ScaleFitStage(request.scale_fit),
    )

    assert [attempt.outlier_count for attempt in accepted.attempts] == [0, 1]
    assert next(attempt for attempt in accepted.attempts if attempt.accepted).outlier_count == 1
    assert outliers.bit_cost.outlier_value_bits == 2 * 8 + 16
    assert outliers.bit_cost.outlier_index_bits == 1
    assert accepted.actual_bit_cost.outlier_value_bits == outliers.bit_cost.outlier_value_bits
    assert accepted.actual_bit_cost.outlier_index_bits == outliers.bit_cost.outlier_index_bits
    assert accepted.extra_retry_bits == outliers.bit_cost.total


def test_matching_final_screen_receipt_is_consumed_as_production_attempt_zero(
    tmp_path: Path,
) -> None:
    plan, source, _residual, context = _fixture(tmp_path)
    plan = replace(
        plan,
        retry=RetryPolicy(1, 0.0, None, None, plan.rank, 0),
    )
    request = ResidentQuantizationRequest(
        tmp_path / "snapshot",
        tmp_path / "run",
        "fixture/model",
        "revision",
        ((1, 2),),
        device="cpu",
        admm=ADMMConfig(outer_iterations=2, inner_iterations=1),
    )
    factor_stage = FactorizationAttemptStage(request.admm)
    outlier_stage = OutlierSelectionStage(
        residual_probe_iterations=1,
        residual_probe_inner_iterations=1,
    )
    scale_stage = ScaleFitStage(request.scale_fit)
    screened, outliers, fitted = _run_resident_factorization_attempts(
        plan,
        source,
        request,
        BudgetState(plan.estimated_cost.total, 0, 0),
        context,
        "screen",
        factor_stage,
        outlier_stage,
        scale_stage,
    )
    candidate = _ProductCodebookProbeCandidate(plan.layer, plan.rank, None)
    evidence = _ProductCodebookOptionEvidence(
        schema_version=3,
        probe_plan=ArtifactRef(
            "product-codebook-probe-plan",
            "sha256-" + "1" * 64,
            1,
        ),
        candidate=candidate,
        source_identity_hash=plan.source_weight.content_hash,
        source_weight_hash=source.content_hash,
        estimated_cost=plan.estimated_cost,
        measured_weighted_error=(
            screened.result.metrics.export_weighted_normalized_error
        ),
        measured_raw_error=screened.result.metrics.raw_normalized_error,
        factor_artifact=screened.result.factors.left_binary.artifact,
        logical_seed=logical_seed(
            request.seed,
            "factorize-attempt",
            plan.layer.block.index,
            plan.layer.path,
            0,
        ),
        outer_iterations=2,
        execution_hash=_product_codebook_execution_hash(
            request,
            request.admm,
            plan,
            candidate,
        ),
        factorization=screened.result,
        outliers=outliers,
        scale_fit=fitted,
        wall_seconds=screened.wall_seconds,
    )
    evidence_reference = _commit_product_codebook_option_evidence(
        request,
        evidence,
        cast(LocalArtifactStore, context.artifact_store),
    )
    artifact_store = cast(LocalArtifactStore, context.artifact_store)
    shutil.rmtree(artifact_store.path_for(evidence_reference.artifact_id))
    assert _load_product_codebook_option_evidence(
        request,
        evidence.probe_plan,
        artifact_store,
    ) == {}
    rebuilt_evidence = replace(evidence, wall_seconds=evidence.wall_seconds + 1.0)
    rebuilt_reference = _commit_product_codebook_option_evidence(
        request,
        rebuilt_evidence,
        artifact_store,
    )
    assert rebuilt_reference != evidence_reference
    loaded = _load_product_codebook_option_evidence(
        request,
        evidence.probe_plan,
        artifact_store,
    )
    assert loaded[evidence.receipt_key] == (rebuilt_reference, rebuilt_evidence)
    evidence_reference = rebuilt_reference
    evidence = _read_product_codebook_option_evidence(
        evidence_reference,
        artifact_store,
    )

    reused, reused_outliers, _reused_fit = _run_resident_factorization_attempts(
        plan,
        source,
        request,
        BudgetState(plan.estimated_cost.total, 0, 0),
        context,
        "production",
        factor_stage,
        outlier_stage,
        scale_stage,
        screening_evidence=evidence,
    )

    assert reused.result.factors.left_binary.artifact == evidence.factor_artifact
    assert reused.wall_seconds == 0
    assert reused_outliers == evidence.outliers
    assert _product_codebook_screen_reuse_reason(
        request,
        plan,
        replace(source, content_hash="changed"),
        evidence,
    ) == "source_weight_changed"
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event["name"] == "product_codebook_probe.production_receipt_consumed"
        for event in events
    )
