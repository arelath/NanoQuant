import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from nanoquant.config.resolution import resolve_config
from nanoquant.config.schema import (
    ActivationStoreKind,
    ExecutorKind,
    MemoryPolicyConfig,
    MemoryPolicyMode,
    MemoryPolicyProfile,
    ModelConfig,
    ResourceLimitsConfig,
    RunConfig,
)
from nanoquant.config.validation import validate
from nanoquant.domain.resources import (
    MemoryPlanRevision,
    ResolvedMemoryPlan,
    ResourceAdmissionError,
    ResourceEnvelope,
    StageMemoryModel,
    geometric_batch_candidates,
    revise_memory_plan_after_oom,
    select_stage_execution_plan,
)
from nanoquant.infrastructure.resource_planning import build_resident_memory_plan, load_memory_plan, persist_memory_plan
from nanoquant.resident_quantization import _resident_config_hash
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    execute_resident_workflow,
    resident_request_from_config,
)
from tests.support.experiments import load_experiment


def _envelope(*, gpu_limit: int = 1_000, pinned_limit: int = 128) -> ResourceEnvelope:
    return ResourceEnvelope(
        device="cuda:0",
        gpu_total_bytes=1_000,
        gpu_free_bytes=1_000,
        gpu_process_allocated_bytes=0,
        gpu_process_reserved_bytes=0,
        gpu_hard_limit_bytes=gpu_limit,
        gpu_reserve_bytes=0,
        host_available_bytes=10_000,
        host_process_bytes=0,
        host_hard_limit_bytes=10_000,
        host_reserve_bytes=0,
        pinned_host_limit_bytes=pinned_limit,
        temporary_disk_free_bytes=10_000,
        temporary_disk_hard_limit_bytes=10_000,
        temporary_disk_reserve_bytes=0,
        observed_at="2026-07-19T00:00:00+00:00",
    )


def _model(stage: str = "tuning", maximum: int = 8) -> StageMemoryModel:
    return StageMemoryModel(
        stage,
        f"{stage}:fixture",
        fixed_gpu_bytes=100,
        gpu_bytes_per_row=100,
        fixed_host_bytes=200,
        host_bytes_per_row=10,
        pinned_bytes_per_prefetch_slot=64,
        temporary_disk_bytes=300,
        largest_indivisible_gpu_allocation_bytes=80,
        minimum_batch_size=1,
        maximum_batch_size=maximum,
    )


def test_adaptive_stage_selects_largest_safe_geometric_batch_and_bounds_prefetch() -> None:
    assert geometric_batch_candidates(8) == (8, 4, 2, 1)

    plan = select_stage_execution_plan(
        _model(),
        _envelope(gpu_limit=700),
        maximum_prefetch_batches=4,
        estimator_error_fraction=0,
        minimum_uncertainty_bytes=0,
        reusable_pool_fraction=0,
    )

    assert plan.batch_size == 4
    assert plan.prefetch_batches == 2
    assert plan.predicted_gpu_bytes == 500
    assert plan.predicted_pinned_host_bytes == 128
    assert plan.resized


def test_admission_fails_with_minimum_and_indivisible_allocation_diagnostics() -> None:
    with pytest.raises(ResourceAdmissionError, match=r"minimum batch 1.*largest indivisible"):
        select_stage_execution_plan(
            _model(),
            _envelope(gpu_limit=150),
            estimator_error_fraction=0,
            minimum_uncertainty_bytes=0,
        )


def test_oom_revision_reduces_physical_batch_without_algorithm_change() -> None:
    envelope = _envelope()
    setup = select_stage_execution_plan(
        _model("model_load", 1), envelope, estimator_error_fraction=0, minimum_uncertainty_bytes=0
    )
    tuning = select_stage_execution_plan(
        _model(),
        envelope,
        maximum_prefetch_batches=2,
        estimator_error_fraction=0,
        minimum_uncertainty_bytes=0,
    )
    plan = ResolvedMemoryPlan(
        1,
        1,
        "adaptive",
        "balanced",
        "sha256:request",
        envelope,
        "resident",
        "ram",
        "inputs",
        False,
        (setup, tuning),
        tuning.predicted_gpu_bytes + 50,
        tuning.predicted_host_bytes,
        tuning.predicted_pinned_host_bytes,
        tuning.predicted_temporary_disk_bytes,
    )

    revised, revision = revise_memory_plan_after_oom(
        plan,
        envelope,
        estimator_error_fraction=0,
        minimum_uncertainty_bytes=0,
    )

    assert revised.revision == 2
    assert revised.stage("tuning").batch_size == 4
    assert revised.stage("tuning").prefetch_batches == 2
    assert not revision.algorithm_changed
    assert revision.parent_revision == 1
    assert revision.action == "reduce_physical_batches"


def test_oom_revision_honors_recipe_fallback_authorization() -> None:
    plan = replace(_workflow_plan("sha256:policy", batch_size=8), activation_gpu_cache="both")

    revised, revision = revise_memory_plan_after_oom(
        plan,
        _envelope(),
        stage="factorized_tuning",
        allowed_actions=("move_activations_down_one_tier", "fail"),
        estimator_error_fraction=0,
        minimum_uncertainty_bytes=0,
    )

    assert revised.stage("tuning").batch_size == 8
    assert revised.activation_gpu_cache == "off"
    assert revision.action == "evict_activation_cache"


def test_memory_plan_artifact_round_trip_reuses_matching_request(tmp_path: Path) -> None:
    envelope = _envelope()
    stage = select_stage_execution_plan(
        _model(), envelope, estimator_error_fraction=0, minimum_uncertainty_bytes=0
    )
    plan = ResolvedMemoryPlan(
        1,
        3,
        "adaptive",
        "balanced",
        "sha256:request",
        envelope,
        "resident",
        "ram",
        "off",
        False,
        (stage,),
        stage.predicted_gpu_bytes,
        stage.predicted_host_bytes,
        stage.predicted_pinned_host_bytes,
        stage.predicted_temporary_disk_bytes,
    )

    reference = persist_memory_plan(plan, tmp_path)

    assert reference.artifact_type == "memory-plan"
    assert load_memory_plan(tmp_path, "sha256:request") == (plan, reference)
    assert load_memory_plan(tmp_path, "sha256:different") is None


def test_memory_policy_and_resource_limits_validate_and_round_trip() -> None:
    base = RunConfig(ModelConfig("fixture"))
    valid = replace(
        base,
        runtime=replace(
            base.runtime,
            resources=ResourceLimitsConfig(
                gpu_memory_gib=12,
                cpu_memory_gib=32,
                pinned_memory_gib=0.5,
                temporary_disk_gib=100,
                workspace_memory_gib=2,
            ),
            memory_policy=MemoryPolicyConfig(
                mode=MemoryPolicyMode.ADAPTIVE,
                profile=MemoryPolicyProfile.THROUGHPUT,
                maximum_stage_retries=2,
            ),
        ),
    )
    assert validate(valid) == ()

    invalid = replace(
        valid,
        runtime=replace(
            valid.runtime,
            resources=replace(valid.runtime.resources, gpu_memory_gib=-1),
            memory_policy=replace(valid.runtime.memory_policy, host_reserve_gib=-1, maximum_stage_retries=-1),
        ),
    )
    assert {issue.code for issue in validate(invalid)} == {"CFG072", "CFG074", "CFG075"}


def test_adaptive_resolution_leaves_executor_and_activation_tier_for_resource_planning() -> None:
    class Resolver:
        def resolve(self, source: str, revision: str | None) -> str:
            return revision or "pinned"

    base = RunConfig(ModelConfig("fixture"))
    adaptive = replace(
        base,
        runtime=replace(
            base.runtime,
            memory_policy=replace(base.runtime.memory_policy, mode=MemoryPolicyMode.ADAPTIVE),
        ),
    )

    resolved = resolve_config(adaptive, Resolver())

    assert resolved.runtime.executor is ExecutorKind.AUTO
    assert resolved.runtime.activations.kind is ActivationStoreKind.AUTO


def test_metadata_only_resident_plan_sizes_real_adapter_inventory(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config_payload = {
        "model_type": "llama",
        "num_hidden_layers": 1,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "max_position_embeddings": 32,
        "torch_dtype": "bfloat16",
    }
    (snapshot / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")
    tensors = {
        "model.embed_tokens.weight": torch.zeros((16, 4), dtype=torch.bfloat16),
        "model.norm.weight": torch.zeros(4, dtype=torch.bfloat16),
        "lm_head.weight": torch.zeros((16, 4), dtype=torch.bfloat16),
    }
    for name, shape in {
        "self_attn.q_proj": (4, 4),
        "self_attn.k_proj": (4, 4),
        "self_attn.v_proj": (4, 4),
        "self_attn.o_proj": (4, 4),
        "mlp.gate_proj": (8, 4),
        "mlp.up_proj": (8, 4),
        "mlp.down_proj": (4, 8),
    }.items():
        tensors[f"model.layers.0.{name}.weight"] = torch.zeros(shape, dtype=torch.bfloat16)
    save_file(tensors, snapshot / "model.safetensors")
    base = RunConfig(
        ModelConfig(
            "fixture/llama",
            revision="revision",
            tokenizer_revision="revision",
            sequence_length=8,
        )
    )
    config = replace(
        base,
        calibration=replace(base.calibration, sample_count=4),
        evaluation=replace(base.evaluation, inline_quality=False),
        runtime=replace(
            base.runtime,
            executor=ExecutorKind.RESIDENT,
            memory_policy=replace(
                base.runtime.memory_policy,
                mode=MemoryPolicyMode.ADAPTIVE,
                gpu_reserve_gib=0,
                host_reserve_gib=0,
                temporary_disk_reserve_gib=0,
            ),
            on_cuda_oom=("reduce_batch_size", "fail"),
        ),
    )
    gib = 2**30
    envelope = replace(
        _envelope(gpu_limit=8 * gib, pinned_limit=gib),
        gpu_total_bytes=8 * gib,
        gpu_free_bytes=8 * gib,
        host_available_bytes=16 * gib,
        host_hard_limit_bytes=16 * gib,
        temporary_disk_free_bytes=32 * gib,
        temporary_disk_hard_limit_bytes=32 * gib,
    )

    plan = build_resident_memory_plan(
        config,
        snapshot,
        tmp_path / "run",
        (4, 8),
        retain_completed_blocks=False,
        envelope=envelope,
    )

    assert plan.executor == "resident"
    assert plan.stage("model_load").model.fixed_gpu_bytes > 0
    assert plan.stage("block_forward").batch_size == 4
    assert plan.stage("calibration").batch_size == 1
    assert plan.peak_temporary_disk_bytes > 0


def _workflow_plan(request_hash: str, batch_size: int, revision: int = 1) -> ResolvedMemoryPlan:
    envelope = _envelope(gpu_limit=10_000)
    stages = tuple(
        select_stage_execution_plan(
            _model(name, 1 if name == "model_load" else batch_size),
            envelope,
            estimator_error_fraction=0,
            minimum_uncertainty_bytes=0,
        )
        for name in ("model_load", "calibration", "block_forward", "tuning", "post_block_refit")
    )
    return ResolvedMemoryPlan(
        1,
        revision,
        "adaptive",
        "balanced",
        request_hash,
        envelope,
        "resident",
        "ram",
        "off",
        True,
        stages,
        max(stage.predicted_gpu_bytes for stage in stages),
        max(stage.predicted_host_bytes for stage in stages),
        max(stage.predicted_pinned_host_bytes for stage in stages),
        max(stage.predicted_temporary_disk_bytes for stage in stages),
    )


def test_adaptive_plan_maps_concrete_batches_but_keeps_revision_identity_stable(tmp_path: Path) -> None:
    base = load_experiment(1).config
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            memory_policy=replace(base.runtime.memory_policy, mode=MemoryPolicyMode.ADAPTIVE),
            on_cuda_oom=("reduce_batch_size", "fail"),
        ),
    )
    tokens = torch.zeros((256, 8), dtype=torch.long)
    inputs = ResolvedResidentInputs(
        tmp_path / "snapshot",
        tmp_path / "output",
        tmp_path,
        tokens,
        tokens[:1],
    )
    first = resident_request_from_config(config, inputs, memory_plan=_workflow_plan("request", 8))
    revised = resident_request_from_config(config, inputs, memory_plan=_workflow_plan("request", 4, 2))

    assert first.block_forward_batch_size == 8
    assert first.tuning_microbatch_size == 8
    assert revised.block_forward_batch_size == 4
    assert revised.tuning_microbatch_size == 4
    assert _resident_config_hash(first) == _resident_config_hash(revised)


def test_workflow_retries_cuda_oom_with_one_persisted_lower_memory_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = load_experiment(1).config
    config = replace(
        base,
        distillation=replace(base.distillation, enabled=False),
        runtime=replace(
            base.runtime,
            memory_policy=replace(base.runtime.memory_policy, mode=MemoryPolicyMode.ADAPTIVE),
            on_cuda_oom=("reduce_batch_size", "fail"),
        ),
    )
    tokens = torch.zeros((256, 8), dtype=torch.long)
    inputs = ResolvedResidentInputs(
        tmp_path / "snapshot",
        tmp_path / "output",
        tmp_path,
        tokens,
        tokens[:1],
    )
    first = _workflow_plan("request", 8)
    second = _workflow_plan("request", 4, 2)
    references = (
        type("Reference", (), {"artifact_id": "sha256-" + "1" * 64})(),
        type("Reference", (), {"artifact_id": "sha256-" + "2" * 64})(),
    )
    monkeypatch.setattr(
        "nanoquant.resident_workflow._resolve_workflow_memory_plan",
        lambda *_args: (first, references[0]),
    )
    revision = MemoryPlanRevision(2, 1, "out_of_memory", None, "reduce_physical_batches", False, 8, 4)
    monkeypatch.setattr(
        "nanoquant.resident_workflow.revise_resident_memory_plan_after_oom",
        lambda *_args, **_kwargs: (second, revision, references[1]),
    )
    observed_batches: list[int] = []
    result = object()

    def quantize(request: Any) -> object:
        observed_batches.append(request.tuning_microbatch_size)
        if len(observed_batches) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return result

    monkeypatch.setattr("nanoquant.resident_workflow.run_resident_quantization", quantize)

    actual = execute_resident_workflow(config, inputs, ResidentExecutionOptions())

    assert observed_batches == [8, 4]
    assert actual.quantization is result
