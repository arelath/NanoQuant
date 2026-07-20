import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from nanoquant.config.codec import to_dict
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
    revise_memory_plan_for_throughput,
    select_fastest_observed_batch,
    select_stage_execution_plan,
    throughput_batch_candidates,
)
from nanoquant.infrastructure.resource_planning import (
    build_resident_memory_plan,
    load_memory_plan,
    load_memory_plan_revision,
    persist_memory_plan,
    revise_resident_memory_plan_for_throughput,
)
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    _ensure_block_commit_disk_capacity,
    _resident_config_hash,
    _resident_manifest,
)
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
        pinned_bytes_per_row_per_prefetch_slot=8,
        temporary_disk_bytes=300,
        largest_indivisible_gpu_allocation_bytes=80,
        minimum_batch_size=1,
        maximum_batch_size=maximum,
    )


def test_adaptive_stage_selects_largest_safe_integer_batch_and_bounds_prefetch() -> None:
    assert geometric_batch_candidates(8) == (8, 4, 2, 1)

    plan = select_stage_execution_plan(
        _model(),
        _envelope(gpu_limit=700),
        maximum_prefetch_batches=4,
        estimator_error_fraction=0,
        minimum_uncertainty_bytes=0,
        reusable_pool_fraction=0,
    )

    assert plan.batch_size == 6
    assert plan.prefetch_batches == 2
    assert plan.predicted_gpu_bytes == 700
    assert plan.predicted_pinned_host_bytes == 96
    assert plan.resized


def test_throughput_selection_searches_safe_candidates_and_requires_material_gain() -> None:
    assert throughput_batch_candidates(19, 8) == (19, 9, 8, 4, 2, 1)

    assert select_fastest_observed_batch(
        ((19, 0.90), (9, 0.80), (8, 0.82), (4, 1.10)),
        baseline_batch=8,
    ) == 8
    assert select_fastest_observed_batch(
        ((19, 0.70), (9, 0.80), (8, 0.82), (4, 1.10)),
        baseline_batch=8,
    ) == 19


def test_throughput_revision_persists_measured_batch_without_algorithm_change() -> None:
    plan = _workflow_plan("sha256:throughput", batch_size=8)

    revised, revision = revise_memory_plan_for_throughput(plan, (("block_forward", 4),))

    assert revised.revision == plan.revision + 1
    assert revised.stage("block_forward").batch_size == 4
    assert revised.stage("block_forward").admitted_maximum_batch_size == 8
    assert revised.stage("block_forward").predicted_gpu_bytes == revised.stage(
        "block_forward"
    ).model.peak_gpu_bytes(4)
    assert revision.action == "select_measured_throughput_batches"
    assert not revision.algorithm_changed


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


def test_measured_throughput_revision_is_durable_for_resume(tmp_path: Path) -> None:
    plan = _workflow_plan("sha256:durable-throughput", batch_size=8)

    revised, revision, reference = revise_resident_memory_plan_for_throughput(
        plan,
        tmp_path,
        (("block_forward", 4),),
    )

    assert load_memory_plan(tmp_path, plan.request_hash) == (revised, reference)
    assert revision.parent_revision == plan.revision
    journal = (tmp_path / "state" / "memory-plan-revisions.jsonl").read_text(encoding="utf-8")
    assert "select_measured_throughput_batches" in journal


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
        calibration=replace(base.calibration, sample_count=9),
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
        (9, 8),
        retain_completed_blocks=False,
        envelope=envelope,
    )

    assert plan.executor == "resident"
    assert plan.stage("model_load").model.fixed_gpu_bytes > 0
    assert plan.stage("block_forward").batch_size == 9
    assert plan.stage("calibration").batch_size == 1
    assert plan.peak_temporary_disk_bytes > 0

    retained = build_resident_memory_plan(
        config,
        snapshot,
        tmp_path / "retained-run",
        (9, 8),
        retain_completed_blocks=True,
        envelope=envelope,
    )
    assert (
        retained.stage("post_block_refit").model.fixed_gpu_bytes
        > plan.stage("post_block_refit").model.fixed_gpu_bytes
    )


def _workflow_plan(request_hash: str, batch_size: int, revision: int = 1) -> ResolvedMemoryPlan:
    envelope = _envelope(gpu_limit=10_000)
    stages = tuple(
        select_stage_execution_plan(
            _model(name, 1 if name in {"model_load", "calibration"} else batch_size),
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
    assert _resident_manifest(first, "resident").config_hash == _resident_manifest(
        revised,
        "resident",
    ).config_hash


def test_memory_plan_revision_journal_loads_exact_revision(tmp_path: Path) -> None:
    journal = tmp_path / "state" / "memory-plan-revisions.jsonl"
    journal.parent.mkdir(parents=True)
    revisions = (
        MemoryPlanRevision(2, 1, "measured throughput autotune", None, "select_batches", False),
        MemoryPlanRevision(3, 2, "out_of_memory", "tuning", "reduce_batch_size", False, 4, 2),
    )
    journal.write_text(
        "".join(json.dumps(to_dict(revision)) + "\n" for revision in revisions),
        encoding="utf-8",
    )

    assert load_memory_plan_revision(tmp_path, 2) == revisions[0]
    assert load_memory_plan_revision(tmp_path, 3) == revisions[1]
    assert load_memory_plan_revision(tmp_path, 4) is None


def test_block_commit_rechecks_live_disk_pressure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_plan = _workflow_plan("disk-guard", 4)
    plan = replace(
        base_plan,
        envelope=replace(
            base_plan.envelope,
            temporary_disk_reserve_bytes=100,
        ),
    )
    request = ResidentQuantizationRequest(
        tmp_path / "snapshot",
        tmp_path,
        "fixture/model",
        "revision",
        ((1, 2),),
        device="cpu",
        memory_plan=plan,
    )
    teacher = torch.zeros((2, 4), dtype=torch.bfloat16)
    compressed = torch.zeros_like(teacher)
    required = teacher.numel() * teacher.element_size() * 2 + 64 * 2**20

    usage = type("DiskUsage", (), {"free": required + 100})()
    monkeypatch.setattr("nanoquant.resident_quantization.shutil.disk_usage", lambda _path: usage)
    assert _ensure_block_commit_disk_capacity(request, teacher, compressed) == (
        required,
        required + 100,
        100,
    )

    usage = type("DiskUsage", (), {"free": required + 99})()
    monkeypatch.setattr("nanoquant.resident_quantization.shutil.disk_usage", lambda _path: usage)
    with pytest.raises(ResourceAdmissionError, match="live safe capacity"):
        _ensure_block_commit_disk_capacity(request, teacher, compressed)


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
