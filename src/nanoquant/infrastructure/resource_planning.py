"""Host inventory, placement selection, and preflight resource refusal."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch

from nanoquant.config.codec import config_hash, from_dict, to_dict
from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    ExecutorKind,
    MemoryPolicyConfig,
    MemoryPolicyMode,
    MemoryPolicyProfile,
    ResourceLimitsConfig,
    RunConfig,
)
from nanoquant.domain.models import ArtifactRef, ArtifactTypes, ModelInventory, TensorSpec
from nanoquant.domain.resources import (
    MemoryPlanRevision,
    ResolvedMemoryPlan,
    ResourceAdmissionError,
    ResourceComponents,
    ResourceEnvelope,
    ResourceMargins,
    ResourcePlan,
    StageExecutionPlan,
    StageMemoryModel,
    gpu_capacity_bytes,
    revise_memory_plan_after_oom,
    revise_memory_plan_for_throughput,
    select_stage_execution_plan,
)
from nanoquant.domain.stages import HostInventory
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.resource_usage import process_memory_snapshot
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource


class InsufficientResourcesError(RuntimeError):
    code = "RES001"


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _available_host_memory() -> int:
    if os.name != "nt":
        sysconf = cast(Any, os).sysconf
        page_size = sysconf("SC_PAGE_SIZE")
        available_pages = sysconf("SC_AVPHYS_PAGES")
        return int(page_size * available_pages)
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys)


def inspect_host(temporary_directory: str | Path, device: str = "cuda:0") -> HostInventory:
    gpu_bytes = 0
    if device.startswith("cuda") and torch.cuda.is_available():
        with torch.cuda.device(device):
            gpu_bytes, _total = torch.cuda.mem_get_info()
    disk_bytes = shutil.disk_usage(Path(temporary_directory).resolve()).free
    return HostInventory(_available_host_memory(), int(gpu_bytes), disk_bytes)


def _gib(value: float | None) -> int | None:
    return None if value is None else int(value * 2**30)


def inspect_resource_envelope(
    temporary_directory: str | Path,
    device: str,
    limits: ResourceLimitsConfig,
    policy: MemoryPolicyConfig,
) -> ResourceEnvelope:
    """Inspect process-aware resources and apply explicit run ceilings/reserves."""

    host_available = _available_host_memory()
    process = process_memory_snapshot()
    gpu_total = 0
    gpu_free = 0
    gpu_allocated = 0
    gpu_reserved = 0
    if device.startswith("cuda") and torch.cuda.is_available():
        with torch.cuda.device(device):
            gpu_free, gpu_total = (int(value) for value in torch.cuda.mem_get_info(device))
            gpu_allocated = int(torch.cuda.memory_allocated(device))
            gpu_reserved = int(torch.cuda.memory_reserved(device))
    gpu_limit = min(gpu_total, _gib(limits.gpu_memory_gib) or gpu_total) if gpu_total else 0
    physical_host_limit = process.working_set_bytes + host_available
    host_limit = min(physical_host_limit, _gib(limits.cpu_memory_gib) or physical_host_limit)
    disk_free = shutil.disk_usage(Path(temporary_directory).resolve()).free
    disk_limit = min(disk_free, _gib(limits.temporary_disk_gib) or disk_free)
    return ResourceEnvelope(
        device=device,
        gpu_total_bytes=gpu_total,
        gpu_free_bytes=gpu_free,
        gpu_process_allocated_bytes=gpu_allocated,
        gpu_process_reserved_bytes=max(gpu_reserved, gpu_allocated),
        gpu_hard_limit_bytes=gpu_limit,
        gpu_reserve_bytes=int(policy.gpu_reserve_gib * 2**30),
        host_available_bytes=host_available,
        host_process_bytes=process.working_set_bytes,
        host_hard_limit_bytes=host_limit,
        host_reserve_bytes=int(policy.host_reserve_gib * 2**30),
        pinned_host_limit_bytes=int(limits.pinned_memory_gib * 2**30),
        temporary_disk_free_bytes=disk_free,
        temporary_disk_hard_limit_bytes=disk_limit,
        temporary_disk_reserve_bytes=int(policy.temporary_disk_reserve_gib * 2**30),
        observed_at=datetime.now(timezone.utc).isoformat(),
    )


_DTYPE_BYTES = {
    "bool": 1,
    "u8": 1,
    "uint8": 1,
    "i8": 1,
    "int8": 1,
    "f16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "i16": 2,
    "int16": 2,
    "f32": 4,
    "float32": 4,
    "i32": 4,
    "int32": 4,
    "f64": 8,
    "float64": 8,
    "i64": 8,
    "int64": 8,
}


def _tensor_bytes(spec: TensorSpec) -> int:
    try:
        element_bytes = _DTYPE_BYTES[spec.dtype.lower().removeprefix("torch.")]
    except KeyError as exc:
        raise ValueError(f"unsupported source tensor dtype for resource planning: {spec.dtype}") from exc
    elements = 1
    for dimension in spec.shape:
        elements *= dimension
    return elements * element_bytes


def _inventory_bytes(inventory: ModelInventory) -> tuple[int, int, int, int]:
    by_key = {
        tensor.source_key: _tensor_bytes(tensor.spec)
        for tensor in (
            *inventory.shared_tensors,
            *(tensor for block in inventory.blocks for tensor in block.source_tensors),
        )
    }
    block_bytes = tuple(
        sum(_tensor_bytes(tensor.spec) for tensor in block.source_tensors) for block in inventory.blocks
    )
    layer_bytes = tuple(
        _tensor_bytes(layer.weight.spec) for block in inventory.blocks for layer in block.quantizable_layers
    )
    return (
        sum(by_key.values()),
        sum(_tensor_bytes(tensor.spec) for tensor in inventory.shared_tensors),
        max(block_bytes, default=0),
        max(layer_bytes, default=0),
    )


def _text_dimension(config: dict[str, object], name: str, default: int) -> int:
    nested = config.get("text_config")
    source = cast(dict[str, object], nested) if isinstance(nested, dict) else config
    value = source.get(name, default)
    return int(value) if isinstance(value, (int, float)) else default


def _policy_parameters(profile: MemoryPolicyProfile) -> tuple[float, int, float]:
    if profile is MemoryPolicyProfile.CONSERVATIVE:
        return 0.35, 2 * 2**30, 0.50
    if profile is MemoryPolicyProfile.THROUGHPUT:
        return 0.15, 768 * 2**20, 1.00
    # Retained 270M, Gemma 1B, Llama 1B, and Gemma 4B runs show that
    # allocator/loss-snapshot transients exceed a shape-only live-tensor model
    # by 0.4-1.3 GiB. Keep that measured floor until stage observations have
    # enough coverage to safely learn a narrower signature-specific margin.
    return 0.25, 1280 * 2**20, 0.75


def _select(
    model: StageMemoryModel,
    envelope: ResourceEnvelope,
    policy: MemoryPolicyConfig,
    *,
    maximum_prefetch_batches: int = 0,
) -> StageExecutionPlan:
    error, uncertainty, reusable = _policy_parameters(policy.profile)
    return select_stage_execution_plan(
        model,
        envelope,
        maximum_prefetch_batches=maximum_prefetch_batches,
        estimator_error_fraction=error,
        minimum_uncertainty_bytes=uncertainty,
        reusable_pool_fraction=reusable,
    )


def build_resident_memory_plan(
    config: RunConfig,
    snapshot: str | Path,
    output: str | Path,
    token_shape: tuple[int, ...],
    *,
    retain_completed_blocks: bool,
    envelope: ResourceEnvelope | None = None,
    revision: int = 1,
) -> ResolvedMemoryPlan:
    """Build a metadata-only, stage-aware plan for the production resident composition."""

    if len(token_shape) != 2 or any(dimension <= 0 for dimension in token_shape):
        raise ValueError("resident memory planning requires a positive [samples, tokens] shape")
    source = SafetensorsModelSource(
        snapshot,
        source=config.model.source,
        revision=str(config.model.revision),
        verify_hashes=False,
    )
    checkpoint = source.inventory()
    adapter = adapter_for_config(checkpoint.config)
    inventory = adapter.model_inventory(source)
    model_bytes, _shared_bytes, active_block_bytes, largest_layer_bytes = _inventory_bytes(inventory)
    samples, sequence_length = token_shape
    hidden = _text_dimension(checkpoint.config, "hidden_size", 1)
    intermediate = _text_dimension(checkpoint.config, "intermediate_size", hidden * 4)
    activation_stream_bytes = samples * sequence_length * hidden * 2
    rolling_commit_bytes = activation_stream_bytes * 4
    packed_bytes = max(1, int(model_bytes * config.allocation.target_bpw / 16.0))
    # The resident research artifacts retain trainable latents, binary factors,
    # scale/outlier state, metrics, and checkpoint generations rather than only
    # the final logical BPW payload. The pinned Gemma 1B evidence is close to ten
    # source-model bytes for non-activation artifacts, so use that measured upper
    # baseline until per-artifact estimators replace it.
    committed_bytes = model_bytes * 10
    scratch_bytes = active_block_bytes * 4
    disk_bytes = rolling_commit_bytes + packed_bytes + committed_bytes + scratch_bytes
    envelope = envelope or inspect_resource_envelope(
        output,
        config.runtime.compute_device,
        config.runtime.resources,
        config.runtime.memory_policy,
    )
    cache_reserve_bytes = int(config.runtime.activations.gpu_reserve_gib * 2**30)
    if cache_reserve_bytes > envelope.gpu_reserve_bytes:
        envelope = replace(envelope, gpu_reserve_bytes=cache_reserve_bytes)

    # Conservative per-sample activation models. They intentionally include block internals,
    # loss temporaries, gradients, and the two-slot staging protocol.
    forward_per_sample = sequence_length * 2 * (hidden * 8 + intermediate * 3)
    if adapter.attention_implementation == "eager":
        resolved_model_config = adapter.definition.config_factory(checkpoint.config)
        attention_heads = int(getattr(resolved_model_config, "num_attention_heads", 1))
        # Eager Gemma attention materializes global FP32 attention scores and
        # softmax/dropout temporaries. Real largest-block probes bound the base
        # workspace at 3.125 BF16-equivalent planes, rising with GQA query
        # expansion (q_out / hidden); 270M's 1.6x expansion requires four.
        attention_score_bytes = attention_heads * sequence_length * sequence_length * 2
        query_outputs = (
            layer.weight.spec.shape[0]
            for block in inventory.blocks
            for layer in block.quantizable_layers
            if layer.layer.path == "self_attn.q_proj"
        )
        query_output = max(query_outputs, default=hidden)
        base_workspace = attention_score_bytes * 25 // 8
        expanded_workspace = attention_score_bytes * 5 * query_output // (2 * hidden)
        forward_per_sample += max(base_workspace, expanded_workspace)
    tuning_per_sample = sequence_length * 2 * (hidden * 16 + intermediate * 8)
    factor_workspace = largest_layer_bytes * 8
    workspace_limit = _gib(config.runtime.resources.workspace_memory_gib)
    if workspace_limit is not None and factor_workspace > workspace_limit:
        raise ResourceAdmissionError(
            f"RES001 factorization workspace requires {factor_workspace} bytes but configured limit is "
            f"{workspace_limit}"
        )
    fixed_mode = config.runtime.memory_policy.mode is MemoryPolicyMode.FIXED
    tuning_logical_max = max(
        config.block_tuning.non_factorized.loop.batch_size,
        config.block_tuning.factorized.loop.batch_size,
    )
    refit_logical_batch = (
        config.block_tuning.post_block_refit.batch_size or config.block_tuning.factorized.loop.batch_size
    )
    if fixed_mode:
        forward_max = min(samples, config.runtime.block_forward_batch_size)
        tuning_max = min(samples, config.block_tuning.microbatch_size or tuning_logical_max, tuning_logical_max)
        refit_max = min(
            samples,
            config.block_tuning.microbatch_size or refit_logical_batch,
            refit_logical_batch,
        )
    else:
        # Adaptive mode replaces host-specific physical batch choices. Logical
        # optimizer batches and the available sample count remain hard semantic
        # bounds; the resource controller chooses the physical subdivision.
        forward_max = samples
        tuning_max = min(samples, tuning_logical_max)
        refit_max = min(samples, refit_logical_batch)
    calibration_max = min(samples, config.calibration.batch_size)
    # One bounded prefetch generation owns two alternating host slots, each
    # containing an input and target BF16 batch row.
    pinned_bytes_per_row = sequence_length * hidden * 8
    largest_allocation_bytes = max(largest_layer_bytes, factor_workspace // 8)
    requested = config.runtime.executor
    if requested not in {ExecutorKind.AUTO, ExecutorKind.RESIDENT, ExecutorKind.CPU_OFFLOAD}:
        raise ResourceAdmissionError(
            f"RES001 resident workflow cannot execute planned executor {requested.value!r}"
        )

    def candidate_stage_plans(executor: ExecutorKind) -> tuple[StageExecutionPlan, ...]:
        if executor is ExecutorKind.CPU_OFFLOAD and (config.evaluation.inline_quality or config.distillation.enabled):
            raise ResourceAdmissionError(
                "RES001 cpu_offload cannot provide inline quality or model-level distillation until teacher "
                "streaming is implemented"
            )
        setup_model = StageMemoryModel(
            "model_load",
            f"model-load:{inventory.model.config_hash}:{executor.value}",
            model_bytes if executor is ExecutorKind.RESIDENT else active_block_bytes,
            0,
            (0 if executor is ExecutorKind.RESIDENT else model_bytes) + activation_stream_bytes * 2,
            0,
            0,
            disk_bytes,
            largest_layer_bytes,
            1,
            1,
            "conservative",
        )
        # Resident execution keeps the complete source model live. Inline full-model
        # evaluation additionally retains dense BF16 factor tensors whose element
        # count is approximately target_bpw times the source parameter count.
        retained_factor_bytes = (
            int(model_bytes * config.allocation.target_bpw)
            if executor is ExecutorKind.RESIDENT and retain_completed_blocks
            else 0
        )
        execution_fixed = (
            model_bytes + retained_factor_bytes
            if executor is ExecutorKind.RESIDENT
            else active_block_bytes
        )
        host_fixed = (model_bytes if executor is ExecutorKind.CPU_OFFLOAD else 0) + activation_stream_bytes * 2
        models = (
            setup_model,
            StageMemoryModel(
                "calibration",
                f"calibration:{inventory.model.config_hash}:{config.calibration.method.value}:{sequence_length}",
                model_bytes if executor is ExecutorKind.RESIDENT else active_block_bytes,
                tuning_per_sample * 2,
                host_fixed,
                0,
                pinned_bytes_per_row,
                disk_bytes,
                largest_allocation_bytes,
                minimum_batch_size=(
                    calibration_max
                    if fixed_mode or config.calibration.method.value in {"online_fisher", "two_phase_fisher"}
                    else 1
                ),
                maximum_batch_size=calibration_max,
                confidence="conservative",
            ),
            StageMemoryModel(
                "block_forward",
                f"block-forward:{inventory.model.config_hash}:{hidden}:{intermediate}:{sequence_length}",
                execution_fixed,
                forward_per_sample,
                host_fixed,
                0,
                pinned_bytes_per_row,
                disk_bytes,
                largest_allocation_bytes,
                minimum_batch_size=forward_max if fixed_mode else 1,
                maximum_batch_size=forward_max,
                confidence="static",
            ),
            StageMemoryModel(
                "tuning",
                f"tuning:{inventory.model.config_hash}:{hidden}:{intermediate}:{sequence_length}",
                execution_fixed + factor_workspace + largest_layer_bytes * 2,
                tuning_per_sample,
                host_fixed,
                0,
                pinned_bytes_per_row,
                disk_bytes,
                largest_allocation_bytes,
                minimum_batch_size=tuning_max if fixed_mode else 1,
                maximum_batch_size=tuning_max,
                confidence="static",
            ),
            StageMemoryModel(
                "post_block_refit",
                f"post-refit:{inventory.model.config_hash}:{hidden}:{intermediate}:{sequence_length}",
                execution_fixed + factor_workspace + active_block_bytes,
                int(tuning_per_sample * 1.5),
                host_fixed,
                0,
                pinned_bytes_per_row,
                disk_bytes,
                largest_allocation_bytes,
                minimum_batch_size=refit_max if fixed_mode else 1,
                maximum_batch_size=refit_max,
                confidence="static",
            ),
        )
        return tuple(
            _select(
                model,
                envelope,
                config.runtime.memory_policy,
                maximum_prefetch_batches=(
                    0 if model.stage == "model_load" else config.runtime.activations.prefetch_batches
                ),
            )
            for model in models
        )

    candidates = (
        (ExecutorKind.RESIDENT, ExecutorKind.CPU_OFFLOAD)
        if requested is ExecutorKind.AUTO
        else (requested,)
    )
    candidate_errors: list[str] = []
    for executor in candidates:
        try:
            stage_plans = candidate_stage_plans(executor)
            break
        except ResourceAdmissionError as exc:
            candidate_errors.append(f"{executor.value}: {exc}")
    else:
        raise ResourceAdmissionError("RES001 no executor is admissible; " + "; ".join(candidate_errors))
    peak_gpu = max(stage.predicted_gpu_bytes for stage in stage_plans)
    _error, _uncertainty, reusable_pool_fraction = _policy_parameters(config.runtime.memory_policy.profile)
    safe_stage_peak = max(stage.predicted_gpu_bytes + stage.uncertainty_bytes for stage in stage_plans)
    remaining_gpu = max(
        0,
        gpu_capacity_bytes(envelope, reusable_pool_fraction=reusable_pool_fraction) - safe_stage_peak,
    )
    configured_cache = config.runtime.activations.gpu_cache
    if (
        configured_cache is ActivationGpuCacheMode.AUTO
        and config.runtime.memory_policy.mode is MemoryPolicyMode.ADAPTIVE
    ):
        if remaining_gpu >= activation_stream_bytes * 2:
            cache = ActivationGpuCacheMode.BOTH
        elif remaining_gpu >= activation_stream_bytes:
            cache = ActivationGpuCacheMode.INPUTS
        else:
            cache = ActivationGpuCacheMode.OFF
    else:
        cache = configured_cache
    requested_cache_bytes = {
        ActivationGpuCacheMode.OFF: 0,
        ActivationGpuCacheMode.AUTO: 0,
        ActivationGpuCacheMode.INPUTS: activation_stream_bytes,
        ActivationGpuCacheMode.BOTH: activation_stream_bytes * 2,
    }[cache]
    if requested_cache_bytes > remaining_gpu:
        raise ResourceAdmissionError(
            f"RES001 explicit activation GPU cache requires {requested_cache_bytes} bytes but only "
            f"{remaining_gpu} planned bytes remain"
        )
    warnings = (
        *candidate_errors,
        *(
            f"{stage.stage} resized from its admissible maximum to batch {stage.batch_size}"
            for stage in stage_plans
            if stage.resized
        ),
    )
    return ResolvedMemoryPlan(
        1,
        revision,
        config.runtime.memory_policy.mode.value,
        config.runtime.memory_policy.profile.value,
        config_hash(config),
        envelope,
        executor.value,
        "ram",
        cache.value,
        retain_completed_blocks,
        stage_plans,
        peak_gpu + requested_cache_bytes,
        max(stage.predicted_host_bytes for stage in stage_plans),
        max(stage.predicted_pinned_host_bytes for stage in stage_plans),
        max(stage.predicted_temporary_disk_bytes for stage in stage_plans),
        warnings,
    )


def persist_memory_plan(plan: ResolvedMemoryPlan, output: str | Path) -> ArtifactRef:
    root = Path(output)
    artifacts = LocalArtifactStore(root / "artifacts")
    with artifacts.begin_write(ArtifactTypes.MEMORY_PLAN, schema_version=1) as writer:
        (writer.path / "memory-plan.json").write_text(
            json.dumps(to_dict(plan), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        descriptor = writer.commit()
    reference = ArtifactRef(ArtifactTypes.MEMORY_PLAN, descriptor.artifact_id, descriptor.schema_version)
    atomic_write_json(root / "state" / "active-memory-plan.json", {"reference": to_dict(reference)})
    return reference


def load_memory_plan(output: str | Path, request_hash: str) -> tuple[ResolvedMemoryPlan, ArtifactRef] | None:
    root = Path(output)
    pointer = root / "state" / "active-memory-plan.json"
    if not pointer.is_file():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        reference = from_dict(ArtifactRef, payload["reference"], path="memory_plan.reference")
        if reference.artifact_type != ArtifactTypes.MEMORY_PLAN or reference.schema_version != 1:
            raise ValueError("active memory plan has the wrong artifact type or schema")
        artifacts = LocalArtifactStore(root / "artifacts", use_persistent_validation_cache=False)
        artifacts.validate(reference.artifact_id)
        plan_payload = json.loads(
            (artifacts.path_for(reference.artifact_id) / "memory-plan.json").read_text(encoding="utf-8")
        )
        plan = from_dict(ResolvedMemoryPlan, plan_payload, path="memory_plan")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("active memory plan is corrupt") from exc
    if plan.request_hash != request_hash:
        return None
    return plan, reference


def load_memory_plan_revision(output: str | Path, revision: int) -> MemoryPlanRevision | None:
    """Load the durable provenance record for one published memory-plan revision."""

    journal = Path(output) / "state" / "memory-plan-revisions.jsonl"
    if not journal.is_file():
        return None
    matched: MemoryPlanRevision | None = None
    try:
        for line_number, line in enumerate(journal.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            record = from_dict(
                MemoryPlanRevision,
                payload,
                path=f"memory_plan_revisions[{line_number}]",
            )
            if record.revision == revision:
                matched = record
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("memory plan revision journal is corrupt") from exc
    return matched


def revise_resident_memory_plan_after_oom(
    plan: ResolvedMemoryPlan,
    config: RunConfig,
    output: str | Path,
    *,
    stage: str | None = None,
) -> tuple[ResolvedMemoryPlan, MemoryPlanRevision, ArtifactRef]:
    """Resample resources, derive one lower-memory revision, and publish it atomically."""

    envelope = inspect_resource_envelope(
        output,
        config.runtime.compute_device,
        config.runtime.resources,
        config.runtime.memory_policy,
    )
    error, uncertainty, reusable = _policy_parameters(config.runtime.memory_policy.profile)
    revised, revision = revise_memory_plan_after_oom(
        plan,
        envelope,
        stage=stage,
        allowed_actions=config.runtime.on_cuda_oom,
        estimator_error_fraction=error,
        minimum_uncertainty_bytes=uncertainty,
        reusable_pool_fraction=reusable,
    )
    reference = persist_memory_plan(revised, output)
    journal = Path(output) / "state" / "memory-plan-revisions.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(to_dict(revision), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return revised, revision, reference


def revise_resident_memory_plan_for_throughput(
    plan: ResolvedMemoryPlan,
    output: str | Path,
    selections: tuple[tuple[str, int], ...],
) -> tuple[ResolvedMemoryPlan, MemoryPlanRevision, ArtifactRef]:
    """Publish empirically selected batches before semantic execution begins."""

    revised, revision = revise_memory_plan_for_throughput(plan, selections)
    reference = persist_memory_plan(revised, output)
    journal = Path(output) / "state" / "memory-plan-revisions.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(to_dict(revision), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return revised, revision, reference


@dataclass(frozen=True, slots=True)
class ResourcePlanningRequest:
    components: ResourceComponents
    requested_executor: str = "auto"
    requested_activation_tier: str = "auto"
    margins: ResourceMargins = ResourceMargins()


def _limit(available: int, margin: float) -> int:
    return int(available * (1 - margin))


def build_resource_plan(request: ResourcePlanningRequest, host: HostInventory) -> ResourcePlan:
    components = request.components
    gpu_limit = _limit(host.gpu_bytes_available, request.margins.gpu_fraction)
    host_limit = _limit(host.cpu_bytes_available, request.margins.host_fraction)
    disk_limit = _limit(host.temporary_disk_bytes_available, request.margins.disk_fraction)
    resident_base_gpu = (
        components.source_checkpoint_bytes
        + components.active_block_bytes
        + components.factor_workspace_bytes
        + components.hessian_bytes
        + components.tuning_state_bytes
    )
    resident_cuda_gpu = resident_base_gpu + components.activation_bytes
    streaming_gpu = (
        components.active_block_bytes
        + components.factor_workspace_bytes
        + components.hessian_bytes
        + components.tuning_state_bytes
    )
    cpu_offload_gpu = streaming_gpu + components.active_block_bytes
    streaming_host = components.active_block_bytes + components.factor_workspace_bytes + components.hessian_bytes
    cpu_offload_host = components.source_checkpoint_bytes + streaming_host
    executor = request.requested_executor
    if executor == "auto":
        if resident_cuda_gpu <= gpu_limit:
            executor = "resident"
        elif cpu_offload_gpu <= gpu_limit and cpu_offload_host <= host_limit:
            executor = "cpu_offload"
        else:
            executor = "streaming"
    if executor not in {"resident", "cpu_offload", "streaming"}:
        raise ValueError(f"unsupported executor: {executor}")
    peak_gpu = {
        "resident": resident_base_gpu,
        "cpu_offload": cpu_offload_gpu,
        "streaming": streaming_gpu,
    }[executor]
    base_host = cpu_offload_host if executor == "cpu_offload" else streaming_host

    tier = request.requested_activation_tier
    if tier == "auto":
        if executor == "resident" and peak_gpu + components.activation_bytes <= gpu_limit:
            tier = "cuda"
        elif base_host + components.activation_bytes <= host_limit:
            tier = "pinned_ram" if components.activation_bytes <= host_limit // 4 else "ram"
        else:
            tier = "mmap"
    if tier not in {"cuda", "pinned_ram", "ram", "mmap"}:
        raise ValueError(f"unsupported activation tier: {tier}")
    if tier == "cuda":
        peak_gpu += components.activation_bytes
    peak_host = base_host + (components.activation_bytes if tier in {"pinned_ram", "ram"} else 0)
    temporary_disk = (
        components.source_checkpoint_bytes
        + components.packed_output_bytes
        + components.committed_artifact_bytes
        + components.temporary_overhead_bytes
        + (components.activation_bytes if tier == "mmap" else 0)
    )
    failures = []
    if peak_gpu > gpu_limit:
        failures.append(f"GPU requires {peak_gpu} bytes but margin-adjusted limit is {gpu_limit}")
    if peak_host > host_limit:
        failures.append(f"host requires {peak_host} bytes but margin-adjusted limit is {host_limit}")
    if temporary_disk > disk_limit:
        failures.append(f"temporary disk requires {temporary_disk} bytes but margin-adjusted limit is {disk_limit}")
    if failures:
        raise InsufficientResourcesError("RES001 " + "; ".join(failures))
    return ResourcePlan(
        executor,
        tier,
        peak_gpu,
        peak_host,
        temporary_disk,
        components.source_checkpoint_bytes + components.active_block_bytes,
        components.packed_output_bytes + components.committed_artifact_bytes + components.activation_bytes,
        gpu_limit,
        host_limit,
        disk_limit,
        components,
    )
