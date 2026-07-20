"""Resource planning types and pure adaptive-memory policy."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


def peak_device_memory_bytes(device: str | torch.device) -> int:
    """Return the CUDA allocator high-water mark that governs future capacity."""
    resolved = str(device)
    if not resolved.startswith("cuda"):
        return 0
    return max(
        int(torch.cuda.max_memory_allocated(resolved)),
        int(torch.cuda.max_memory_reserved(resolved)),
    )


@dataclass(frozen=True, slots=True)
class ResourceComponents:
    source_checkpoint_bytes: int
    packed_output_bytes: int
    active_block_bytes: int
    factor_workspace_bytes: int
    hessian_bytes: int
    activation_bytes: int
    tuning_state_bytes: int
    committed_artifact_bytes: int
    temporary_overhead_bytes: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.as_tuple()):
            raise ValueError("resource component sizes cannot be negative")

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.source_checkpoint_bytes,
            self.packed_output_bytes,
            self.active_block_bytes,
            self.factor_workspace_bytes,
            self.hessian_bytes,
            self.activation_bytes,
            self.tuning_state_bytes,
            self.committed_artifact_bytes,
            self.temporary_overhead_bytes,
        )


@dataclass(frozen=True, slots=True)
class ResourceMargins:
    gpu_fraction: float = 0.10
    host_fraction: float = 0.10
    disk_fraction: float = 0.10

    def __post_init__(self) -> None:
        if any(value < 0 or value >= 1 for value in (self.gpu_fraction, self.host_fraction, self.disk_fraction)):
            raise ValueError("resource safety margins must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    executor: str
    activation_tier: str
    peak_gpu_bytes: int
    peak_host_bytes: int
    temporary_disk_bytes: int
    bytes_read: int
    bytes_written: int
    gpu_limit_after_margin: int
    host_limit_after_margin: int
    disk_limit_after_margin: int
    components: ResourceComponents
    warnings: tuple[str, ...] = ()


class ResourceAdmissionError(RuntimeError):
    """Raised when even a stage's minimum execution shape cannot fit."""

    code = "RES001"


@dataclass(frozen=True, slots=True)
class ResourceEnvelope:
    """One timestamped view of capacity available to the NanoQuant process."""

    device: str
    gpu_total_bytes: int
    gpu_free_bytes: int
    gpu_process_allocated_bytes: int
    gpu_process_reserved_bytes: int
    gpu_hard_limit_bytes: int
    gpu_reserve_bytes: int
    host_available_bytes: int
    host_process_bytes: int
    host_hard_limit_bytes: int
    host_reserve_bytes: int
    pinned_host_limit_bytes: int
    temporary_disk_free_bytes: int
    temporary_disk_hard_limit_bytes: int
    temporary_disk_reserve_bytes: int
    observed_at: str

    def __post_init__(self) -> None:
        byte_values = (
            self.gpu_total_bytes,
            self.gpu_free_bytes,
            self.gpu_process_allocated_bytes,
            self.gpu_process_reserved_bytes,
            self.gpu_hard_limit_bytes,
            self.gpu_reserve_bytes,
            self.host_available_bytes,
            self.host_process_bytes,
            self.host_hard_limit_bytes,
            self.host_reserve_bytes,
            self.pinned_host_limit_bytes,
            self.temporary_disk_free_bytes,
            self.temporary_disk_hard_limit_bytes,
            self.temporary_disk_reserve_bytes,
        )
        if any(value < 0 for value in byte_values):
            raise ValueError("resource envelope byte counts cannot be negative")
        if self.gpu_process_reserved_bytes < self.gpu_process_allocated_bytes:
            raise ValueError("CUDA reservation cannot be smaller than live allocation")
        if self.gpu_total_bytes and self.gpu_free_bytes > self.gpu_total_bytes:
            raise ValueError("free CUDA bytes cannot exceed total CUDA bytes")
        if not self.observed_at:
            raise ValueError("resource envelope requires an observation timestamp")


@dataclass(frozen=True, slots=True)
class StageMemoryModel:
    """Auditable fixed-plus-scalable peak model for one execution stage."""

    stage: str
    signature: str
    fixed_gpu_bytes: int
    gpu_bytes_per_row: int
    fixed_host_bytes: int
    host_bytes_per_row: int
    pinned_bytes_per_row_per_prefetch_slot: int
    temporary_disk_bytes: int
    largest_indivisible_gpu_allocation_bytes: int
    minimum_batch_size: int
    maximum_batch_size: int
    confidence: str = "static"

    def __post_init__(self) -> None:
        if not self.stage or not self.signature:
            raise ValueError("stage memory model requires a stage and signature")
        sizes = (
            self.fixed_gpu_bytes,
            self.gpu_bytes_per_row,
            self.fixed_host_bytes,
            self.host_bytes_per_row,
            self.pinned_bytes_per_row_per_prefetch_slot,
            self.temporary_disk_bytes,
            self.largest_indivisible_gpu_allocation_bytes,
        )
        if any(value < 0 for value in sizes):
            raise ValueError("stage memory sizes cannot be negative")
        if self.minimum_batch_size <= 0 or self.maximum_batch_size < self.minimum_batch_size:
            raise ValueError("stage batch bounds are invalid")
        if self.confidence not in {"static", "measured", "conservative"}:
            raise ValueError("unsupported stage-memory confidence")

    def peak_gpu_bytes(self, batch_size: int) -> int:
        self._validate_batch(batch_size)
        return self.fixed_gpu_bytes + self.gpu_bytes_per_row * batch_size

    def peak_host_bytes(self, batch_size: int) -> int:
        self._validate_batch(batch_size)
        return self.fixed_host_bytes + self.host_bytes_per_row * batch_size

    def pinned_host_bytes(self, batch_size: int, prefetch_batches: int) -> int:
        self._validate_batch(batch_size)
        if prefetch_batches < 0:
            raise ValueError("prefetch batch count cannot be negative")
        return self.pinned_bytes_per_row_per_prefetch_slot * batch_size * prefetch_batches

    def _validate_batch(self, batch_size: int) -> None:
        if not self.minimum_batch_size <= batch_size <= self.maximum_batch_size:
            raise ValueError("batch size is outside the stage model bounds")


@dataclass(frozen=True, slots=True)
class StageExecutionPlan:
    stage: str
    signature: str
    model: StageMemoryModel
    batch_size: int
    prefetch_batches: int
    predicted_gpu_bytes: int
    predicted_host_bytes: int
    predicted_pinned_host_bytes: int
    predicted_temporary_disk_bytes: int
    gpu_capacity_bytes: int
    host_capacity_bytes: int
    disk_capacity_bytes: int
    uncertainty_bytes: int
    resized: bool
    admitted_maximum_batch_size: int | None = None


@dataclass(frozen=True, slots=True)
class ResolvedMemoryPlan:
    schema_version: int
    revision: int
    mode: str
    profile: str
    request_hash: str
    envelope: ResourceEnvelope
    executor: str
    activation_tier: str
    activation_gpu_cache: str
    retain_completed_blocks: bool
    stages: tuple[StageExecutionPlan, ...]
    peak_gpu_bytes: int
    peak_host_bytes: int
    peak_pinned_host_bytes: int
    peak_temporary_disk_bytes: int
    warnings: tuple[str, ...] = ()

    def stage(self, name: str) -> StageExecutionPlan:
        matches = tuple(stage for stage in self.stages if stage.stage == name)
        if len(matches) != 1:
            raise KeyError(f"memory plan does not contain exactly one stage named {name!r}")
        return matches[0]


@dataclass(frozen=True, slots=True)
class MemoryPlanRevision:
    revision: int
    parent_revision: int | None
    reason: str
    stage: str | None
    action: str
    algorithm_changed: bool
    previous_batch_size: int | None = None
    next_batch_size: int | None = None


def geometric_batch_candidates(maximum: int, minimum: int = 1) -> tuple[int, ...]:
    """Return deterministic descending candidates without probing every integer."""

    if minimum <= 0 or maximum < minimum:
        raise ValueError("batch candidate bounds are invalid")
    candidates: list[int] = []
    current = maximum
    while current > minimum:
        candidates.append(current)
        current = max(minimum, current // 2)
    candidates.append(minimum)
    return tuple(dict.fromkeys(candidates))


def throughput_batch_candidates(maximum_safe: int, preferred: int) -> tuple[int, ...]:
    """Return a small deterministic search set within a proven-safe upper bound."""

    if maximum_safe <= 0 or preferred <= 0:
        raise ValueError("throughput batch bounds must be positive")
    baseline = min(maximum_safe, preferred)
    return tuple(sorted({*geometric_batch_candidates(maximum_safe), baseline}, reverse=True))


def select_fastest_observed_batch(
    observations: tuple[tuple[int, float], ...],
    *,
    baseline_batch: int,
    minimum_improvement_fraction: float = 0.05,
) -> int:
    """Select a measured winner, retaining the baseline for immaterial gains."""

    if not observations or baseline_batch <= 0:
        raise ValueError("throughput selection requires observations and a positive baseline")
    if not 0 <= minimum_improvement_fraction < 1:
        raise ValueError("minimum throughput improvement must be in [0, 1)")
    timings = {batch: seconds for batch, seconds in observations}
    if len(timings) != len(observations) or any(batch <= 0 or seconds <= 0 for batch, seconds in observations):
        raise ValueError("throughput observations must contain unique positive batches and timings")
    if baseline_batch not in timings:
        raise ValueError("throughput observations must include the baseline batch")
    winner_batch, winner_seconds = min(observations, key=lambda item: (item[1], item[0]))
    baseline_seconds = timings[baseline_batch]
    improvement = (baseline_seconds - winner_seconds) / baseline_seconds
    return winner_batch if improvement >= minimum_improvement_fraction else baseline_batch


def revise_memory_plan_for_throughput(
    plan: ResolvedMemoryPlan,
    selections: tuple[tuple[str, int], ...],
) -> tuple[ResolvedMemoryPlan, MemoryPlanRevision]:
    """Persist measured physical batches while retaining conservative admission data."""

    selected_by_stage = dict(selections)
    if not selected_by_stage or len(selected_by_stage) != len(selections):
        raise ValueError("throughput revision requires unique stage selections")
    revised_stages = []
    previous_batches: list[int] = []
    next_batches: list[int] = []
    for stage in plan.stages:
        selected = selected_by_stage.get(stage.stage)
        if selected is None:
            revised_stages.append(stage)
            continue
        admitted_maximum = stage.admitted_maximum_batch_size or stage.batch_size
        if not stage.model.minimum_batch_size <= selected <= admitted_maximum:
            raise ValueError("throughput batch must remain within the admitted stage range")
        previous_batches.append(stage.batch_size)
        next_batches.append(selected)
        revised_stages.append(
            replace(
                stage,
                batch_size=selected,
                predicted_gpu_bytes=stage.model.peak_gpu_bytes(selected),
                predicted_host_bytes=stage.model.peak_host_bytes(selected),
                predicted_pinned_host_bytes=stage.model.pinned_host_bytes(selected, stage.prefetch_batches),
                resized=selected < stage.model.maximum_batch_size,
                admitted_maximum_batch_size=admitted_maximum,
            )
        )
    if set(selected_by_stage) - {stage.stage for stage in plan.stages}:
        raise KeyError("throughput revision names an unknown stage")
    original_stage_peak = max(stage.predicted_gpu_bytes for stage in plan.stages)
    cache_bytes = max(0, plan.peak_gpu_bytes - original_stage_peak)
    stages = tuple(revised_stages)
    revised = replace(
        plan,
        revision=plan.revision + 1,
        stages=stages,
        peak_gpu_bytes=max(stage.predicted_gpu_bytes for stage in stages) + cache_bytes,
        peak_host_bytes=max(stage.predicted_host_bytes for stage in stages),
        peak_pinned_host_bytes=max(stage.predicted_pinned_host_bytes for stage in stages),
        warnings=(
            *plan.warnings,
            *(f"{stage} selected measured-throughput batch {batch}" for stage, batch in selections),
        ),
    )
    revision = MemoryPlanRevision(
        revised.revision,
        plan.revision,
        "measured throughput autotune",
        None,
        "select_measured_throughput_batches",
        False,
        previous_batches[0] if len(previous_batches) == 1 else None,
        next_batches[0] if len(next_batches) == 1 else None,
    )
    return revised, revision


def gpu_capacity_bytes(envelope: ResourceEnvelope, *, reusable_pool_fraction: float = 0.75) -> int:
    """Return the safe total process allocation supported by the current CUDA envelope."""

    if not 0 <= reusable_pool_fraction <= 1:
        raise ValueError("reusable CUDA pool fraction must be in [0, 1]")
    if envelope.gpu_total_bytes == 0:
        return 0
    reusable = max(0, envelope.gpu_process_reserved_bytes - envelope.gpu_process_allocated_bytes)
    physical = (
        envelope.gpu_process_allocated_bytes
        + envelope.gpu_free_bytes
        + int(reusable * reusable_pool_fraction)
    )
    return max(0, min(envelope.gpu_hard_limit_bytes, physical) - envelope.gpu_reserve_bytes)


def host_capacity_bytes(envelope: ResourceEnvelope) -> int:
    physical = envelope.host_process_bytes + envelope.host_available_bytes
    return max(0, min(envelope.host_hard_limit_bytes, physical) - envelope.host_reserve_bytes)


def disk_capacity_bytes(envelope: ResourceEnvelope) -> int:
    return max(
        0,
        min(envelope.temporary_disk_hard_limit_bytes, envelope.temporary_disk_free_bytes)
        - envelope.temporary_disk_reserve_bytes,
    )


def select_stage_execution_plan(
    model: StageMemoryModel,
    envelope: ResourceEnvelope,
    *,
    maximum_prefetch_batches: int = 0,
    estimator_error_fraction: float = 0.10,
    minimum_uncertainty_bytes: int = 64 * 2**20,
    reusable_pool_fraction: float = 0.75,
) -> StageExecutionPlan:
    """Choose the largest predicted-safe physical batch and prefetch depth."""

    if maximum_prefetch_batches < 0:
        raise ValueError("maximum prefetch batches cannot be negative")
    if not 0 <= estimator_error_fraction < 1 or minimum_uncertainty_bytes < 0:
        raise ValueError("invalid estimator uncertainty policy")
    gpu_capacity = gpu_capacity_bytes(envelope, reusable_pool_fraction=reusable_pool_fraction)
    host_capacity = host_capacity_bytes(envelope)
    disk_capacity = disk_capacity_bytes(envelope)
    if model.temporary_disk_bytes > disk_capacity:
        raise ResourceAdmissionError(
            f"RES001 {model.stage} requires {model.temporary_disk_bytes} temporary disk bytes "
            f"but safe capacity is {disk_capacity}"
        )
    selected: tuple[int, int, int, int] | None = None
    # The memory model is monotonic, so checking every integer is cheap and
    # avoids the almost-2x utilization loss of power-of-two-only candidates.
    for batch_size in range(model.maximum_batch_size, model.minimum_batch_size - 1, -1):
        predicted_gpu = model.peak_gpu_bytes(batch_size)
        predicted_host = model.peak_host_bytes(batch_size)
        uncertainty = max(minimum_uncertainty_bytes, int(predicted_gpu * estimator_error_fraction))
        pinned_per_prefetch = model.pinned_host_bytes(batch_size, 1)
        prefetch = (
            maximum_prefetch_batches
            if pinned_per_prefetch == 0
            else min(maximum_prefetch_batches, envelope.pinned_host_limit_bytes // pinned_per_prefetch)
        )
        pinned_admitted = maximum_prefetch_batches == 0 or prefetch > 0
        if (
            predicted_gpu + uncertainty <= gpu_capacity
            and predicted_host <= host_capacity
            and pinned_admitted
            and model.largest_indivisible_gpu_allocation_bytes + envelope.gpu_process_allocated_bytes
            <= gpu_capacity
        ):
            selected = batch_size, prefetch, predicted_gpu, predicted_host
            break
    if selected is None:
        minimum_gpu = model.peak_gpu_bytes(model.minimum_batch_size)
        minimum_host = model.peak_host_bytes(model.minimum_batch_size)
        raise ResourceAdmissionError(
            f"RES001 {model.stage} minimum batch {model.minimum_batch_size} requires "
            f"{minimum_gpu} GPU and {minimum_host} host bytes; safe capacities are "
            f"{gpu_capacity} GPU and {host_capacity} host bytes; largest indivisible CUDA allocation is "
            f"{model.largest_indivisible_gpu_allocation_bytes} bytes"
        )
    batch_size, prefetch, predicted_gpu, predicted_host = selected
    uncertainty = max(minimum_uncertainty_bytes, int(predicted_gpu * estimator_error_fraction))
    return StageExecutionPlan(
        model.stage,
        model.signature,
        model,
        batch_size,
        prefetch,
        predicted_gpu,
        predicted_host,
        model.pinned_host_bytes(batch_size, prefetch),
        model.temporary_disk_bytes,
        gpu_capacity,
        host_capacity,
        disk_capacity,
        uncertainty,
        batch_size != model.maximum_batch_size or prefetch != maximum_prefetch_batches,
        batch_size,
    )


def revise_memory_plan_after_oom(
    plan: ResolvedMemoryPlan,
    envelope: ResourceEnvelope,
    *,
    stage: str | None = None,
    allowed_actions: tuple[str, ...] = (
        "reduce_batch_size",
        "move_activations_down_one_tier",
        "fail",
    ),
    estimator_error_fraction: float = 0.10,
    minimum_uncertainty_bytes: int = 64 * 2**20,
    reusable_pool_fraction: float = 0.75,
) -> tuple[ResolvedMemoryPlan, MemoryPlanRevision]:
    """Create one finite, lower-memory adaptive revision after rollback."""

    if plan.mode != "adaptive":
        raise ResourceAdmissionError("RES002 fixed memory plans cannot be revised after OOM")
    if stage == "model_load":
        raise ResourceAdmissionError("RES002 model-load OOM cannot be recovered by reducing a physical batch")
    action_set = set(allowed_actions)
    allow_batch_reduction = bool(action_set & {"reduce_batch_size", "reduce_stage_batch_size"})
    allow_cache_eviction = bool(
        action_set & {"move_activations_down_one_tier", "move_activation_store_to_pageable_ram"}
    )
    stage_aliases = {
        "nonfactorized_tuning": "tuning",
        "factorized_tuning": "tuning",
        "calibration": "calibration",
        "calibration_block": "calibration",
        "post_block_refit": "post_block_refit",
        "prefix_activation_capture": "block_forward",
        "block_entry_loss": "block_forward",
        "block_propagation": "block_forward",
        "reference_quality": "block_forward",
        "quality_evaluation": "block_forward",
    }
    selected_stage = stage_aliases.get(stage or "")
    revised_stages: list[StageExecutionPlan] = []
    changed_batches = False
    for planned_stage in plan.stages:
        current = planned_stage.batch_size
        if (
            not allow_batch_reduction
            or planned_stage.stage == "model_load"
            or current <= planned_stage.model.minimum_batch_size
            or (selected_stage is not None and planned_stage.stage != selected_stage)
        ):
            revised_stages.append(planned_stage)
            continue
        maximum = max(planned_stage.model.minimum_batch_size, current // 2)
        revised_model = replace(planned_stage.model, maximum_batch_size=maximum)
        revised_stages.append(
            select_stage_execution_plan(
                revised_model,
                envelope,
                maximum_prefetch_batches=planned_stage.prefetch_batches,
                estimator_error_fraction=estimator_error_fraction,
                minimum_uncertainty_bytes=minimum_uncertainty_bytes,
                reusable_pool_fraction=reusable_pool_fraction,
            )
        )
        changed_batches = True
    cache_changed = allow_cache_eviction and plan.activation_gpu_cache != "off"
    if not changed_batches and not cache_changed:
        raise ResourceAdmissionError("RES002 adaptive memory fallbacks are exhausted at batch one")
    new_cache = "off" if not changed_batches else plan.activation_gpu_cache
    old_stage_peak = max(stage.predicted_gpu_bytes for stage in plan.stages)
    cache_bytes = max(0, plan.peak_gpu_bytes - old_stage_peak) if new_cache != "off" else 0
    peak_gpu = max(stage.predicted_gpu_bytes for stage in revised_stages) + cache_bytes
    action_parts = []
    if changed_batches:
        action_parts.append("reduce_physical_batches")
    if new_cache != plan.activation_gpu_cache:
        action_parts.append("evict_activation_cache")
    action = "+".join(action_parts)
    revision = MemoryPlanRevision(
        plan.revision + 1,
        plan.revision,
        "out_of_memory",
        stage,
        action,
        False,
        max((stage.batch_size for stage in plan.stages if stage.stage != "model_load"), default=None),
        max((stage.batch_size for stage in revised_stages if stage.stage != "model_load"), default=None),
    )
    warnings = (*plan.warnings, f"revision {revision.revision}: {action} after OOM")
    return (
        replace(
            plan,
            revision=revision.revision,
            envelope=envelope,
            activation_gpu_cache=new_cache,
            stages=tuple(revised_stages),
            peak_gpu_bytes=peak_gpu,
            peak_host_bytes=max(stage.predicted_host_bytes for stage in revised_stages),
            peak_pinned_host_bytes=max(stage.predicted_pinned_host_bytes for stage in revised_stages),
            peak_temporary_disk_bytes=max(stage.predicted_temporary_disk_bytes for stage in revised_stages),
            warnings=warnings,
        ),
        revision,
    )
