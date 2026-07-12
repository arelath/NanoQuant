"""Auditable resident quantization composition for pinned Transformers snapshots."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from nanoquant.application.assembly import assemble_frozen_model
from nanoquant.application.calibration import MaterializedLayerCalibration, calibrate_block
from nanoquant.application.calibration_artifacts import build_objectives, persist_calibration
from nanoquant.application.layers import BlockEditor, LayerFreezer, TrainableFactorizedLinear
from nanoquant.application.loss_snapshots import BlockLossRecorder
from nanoquant.application.planning import PlanningRequest, build_quantization_plan, persist_plan
from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.application.quantization_stages import (
    FactorizationAttemptStage,
    OutlierSelectionStage,
)
from nanoquant.application.reconstruction_report import render_reconstruction_tables
from nanoquant.application.stages import StageContext, execute_stage
from nanoquant.config.codec import canonical_json, to_dict
from nanoquant.config.schema import (
    ADMMConfig,
    AllocationStrategy,
    ObjectiveConfig,
    OutlierConfig,
    RankAllocationConfig,
    RankBoundsConfig,
    RankRetryConfig,
)
from nanoquant.domain.models import (
    ArtifactRef,
    AttemptSummary,
    BlockResult,
    DatasetIdentity,
    FactorizationRequest,
    FrozenBlockState,
    FrozenModelResult,
    LayerId,
    LayerResult,
    ModelInventory,
    OutlierSelectionRequest,
    QuantizationPlan,
)
from nanoquant.domain.runs import BudgetState
from nanoquant.domain.seeds import logical_seed
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import (
    CommitIdentity,
    commit_block,
    commit_layer,
    load_block_activations,
    load_committed_block,
    load_committed_layer,
)
from nanoquant.infrastructure.events import JsonlEventSink
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.progress import ProgressJournal
from nanoquant.infrastructure.resident_executor import Cancellation, ResidentExecutor
from nanoquant.infrastructure.resource_usage import peak_process_memory_bytes
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.tensor_store import LocalTensorStore


@dataclass(frozen=True, slots=True)
class ResidentQuantizationRequest:
    snapshot: Path
    output: Path
    source: str
    revision: str
    token_ids: tuple[tuple[int, ...], ...]
    device: str = "cuda"
    target_bpw: float = 1.0
    rank_multiple: int = 32
    allocation_strategy: AllocationStrategy = AllocationStrategy.SENSITIVITY
    rank_floor_fraction: float = 0.5
    rank_ceiling_fraction: float = 4.5
    admm: ADMMConfig = ADMMConfig(outer_iterations=1, inner_iterations=1)
    seed: int = 0
    verify_hashes: bool = True
    interrupt_after_layer_commits: int | None = None


@dataclass(frozen=True, slots=True)
class ResidentQuantizationResult:
    inventory: ModelInventory
    plan: QuantizationPlan
    identity: CommitIdentity
    frozen_model: FrozenModelResult
    blocks: tuple[BlockResult, ...]
    report: ArtifactRef
    reference_nll: float
    compressed_nll: float
    logit_mse: float
    argmax_agreement: float
    peak_device_bytes: int
    peak_host_bytes: int
    artifact_bytes: int
    elapsed_seconds: float
    reused_commit_count: int


def _checkpoint_dtype(config: dict[str, object]) -> torch.dtype:
    value = config.get("torch_dtype")
    if not isinstance(value, str):
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(value, torch.float32)


def _decoder_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    base = getattr(model, "model", None)
    values = getattr(base, "layers", None)
    if not isinstance(values, nn.ModuleList):
        raise TypeError("model does not expose a supported decoder layer stack")
    return tuple(values)


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction.detach().float() - target.detach().float()).square().mean())


def _block_loss(
    adapter: Any,
    block: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    metadata: dict[str, object],
) -> float:
    with torch.no_grad():
        return _mse(adapter.run_block(block, inputs, **metadata), targets)


def _nll(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    prediction = logits[:, :-1].float().reshape(-1, logits.shape[-1])
    target = tokens[:, 1:].reshape(-1)
    return float(torch.nn.functional.cross_entropy(prediction, target))


def _artifact_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def run_resident_quantization(request: ResidentQuantizationRequest) -> ResidentQuantizationResult:
    """Quantize all decoder linears while the source model remains resident on one device."""
    started = time.perf_counter()
    if not request.token_ids or len({len(row) for row in request.token_ids}) != 1:
        raise ValueError("resident quantization requires equal-length, non-empty token rows")
    if request.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA resident quantization requested without CUDA")
    artifacts = LocalArtifactStore(request.output / "artifacts")
    tensors = LocalTensorStore(artifacts)
    executor = ResidentExecutor()
    events = JsonlEventSink(request.output / "events.jsonl", "resident-quantization")
    context = StageContext("resident-quantization", executor, artifacts, tensors, events, Cancellation())
    source = SafetensorsModelSource(
        request.snapshot,
        source=request.source,
        revision=request.revision,
        verify_hashes=request.verify_hashes,
    )
    checkpoint = source.inventory()
    adapter = adapter_for_config(checkpoint.config)
    inventory = adapter.model_inventory(source)
    tokens = torch.tensor(request.token_ids, dtype=torch.long, device=request.device)
    if request.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(request.device)
    model = cast(
        nn.Module,
        AutoModelForCausalLM.from_pretrained(
            request.snapshot,
            local_files_only=True,
            torch_dtype=_checkpoint_dtype(checkpoint.config),
        ),
    ).to(request.device)
    model.eval()
    decoder_layers = _decoder_layers(model)
    with torch.no_grad():
        reference_logits = cast(torch.Tensor, cast(Any, model)(input_ids=tokens, use_cache=False).logits).detach()
    capture = capture_prefix_invocations(
        decoder_layers[0], (lambda: cast(Any, model)(input_ids=tokens, use_cache=False),)
    )[0]
    initial_inputs = capture.positional[0]
    if not isinstance(initial_inputs, torch.Tensor):
        raise TypeError("captured first-block hidden state is not a tensor")
    metadata = capture.keyword

    calibration_values: list[tuple[LayerId, MaterializedLayerCalibration]] = []
    calibration_inputs = initial_inputs
    for block_inventory, block in zip(inventory.blocks, decoder_layers, strict=True):
        paths = tuple(layer.path for layer in adapter.quantizable_layers(block, block_inventory.block))

        def calibration_runner(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
            return adapter.run_block(module, value, **metadata)

        stats = calibrate_block(
            block,
            (calibration_inputs,),
            paths,
            calibration_runner,
            method="forward_only",
        )
        calibration_values.extend((LayerId(block_inventory.block, item.path), item) for item in stats)
        with torch.no_grad():
            calibration_inputs = calibration_runner(block, calibration_inputs).detach()
    token_bytes = tokens.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    dataset = DatasetIdentity(
        "sha256:" + hashlib.sha256(token_bytes).hexdigest(),
        ("deterministic-token-fixture",),
        ("1",),
        checkpoint.tokenizer_hash,
        "raw-token-ids-v1",
    )
    calibration = persist_calibration(
        tuple(calibration_values),
        inventory.model,
        dataset,
        "forward_only",
        "float32",
        artifacts,
        tensors,
        total_tokens=tokens.numel(),
    )
    objectives = build_objectives(calibration, ObjectiveConfig(), artifacts)
    allocation = RankAllocationConfig(
        target_bpw=request.target_bpw,
        strategy=request.allocation_strategy,
        bounds=RankBoundsConfig(
            multiple=request.rank_multiple,
            floor_fraction_of_uniform=request.rank_floor_fraction,
            ceiling_fraction_of_uniform=request.rank_ceiling_fraction,
            edge_block_boost=0.0,
        ),
        retry=RankRetryConfig(enabled=False, maximum_attempts=1),
    )
    plan = build_quantization_plan(
        PlanningRequest(
            inventory,
            calibration.stats,
            calibration.reference,
            objectives.objectives,
            allocation,
            OutlierConfig(),
        )
    )
    persisted_plan = persist_plan(plan, artifacts)
    config_hash = (
        "sha256:"
        + hashlib.sha256(
            canonical_json(
                {
                    "target_bpw": request.target_bpw,
                    "rank_multiple": request.rank_multiple,
                    "allocation_strategy": request.allocation_strategy,
                    "rank_floor_fraction": request.rank_floor_fraction,
                    "rank_ceiling_fraction": request.rank_ceiling_fraction,
                    "admm": request.admm,
                    "seed": request.seed,
                }
            ).encode()
        ).hexdigest()
    )
    identity = CommitIdentity(config_hash, inventory.model.config_hash, persisted_plan.reference.artifact_id)
    journal = ProgressJournal(request.output / "state", "resident-quantization", artifacts)
    discovery = journal.discover(plan, identity)
    discovered_records = (*discovery.valid_records, *discovery.orphan_records)
    block_records = sorted(
        (record for record in discovered_records if record.kind == "block"), key=lambda record: record.block
    )
    committed_blocks = [
        (
            ArtifactRef("block-result", record.artifact_id, 1),
            load_committed_block(ArtifactRef("block-result", record.artifact_id, 1), artifacts, identity).result,
        )
        for record in block_records
    ]
    accepted_bits = sum(layer.actual_bit_cost.total for _, block in committed_blocks for layer in block.layers)
    budget = BudgetState(plan.planned_cost.total, accepted_bits, 0)
    if committed_blocks:
        teacher_inputs, compressed_inputs = load_block_activations(committed_blocks[-1][0], artifacts, request.device)
    else:
        teacher_inputs = initial_inputs
        compressed_inputs = initial_inputs
    completed_block_indexes = {block.block.index for _, block in committed_blocks}
    partial_layer_records = {
        (record.block, record.layer): record
        for record in discovered_records
        if record.kind == "layer" and record.block not in completed_block_indexes
    }
    peak_device_bytes = 0
    factorization_wall_seconds = 0.0
    new_layer_commits = 0
    factor_stage = FactorizationAttemptStage(request.admm, device=request.device)

    for block_plan in plan.blocks:
        if block_plan.block.index in completed_block_indexes:
            continue
        block_started = time.perf_counter()
        block_index = block_plan.block.index
        source_block = decoder_layers[block_index]
        working_block = adapter.load_block(source, block_plan.block, request.device)
        working_block.eval()
        with torch.no_grad():
            teacher_outputs = adapter.run_block(source_block, teacher_inputs, **metadata).detach()
        recorder = BlockLossRecorder()
        recorder.record_source_reference(_mse(teacher_outputs, teacher_outputs))
        recorder.record_block_entry(_block_loss(adapter, working_block, compressed_inputs, teacher_outputs, metadata))
        layer_results: list[LayerResult] = []
        frozen_states = []

        for layer_plan in block_plan.layers:
            prior_record = partial_layer_records.get((block_index, layer_plan.layer.path))
            if prior_record is not None:
                prior = load_committed_layer(
                    ArtifactRef("layer-result", prior_record.artifact_id, 1), artifacts, identity
                ).result
                frozen = LayerFreezer().load(
                    prior.frozen_state,
                    tensors,
                    device=request.device,
                    dtype=compressed_inputs.dtype,
                )
                BlockEditor().install_frozen_layer(working_block, layer_plan.layer.path, frozen.module)
                frozen_states.append(prior.frozen_state)
                layer_results.append(prior)
                budget = replace(budget, accepted_bits=budget.accepted_bits + prior.actual_bit_cost.total)
                recorder.record_after_layer(
                    layer_plan.layer,
                    _block_loss(adapter, working_block, compressed_inputs, teacher_outputs, metadata),
                )
                continue
            with source.read_tensor(layer_plan.source_weight, device="cpu") as source_weight:
                source_ref = tensors.put("source-layer", {"weight": source_weight})["weight"]
            outliers = execute_stage(
                OutlierSelectionStage(),
                OutlierSelectionRequest(
                    layer_plan.layer,
                    source_ref,
                    layer_plan.objective,
                    layer_plan.outliers,
                    layer_plan.rank,
                    logical_seed(request.seed, "outliers", block_index, layer_plan.layer.path, 0),
                ),
                context,
            )
            factorized = execute_stage(
                factor_stage,
                FactorizationRequest(
                    1,
                    layer_plan.layer,
                    source_ref,
                    outliers.residual_weight,
                    layer_plan.objective,
                    layer_plan.rank,
                    logical_seed(request.seed, "factorize", block_index, layer_plan.layer.path, 0),
                    config_hash,
                ),
                context,
            )
            peak_device_bytes = max(peak_device_bytes, factorized.peak_workspace_bytes)
            factorization_wall_seconds += factorized.wall_seconds
            mid_ref = factorized.factors.scales.mid
            if mid_ref is None:
                raise AssertionError("factorizer omitted required mid scale")
            with (
                tensors.read(factorized.factors.left_latent, request.device) as left,
                tensors.read(factorized.factors.right_latent, request.device) as right,
                tensors.read(factorized.factors.scales.pre, request.device) as scale_pre,
                tensors.read(mid_ref, request.device) as scale_mid,
                tensors.read(factorized.factors.scales.post, request.device) as scale_post,
            ):
                trainable = TrainableFactorizedLinear(left, right, scale_pre, scale_mid, scale_post)
            frozen = LayerFreezer().freeze(layer_plan.layer, trainable, tensors)
            frozen_module = frozen.module.to(device=request.device, dtype=compressed_inputs.dtype)
            BlockEditor().install_frozen_layer(working_block, layer_plan.layer.path, frozen_module)
            frozen_states.append(frozen.state)
            attempt = AttemptSummary(
                0,
                layer_plan.rank,
                factorized.factors.left_binary.artifact,
                factorized.metrics.export_weighted_normalized_error,
                factorized.metrics.raw_normalized_error,
                layer_plan.estimated_cost,
                factorized.metrics.export_weighted_normalized_error,
                True,
                "accepted",
            )
            layer_result = LayerResult(
                1,
                layer_plan.layer,
                layer_plan,
                (attempt,),
                0,
                factorized.factors.left_binary.artifact,
                None,
                None,
                frozen.state,
                factorized.metrics,
                layer_plan.estimated_cost,
                0,
                ("scale_fit_disabled", "tuning_disabled"),
            )
            committed_layer = commit_layer(layer_result, artifacts, identity)
            journal.append(
                "layer",
                block_index,
                layer_plan.layer.path,
                committed_layer.reference.artifact_id,
                identity,
            )
            new_layer_commits += 1
            if (
                request.interrupt_after_layer_commits is not None
                and new_layer_commits >= request.interrupt_after_layer_commits
            ):
                raise InterruptedError(f"injected interruption after {new_layer_commits} new layer commits")
            layer_results.append(layer_result)
            budget = replace(budget, accepted_bits=budget.accepted_bits + layer_plan.estimated_cost.total)
            recorder.record_after_layer(
                layer_plan.layer,
                _block_loss(adapter, working_block, compressed_inputs, teacher_outputs, metadata),
            )
        with torch.no_grad():
            compressed_outputs = adapter.run_block(working_block, compressed_inputs, **metadata).detach()
        recorder.record_final_frozen_pre_kd(_mse(compressed_outputs, teacher_outputs))
        frozen_block = FrozenBlockState(block_plan.block, tuple(frozen_states), ())
        block_peak = int(torch.cuda.max_memory_allocated(request.device)) if request.device.startswith("cuda") else 0
        peak_device_bytes = max(peak_device_bytes, block_peak)
        committed = commit_block(
            block_plan.block,
            tuple(layer_results),
            frozen_block,
            recorder.finalize(),
            teacher_outputs,
            compressed_outputs,
            budget.retry_bits_spent,
            artifacts,
            identity,
            wall_seconds=time.perf_counter() - block_started,
            peak_gpu_bytes=block_peak,
            warnings=("scale_fit_disabled", "tuning_disabled"),
        )
        journal.append("block", block_index, None, committed.reference.artifact_id, identity)
        committed_blocks.append((committed.reference, committed.result))
        teacher_inputs = teacher_outputs
        compressed_inputs = compressed_outputs
        del working_block

    with torch.no_grad():
        compressed_logits = adapter.run_suffix(model, compressed_inputs).detach()
    original_elements = sum(
        layer.in_features * layer.out_features for block in inventory.blocks for layer in block.quantizable_layers
    )
    frozen_model = assemble_frozen_model(
        inventory.model,
        persisted_plan.reference,
        tuple(committed_blocks),
        (),
        original_elements,
    )
    reference_nll = _nll(reference_logits, tokens)
    compressed_nll = _nll(compressed_logits, tokens)
    logit_mse = _mse(compressed_logits, reference_logits)
    argmax_agreement = float((compressed_logits.argmax(-1) == reference_logits.argmax(-1)).float().mean())
    elapsed = time.perf_counter() - started
    peak_host_bytes = peak_process_memory_bytes()
    ranks = [layer.rank for block in plan.blocks for layer in block.layers]
    artifact_bytes_before_report = _artifact_bytes(artifacts.root)
    report_payload = {
        "schema_version": 1,
        "source": request.source,
        "revision": request.revision,
        "model": to_dict(inventory.model),
        "plan": persisted_plan.reference.artifact_id,
        "block_count": len(committed_blocks),
        "layer_count": sum(len(block.layers) for _, block in committed_blocks),
        "target_bpw": request.target_bpw,
        "effective_bpw": frozen_model.effective_bpw,
        "actual_total_bits": frozen_model.actual_total_bits,
        "rank_minimum": min(ranks),
        "rank_maximum": max(ranks),
        "rank_mean": sum(ranks) / len(ranks),
        "mean_final_block_loss": sum(block.losses.final_frozen_pre_kd for _, block in committed_blocks)
        / len(committed_blocks),
        "factorization_wall_seconds": factorization_wall_seconds,
        "reference_nll": reference_nll,
        "compressed_nll": compressed_nll,
        "logit_mse": logit_mse,
        "argmax_agreement": argmax_agreement,
        "peak_device_bytes": peak_device_bytes,
        "peak_host_bytes": peak_host_bytes,
        "artifact_bytes_before_report": artifact_bytes_before_report,
        "elapsed_seconds": elapsed,
        "admm": to_dict(request.admm),
        "warnings": ["scale_fit_disabled", "tuning_disabled", "single_fixture_quality_only"],
        "reused_commit_count": len(discovered_records),
    }
    with artifacts.begin_write("resident-quantization-report") as writer:
        (writer.path / "report.json").write_text(json.dumps(report_payload, sort_keys=True, indent=2), encoding="utf-8")
        (writer.path / "reconstruction.md").write_text(
            render_reconstruction_tables(tuple(block for _, block in committed_blocks)), encoding="utf-8"
        )
        descriptor = writer.commit()
    report = ArtifactRef("resident-quantization-report", descriptor.artifact_id, descriptor.schema_version)
    artifact_bytes = _artifact_bytes(artifacts.root)
    executor.release()
    return ResidentQuantizationResult(
        inventory,
        plan,
        identity,
        frozen_model,
        tuple(block for _, block in committed_blocks),
        report,
        reference_nll,
        compressed_nll,
        logit_mse,
        argmax_agreement,
        peak_device_bytes,
        peak_host_bytes,
        artifact_bytes,
        elapsed,
        len(discovered_records),
    )
