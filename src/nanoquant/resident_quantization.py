"""Auditable resident quantization composition for pinned Transformers snapshots."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from nanoquant.application.assembly import assemble_frozen_model
from nanoquant.application.calibration import MaterializedLayerCalibration, calibrate_block, calibrate_causal_model
from nanoquant.application.calibration_artifacts import build_objectives, persist_calibration
from nanoquant.application.layers import BlockEditor, LayerFreezer, TrainableFactorizedLinear
from nanoquant.application.loss_snapshots import BlockLossRecorder
from nanoquant.application.planning import PlanningRequest, build_quantization_plan, persist_plan
from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.application.quantization_stages import (
    FactorizationAttemptStage,
    MaterializedScaleFitStageRequest,
    OutlierSelectionStage,
    ScaleFitStage,
)
from nanoquant.application.reconstruction_report import render_reconstruction_tables
from nanoquant.application.stages import StageContext, execute_stage
from nanoquant.application.tuning import TuningRequest, tune_factorized
from nanoquant.config.codec import canonical_json, to_dict
from nanoquant.config.schema import (
    ADMMConfig,
    AllocationStrategy,
    ObjectiveConfig,
    OutlierConfig,
    RankAllocationConfig,
    RankBoundsConfig,
    RankRetryConfig,
    ScaleFitConfig,
)
from nanoquant.domain.metrics import reconstruction_metrics
from nanoquant.domain.models import (
    ArtifactRef,
    AttemptSummary,
    BlockResult,
    DatasetIdentity,
    FactorizationRequest,
    FrozenBlockState,
    FrozenModelResult,
    FrozenOutlierState,
    LayerId,
    LayerResult,
    ModelInventory,
    OutlierSelectionRequest,
    QuantizationPlan,
    ScaleFitRequest,
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
    token_ids: torch.Tensor | tuple[tuple[int, ...], ...]
    device: str = "cuda"
    target_bpw: float = 1.0
    rank_multiple: int = 32
    allocation_strategy: AllocationStrategy = AllocationStrategy.SENSITIVITY
    rank_floor_fraction: float = 0.5
    rank_ceiling_fraction: float = 4.5
    rank_sensitivity_alpha: float = 0.5
    rank_edge_boost: float = 0.0
    layer_order: tuple[str, ...] = ()
    admm: ADMMConfig = ADMMConfig(outer_iterations=1, inner_iterations=1)
    outliers: OutlierConfig = OutlierConfig()
    scale_fit: ScaleFitConfig = ScaleFitConfig(enabled=False)
    factorized_tuning_epochs: int = 0
    factorized_tuning_batch_size: int = 8
    factorized_tuning_learning_rate: float = 1e-5
    seed: int = 0
    verify_hashes: bool = True
    interrupt_after_layer_commits: int | None = None
    block_forward_batch_size: int = 8
    quality_token_ids: torch.Tensor | tuple[tuple[int, ...], ...] | None = None
    calibration_method: str = "forward_only"
    calibration_shrinkage: float = 0.0
    calibration_batch_size: int = 1


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
    if prediction.shape != target.shape:
        raise ValueError("MSE prediction and target shapes differ")
    if prediction.ndim == 0:
        return float((prediction.detach().float() - target.detach().float()).square())
    total = torch.zeros((), device=prediction.device)
    for index in range(prediction.shape[0]):
        total += (prediction[index].detach().float() - target[index].detach().float()).square().sum()
    return float(total / prediction.numel())


def _block_loss(
    adapter: Any,
    block: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    metadata: dict[str, object],
    batch_size: int,
) -> float:
    with torch.no_grad():
        squared_error = torch.zeros((), device=inputs.device)
        elements = 0
        for start in range(0, inputs.shape[0], batch_size):
            end = min(start + batch_size, inputs.shape[0])
            prediction = adapter.run_block(block, inputs[start:end], **metadata)
            target = targets[start:end]
            squared_error += (prediction.float() - target.float()).square().sum()
            elements += target.numel()
        return float(squared_error / elements)


def _run_block_batched(
    adapter: Any,
    block: nn.Module,
    inputs: torch.Tensor,
    metadata: dict[str, object],
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("block forward batch size must be positive")
    result: torch.Tensor | None = None
    for start in range(0, inputs.shape[0], batch_size):
        end = min(start + batch_size, inputs.shape[0])
        output = adapter.run_block(block, inputs[start:end], **metadata)
        if result is None:
            result = torch.empty(
                (inputs.shape[0], *output.shape[1:]),
                device=output.device,
                dtype=output.dtype,
            )
        result[start:end].copy_(output)
    if result is None:
        raise ValueError("cannot run a block over empty inputs")
    return result


def _nll(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    prediction = logits[:, :-1].float().reshape(-1, logits.shape[-1])
    target = tokens[:, 1:].reshape(-1)
    return float(torch.nn.functional.cross_entropy(prediction, target))


def _artifact_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _token_tensor(value: torch.Tensor | tuple[tuple[int, ...], ...], device: str) -> torch.Tensor:
    result = value.detach().clone() if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("resident quantization tokens must be a non-empty rectangular rank-2 tensor")
    return result.to(device=device, dtype=torch.long)


def _layer_type_multiplier(path: str) -> float:
    if path.endswith(("gate_proj", "up_proj", "fc1")):
        return 1.4
    if path.endswith(("down_proj", "fc2", "v_proj")):
        return 1.2
    if path.endswith(("q_proj", "k_proj")):
        return 0.8
    return 1.0


def _legacy_sensitivity_profile(
    inventory: ModelInventory,
    calibration: Any,
    source: SafetensorsModelSource,
    tensors: LocalTensorStore,
    *,
    alpha: float,
    edge_boost: float,
) -> tuple[tuple[str, float], ...]:
    stats = {item.layer: item for item in calibration.stats.layers}
    entries: list[dict[str, Any]] = []
    for block in inventory.blocks:
        for layer in block.quantizable_layers:
            layer_stats = stats[layer.layer]
            with (
                source.read_tensor(layer.weight, device="cpu") as weight,
                tensors.read(layer_stats.input_importance, "cpu") as input_importance,
                tensors.read(layer_stats.output_importance, "cpu") as output_importance,
            ):
                energy = float(
                    (
                        weight.float().square()
                        * output_importance.float()[:, None].clamp_min(1e-12)
                        * input_importance.float()[None, :].clamp_min(1e-12)
                    )
                    .mean()
                    .sqrt()
                )
            entries.append({"block": block.block.index, "path": layer.layer.path, "energy": energy})
    block_medians = {
        block.block.index: max(
            statistics.median(item["energy"] for item in entries if item["block"] == block.block.index), 1e-12
        )
        for block in inventory.blocks
    }
    for item in entries:
        item["relative"] = item["energy"] / block_medians[item["block"]]
    type_medians = {
        path: max(statistics.median(item["relative"] for item in entries if item["path"] == path), 1e-12)
        for path in {item["path"] for item in entries}
    }
    last_block = len(inventory.blocks) - 1
    result = []
    for item in entries:
        residual = max(item["relative"] / type_medians[item["path"]], 1e-12)
        edge_distance = min(item["block"], last_block - item["block"])
        edge_score = 1.0 - edge_distance / max(1.0, last_block / 2.0)
        edge = 1.0 + edge_boost * max(0.0, edge_score)
        score = residual**alpha * _layer_type_multiplier(item["path"]) * edge
        result.append((f"{item['block']}:{item['path']}", score))
    return tuple(result)


def run_resident_quantization(request: ResidentQuantizationRequest) -> ResidentQuantizationResult:
    """Quantize all decoder linears while the source model remains resident on one device."""
    started = time.perf_counter()
    if request.block_forward_batch_size <= 0:
        raise ValueError("resident quantization block forward batch size must be positive")
    if request.calibration_batch_size <= 0:
        raise ValueError("resident quantization calibration batch size must be positive")
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
    if request.layer_order:
        reordered_blocks = []
        for inventory_block in inventory.blocks:
            by_path = {layer.layer.path: layer for layer in inventory_block.quantizable_layers}
            if set(by_path) != set(request.layer_order):
                raise ValueError("requested layer order does not exactly match adapter quantizable layers")
            reordered_blocks.append(
                replace(
                    inventory_block,
                    quantizable_layers=tuple(by_path[path] for path in request.layer_order),
                )
            )
        inventory = replace(inventory, blocks=tuple(reordered_blocks))
    tokens = _token_tensor(request.token_ids, request.device)
    quality_tokens = _token_tensor(
        request.token_ids if request.quality_token_ids is None else request.quality_token_ids,
        request.device,
    )
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
        reference_logits = cast(
            torch.Tensor,
            cast(Any, model)(input_ids=quality_tokens, use_cache=False).logits,
        ).detach()
    text_model = getattr(model, "model", model)
    capture = capture_prefix_invocations(
        decoder_layers[0],
        (lambda: cast(Any, text_model)(input_ids=tokens[:1], use_cache=False),),
    )[0]
    captured_input = capture.positional[0]
    if not isinstance(captured_input, torch.Tensor):
        raise TypeError("captured first-block hidden state is not a tensor")
    initial_inputs = adapter.run_prefix(model, tokens).detach()
    if not torch.equal(initial_inputs[:1], captured_input):
        raise ValueError("adapter prefix does not match the model's first-block input")
    metadata = capture.keyword

    calibration_values: list[tuple[LayerId, MaterializedLayerCalibration]] = []
    if request.calibration_method in {"online_fisher", "two_phase_fisher"}:
        causal_layers: list[tuple[str, nn.Linear]] = []
        causal_ids: dict[str, LayerId] = {}
        for block_inventory, block in zip(inventory.blocks, decoder_layers, strict=True):
            block_modules = dict(block.named_modules())
            for layer in adapter.quantizable_layers(block, block_inventory.block):
                module = block_modules.get(layer.path)
                if not isinstance(module, nn.Linear):
                    raise TypeError(f"causal calibration target is not a linear layer: {layer}")
                key = f"block.{block_inventory.block.index}.{layer.path}"
                causal_layers.append((key, module))
                causal_ids[key] = layer
        stats = calibrate_causal_model(
            model,
            tuple(
                tokens[start : start + request.calibration_batch_size]
                for start in range(0, tokens.shape[0], request.calibration_batch_size)
            ),
            tuple(causal_layers),
            method=request.calibration_method,
            shrinkage=request.calibration_shrinkage,
        )
        calibration_values.extend((causal_ids[item.path], item) for item in stats)
    elif request.calibration_method == "forward_only":
        calibration_inputs = initial_inputs
        for block_inventory, block in zip(inventory.blocks, decoder_layers, strict=True):
            paths = tuple(layer.path for layer in adapter.quantizable_layers(block, block_inventory.block))

            def calibration_runner(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
                return adapter.run_block(module, value, **metadata)

            stats = calibrate_block(
                block,
                tuple(
                    calibration_inputs[start : start + request.block_forward_batch_size]
                    for start in range(0, calibration_inputs.shape[0], request.block_forward_batch_size)
                ),
                paths,
                calibration_runner,
                method="forward_only",
                shrinkage=request.calibration_shrinkage,
            )
            calibration_values.extend((LayerId(block_inventory.block, item.path), item) for item in stats)
            with torch.no_grad():
                calibration_inputs = _run_block_batched(
                    adapter,
                    block,
                    calibration_inputs,
                    metadata,
                    request.block_forward_batch_size,
                ).detach()
    else:
        raise ValueError(f"unsupported resident calibration method: {request.calibration_method}")
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
        request.calibration_method,
        "float32",
        artifacts,
        tensors,
        total_tokens=tokens.numel(),
    )
    objectives = build_objectives(calibration, ObjectiveConfig(), artifacts)
    sensitivity_profile = (
        _legacy_sensitivity_profile(
            inventory,
            calibration,
            source,
            tensors,
            alpha=request.rank_sensitivity_alpha,
            edge_boost=request.rank_edge_boost,
        )
        if request.allocation_strategy is AllocationStrategy.SENSITIVITY
        else ()
    )
    allocation = RankAllocationConfig(
        target_bpw=request.target_bpw,
        strategy=request.allocation_strategy,
        sensitivity_alpha=request.rank_sensitivity_alpha,
        bounds=RankBoundsConfig(
            multiple=request.rank_multiple,
            floor_fraction_of_uniform=request.rank_floor_fraction,
            ceiling_fraction_of_uniform=request.rank_ceiling_fraction,
            edge_block_boost=request.rank_edge_boost,
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
            request.outliers,
            sensitivity_profile,
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
                    "rank_sensitivity_alpha": request.rank_sensitivity_alpha,
                    "rank_edge_boost": request.rank_edge_boost,
                    "layer_order": request.layer_order,
                    "admm": request.admm,
                    "outliers": request.outliers,
                    "scale_fit": request.scale_fit,
                    "factorized_tuning_epochs": request.factorized_tuning_epochs,
                    "factorized_tuning_batch_size": request.factorized_tuning_batch_size,
                    "factorized_tuning_learning_rate": request.factorized_tuning_learning_rate,
                    "calibration_method": request.calibration_method,
                    "calibration_shrinkage": request.calibration_shrinkage,
                    "calibration_batch_size": request.calibration_batch_size,
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
    layer_container = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layer_container, nn.ModuleList):
        raise TypeError("model does not expose a mutable decoder layer stack")
    for _, completed_block in committed_blocks:
        for state in completed_block.frozen_state.quantized_layers:
            frozen = LayerFreezer().load(
                state,
                tensors,
                device=request.device,
                dtype=compressed_inputs.dtype,
                backend="factorized",
            )
            BlockEditor().install_frozen_layer(
                layer_container[completed_block.block.index],
                state.layer.path,
                frozen.module,
            )
    del decoder_layers
    partial_layer_records = {
        (record.block, record.layer): record
        for record in discovered_records
        if record.kind == "layer" and record.block not in completed_block_indexes
    }
    peak_device_bytes = 0
    factorization_wall_seconds = 0.0
    new_layer_commits = 0
    factor_stage = FactorizationAttemptStage(request.admm, device=request.device)
    outlier_stage = OutlierSelectionStage(
        device=request.device,
        residual_probe_iterations=request.outliers.residual_probe.iterations,
    )
    scale_stage = ScaleFitStage(request.scale_fit, device=request.device)

    for block_plan in plan.blocks:
        if block_plan.block.index in completed_block_indexes:
            continue
        block_started = time.perf_counter()
        block_index = block_plan.block.index
        source_block = layer_container[block_index]
        working_block = adapter.load_block(source, block_plan.block, request.device)
        working_block.eval()
        with torch.no_grad():
            teacher_outputs = _run_block_batched(
                adapter,
                source_block,
                teacher_inputs,
                metadata,
                request.block_forward_batch_size,
            ).detach()
        recorder = BlockLossRecorder()
        recorder.record_source_reference(_mse(teacher_outputs, teacher_outputs))
        recorder.record_block_entry(
            _block_loss(
                adapter,
                working_block,
                compressed_inputs,
                teacher_outputs,
                metadata,
                request.block_forward_batch_size,
            )
        )
        block_output_stats = next(
            (
                item
                for item in calibration.stats.layers
                if item.layer.block.index == block_index and item.layer.path == "mlp.down_proj"
            ),
            next(item for item in calibration.stats.layers if item.layer.block.index == block_index),
        )
        with tensors.read(block_output_stats.output_importance, request.device) as value:
            block_output_importance = value.clone()
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
                    _block_loss(
                        adapter,
                        working_block,
                        compressed_inputs,
                        teacher_outputs,
                        metadata,
                        request.block_forward_batch_size,
                    ),
                )
                continue
            with source.read_tensor(layer_plan.source_weight, device="cpu") as source_weight:
                source_ref = tensors.put("source-layer", {"weight": source_weight})["weight"]
            outliers = execute_stage(
                outlier_stage,
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
            fitted = None
            scales = factorized.factors.scales
            if request.scale_fit.enabled:
                fitted = execute_stage(
                    scale_stage,
                    MaterializedScaleFitStageRequest(
                        ScaleFitRequest(
                            layer_plan.layer,
                            outliers.residual_weight,
                            factorized.factors,
                            layer_plan.objective,
                            outliers.indices,
                        ),
                        layer_plan.objective.input_importance,
                        layer_plan.objective.output_importance,
                    ),
                    context,
                )
                scales = fitted.scales
            mid_ref = scales.mid
            if mid_ref is None:
                raise AssertionError("factorizer omitted required mid scale")
            outlier_indices = None
            outlier_values = None
            outlier_scales = None
            if layer_plan.outliers.count:
                with (
                    tensors.read(outliers.indices, request.device) as indices,
                    tensors.read(outliers.values, request.device) as values,
                ):
                    outlier_indices = indices.clone()
                    outlier_values = values.clone()
                if outliers.scales is not None:
                    with tensors.read(outliers.scales, request.device) as values:
                        outlier_scales = values.clone()
            with (
                tensors.read(factorized.factors.left_latent, request.device) as left,
                tensors.read(factorized.factors.right_latent, request.device) as right,
                tensors.read(scales.pre, request.device) as scale_pre,
                tensors.read(mid_ref, request.device) as scale_mid,
                tensors.read(scales.post, request.device) as scale_post,
            ):
                trainable = TrainableFactorizedLinear(
                    left,
                    right,
                    scale_pre,
                    scale_mid,
                    scale_post,
                    outlier_indices=outlier_indices,
                    outlier_values=outlier_values,
                    outlier_scales=outlier_scales,
                )
            tuning = None
            if request.factorized_tuning_epochs > 0:
                BlockEditor().install_trainable_layer(working_block, layer_plan.layer.path, trainable)
                tuning = tune_factorized(
                    working_block,
                    layer_plan.layer.path,
                    TuningRequest(
                        compressed_inputs,
                        teacher_outputs,
                        request.factorized_tuning_epochs,
                        request.factorized_tuning_batch_size,
                        request.factorized_tuning_learning_rate,
                        output_importance=block_output_importance,
                        seed=logical_seed(request.seed, "factorized-tuning", block_index, layer_plan.layer.path, 0),
                    ),
                    lambda module, value: adapter.run_block(module, value, **metadata),
                )
            frozen_outliers = (
                None
                if layer_plan.outliers.count == 0
                else FrozenOutlierState(outliers.indices, outliers.values, outliers.scales)
            )
            frozen = LayerFreezer().freeze(
                layer_plan.layer,
                trainable,
                tensors,
                outliers=frozen_outliers,
            )
            with (
                tensors.read(source_ref, request.device) as source_value,
                tensors.read(layer_plan.objective.input_importance, request.device) as input_importance,
                tensors.read(layer_plan.objective.output_importance, request.device) as output_importance,
            ):
                final_metrics = reconstruction_metrics(
                    source_value,
                    frozen.module.dense_weight(),
                    input_importance,
                    output_importance,
                )
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
                fitted,
                tuning,
                frozen.state,
                final_metrics,
                layer_plan.estimated_cost,
                0,
                ()
                if request.factorized_tuning_epochs > 0 and request.scale_fit.enabled
                else (("tuning_disabled",) if request.scale_fit.enabled else ("scale_fit_disabled", "tuning_disabled")),
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
                _block_loss(
                    adapter,
                    working_block,
                    compressed_inputs,
                    teacher_outputs,
                    metadata,
                    request.block_forward_batch_size,
                ),
            )
        with torch.no_grad():
            compressed_outputs = _run_block_batched(
                adapter,
                working_block,
                compressed_inputs,
                metadata,
                request.block_forward_batch_size,
            ).detach()
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
            warnings=()
            if request.factorized_tuning_epochs > 0 and request.scale_fit.enabled
            else (("tuning_disabled",) if request.scale_fit.enabled else ("scale_fit_disabled", "tuning_disabled")),
        )
        journal.append("block", block_index, None, committed.reference.artifact_id, identity)
        committed_blocks.append((committed.reference, committed.result))
        teacher_inputs = teacher_outputs
        compressed_inputs = compressed_outputs
        layer_container[block_index] = working_block
        del working_block

    with torch.no_grad():
        compressed_logits = cast(
            torch.Tensor,
            cast(Any, model)(input_ids=quality_tokens, use_cache=False).logits,
        ).detach()
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
    reference_nll = _nll(reference_logits, quality_tokens)
    compressed_nll = _nll(compressed_logits, quality_tokens)
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
        "warnings": [
            *([] if request.scale_fit.enabled else ["scale_fit_disabled"]),
            *([] if request.factorized_tuning_epochs > 0 else ["tuning_disabled"]),
            "single_fixture_quality_only",
        ],
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
