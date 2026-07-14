"""Deterministic end-to-end composition over the real rewrite components."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from nanoquant.application.assembly import assemble_frozen_model
from nanoquant.application.calibration import MaterializedLayerCalibration, calibrate_block
from nanoquant.application.calibration_artifacts import build_objectives, persist_calibration
from nanoquant.application.layers import BlockEditor, LayerFreezer, TrainableFactorizedLinear
from nanoquant.application.loss_snapshots import BlockLossRecorder
from nanoquant.application.planning import PlanningRequest, build_quantization_plan, persist_plan
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
    AllocationStrategy,
    ObjectiveConfig,
    ObservabilityConfig,
    OutlierConfig,
    ProfilingConfig,
    ProfilingLevel,
    RankAllocationConfig,
    RankBoundsConfig,
    RankRetryConfig,
)
from nanoquant.domain.metrics import reconstruction_metrics
from nanoquant.domain.models import (
    AttemptSummary,
    BlockId,
    BlockInventory,
    BlockResult,
    ComponentRef,
    DatasetIdentity,
    FactorizationRequest,
    FrozenBlockState,
    FrozenModelResult,
    LayerId,
    LayerInventory,
    LayerResult,
    ModelIdentity,
    ModelInventory,
    OutlierSelectionRequest,
    QuantizationPlan,
    ScaleFitRequest,
    SourceTensor,
    TensorId,
)
from nanoquant.domain.profiling import NULL_RECORDER, PhaseRecorder
from nanoquant.domain.runs import BudgetState, RunStatus
from nanoquant.domain.seeds import logical_seed
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity, commit_block, commit_layer
from nanoquant.infrastructure.environment import capture_environment
from nanoquant.infrastructure.profiling import profiled_run
from nanoquant.infrastructure.progress import ProgressJournal
from nanoquant.infrastructure.resident_executor import Cancellation, ResidentExecutor
from nanoquant.infrastructure.run_session import open_run_session
from nanoquant.infrastructure.runs import (
    RunDirectory,
    initial_manifest_from_resolved,
    launcher_provenance,
    transition,
)
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.infrastructure.tiny_model import TinyCausalTransformer, TinyModelAdapter, TinyModelConfig
from nanoquant.ports.event_sink import EventSink

_DEFAULT_PROFILING = ProfilingConfig()
_DEFAULT_OBSERVABILITY = ObservabilityConfig()


@dataclass(frozen=True, slots=True)
class TinyPipelineResult:
    frozen_model: FrozenModelResult
    plan: QuantizationPlan
    blocks: tuple[BlockResult, ...]
    teacher_logits: torch.Tensor
    compressed_logits: torch.Tensor
    report: str
    elapsed_seconds: float
    run_root: Path


def _module(block: nn.Module, path: str) -> nn.Linear:
    value = dict(block.named_modules()).get(path)
    if not isinstance(value, nn.Linear):
        raise TypeError(f"expected linear at {path}")
    return value


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction.detach().float() - target.detach().float()).square().mean())


def _run_tiny_pipeline(
    root: str | Path,
    *,
    seed: int,
    recorder: PhaseRecorder,
    profiling: ProfilingConfig,
    events: EventSink,
    run_id: str,
) -> TinyPipelineResult:
    started = time.perf_counter()
    root = Path(root)
    micro_recorder = recorder if profiling.level is ProfilingLevel.MICRO else NULL_RECORDER
    artifacts = LocalArtifactStore(root / "artifacts", recorder=micro_recorder)
    tensors = LocalTensorStore(artifacts)
    context = StageContext(run_id, ResidentExecutor(), artifacts, tensors, events, Cancellation(), recorder)
    config = TinyModelConfig(vocabulary_size=32, hidden_size=8, intermediate_size=12, block_count=2)
    teacher = TinyCausalTransformer(config, seed=seed)
    working = TinyCausalTransformer(config, seed=seed)
    adapter = TinyModelAdapter(config)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    layer_paths = tuple(layer.path for layer in adapter.quantizable_layers(teacher.blocks[0], BlockId(0)))

    source_values = {}
    for block_index, block in enumerate(teacher.blocks):
        for path in layer_paths:
            source_values[f"block_{block_index}.{path}.weight"] = _module(block, path).weight.detach()
    source_refs = tensors.put("tiny-source-weights", source_values)
    model_hash = hashlib.sha256(str(seed).encode()).hexdigest()
    model_identity = ModelIdentity(
        "offline/tiny",
        f"seed-{seed}",
        f"sha256:{model_hash}",
        "offline/tiny-tokenizer",
        "v1",
        ComponentRef("tiny", "1"),
    )

    teacher_inputs = teacher.embed(tokens).detach()
    calibration_materialized: list[tuple[LayerId, MaterializedLayerCalibration]] = []
    block_inventories = []
    calibration_inputs = teacher_inputs
    for block_index, block in enumerate(teacher.blocks):
        block_id = BlockId(block_index)
        calibrated = calibrate_block(
            block,
            (calibration_inputs,),
            layer_paths,
            lambda module, value: module(value),
            method="forward_only",
            recorder=micro_recorder,
        )
        calibration_materialized.extend((LayerId(block_id, stats.path), stats) for stats in calibrated)
        layer_inventory = []
        source_tensors = []
        for path in layer_paths:
            layer_id = LayerId(block_id, path)
            key = f"block_{block_index}.{path}.weight"
            reference = source_refs[key]
            source = SourceTensor(
                TensorId(layer_id, "weight"),
                key,
                reference.artifact.artifact_id,
                reference.spec,
                reference.content_hash,
            )
            source_tensors.append(source)
            layer_inventory.append(
                LayerInventory(layer_id, source, None, reference.spec.shape[1], reference.spec.shape[0])
            )
        block_inventories.append(BlockInventory(block_id, tuple(source_tensors), tuple(layer_inventory)))
        with torch.no_grad():
            calibration_inputs = block(calibration_inputs).detach()
    inventory = ModelInventory(
        1,
        model_identity,
        tuple(block_inventories),
        (),
        sum(value.numel() * value.element_size() for value in source_values.values()),
    )
    dataset_identity = DatasetIdentity(
        "sha256:tiny-dataset", ("offline/tiny",), ("v1",), "sha256:tiny-tokenizer", "tokens-v1"
    )
    calibration = persist_calibration(
        tuple(calibration_materialized),
        model_identity,
        dataset_identity,
        "forward_only",
        "float32",
        artifacts,
        tensors,
        total_tokens=tokens.numel(),
    )
    objectives = build_objectives(calibration, ObjectiveConfig(), artifacts)
    allocation = RankAllocationConfig(
        target_bpw=8.0,
        strategy=AllocationStrategy.UNIFORM,
        bounds=RankBoundsConfig(
            multiple=1, floor_fraction_of_uniform=1.0, ceiling_fraction_of_uniform=1.0, edge_block_boost=0
        ),
        retry=RankRetryConfig(enabled=False, maximum_attempts=1),
    )
    plan = build_quantization_plan(
        PlanningRequest(
            inventory, calibration.stats, calibration.reference, objectives.objectives, allocation, OutlierConfig()
        )
    )
    persisted_plan = persist_plan(plan, artifacts)
    identity = CommitIdentity("tiny-config-v1", model_identity.config_hash, persisted_plan.reference.artifact_id)
    journal = ProgressJournal(root / "state", run_id, artifacts)
    budget = BudgetState(plan.planned_cost.total, 0, 0)
    teacher_inputs = teacher.embed(tokens).detach()
    compressed_inputs = working.embed(tokens).detach()
    committed_blocks = []

    for block_plan in plan.blocks:
        block_index = block_plan.block.index
        source_block = teacher.blocks[block_index]
        working_block = working.blocks[block_index]
        with torch.no_grad():
            teacher_outputs = source_block(teacher_inputs).detach()
        loss_recorder = BlockLossRecorder()
        loss_recorder.record_source_reference(_mse(source_block(teacher_inputs), teacher_outputs))
        loss_recorder.record_block_entry(_mse(working_block(compressed_inputs), teacher_outputs))
        layer_results = []
        frozen_states = []
        for layer_plan in block_plan.layers:
            path = layer_plan.layer.path
            source_ref = source_refs[f"block_{block_index}.{path}.weight"]
            outlier_request = OutlierSelectionRequest(
                layer_plan.layer,
                source_ref,
                layer_plan.objective,
                layer_plan.outliers,
                layer_plan.rank,
                logical_seed(seed, "outliers", block_index, path, 0),
            )
            outliers = execute_stage(OutlierSelectionStage(), outlier_request, context)
            factor_objective = replace(
                layer_plan.objective,
                input_importance=outliers.factor_input_importance,
            )
            factorized = execute_stage(
                FactorizationAttemptStage(),
                FactorizationRequest(
                    1,
                    layer_plan.layer,
                    source_ref,
                    outliers.residual_weight,
                    factor_objective,
                    layer_plan.rank,
                    logical_seed(seed, "factorize", block_index, path, 0),
                    "tiny-factor-v1",
                    outliers.factor_generator_state,
                ),
                context,
            )
            fitted = execute_stage(
                ScaleFitStage(),
                MaterializedScaleFitStageRequest(
                    ScaleFitRequest(
                        layer_plan.layer,
                        outliers.residual_weight,
                        factorized.factors,
                        factor_objective,
                        outliers.indices,
                    ),
                    factor_objective.input_importance,
                    layer_plan.objective.output_importance,
                ),
                context,
            )
            if fitted.scales.mid is None:
                raise AssertionError("tiny factorization omitted mid scale")
            with (
                tensors.read(factorized.factors.left_latent) as left,
                tensors.read(factorized.factors.right_latent) as right,
                tensors.read(fitted.scales.pre) as scale_pre,
                tensors.read(fitted.scales.mid) as scale_mid,
                tensors.read(fitted.scales.post) as scale_post,
            ):
                trainable = TrainableFactorizedLinear(left, right, scale_pre, scale_mid, scale_post)
            parent, name = path.rsplit(".", 1)
            container = dict(working_block.named_modules())[parent]
            if isinstance(container, nn.ModuleDict):
                container[name] = trainable
            else:
                setattr(container, name, trainable)
            tuning = tune_factorized(
                working_block,
                path,
                TuningRequest(compressed_inputs, teacher_outputs, 1, 2, 1e-2),
                lambda module, value: module(value),
                micro_recorder,
            )
            frozen = LayerFreezer().freeze(layer_plan.layer, trainable, tensors)
            BlockEditor().install_frozen_layer(working_block, path, frozen.module)
            frozen_states.append(frozen.state)
            with (
                tensors.read(layer_plan.objective.input_importance) as input_importance,
                tensors.read(layer_plan.objective.output_importance) as output_importance,
            ):
                source_weight = _module(source_block, path).weight.detach()
                final_metrics = reconstruction_metrics(
                    source_weight, frozen.module.dense_weight(), input_importance, output_importance
                )
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
                (),
            )
            committed_layer = commit_layer(layer_result, artifacts, identity)
            journal.append("layer", block_index, path, committed_layer.reference.artifact_id, identity)
            layer_results.append(layer_result)
            budget = replace(budget, accepted_bits=budget.accepted_bits + layer_plan.estimated_cost.total)
            loss_recorder.record_after_layer(
                layer_plan.layer, _mse(working_block(compressed_inputs), teacher_outputs)
            )
        with torch.no_grad():
            compressed_outputs = working_block(compressed_inputs).detach()
        loss_recorder.record_final_frozen_pre_kd(_mse(compressed_outputs, teacher_outputs))
        frozen_block = FrozenBlockState(block_plan.block, tuple(frozen_states), ())
        committed = commit_block(
            block_plan.block,
            tuple(layer_results),
            frozen_block,
            loss_recorder.finalize(),
            teacher_outputs,
            compressed_outputs,
            budget.retry_bits_spent,
            artifacts,
            identity,
        )
        journal.append("block", block_index, None, committed.reference.artifact_id, identity)
        committed_blocks.append((committed.reference, committed.result))
        teacher_inputs = teacher_outputs
        compressed_inputs = compressed_outputs
    with torch.no_grad():
        teacher_logits = teacher.lm_head(teacher.final_norm(teacher_inputs)).detach()
        compressed_logits = working.lm_head(working.final_norm(compressed_inputs)).detach()
    frozen_model = assemble_frozen_model(
        model_identity,
        persisted_plan.reference,
        tuple(committed_blocks),
        (),
        sum(value.numel() for value in source_values.values()),
    )
    report = render_reconstruction_tables(tuple(result for _, result in committed_blocks))
    (root / "report.md").write_text(report, encoding="utf-8")
    return TinyPipelineResult(
        frozen_model,
        plan,
        tuple(result for _, result in committed_blocks),
        teacher_logits,
        compressed_logits,
        report,
        time.perf_counter() - started,
        root,
    )


def run_tiny_pipeline(
    root: str | Path,
    *,
    seed: int = 0,
    profiling: ProfilingConfig = _DEFAULT_PROFILING,
    observability: ObservabilityConfig = _DEFAULT_OBSERVABILITY,
    registry_root: Path | None = None,
) -> TinyPipelineResult:
    output = Path(root)
    resolved = {
        "component": "tiny-pipeline",
        "seed": seed,
        "profiling": to_dict(profiling),
        "observability": to_dict(observability),
    }
    resolved_hash = "sha256:" + hashlib.sha256(canonical_json(resolved).encode()).hexdigest()
    proposed = initial_manifest_from_resolved(
        resolved_hash,
        resolved,
        launcher_provenance(__file__, None),
        capture_environment(),
    )
    with open_run_session(
        output,
        manifest=proposed,
        observability=observability,
        registry_root=registry_root,
        console=False,
    ) as session:
        directory = RunDirectory(output.parent, output.name)
        manifest = transition(session.manifest, RunStatus.RUNNING)
        directory.write_manifest(manifest)
        session.events.emit("run", "info", "run.started", config_hash=manifest.config_hash)
        try:
            with profiled_run(profiling, output, session.events, run_id=session.run_id) as recorder:
                with recorder.phase("run"):
                    result = _run_tiny_pipeline(
                        output,
                        seed=seed,
                        recorder=recorder,
                        profiling=profiling,
                        events=session.events,
                        run_id=session.run_id,
                    )
        except BaseException as exc:
            session.events.emit(
                "run",
                "error",
                "run.failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            manifest = transition(
                manifest,
                RunStatus.FAILED,
                failure={"type": type(exc).__name__, "message": str(exc)},
            )
            directory.write_manifest(manifest)
            raise
        session.events.emit("run", "info", "run.completed")
        manifest = transition(manifest, RunStatus.COMPLETED)
        directory.write_manifest(manifest)
        return result
