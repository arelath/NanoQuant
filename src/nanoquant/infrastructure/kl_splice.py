"""Dense reconstruction splicing against a cached causal-language-model teacher."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from nanoquant.application.kl_budget import (
    KlBudgetArmResult,
    KlSequenceResult,
    causal_kl_nll_per_sequence_from_logits,
)
from nanoquant.application.layers import (
    FrozenReferenceLinear,
    LayerFreezer,
    SharedInputGroupFreezer,
    SharedInputProjectionView,
)
from nanoquant.config.codec import from_dict
from nanoquant.domain.models import ArtifactRef, ArtifactTypes, LayerId, QuantizationPlan
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import (
    CommitIdentity,
    latest_complete_identity,
    load_committed_block,
)
from nanoquant.infrastructure.frozen_model_loader import LoadedFrozenModel
from nanoquant.infrastructure.global_tuning import active_global_tuning, load_global_tuning
from nanoquant.infrastructure.tensor_store import LocalTensorStore


@dataclass(frozen=True, slots=True)
class SpliceReconstruction:
    layer: LayerId
    weight: torch.Tensor
    bias: torch.Tensor | None
    weighted_normalized_squared_error: float


@dataclass(frozen=True, slots=True)
class SpliceReconstructionSet:
    layers: tuple[SpliceReconstruction, ...]
    unit_members: tuple[tuple[str, tuple[LayerId, ...]], ...]
    unit_weighted_normalized_squared_errors: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class LoadedSpliceReconstructionRun:
    reconstructions: SpliceReconstructionSet
    identity: CommitIdentity
    global_tuning: ArtifactRef | None


def _module_at_path(block: nn.Module, path: str) -> nn.Module:
    current: nn.Module = block
    for part in path.split("."):
        child = current[part] if isinstance(current, nn.ModuleDict) else getattr(current, part, None)
        if not isinstance(child, nn.Module):
            raise KeyError(f"module path not found: {path}")
        current = child
    return current


def _decoder_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("model does not expose a supported decoder layer stack")
    return tuple(layers)


def collect_splice_reconstructions(loaded: LoadedFrozenModel) -> SpliceReconstructionSet:
    """Materialize per-logical-layer dense reconstructions from committed factors."""

    blocks = _decoder_layers(loaded.model)
    reconstructions: list[SpliceReconstruction] = []
    unit_members: list[tuple[str, tuple[LayerId, ...]]] = []
    unit_errors: list[tuple[str, float]] = []
    for block_result, block in zip(loaded.blocks, blocks, strict=True):
        for layer_result in block_result.layers:
            module = _module_at_path(block, layer_result.layer.path)
            if not isinstance(module, FrozenReferenceLinear):
                raise TypeError(f"frozen splice layer is not reconstructable: {layer_result.layer.path}")
            reconstructions.append(
                SpliceReconstruction(
                    layer_result.layer,
                    module.dense_weight().detach().cpu().clone(),
                    None if module.bias is None else module.bias.detach().cpu().clone(),
                    layer_result.final_reconstruction.export_weighted_normalized_error,
                )
            )
            unit_id = f"{layer_result.layer.block.index}:{layer_result.layer.path}"
            unit_members.append((unit_id, (layer_result.layer,)))
            unit_errors.append((unit_id, layer_result.final_reconstruction.export_weighted_normalized_error))
        for group_result in block_result.shared_input_groups:
            members = tuple(member.layer for member in group_result.frozen_state.members)
            metrics = dict(group_result.member_reconstruction)
            for member_slice in group_result.frozen_state.members:
                view = _module_at_path(block, member_slice.layer.path)
                if not isinstance(view, SharedInputProjectionView) or not isinstance(view.owner, FrozenReferenceLinear):
                    raise TypeError(f"frozen shared-input splice member is invalid: {member_slice.layer.path}")
                owner = view.owner
                bias = (
                    None
                    if owner.bias is None
                    else owner.bias[member_slice.row_start : member_slice.row_end].detach().cpu().clone()
                )
                reconstructions.append(
                    SpliceReconstruction(
                        member_slice.layer,
                        owner.dense_weight()[member_slice.row_start : member_slice.row_end].detach().cpu().clone(),
                        bias,
                        metrics[member_slice.layer].export_weighted_normalized_error,
                    )
                )
            unit_id = f"{group_result.block.index}:{group_result.name}"
            unit_members.append((unit_id, members))
            unit_errors.append((unit_id, group_result.final_reconstruction.export_weighted_normalized_error))
    return SpliceReconstructionSet(tuple(reconstructions), tuple(unit_members), tuple(unit_errors))


def load_splice_reconstructions_from_run(
    run_output: str | Path,
    expected_blocks: int,
    *,
    device: str,
    source: str,
    revision: str,
    model_config_hash: str,
    use_global_tuning: bool = False,
) -> LoadedSpliceReconstructionRun:
    """Materialize committed splice weights without constructing a candidate model shell."""

    if expected_blocks <= 0:
        raise ValueError("splice reconstruction block count must be positive")
    run_root = Path(run_output)
    artifacts = LocalArtifactStore(run_root / "artifacts")
    tensors = LocalTensorStore(artifacts)
    records = [
        json.loads(line)
        for line in (run_root / "state" / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identity, block_records = latest_complete_identity(records, expected_blocks)
    if identity.model_hash != model_config_hash:
        raise ValueError("splice run commit identity belongs to a different model config")
    plan_descriptor = artifacts.validate(identity.plan_hash)
    if plan_descriptor.artifact_type != ArtifactTypes.QUANTIZATION_PLAN:
        raise ValueError("splice run plan reference is not a quantization plan")
    plan_payload = json.loads(
        (artifacts.path_for(identity.plan_hash) / "plan.json").read_text(encoding="utf-8")
    )
    plan = from_dict(QuantizationPlan, plan_payload, path="plan")
    if (
        plan.model.source != source
        or plan.model.revision != revision
        or plan.model.config_hash != model_config_hash
        or len(plan.blocks) != expected_blocks
    ):
        raise ValueError("splice run plan belongs to a different model or block inventory")
    committed = tuple(
        load_committed_block(
            ArtifactRef(
                ArtifactTypes.BLOCK_RESULT,
                str(block_records[index]["artifact_id"]),
                1,
            ),
            artifacts,
            identity,
        ).result
        for index in range(expected_blocks)
    )
    global_reference = active_global_tuning(run_root) if use_global_tuning else None
    global_result = (
        None
        if global_reference is None
        else load_global_tuning(global_reference, artifacts).result
    )
    if use_global_tuning and global_reference is None:
        raise ValueError("tuned KL splice profile requires an active global tuning artifact")
    source_blocks = tuple(block.teacher_outputs.artifact for block in committed)
    if global_result is not None:
        if global_result.source_blocks != source_blocks:
            raise ValueError("global tuning result does not match the splice run's committed blocks")
        if tuple(state.block.index for state in global_result.tuned_blocks) != tuple(range(expected_blocks)):
            raise ValueError("global tuning result does not contain complete contiguous splice blocks")
    frozen_blocks = (
        tuple(block.frozen_state for block in committed)
        if global_result is None
        else global_result.tuned_blocks
    )
    reconstructions: list[SpliceReconstruction] = []
    unit_members: list[tuple[str, tuple[LayerId, ...]]] = []
    unit_errors: list[tuple[str, float]] = []
    layer_freezer = LayerFreezer()
    group_freezer = SharedInputGroupFreezer()
    with torch.inference_mode():
        for block_result, frozen_block in zip(committed, frozen_blocks, strict=True):
            if block_result.block != frozen_block.block:
                raise ValueError("splice block result and selected frozen state differ")
            layer_states = {state.layer: state for state in frozen_block.quantized_layers}
            group_states = {state.name: state for state in frozen_block.shared_input_groups}
            result_layers = {result.layer for result in block_result.layers}
            result_groups = {result.name for result in block_result.shared_input_groups}
            if set(layer_states) != result_layers or set(group_states) != result_groups:
                raise ValueError("splice result and frozen-state owner inventories differ")
            for layer_result in block_result.layers:
                layer_state = layer_states.get(layer_result.layer)
                if layer_state is None:
                    raise ValueError(f"splice frozen state is missing layer {layer_result.layer}")
                frozen = layer_freezer.load(
                    layer_state,
                    tensors,
                    device=device,
                    backend="dense",
                    compact_dense=True,
                )
                reconstructions.append(
                    SpliceReconstruction(
                        layer_result.layer,
                        frozen.module.dense_weight().detach().cpu().clone(),
                        None if frozen.module.bias is None else frozen.module.bias.detach().cpu().clone(),
                        layer_result.final_reconstruction.export_weighted_normalized_error,
                    )
                )
                unit_id = f"{layer_result.layer.block.index}:{layer_result.layer.path}"
                unit_members.append((unit_id, (layer_result.layer,)))
                unit_errors.append((unit_id, layer_result.final_reconstruction.export_weighted_normalized_error))
                del frozen
            for group_result in block_result.shared_input_groups:
                group_state = group_states.get(group_result.name)
                if group_state is None:
                    raise ValueError(f"splice frozen state is missing group {group_result.name}")
                if group_state.members != group_result.frozen_state.members:
                    raise ValueError(f"splice frozen group members differ for {group_result.name}")
                frozen_group = group_freezer.load(
                    group_state,
                    tensors,
                    device=device,
                    backend="dense",
                )
                dense = frozen_group.owner.dense_weight()
                metrics = dict(group_result.member_reconstruction)
                for member in group_state.members:
                    bias = (
                        None
                        if frozen_group.owner.bias is None
                        else frozen_group.owner.bias[member.row_start : member.row_end].detach().cpu().clone()
                    )
                    reconstructions.append(
                        SpliceReconstruction(
                            member.layer,
                            dense[member.row_start : member.row_end].detach().cpu().clone(),
                            bias,
                            metrics[member.layer].export_weighted_normalized_error,
                        )
                    )
                members = tuple(member.layer for member in group_state.members)
                unit_id = f"{group_state.block.index}:{group_state.name}"
                unit_members.append((unit_id, members))
                unit_errors.append((unit_id, group_result.final_reconstruction.export_weighted_normalized_error))
                del dense, frozen_group
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    if len(reconstructions) != sum(
        len(block.layers) + sum(len(group.frozen_state.members) for group in block.shared_input_groups)
        for block in committed
    ):
        raise ValueError("splice reconstruction inventory is incomplete")
    return LoadedSpliceReconstructionRun(
        SpliceReconstructionSet(
            tuple(reconstructions),
            tuple(unit_members),
            tuple(unit_errors),
        ),
        identity,
        global_reference,
    )


@dataclass(slots=True)
class _OriginalLinear:
    module: nn.Linear
    weight: torch.Tensor
    bias: torch.Tensor | None


class DenseKlSpliceEvaluator:
    """Apply one reconstruction arm, evaluate it, and restore clean teacher weights."""

    def __init__(
        self,
        teacher: nn.Module,
        reconstructions: SpliceReconstructionSet,
        token_ids: torch.Tensor,
        *,
        device: str,
        batch_size: int = 1,
        token_chunk_size: int = 128,
        teacher_cache_dtype: torch.dtype = torch.float16,
        teacher_cache_mode: Literal["cpu", "on_the_fly"] = "cpu",
    ) -> None:
        if token_ids.ndim != 2 or token_ids.shape[1] < 2 or batch_size <= 0:
            raise ValueError("KL splice evaluator token dimensions or batch size are invalid")
        if teacher_cache_mode not in {"cpu", "on_the_fly"}:
            raise ValueError("KL teacher cache mode must be cpu or on_the_fly")
        self.teacher = teacher
        self.reconstructions = reconstructions
        self.token_ids = token_ids.detach().cpu()
        self.device = device
        self.batch_size = batch_size
        self.token_chunk_size = token_chunk_size
        self.teacher_cache_dtype = teacher_cache_dtype
        self.teacher_cache_mode = teacher_cache_mode
        self._blocks = _decoder_layers(teacher)
        self._by_layer = {item.layer: item for item in reconstructions.layers}
        self._unit_members = dict(reconstructions.unit_members)
        self._unit_normalized_squared_errors = dict(reconstructions.unit_weighted_normalized_squared_errors)
        self._originals: dict[LayerId, _OriginalLinear] = {}
        for layer in self._by_layer:
            module = _module_at_path(self._blocks[layer.block.index], layer.path)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"teacher splice target is not a dense linear: {layer.path}")
            self._originals[layer] = _OriginalLinear(
                module,
                module.weight.detach().cpu().clone(),
                None if module.bias is None else module.bias.detach().cpu().clone(),
            )
        self._teacher_log_probs: tuple[torch.Tensor, ...] = ()
        self._baseline_nll = math.nan

    @property
    def baseline_negative_log_likelihood(self) -> float:
        if math.isnan(self._baseline_nll) and self.teacher_cache_mode == "cpu":
            self.cache_teacher()
        elif math.isnan(self._baseline_nll):
            self._baseline_nll = self._measure_baseline_nll()
        return self._baseline_nll

    def teacher_cache_state(self) -> tuple[float, tuple[torch.Tensor, ...]]:
        """Expose the immutable CPU cache after validating it is materialized."""

        if not self._teacher_log_probs:
            self.cache_teacher()
        return self._baseline_nll, self._teacher_log_probs

    def install_teacher_cache(
        self,
        baseline_negative_log_likelihood: float,
        batches: tuple[torch.Tensor, ...],
    ) -> None:
        """Install a persistent cache only when it exactly matches this request."""

        if not math.isfinite(baseline_negative_log_likelihood):
            raise ValueError("KL teacher-cache baseline NLL must be finite")
        expected_starts = tuple(range(0, self.token_ids.shape[0], self.batch_size))
        if len(batches) != len(expected_starts):
            raise ValueError("KL teacher-cache batch count differs from the token request")
        vocabulary_size: int | None = None
        for batch, start in zip(batches, expected_starts, strict=True):
            expected_rows = min(self.batch_size, self.token_ids.shape[0] - start)
            if (
                batch.device.type != "cpu"
                or batch.dtype != self.teacher_cache_dtype
                or batch.ndim != 3
                or batch.shape[0] != expected_rows
                or batch.shape[1] != self.token_ids.shape[1]
                or batch.shape[2] <= 0
            ):
                raise ValueError("KL teacher-cache tensor inventory differs from the token request")
            if vocabulary_size is None:
                vocabulary_size = batch.shape[2]
            elif vocabulary_size != batch.shape[2]:
                raise ValueError("KL teacher-cache vocabulary dimensions are inconsistent")
        self._teacher_log_probs = tuple(batch.contiguous() for batch in batches)
        self._baseline_nll = baseline_negative_log_likelihood

    def _measure_baseline_nll(self) -> float:
        total_nll = 0.0
        total_tokens = 0
        self.teacher.eval()
        with torch.no_grad():
            for start in range(0, self.token_ids.shape[0], self.batch_size):
                batch = self.token_ids[start : start + self.batch_size].to(self.device)
                logits = cast(Any, self.teacher)(input_ids=batch, use_cache=False).logits
                log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                labels = batch[:, 1:].reshape(-1, 1)
                total_nll -= float(log_probs.reshape(-1, log_probs.shape[-1]).gather(1, labels).sum())
                total_tokens += labels.numel()
                del logits, log_probs
        if total_tokens <= 0:
            raise ValueError("KL teacher baseline contains no next-token targets")
        return total_nll / total_tokens

    def cache_teacher(self) -> None:
        cached: list[torch.Tensor] = []
        total_nll = 0.0
        total_tokens = 0
        self.teacher.eval()
        with torch.no_grad():
            for start in range(0, self.token_ids.shape[0], self.batch_size):
                batch = self.token_ids[start : start + self.batch_size].to(self.device)
                logits = cast(Any, self.teacher)(input_ids=batch, use_cache=False).logits
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                labels = batch[:, 1:].reshape(-1, 1)
                shifted = log_probs[:, :-1]
                total_nll -= float(shifted.reshape(-1, shifted.shape[-1]).gather(1, labels).sum())
                total_tokens += labels.numel()
                cached.append(log_probs.to(dtype=self.teacher_cache_dtype, device="cpu"))
                del logits, log_probs
        if total_tokens <= 0:
            raise ValueError("KL teacher cache contains no next-token targets")
        self._teacher_log_probs = tuple(cached)
        self._baseline_nll = total_nll / total_tokens

    def _selected_layers(self, arm: str) -> tuple[LayerId, ...]:
        available = tuple(self._by_layer)
        if arm == "full":
            return available
        if arm.startswith("block:"):
            block = int(arm[6:])
            selected = tuple(layer for layer in available if layer.block.index == block)
        elif arm.startswith("type:"):
            unit_type = arm[5:]
            selected = tuple(layer for layer in available if layer.path == unit_type)
            if not selected:
                selected = tuple(
                    member
                    for unit_id, members in self._unit_members.items()
                    if unit_id.split(":", 1)[1] == unit_type
                    for member in members
                )
        elif arm.startswith("unit:"):
            unit_id = arm[5:]
            selected = self._unit_members.get(unit_id, ())
        else:
            raise ValueError(f"unsupported KL splice arm: {arm}")
        if not selected:
            raise ValueError(f"KL splice arm selects no reconstruction: {arm}")
        return selected

    def _install(self, layers: tuple[LayerId, ...]) -> None:
        with torch.no_grad():
            for layer in layers:
                reconstruction = self._by_layer[layer]
                target = self._originals[layer].module
                target.weight.copy_(reconstruction.weight.to(device=target.weight.device, dtype=target.weight.dtype))
                if reconstruction.bias is None:
                    cast(Any, target).bias = None
                else:
                    cast(Any, target).bias = nn.Parameter(
                        reconstruction.bias.to(device=target.weight.device, dtype=target.weight.dtype),
                        requires_grad=False,
                    )

    def _restore(self, layers: tuple[LayerId, ...]) -> None:
        with torch.no_grad():
            for layer in layers:
                original = self._originals[layer]
                original.module.weight.copy_(
                    original.weight.to(device=original.module.weight.device, dtype=original.module.weight.dtype)
                )
                cast(Any, original.module).bias = (
                    None
                    if original.bias is None
                    else nn.Parameter(
                        original.bias.to(device=original.module.weight.device, dtype=original.module.weight.dtype),
                        requires_grad=False,
                    )
                )

    def _evaluate_cached(self, layers: tuple[LayerId, ...]) -> tuple[KlSequenceResult, ...]:
        self._install(layers)
        sequences: list[KlSequenceResult] = []
        try:
            with torch.no_grad():
                for batch_index, start in enumerate(range(0, self.token_ids.shape[0], self.batch_size)):
                    batch = self.token_ids[start : start + self.batch_size].to(self.device)
                    logits = cast(Any, self.teacher)(input_ids=batch, use_cache=False).logits
                    sequences.extend(
                        causal_kl_nll_per_sequence_from_logits(
                            self._teacher_log_probs[batch_index].to(self.device),
                            logits,
                            batch,
                            token_chunk_size=self.token_chunk_size,
                            teacher_is_log_probs=True,
                        )
                    )
                    del logits
        finally:
            self._restore(layers)
        return tuple(sequences)

    def _evaluate_on_the_fly(self, layers: tuple[LayerId, ...]) -> tuple[KlSequenceResult, ...]:
        sequences: list[KlSequenceResult] = []
        installed = False
        try:
            with torch.no_grad():
                for start in range(0, self.token_ids.shape[0], self.batch_size):
                    batch = self.token_ids[start : start + self.batch_size].to(self.device)
                    teacher_logits = cast(Any, self.teacher)(input_ids=batch, use_cache=False).logits
                    self._install(layers)
                    installed = True
                    student_logits = cast(Any, self.teacher)(input_ids=batch, use_cache=False).logits
                    self._restore(layers)
                    installed = False
                    teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1).to(
                        self.teacher_cache_dtype
                    )
                    sequences.extend(
                        causal_kl_nll_per_sequence_from_logits(
                            teacher_log_probs,
                            student_logits,
                            batch,
                            token_chunk_size=self.token_chunk_size,
                            teacher_is_log_probs=True,
                        )
                    )
                    del teacher_logits, teacher_log_probs, student_logits
        finally:
            if installed:
                self._restore(layers)
        return tuple(sequences)

    def __call__(self, arm: str) -> KlBudgetArmResult:
        layers = self._selected_layers(arm)
        if self.teacher_cache_mode == "cpu":
            if not self._teacher_log_probs:
                self.cache_teacher()
            sequences = self._evaluate_cached(layers)
        else:
            if math.isnan(self._baseline_nll):
                self._baseline_nll = self._measure_baseline_nll()
            sequences = self._evaluate_on_the_fly(layers)
        unit_id = arm[5:] if arm.startswith("unit:") else None
        normalized_squared_error = (
            self._unit_normalized_squared_errors[unit_id] if unit_id is not None else None
        )
        total_tokens = sum(sequence.token_count for sequence in sequences)
        total_nll = math.fsum(sequence.negative_log_likelihood * sequence.token_count for sequence in sequences)
        total_kl = math.fsum(sequence.kl_nats_per_token * sequence.token_count for sequence in sequences)
        return KlBudgetArmResult(
            arm,
            total_nll / total_tokens,
            max(0.0, total_kl / total_tokens),
            total_tokens,
            normalized_squared_error,
            sequences,
        )


__all__ = [
    "DenseKlSpliceEvaluator",
    "LoadedSpliceReconstructionRun",
    "SpliceReconstruction",
    "SpliceReconstructionSet",
    "collect_splice_reconstructions",
    "load_splice_reconstructions_from_run",
]
