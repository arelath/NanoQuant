"""Selective additive rank expansion for a completed packed NanoQuant model."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nanoquant.domain.factorization import factorize_admm
from nanoquant.domain.linear_math import functional_dense_reconstruction
from nanoquant.domain.metrics import reconstruction_metrics
from nanoquant.domain.models import ArtifactRef, ArtifactTypes, BlockResult
from nanoquant.domain.rank_expansion import fit_residual_middle_scales
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import latest_complete_identity, load_committed_block
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.runtime import (
    LogicalLayerState,
    OpenPackedArtifact,
    PackedLayerState,
    open_packed_artifact,
    pack_logical_layer,
    packed_row_stride_bytes,
    write_packed_artifact_stream,
)


@dataclass(frozen=True, slots=True)
class RankExpansionRequest:
    parent_run: Path
    source_packed: Path
    snapshot: Path
    output_packed: Path
    report_output: Path
    source: str
    revision: str
    expected_blocks: int
    layer_suffix: str = "self_attn.v_proj"
    bit_multiplier: float = 1.30
    rank_multiple: int = 32
    device: str = "cuda:0"
    seed: int = 0
    outer_iterations: int = 800
    inner_iterations: int = 5
    regularization: float = 3e-2
    penalty_schedule: str = "cubic"
    convergence_check_interval: int = 100
    early_stop_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.expected_blocks <= 0:
            raise ValueError("rank expansion expected block count must be positive")
        if not self.layer_suffix or self.layer_suffix.startswith("."):
            raise ValueError("rank expansion layer suffix must be a canonical relative path")
        if self.bit_multiplier <= 1:
            raise ValueError("rank expansion bit multiplier must exceed one")
        if self.rank_multiple <= 0:
            raise ValueError("rank expansion rank multiple must be positive")
        if self.outer_iterations < 0 or self.inner_iterations <= 0:
            raise ValueError("rank expansion ADMM iteration settings are invalid")


def _journal_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"parent rank-expansion journal line {number} is invalid") from error
        if not isinstance(value, dict):
            raise ValueError(f"parent rank-expansion journal line {number} is not an object")
        records.append(value)
    return records


def _parent_blocks(request: RankExpansionRequest) -> tuple[tuple[BlockResult, ...], LocalTensorStore]:
    artifacts = LocalArtifactStore(
        request.parent_run / "artifacts",
        use_persistent_validation_cache=False,
    )
    tensors = LocalTensorStore(artifacts)
    identity, records = latest_complete_identity(
        _journal_records(request.parent_run / "state" / "journal.jsonl"),
        request.expected_blocks,
    )
    blocks = tuple(
        load_committed_block(
            ArtifactRef(ArtifactTypes.BLOCK_RESULT, str(records[index]["artifact_id"]), 1),
            artifacts,
            identity,
        ).result
        for index in range(request.expected_blocks)
    )
    return blocks, tensors


def _packed_tensors(state: PackedLayerState) -> dict[str, torch.Tensor]:
    result = {
        "factor_left_words": state.left_words,
        "factor_right_words": state.right_words,
        "scale_pre": state.scale_pre,
        "scale_mid": state.scale_mid,
        "scale_post": state.scale_post,
    }
    for name, value in (
        ("bias", state.bias),
        ("outlier_indices", state.outlier_indices),
        ("outlier_values", state.outlier_values),
        ("outlier_scales", state.outlier_scales),
    ):
        if value is not None:
            result[name] = value
    return {name: value.detach().cpu().contiguous() for name, value in result.items()}


def _packed_bits_at_rank(state: PackedLayerState, rank: int) -> int:
    current = sum(value.numel() * value.element_size() * 8 for value in _packed_tensors(state).values())
    old_factor_and_mid = (
        state.left_words.numel() * state.left_words.element_size()
        + state.right_words.numel() * state.right_words.element_size()
        + state.scale_mid.numel() * state.scale_mid.element_size()
    )
    new_factor_and_mid = (
        state.spec.out_features * packed_row_stride_bytes(rank)
        + rank * packed_row_stride_bytes(state.spec.in_features)
        + rank * state.scale_mid.element_size()
    )
    return (current // 8 - old_factor_and_mid + new_factor_and_mid) * 8


def _target_rank(state: PackedLayerState, multiplier: float, multiple: int) -> tuple[int, int, int]:
    old_bits = _packed_bits_at_rank(state, state.spec.rank)
    requested_bits = math.ceil(old_bits * multiplier)
    cap = min(state.spec.in_features, state.spec.out_features)
    candidates = range(state.spec.rank + multiple, cap + 1, multiple)
    target = next((rank for rank in candidates if _packed_bits_at_rank(state, rank) >= requested_bits), cap)
    target = min(cap, max(state.spec.rank + 1, target))
    if target != cap and target % multiple:
        raise AssertionError("rank expansion target is not aligned")
    return target, old_bits, _packed_bits_at_rank(state, target)


def _work_identity(request: RankExpansionRequest, source_packed: OpenPackedArtifact) -> str:
    payload = {
        "schema_version": 1,
        "source_descriptor_sha256": hash_file(source_packed.root / "nanoquant-packed-model.json"),
        "parent_run": str(request.parent_run.resolve()),
        "source": request.source,
        "revision": request.revision,
        "expected_blocks": request.expected_blocks,
        "layer_suffix": request.layer_suffix,
        "bit_multiplier": request.bit_multiplier,
        "rank_multiple": request.rank_multiple,
        "seed": request.seed,
        "outer_iterations": request.outer_iterations,
        "inner_iterations": request.inner_iterations,
        "regularization": request.regularization,
        "penalty_schedule": request.penalty_schedule,
        "convergence_check_interval": request.convergence_check_interval,
        "early_stop_tolerance": request.early_stop_tolerance,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_root(request: RankExpansionRequest) -> Path:
    return request.output_packed.with_name(request.output_packed.name + ".rank-expansion")


def _checkpoint_records(path: Path, identity: str) -> dict[int, ArtifactRef]:
    if not path.is_file():
        return {}
    result = {}
    for value in _journal_records(path):
        if value.get("identity") == identity and value.get("kind") == "expanded_layer":
            result[int(value["block"])] = ArtifactRef(
                "rank-expanded-layer",
                str(value["artifact_id"]),
                1,
            )
    return result


def _append_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_expanded_checkpoint(
    reference: ArtifactRef,
    artifacts: LocalArtifactStore,
    source_state: PackedLayerState,
    identity: str,
) -> tuple[PackedLayerState, dict[str, Any]]:
    descriptor = artifacts.validate(reference.artifact_id)
    if descriptor.artifact_type != "rank-expanded-layer":
        raise ValueError("rank expansion checkpoint has the wrong artifact type")
    root = artifacts.path_for(reference.artifact_id)
    result = cast(dict[str, Any], json.loads((root / "result.json").read_text(encoding="utf-8")))
    if result.get("identity") != identity:
        raise ValueError("rank expansion checkpoint identity differs")
    target_rank = int(result["target_rank"])
    with safe_open(root / "state.safetensors", framework="pt", device="cpu") as handle:
        values = {name: handle.get_tensor(name) for name in handle.keys()}
    spec = replace(source_state.spec, rank=target_rank)
    state = PackedLayerState(
        spec,
        source_state.layout,
        values["factor_left_words"],
        values["factor_right_words"],
        values["scale_pre"],
        values["scale_mid"],
        values["scale_post"],
        values.get("bias"),
        values.get("outlier_indices"),
        values.get("outlier_values"),
        values.get("outlier_scales"),
    )
    return state, result


def _commit_expanded_checkpoint(
    state: PackedLayerState,
    result: dict[str, Any],
    artifacts: LocalArtifactStore,
) -> ArtifactRef:
    with artifacts.begin_write("rank-expanded-layer") as writer:
        save_file(_packed_tensors(state), writer.path / "state.safetensors")
        atomic_write_json(writer.path / "result.json", result)
        descriptor = writer.commit()
    return ArtifactRef("rank-expanded-layer", descriptor.artifact_id, 1)


def _expand_layer(
    request: RankExpansionRequest,
    block_index: int,
    source_state: PackedLayerState,
    block: BlockResult,
    source_weights: SafetensorsModelSource,
    tensors: LocalTensorStore,
    identity: str,
) -> tuple[PackedLayerState, dict[str, Any]]:
    layer_result = next(layer for layer in block.layers if layer.layer.path == request.layer_suffix)
    target_rank, old_bits, target_bits = _target_rank(
        source_state,
        request.bit_multiplier,
        request.rank_multiple,
    )
    added_rank = target_rank - source_state.spec.rank
    if added_rank <= 0:
        raise ValueError(f"rank expansion layer is already at its cap: {source_state.spec.name}")
    device = torch.device(request.device)
    logical = source_state.to_logical()
    started = time.perf_counter()
    with (
        source_weights.read_tensor(layer_result.plan.source_weight, str(device)) as target,
        tensors.read(layer_result.plan.objective.input_importance, str(device)) as input_importance,
        tensors.read(layer_result.plan.objective.output_importance, str(device)) as output_importance,
    ):
        target32 = target.float()
        current = functional_dense_reconstruction(
            logical.left_binary.to(device),
            logical.right_binary.to(device),
            logical.scale_pre.to(device),
            logical.scale_mid.to(device),
            logical.scale_post.to(device),
            None if logical.outlier_indices is None else logical.outlier_indices.to(device),
            None if logical.outlier_values is None else logical.outlier_values.to(device),
            None if logical.outlier_scales is None else logical.outlier_scales.to(device),
        ).float()
        residual = target32 - current
        protected = None if logical.outlier_indices is None else logical.outlier_indices.to(device)
        if protected is not None:
            residual[:, protected.long()] = 0
        generator = torch.Generator(device=device).manual_seed(request.seed + block_index * 1_000_003)
        factors = factorize_admm(
            residual,
            input_importance,
            output_importance,
            added_rank,
            generator,
            outer_iterations=request.outer_iterations,
            inner_iterations=request.inner_iterations,
            regularization=request.regularization,
            penalty_schedule=request.penalty_schedule,
            convergence_check_interval=request.convergence_check_interval,
            early_stop_tolerance=request.early_stop_tolerance,
            transpose_wide=False,
        )
        fit = fit_residual_middle_scales(
            residual,
            factors.left_binary,
            factors.right_binary,
            logical.scale_pre.to(device),
            logical.scale_post.to(device),
            input_importance,
            output_importance,
            protected_columns=protected,
        )
        if not fit.accepted or fit.after_error >= fit.before_error:
            raise RuntimeError(f"rank expansion did not improve weighted residual: {source_state.spec.name}")
        exported_middle = fit.scale_mid.to(logical.scale_mid.dtype)
        exported_correction = functional_dense_reconstruction(
            factors.left_binary,
            factors.right_binary,
            logical.scale_pre.to(device),
            exported_middle,
            logical.scale_post.to(device),
            protected,
        ).float()
        candidate = current + exported_correction
        before_metrics = reconstruction_metrics(target32, current, input_importance, output_importance)
        after_metrics = reconstruction_metrics(target32, candidate, input_importance, output_importance)
        if after_metrics.export_weighted_error >= before_metrics.export_weighted_error:
            raise RuntimeError(
                f"rank expansion regressed after storage-dtype conversion: {source_state.spec.name}"
            )
        expanded = LogicalLayerState(
            replace(logical.spec, rank=target_rank),
            torch.cat((logical.left_binary, factors.left_binary.detach().cpu()), dim=1),
            torch.cat((logical.right_binary, factors.right_binary.detach().cpu()), dim=0),
            logical.scale_pre,
            torch.cat(
                (
                    logical.scale_mid,
                    exported_middle.detach().cpu(),
                )
            ),
            logical.scale_post,
            logical.bias,
            logical.outlier_indices,
            logical.outlier_values,
            logical.outlier_scales,
        )
    result = {
        "schema_version": 1,
        "identity": identity,
        "block": block_index,
        "layer": source_state.spec.name,
        "old_rank": source_state.spec.rank,
        "target_rank": target_rank,
        "added_rank": added_rank,
        "old_bits": old_bits,
        "target_bits": target_bits,
        "realized_bit_multiplier": target_bits / old_bits,
        "before": asdict(before_metrics),
        "after": asdict(after_metrics),
        "residual_fit_before_error": fit.before_error,
        "residual_fit_after_error": fit.after_error,
        "admm_iterations_completed": factors.iterations_completed,
        "admm_stopped_early": factors.stopped_early,
        "wall_seconds": time.perf_counter() - started,
    }
    return pack_logical_layer(expanded), result


def _same_optional(left: torch.Tensor | None, right: torch.Tensor | None) -> bool:
    return left is None and right is None or (
        left is not None and right is not None and torch.equal(left, right)
    )


def _verify_derivative(
    source: OpenPackedArtifact,
    output: OpenPackedArtifact,
    suffix: str,
) -> tuple[int, int]:
    exact_non_target = 0
    exact_prefixes = 0
    for source_block, output_block in zip(source.manifest.blocks, output.manifest.blocks, strict=True):
        for source_entry, output_entry in zip(source_block.layers, output_block.layers, strict=True):
            old = source.load_layer(source_entry.spec.name)
            new = output.load_layer(output_entry.spec.name)
            if not source_entry.spec.name.endswith(suffix):
                if old.spec != new.spec or any(
                    not _same_optional(left, right)
                    for left, right in zip(
                        (
                            old.left_words,
                            old.right_words,
                            old.scale_pre,
                            old.scale_mid,
                            old.scale_post,
                            old.bias,
                            old.outlier_indices,
                            old.outlier_values,
                            old.outlier_scales,
                        ),
                        (
                            new.left_words,
                            new.right_words,
                            new.scale_pre,
                            new.scale_mid,
                            new.scale_post,
                            new.bias,
                            new.outlier_indices,
                            new.outlier_values,
                            new.outlier_scales,
                        ),
                        strict=True,
                    )
                ):
                    raise ValueError(f"non-target packed layer changed: {source_entry.spec.name}")
                exact_non_target += 1
                continue
            old_logical = old.to_logical()
            new_logical = new.to_logical()
            rank = old.spec.rank
            if (
                not torch.equal(old_logical.left_binary, new_logical.left_binary[:, :rank])
                or not torch.equal(old_logical.right_binary, new_logical.right_binary[:rank])
                or not torch.equal(old.scale_mid, new.scale_mid[:rank])
                or any(
                    not _same_optional(left, right)
                    for left, right in (
                        (old.scale_pre, new.scale_pre),
                        (old.scale_post, new.scale_post),
                        (old.bias, new.bias),
                        (old.outlier_indices, new.outlier_indices),
                        (old.outlier_values, new.outlier_values),
                        (old.outlier_scales, new.outlier_scales),
                    )
                )
            ):
                raise ValueError(f"target packed layer did not preserve its original prefix: {source_entry.spec.name}")
            exact_prefixes += 1
    return exact_non_target, exact_prefixes


def execute_rank_expansion(
    request: RankExpansionRequest,
    *,
    safe_point: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Create a resumable packed derivative with additive rank only on one layer family."""

    if safe_point is not None:
        safe_point()
    source_packed = open_packed_artifact(request.source_packed, verify_hashes=True)
    identity = _work_identity(request, source_packed)
    if request.output_packed.exists() or request.report_output.exists():
        if not request.output_packed.is_dir() or not request.report_output.is_file():
            raise FileExistsError("rank expansion output exists only partially")
        report = cast(dict[str, Any], json.loads(request.report_output.read_text(encoding="utf-8")))
        if report.get("identity") != identity:
            raise ValueError("existing rank expansion output has a different identity")
        output = open_packed_artifact(request.output_packed, verify_hashes=True)
        _verify_derivative(source_packed, output, request.layer_suffix)
        return report

    blocks, tensors = _parent_blocks(request)
    source_weights = SafetensorsModelSource(
        request.snapshot,
        source=request.source,
        revision=request.revision,
        verify_hashes=False,
    )
    work = _checkpoint_root(request)
    work.mkdir(parents=True, exist_ok=True)
    work_artifacts = LocalArtifactStore(work / "artifacts")
    journal = work / "journal.jsonl"
    completed = _checkpoint_records(journal, identity)
    results: dict[int, dict[str, Any]] = {}
    expanded_by_block: dict[int, PackedLayerState] = {}
    for block_index, block in enumerate(blocks):
        name = f"blocks.{block_index}.{request.layer_suffix}"
        source_state = source_packed.load_layer(name)
        if block_index in completed:
            expanded, result = _load_expanded_checkpoint(
                completed[block_index],
                work_artifacts,
                source_state,
                identity,
            )
        else:
            expanded, result = _expand_layer(
                request,
                block_index,
                source_state,
                block,
                source_weights,
                tensors,
                identity,
            )
            reference = _commit_expanded_checkpoint(expanded, result, work_artifacts)
            _append_checkpoint(
                journal,
                {
                    "schema_version": 1,
                    "kind": "expanded_layer",
                    "identity": identity,
                    "block": block_index,
                    "artifact_id": reference.artifact_id,
                },
            )
        expanded_by_block[block_index] = expanded
        results[block_index] = result
        del expanded, source_state
        gc.collect()
        if torch.cuda.is_available() and request.device.startswith("cuda"):
            torch.cuda.empty_cache()
        if safe_point is not None:
            safe_point()

    def output_blocks() -> Iterator[tuple[int, list[PackedLayerState]]]:
        for block in source_packed.manifest.blocks:
            states = []
            for entry in block.layers:
                states.append(
                    expanded_by_block[block.index]
                    if entry.spec.name.endswith(request.layer_suffix)
                    else source_packed.load_layer(entry.spec.name)
                )
            yield block.index, states

    output = write_packed_artifact_stream(
        request.output_packed,
        source_packed.manifest.model,
        identity,
        output_blocks(),
    )
    exact_non_target, exact_prefixes = _verify_derivative(
        source_packed,
        output,
        request.layer_suffix,
    )
    rows = [results[index] for index in sorted(results)]
    old_bits = sum(int(row["old_bits"]) for row in rows)
    target_bits = sum(int(row["target_bits"]) for row in rows)
    before_weighted = sum(float(cast(dict[str, Any], row["before"])["export_weighted_error"]) for row in rows)
    after_weighted = sum(float(cast(dict[str, Any], row["after"])["export_weighted_error"]) for row in rows)
    report = {
        "schema_version": 1,
        "identity": identity,
        "request": {
            **asdict(request),
            "parent_run": str(request.parent_run.resolve()),
            "source_packed": str(request.source_packed.resolve()),
            "snapshot": str(request.snapshot.resolve()),
            "output_packed": str(request.output_packed.resolve()),
            "report_output": str(request.report_output.resolve()),
        },
        "source_descriptor_sha256": hash_file(source_packed.root / "nanoquant-packed-model.json"),
        "output_descriptor_sha256": hash_file(output.root / "nanoquant-packed-model.json"),
        "source_packed_bytes": source_packed.manifest.weight_bytes,
        "output_packed_bytes": output.manifest.weight_bytes,
        "target_layer_count": len(rows),
        "exact_non_target_layer_count": exact_non_target,
        "exact_target_prefix_count": exact_prefixes,
        "target_bits_before": old_bits,
        "target_bits_after": target_bits,
        "target_bit_multiplier": target_bits / old_bits,
        "whole_packed_storage_multiplier": output.manifest.weight_bytes / source_packed.manifest.weight_bytes,
        "weighted_error_before": before_weighted,
        "weighted_error_after": after_weighted,
        "weighted_error_relative_change": after_weighted / before_weighted - 1.0,
        "layers": rows,
    }
    atomic_write_json(request.report_output, report)
    if safe_point is not None:
        safe_point()
    return report


__all__ = ["RankExpansionRequest", "execute_rank_expansion"]
