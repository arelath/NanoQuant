"""Stream committed research artifacts into the deployment logical format."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nanoquant.domain.models import (
    ArtifactRef,
    ArtifactTypes,
    FrozenBlockState,
    FrozenNanoQuantState,
    FrozenSharedInputGroupState,
)
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity, latest_complete_identity, load_committed_block
from nanoquant.infrastructure.global_tuning import active_global_tuning, load_global_tuning
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.runtime import (
    LOGICAL_FORMAT_VERSION,
    LogicalLayerState,
    QuantizedLinearSpec,
    RuntimeModelMetadata,
    canonical_torch_dtype,
    open_logical_artifact,
    write_logical_artifact_stream,
)
from nanoquant.runtime.backend import ProjectionMemberSpec


@dataclass(frozen=True, slots=True)
class LogicalRunExportResult:
    output: Path
    identity: CommitIdentity
    global_tuning: ArtifactRef | None
    block_count: int
    layer_count: int
    weight_bytes: int


@dataclass(frozen=True, slots=True)
class LogicalRunExportValidation:
    output: Path
    identity: CommitIdentity
    global_tuning: ArtifactRef | None
    block_count: int
    layer_count: int
    tensor_count: int
    tensor_bytes: int
    weight_bytes: int
    exact: bool


@dataclass(frozen=True, slots=True)
class FrozenRunAuxiliaryState:
    """Named non-linear parameters selected by a committed frozen run."""

    identity: CommitIdentity
    global_tuning: ArtifactRef | None
    parameters: tuple[tuple[str, torch.Tensor], ...]


@dataclass(frozen=True, slots=True)
class _ResolvedFrozenRun:
    identity: CommitIdentity
    global_tuning: ArtifactRef | None
    blocks: tuple[FrozenBlockState, ...]
    tensors: LocalTensorStore


def _read_journal(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read frozen run journal: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"frozen run journal line {line_number} is invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"frozen run journal line {line_number} is not an object")
        records.append(value)
    if not records:
        raise ValueError("frozen run journal is empty")
    return records


def _validate_run_manifest(run_root: Path, model: RuntimeModelMetadata) -> None:
    path = run_root / "manifest.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("frozen run manifest is invalid JSON") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("resolved_config"), dict):
        raise ValueError("frozen run manifest has no resolved configuration")
    resolved = payload["resolved_config"]
    for field, expected in (("source", model.source), ("revision", model.revision)):
        actual = resolved.get(field)
        if actual != expected:
            raise ValueError(f"runtime model {field} does not match the frozen run: {expected!r} != {actual!r}")


def _runtime_state(
    frozen: FrozenNanoQuantState,
    tensors: LocalTensorStore,
    stack: ExitStack,
) -> LogicalLayerState:
    if frozen.logical_format != LOGICAL_FORMAT_VERSION:
        raise ValueError(
            f"frozen layer uses unsupported logical format {frozen.logical_format!r}: "
            f"block {frozen.layer.block.index} {frozen.layer.path}"
        )
    if frozen.scales.mid is None:
        raise ValueError(f"frozen layer has no middle scale: block {frozen.layer.block.index} {frozen.layer.path}")
    left = stack.enter_context(tensors.read(frozen.left_binary))
    right = stack.enter_context(tensors.read(frozen.right_binary))
    scale_pre = stack.enter_context(tensors.read(frozen.scales.pre))
    scale_mid = stack.enter_context(tensors.read(frozen.scales.mid))
    scale_post = stack.enter_context(tensors.read(frozen.scales.post))
    bias = None if frozen.bias is None else stack.enter_context(tensors.read(frozen.bias))
    indices = None
    values = None
    outlier_scales = None
    if frozen.outliers is not None:
        indices = stack.enter_context(tensors.read(frozen.outliers.indices))
        values = stack.enter_context(tensors.read(frozen.outliers.values))
        if frozen.outliers.scales is not None:
            outlier_scales = stack.enter_context(tensors.read(frozen.outliers.scales))
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"frozen factors must be matrices: {frozen.layer.path}")
    if left.shape[1] != frozen.rank or right.shape[0] != frozen.rank:
        raise ValueError(f"frozen factor rank differs from metadata: {frozen.layer.path}")
    factor_dtype = canonical_torch_dtype(left.dtype)
    if canonical_torch_dtype(right.dtype) != factor_dtype:
        raise ValueError(f"frozen factor dtypes differ: {frozen.layer.path}")
    scale_dtype = canonical_torch_dtype(scale_pre.dtype)
    if any(canonical_torch_dtype(value.dtype) != scale_dtype for value in (scale_mid, scale_post)):
        raise ValueError(f"frozen scale dtypes differ: {frozen.layer.path}")
    outlier_count = 0 if indices is None else int(indices.numel())
    spec = QuantizedLinearSpec(
        name=f"blocks.{frozen.layer.block.index}.{frozen.layer.path}",
        logical_format=frozen.logical_format,
        in_features=int(right.shape[1]),
        out_features=int(left.shape[0]),
        rank=frozen.rank,
        factor_dtype=factor_dtype,
        scale_dtype=scale_dtype,
        outlier_count=outlier_count,
        outlier_value_dtype=None if values is None else canonical_torch_dtype(values.dtype),
        has_outlier_scales=outlier_scales is not None,
        has_bias=bias is not None,
    )
    return LogicalLayerState(
        spec,
        left,
        right,
        scale_pre,
        scale_mid,
        scale_post,
        bias,
        indices,
        values,
        outlier_scales,
    )


def _runtime_group_state(
    frozen: FrozenSharedInputGroupState,
    tensors: LocalTensorStore,
    stack: ExitStack,
) -> LogicalLayerState:
    if frozen.scales.mid is None:
        raise ValueError(f"frozen shared-input group has no middle scale: {frozen.name}")
    left = stack.enter_context(tensors.read(frozen.left_binary))
    right = stack.enter_context(tensors.read(frozen.right_binary))
    scale_pre = stack.enter_context(tensors.read(frozen.scales.pre))
    scale_mid = stack.enter_context(tensors.read(frozen.scales.mid))
    scale_post = stack.enter_context(tensors.read(frozen.scales.post))
    bias = None if frozen.bias is None else stack.enter_context(tensors.read(frozen.bias))
    indices = values = outlier_scales = None
    if frozen.outliers is not None:
        indices = stack.enter_context(tensors.read(frozen.outliers.indices))
        values = stack.enter_context(tensors.read(frozen.outliers.values))
        if frozen.outliers.scales is not None:
            outlier_scales = stack.enter_context(tensors.read(frozen.outliers.scales))
    factor_dtype = canonical_torch_dtype(left.dtype)
    scale_dtype = canonical_torch_dtype(scale_pre.dtype)
    prefix = f"blocks.{frozen.block.index}."
    spec = QuantizedLinearSpec(
        name=prefix + frozen.name,
        logical_format=frozen.logical_format,
        in_features=int(right.shape[1]),
        out_features=int(left.shape[0]),
        rank=frozen.rank,
        factor_dtype=factor_dtype,
        scale_dtype=scale_dtype,
        outlier_count=0 if indices is None else int(indices.numel()),
        outlier_value_dtype=None if values is None else canonical_torch_dtype(values.dtype),
        has_outlier_scales=outlier_scales is not None,
        has_bias=bias is not None,
        members=tuple(
            ProjectionMemberSpec(prefix + member.layer.path, member.row_start, member.row_end)
            for member in frozen.members
        ),
    )
    return LogicalLayerState(
        spec,
        left,
        right,
        scale_pre,
        scale_mid,
        scale_post,
        bias,
        indices,
        values,
        outlier_scales,
    )


def _stream_logical_blocks(
    blocks: Sequence[FrozenBlockState],
    tensors: LocalTensorStore,
) -> Iterator[tuple[int, list[LogicalLayerState]]]:
    for expected_index, block in enumerate(blocks):
        if block.block.index != expected_index:
            raise ValueError(
                f"frozen block states are not contiguous: expected {expected_index}, received {block.block.index}"
            )
        with ExitStack() as stack:
            if any(state.layer.block != block.block for state in block.quantized_layers):
                raise ValueError(f"frozen block {expected_index} contains a layer from another block")
            states = [
                *[_runtime_state(state, tensors, stack) for state in block.quantized_layers],
                *[_runtime_group_state(state, tensors, stack) for state in block.shared_input_groups],
            ]
            yield expected_index, states
            del states


def _resolve_frozen_run(
    run_root: Path,
    expected_blocks: int,
    *,
    use_global_tuning: bool,
    fresh_validation: bool,
) -> _ResolvedFrozenRun:
    artifact_root = run_root / "artifacts"
    if not artifact_root.is_dir():
        raise ValueError(f"frozen run artifact store is missing: {artifact_root}")
    artifacts = LocalArtifactStore(
        artifact_root,
        use_persistent_validation_cache=not fresh_validation,
    )
    tensors = LocalTensorStore(artifacts)
    identity, block_records = latest_complete_identity(
        _read_journal(run_root / "state" / "journal.jsonl"),
        expected_blocks,
    )
    committed = tuple(
        load_committed_block(
            ArtifactRef(ArtifactTypes.BLOCK_RESULT, str(block_records[index]["artifact_id"]), 1),
            artifacts,
            identity,
        ).result
        for index in range(expected_blocks)
    )
    global_reference = active_global_tuning(run_root) if use_global_tuning else None
    global_result = None if global_reference is None else load_global_tuning(global_reference, artifacts).result
    source_blocks = tuple(block.teacher_outputs.artifact for block in committed)
    if global_result is not None:
        if global_result.source_blocks != source_blocks:
            raise ValueError("global tuning result does not match the run's committed blocks")
        if tuple(state.block.index for state in global_result.tuned_blocks) != tuple(range(expected_blocks)):
            raise ValueError("global tuning result does not contain complete contiguous block states")
    frozen_blocks = (
        tuple(block.frozen_state for block in committed) if global_result is None else global_result.tuned_blocks
    )
    return _ResolvedFrozenRun(identity, global_reference, frozen_blocks, tensors)


def load_frozen_run_auxiliary(
    run_output: str | Path,
    expected_blocks: int,
    *,
    use_global_tuning: bool = True,
    fresh_validation: bool = True,
) -> FrozenRunAuxiliaryState:
    """Load the complete named block/global auxiliary override inventory on CPU."""

    if expected_blocks <= 0:
        raise ValueError("expected block count must be positive")
    resolved = _resolve_frozen_run(
        Path(run_output),
        expected_blocks,
        use_global_tuning=use_global_tuning,
        fresh_validation=fresh_validation,
    )
    parameters: dict[str, torch.Tensor] = {}
    for block in resolved.blocks:
        for local_name, reference in block.auxiliary_parameters:
            name = f"model.layers.{block.block.index}.{local_name}"
            if name in parameters:
                raise ValueError(f"frozen auxiliary parameter is duplicated: {name}")
            with resolved.tensors.read(reference, "cpu") as value:
                parameters[name] = value.detach().clone()
    if resolved.global_tuning is not None:
        result = load_global_tuning(resolved.global_tuning, resolved.tensors.artifacts).result
        for name, reference in result.auxiliary_parameters:
            with resolved.tensors.read(reference, "cpu") as value:
                parameters[name] = value.detach().clone()
    return FrozenRunAuxiliaryState(
        resolved.identity,
        resolved.global_tuning,
        tuple(sorted(parameters.items())),
    )


def export_frozen_run_logical(
    run_output: str | Path,
    output: str | Path,
    model: RuntimeModelMetadata,
    expected_blocks: int,
    *,
    use_global_tuning: bool = True,
    fresh_validation: bool = True,
) -> LogicalRunExportResult:
    """Export a complete committed run without loading a source model or using CUDA."""

    if expected_blocks <= 0:
        raise ValueError("expected block count must be positive")
    run_root = Path(run_output)
    _validate_run_manifest(run_root, model)
    resolved = _resolve_frozen_run(
        run_root,
        expected_blocks,
        use_global_tuning=use_global_tuning,
        fresh_validation=fresh_validation,
    )
    if model.config_hash != resolved.identity.model_hash:
        raise ValueError(
            "runtime model config hash does not match the committed run: "
            f"{model.config_hash} != {resolved.identity.model_hash}"
        )
    artifact = write_logical_artifact_stream(
        output,
        model,
        _stream_logical_blocks(resolved.blocks, resolved.tensors),
    )
    return LogicalRunExportResult(
        artifact.root,
        resolved.identity,
        resolved.global_tuning,
        len(artifact.manifest.blocks),
        artifact.manifest.layer_count,
        artifact.manifest.weight_bytes,
    )


def _logical_values(state: LogicalLayerState) -> tuple[tuple[str, torch.Tensor | None], ...]:
    return (
        ("factor_left", state.left_binary),
        ("factor_right", state.right_binary),
        ("scale_pre", state.scale_pre),
        ("scale_mid", state.scale_mid),
        ("scale_post", state.scale_post),
        ("bias", state.bias),
        ("outlier_indices", state.outlier_indices),
        ("outlier_values", state.outlier_values),
        ("outlier_scales", state.outlier_scales),
    )


def validate_frozen_run_logical(
    run_output: str | Path,
    artifact_root: str | Path,
    expected_blocks: int,
    *,
    use_global_tuning: bool = True,
    fresh_validation: bool = True,
) -> LogicalRunExportValidation:
    """Prove every exported specification and tensor equals the selected frozen run."""

    if expected_blocks <= 0:
        raise ValueError("expected block count must be positive")
    artifact = open_logical_artifact(artifact_root, verify_hashes=True)
    run_root = Path(run_output)
    _validate_run_manifest(run_root, artifact.manifest.model)
    resolved = _resolve_frozen_run(
        run_root,
        expected_blocks,
        use_global_tuning=use_global_tuning,
        fresh_validation=fresh_validation,
    )
    if artifact.manifest.model.config_hash != resolved.identity.model_hash:
        raise ValueError("logical artifact model config hash differs from the committed run")
    expected_names = [
        name
        for block in resolved.blocks
        for name in (
            *(f"blocks.{block.block.index}.{state.layer.path}" for state in block.quantized_layers),
            *(f"blocks.{block.block.index}.{state.name}" for state in block.shared_input_groups),
        )
    ]
    actual_names = [layer.spec.name for block in artifact.manifest.blocks for layer in block.layers]
    if actual_names != expected_names:
        raise ValueError("logical artifact layer inventory or ordering differs from the frozen run")
    tensor_count = 0
    tensor_bytes = 0
    for block in resolved.blocks:
        with ExitStack() as stack:
            expected_states = (
                *(_runtime_state(state, resolved.tensors, stack) for state in block.quantized_layers),
                *(_runtime_group_state(state, resolved.tensors, stack) for state in block.shared_input_groups),
            )
            for expected in expected_states:
                actual = artifact.load_layer(expected.spec.name)
                if actual.spec != expected.spec:
                    raise ValueError(f"logical artifact layer specification differs: {expected.spec.name}")
                actual_values = dict(_logical_values(actual))
                for role, expected_value in _logical_values(expected):
                    actual_value = actual_values[role]
                    if (expected_value is None) != (actual_value is None):
                        raise ValueError(f"logical artifact tensor presence differs: {expected.spec.name}:{role}")
                    if expected_value is not None and actual_value is not None:
                        if not torch.equal(expected_value, actual_value):
                            raise ValueError(f"logical artifact tensor differs: {expected.spec.name}:{role}")
                        tensor_count += 1
                        tensor_bytes += expected_value.numel() * expected_value.element_size()
            del actual, actual_value, actual_values, expected, expected_states, expected_value
    return LogicalRunExportValidation(
        artifact.root,
        resolved.identity,
        resolved.global_tuning,
        len(artifact.manifest.blocks),
        artifact.manifest.layer_count,
        tensor_count,
        tensor_bytes,
        artifact.manifest.weight_bytes,
        True,
    )
