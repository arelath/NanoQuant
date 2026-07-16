"""Shared validated deployment export for every completed compression run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from nanoquant.config.codec import to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.infrastructure.gguf_export import (
    DEFAULT_TOKEN_EMBEDDING_TYPE,
    GgufExportResult,
    export_llamacpp_gguf,
    normalize_token_embedding_type,
)
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.live_reconstruction import initialize_live_weight_error_report
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.runtime_export import (
    export_frozen_run_logical,
    validate_frozen_run_logical,
)
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResidentWorkflowResult,
    ResolvedResidentInputs,
    execute_resident_workflow,
)
from nanoquant.runtime import (
    RuntimeModelMetadata,
    convert_logical_to_packed,
    open_logical_artifact,
    open_packed_artifact,
    validate_packed_conversion,
)


@dataclass(frozen=True, slots=True)
class CompressionExportRecipe:
    """Material deployment outputs required after successful compression."""

    logical_output: Path
    packed_output: Path
    checkpoint_output: Path
    gguf_output: Path
    llama_cpp_root: Path
    runtime_family: str = "gemma3"
    token_embedding_type: str = DEFAULT_TOKEN_EMBEDDING_TYPE

    def __post_init__(self) -> None:
        if not self.runtime_family:
            raise ValueError("compression export runtime family is required")
        if self.gguf_output.suffix.lower() != ".gguf":
            raise ValueError("compression export output must use the .gguf extension")
        object.__setattr__(
            self,
            "token_embedding_type",
            normalize_token_embedding_type(self.token_embedding_type),
        )


@dataclass(frozen=True, slots=True)
class ResolvedCompressionExportRecipe:
    logical_output: Path
    packed_output: Path
    checkpoint_output: Path
    gguf_output: Path
    llama_cpp_root: Path
    runtime_family: str
    token_embedding_type: str = DEFAULT_TOKEN_EMBEDDING_TYPE


@dataclass(frozen=True, slots=True)
class CompressionExportResult:
    logical: dict[str, Any]
    packed: dict[str, Any]
    gguf: GgufExportResult
    summary_output: Path


@dataclass(frozen=True, slots=True)
class CompleteCompressionResult:
    workflow: ResidentWorkflowResult
    exports: CompressionExportResult


def _repository_path(path: Path, repository_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def resolve_compression_export_recipe(
    recipe: CompressionExportRecipe,
    repository_root: str | Path,
) -> ResolvedCompressionExportRecipe:
    root = Path(repository_root).resolve()
    return ResolvedCompressionExportRecipe(
        _repository_path(recipe.logical_output, root),
        _repository_path(recipe.packed_output, root),
        _repository_path(recipe.checkpoint_output, root),
        _repository_path(recipe.gguf_output, root),
        _repository_path(recipe.llama_cpp_root, root),
        recipe.runtime_family,
        recipe.token_embedding_type,
    )


def _runtime_metadata(
    config: RunConfig,
    snapshot: Path,
    runtime_family: str,
) -> RuntimeModelMetadata:
    source = SafetensorsModelSource(
        snapshot,
        source=config.model.source,
        revision=str(config.model.revision),
        verify_hashes=False,
    )
    checkpoint = source.inventory()
    model = adapter_for_config(checkpoint.config).model_inventory(source).model
    return RuntimeModelMetadata(
        config.model.source,
        str(config.model.revision),
        runtime_family,
        model.config_hash,
        checkpoint.tokenizer_hash,
    )


def _ensure_logical_export(
    run_output: Path,
    resolved: ResolvedCompressionExportRecipe,
    metadata: RuntimeModelMetadata,
    expected_blocks: int,
    *,
    use_global_tuning: bool,
) -> dict[str, Any]:
    if resolved.logical_output.exists():
        artifact = open_logical_artifact(resolved.logical_output, verify_hashes=True)
        if artifact.manifest.model != metadata:
            raise ValueError("existing logical export belongs to a different model")
    else:
        export_frozen_run_logical(
            run_output,
            resolved.logical_output,
            metadata,
            expected_blocks,
            use_global_tuning=use_global_tuning,
            fresh_validation=True,
        )
    return cast(
        dict[str, Any],
        to_dict(
            validate_frozen_run_logical(
                run_output,
                resolved.logical_output,
                expected_blocks,
                use_global_tuning=use_global_tuning,
                fresh_validation=True,
            )
        ),
    )


def _ensure_packed_export(resolved: ResolvedCompressionExportRecipe) -> dict[str, Any]:
    if resolved.packed_output.exists():
        open_packed_artifact(resolved.packed_output, verify_hashes=True)
    else:
        convert_logical_to_packed(resolved.logical_output, resolved.packed_output)
    return cast(
        dict[str, Any],
        to_dict(validate_packed_conversion(resolved.logical_output, resolved.packed_output)),
    )


def execute_compression_export(
    config: RunConfig,
    recipe: CompressionExportRecipe,
    *,
    repository_root: str | Path,
    run_output: str | Path,
    snapshot: str | Path,
    expected_blocks: int,
) -> CompressionExportResult:
    """Validate and export one complete committed run without recompressing it."""

    if expected_blocks <= 0:
        raise ValueError("compression export expected block count must be positive")
    resolved = resolve_compression_export_recipe(recipe, repository_root)
    run = Path(run_output).resolve()
    source_snapshot = Path(snapshot).resolve()
    metadata = _runtime_metadata(config, source_snapshot, resolved.runtime_family)
    logical = _ensure_logical_export(
        run,
        resolved,
        metadata,
        expected_blocks,
        use_global_tuning=config.distillation.enabled,
    )
    packed = _ensure_packed_export(resolved)
    gguf = export_llamacpp_gguf(
        resolved.packed_output,
        source_snapshot,
        resolved.checkpoint_output,
        resolved.gguf_output,
        resolved.llama_cpp_root,
        token_embedding_type=resolved.token_embedding_type,
    )
    summary_output = resolved.gguf_output.with_suffix(".export-summary.json")
    atomic_write_json(
        summary_output,
        {
            "schema_version": 3,
            "run_output": str(run),
            "logical": logical,
            "packed": packed,
            "gguf": {
                "output": str(gguf.output),
                "checkpoint": str(gguf.checkpoint),
                "converter": str(gguf.converter),
                "quantizer": None if gguf.quantizer is None else str(gguf.quantizer),
                "token_embedding_type": gguf.token_embedding_type,
                "bytes": gguf.bytes,
                "sha256": gguf.sha256,
                "reused": gguf.reused,
                "receipt": str(gguf.output.with_suffix(gguf.output.suffix + ".export.json")),
            },
            "mmproj": (
                None
                if gguf.mmproj is None
                else {
                    "output": str(gguf.mmproj.output),
                    "converter": str(gguf.mmproj.converter),
                    "bytes": gguf.mmproj.bytes,
                    "sha256": gguf.mmproj.sha256,
                    "tensor_count": gguf.mmproj.tensor_count,
                    "tensor_types": gguf.mmproj.tensor_types,
                    "reused": gguf.mmproj.reused,
                    "receipt": str(
                        gguf.mmproj.output.with_suffix(gguf.mmproj.output.suffix + ".export.json")
                    ),
                }
            ),
        },
    )
    return CompressionExportResult(logical, packed, gguf, summary_output)


def execute_complete_compression(
    config: RunConfig,
    inputs: ResolvedResidentInputs,
    recipe: CompressionExportRecipe,
    *,
    expected_blocks: int,
    options: ResidentExecutionOptions | None = None,
) -> CompleteCompressionResult:
    """Run compression and require its validated GGUF before reporting completion."""

    if inputs.launcher_path is None:
        raise ValueError("complete compression requires launcher provenance")
    experiment_number = config.intent.experiment_number
    if experiment_number is None:
        raise ValueError("complete compression requires a numbered experiment")
    repository_root = inputs.launcher_path.resolve().parent.parent
    initialize_live_weight_error_report(
        repository_root,
        experiment_number,
        inputs.output,
        expected_blocks=expected_blocks,
        layer_order=config.block_tuning.layer_order,
    )
    workflow = execute_resident_workflow(
        config,
        inputs,
        ResidentExecutionOptions() if options is None else options,
    )
    block_count = len(workflow.quantization.inventory.blocks)
    if block_count != expected_blocks:
        raise ValueError(
            f"resolved model block count differs from compression recipe: {block_count} != {expected_blocks}"
        )
    exports = execute_compression_export(
        config,
        recipe,
        repository_root=repository_root,
        run_output=inputs.output,
        snapshot=inputs.snapshot,
        expected_blocks=block_count,
    )
    return CompleteCompressionResult(workflow, exports)


__all__ = [
    "CompressionExportRecipe",
    "CompressionExportResult",
    "CompleteCompressionResult",
    "ResolvedCompressionExportRecipe",
    "execute_compression_export",
    "execute_complete_compression",
    "resolve_compression_export_recipe",
]
