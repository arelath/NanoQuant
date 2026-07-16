"""End-to-end compression, deployment export, and quality benchmark composition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nanoquant.config.codec import to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.infrastructure.gguf_export import GgufExportResult, export_llamacpp_gguf
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.publication import (
    PublishableArtifact,
    PublishableArtifactKind,
    publish_experiment_artifacts,
)
from nanoquant.infrastructure.runtime_export import (
    export_frozen_run_logical,
    validate_frozen_run_logical,
)
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.quality_evaluation import QualityEvaluationRequest, execute_quality_evaluation
from nanoquant.resident_workflow import (
    ResidentWorkflowResult,
    ResolvedResidentInputs,
    execute_resident_workflow,
    resolve_resident_experiment_inputs,
)
from nanoquant.runtime import (
    RuntimeModelMetadata,
    convert_logical_to_packed,
    open_logical_artifact,
    open_packed_artifact,
    validate_packed_conversion,
)

LEGACY_007_TASKS = (
    "piqa",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "boolq",
)


@dataclass(frozen=True, slots=True)
class CompressionBenchmarkExperiment:
    """Repository-relative material outputs and the comparison protocol."""

    logical_output: Path
    packed_output: Path
    checkpoint_output: Path
    gguf_output: Path
    benchmark_output: Path
    llama_cpp_root: Path
    runtime_family: str = "gemma3"
    expected_blocks: int = 26
    wikitext_samples: int = 64
    wikitext_sequence_length: int = 128
    wikitext_batch_size: int = 1
    task_names: tuple[str, ...] = LEGACY_007_TASKS
    task_limit: int = 200
    task_batch_size: int = 1
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if not self.runtime_family:
            raise ValueError("runtime family is required")
        if self.expected_blocks <= 0:
            raise ValueError("expected block count must be positive")
        if self.wikitext_samples <= 0 or self.wikitext_sequence_length < 2:
            raise ValueError("WikiText protocol dimensions are invalid")
        if self.wikitext_batch_size <= 0 or self.task_limit <= 0 or self.task_batch_size <= 0:
            raise ValueError("benchmark limits and batch sizes must be positive")
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError("benchmark task names must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class ResolvedCompressionBenchmarkExperiment:
    inputs: ResolvedResidentInputs
    logical_output: Path
    packed_output: Path
    checkpoint_output: Path
    gguf_output: Path
    benchmark_output: Path
    llama_cpp_root: Path


def _repository_path(path: Path, repository_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def resolve_compression_benchmark_experiment(
    config: RunConfig,
    experiment: CompressionBenchmarkExperiment,
    *,
    launcher_path: str | Path,
) -> ResolvedCompressionBenchmarkExperiment:
    """Resolve pinned model/calibration inputs and every repository-relative output."""

    launcher = Path(launcher_path).resolve()
    repository_root = launcher.parent.parent
    inputs = resolve_resident_experiment_inputs(config, launcher_path=launcher)
    return ResolvedCompressionBenchmarkExperiment(
        inputs,
        _repository_path(experiment.logical_output, repository_root),
        _repository_path(experiment.packed_output, repository_root),
        _repository_path(experiment.checkpoint_output, repository_root),
        _repository_path(experiment.gguf_output, repository_root),
        _repository_path(experiment.benchmark_output, repository_root),
        _repository_path(experiment.llama_cpp_root, repository_root),
    )


def _runtime_metadata(
    config: RunConfig,
    experiment: CompressionBenchmarkExperiment,
    resolved: ResolvedCompressionBenchmarkExperiment,
    workflow: ResidentWorkflowResult,
) -> RuntimeModelMetadata:
    checkpoint = SafetensorsModelSource(
        resolved.inputs.snapshot,
        source=config.model.source,
        revision=str(config.model.revision),
        verify_hashes=False,
    ).inventory()
    return RuntimeModelMetadata(
        config.model.source,
        str(config.model.revision),
        experiment.runtime_family,
        workflow.quantization.inventory.model.config_hash,
        checkpoint.tokenizer_hash,
    )


def _ensure_logical_export(
    resolved: ResolvedCompressionBenchmarkExperiment,
    metadata: RuntimeModelMetadata,
    expected_blocks: int,
) -> dict[str, Any]:
    if resolved.logical_output.exists():
        artifact = open_logical_artifact(resolved.logical_output, verify_hashes=True)
        if artifact.manifest.model != metadata:
            raise ValueError("existing logical export belongs to a different model")
    else:
        export_frozen_run_logical(
            resolved.inputs.output,
            resolved.logical_output,
            metadata,
            expected_blocks,
            use_global_tuning=True,
            fresh_validation=True,
        )
    return asdict(
        validate_frozen_run_logical(
            resolved.inputs.output,
            resolved.logical_output,
            expected_blocks,
            use_global_tuning=True,
            fresh_validation=True,
        )
    )


def _ensure_packed_export(resolved: ResolvedCompressionBenchmarkExperiment) -> dict[str, Any]:
    if resolved.packed_output.exists():
        open_packed_artifact(resolved.packed_output, verify_hashes=True)
    else:
        convert_logical_to_packed(resolved.logical_output, resolved.packed_output)
    return asdict(validate_packed_conversion(resolved.logical_output, resolved.packed_output))


def execute_compression_benchmark_experiment(
    config: RunConfig,
    experiment: CompressionBenchmarkExperiment,
    resolved: ResolvedCompressionBenchmarkExperiment,
) -> dict[str, Any]:
    """Compress Gemma, export its committed state to GGUF, then compare quality."""

    workflow = execute_resident_workflow(config, resolved.inputs)
    block_count = len(workflow.quantization.inventory.blocks)
    if block_count != experiment.expected_blocks:
        raise ValueError(
            f"resolved model block count differs from experiment: {block_count} != "
            f"{experiment.expected_blocks}"
        )
    metadata = _runtime_metadata(config, experiment, resolved, workflow)
    logical = _ensure_logical_export(resolved, metadata, block_count)
    packed = _ensure_packed_export(resolved)
    gguf: GgufExportResult = export_llamacpp_gguf(
        resolved.packed_output,
        resolved.inputs.snapshot,
        resolved.checkpoint_output,
        resolved.gguf_output,
        resolved.llama_cpp_root,
    )
    quality = execute_quality_evaluation(
        QualityEvaluationRequest(
            snapshot=resolved.inputs.snapshot,
            source=config.model.source,
            revision=str(config.model.revision),
            run_output=resolved.inputs.output,
            device=config.runtime.compute_device,
            backend="factorized",
            use_global_tuning=True,
            wikitext_samples=experiment.wikitext_samples,
            wikitext_sequence_length=experiment.wikitext_sequence_length,
            wikitext_batch_size=experiment.wikitext_batch_size,
            task_names=experiment.task_names,
            task_limit=experiment.task_limit,
            task_batch_size=experiment.task_batch_size,
            local_files_only=experiment.local_files_only,
        )
    )
    experiment_number = config.intent.experiment_number
    if experiment_number is None:
        raise ValueError("compression benchmark requires an experiment number")
    launcher = resolved.inputs.launcher_path
    if launcher is None:
        raise ValueError("compression benchmark requires launcher provenance")
    repository_root = launcher.resolve().parent.parent
    publication_directory = repository_root / "Results" / f"{experiment_number:03d}"
    payload = {
        "schema_version": 1,
        "passed": bool(quality.get("passed")),
        "experiment": {
            "config": to_dict(config),
            "launcher": str(resolved.inputs.launcher_path),
            "comparison_labels": {"base": "bf16", "frozen": "nanoquant"},
        },
        "compression": {
            "run_output": str(resolved.inputs.output.resolve()),
            "commit_identity": to_dict(workflow.quantization.identity),
            "effective_bpw": workflow.quantization.frozen_model.effective_bpw,
            "peak_device_bytes": workflow.quantization.peak_device_bytes,
            "peak_host_bytes": workflow.quantization.peak_host_bytes,
            "artifact_bytes": workflow.quantization.artifact_bytes,
            "elapsed_seconds": workflow.quantization.elapsed_seconds,
            "reused_commit_count": workflow.quantization.reused_commit_count,
            "distillation": None if workflow.distillation is None else to_dict(workflow.distillation.metrics),
        },
        "exports": {
            "logical": logical,
            "packed": packed,
            "gguf": {
                "output": str(gguf.output),
                "checkpoint": str(gguf.checkpoint),
                "converter": str(gguf.converter),
                "bytes": gguf.bytes,
                "sha256": gguf.sha256,
                "reused": gguf.reused,
            },
        },
        "benchmarks": quality,
        "publication": {
            "directory": str(publication_directory),
            "manifest": str(publication_directory / "publication.json"),
        },
    }
    atomic_write_json(resolved.benchmark_output, payload)
    profile_json = sorted(resolved.inputs.output.glob("profile*.json"))
    profile_markdown = sorted(resolved.inputs.output.glob("profile*.md"))
    publish_experiment_artifacts(
        repository_root,
        experiment_number,
        (
            PublishableArtifact(gguf.output, PublishableArtifactKind.MODEL),
            PublishableArtifact(resolved.benchmark_output, PublishableArtifactKind.STATISTICS),
            *(PublishableArtifact(path, PublishableArtifactKind.STATISTICS) for path in profile_json),
            *(PublishableArtifact(path, PublishableArtifactKind.REPORT) for path in profile_markdown),
        ),
    )
    return payload


def run_compression_benchmark_experiment(
    config: RunConfig,
    experiment: CompressionBenchmarkExperiment,
    *,
    launcher_path: str | Path,
) -> int:
    resolved = resolve_compression_benchmark_experiment(
        config,
        experiment,
        launcher_path=launcher_path,
    )
    execute_compression_benchmark_experiment(config, experiment, resolved)
    return 0


__all__ = [
    "LEGACY_007_TASKS",
    "CompressionBenchmarkExperiment",
    "ResolvedCompressionBenchmarkExperiment",
    "execute_compression_benchmark_experiment",
    "resolve_compression_benchmark_experiment",
    "run_compression_benchmark_experiment",
]
