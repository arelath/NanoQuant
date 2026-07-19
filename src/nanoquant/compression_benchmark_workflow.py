"""End-to-end compression, deployment export, and quality benchmark composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoquant.compression_export_workflow import (
    CompressionExportRecipe,
    ResolvedCompressionExportRecipe,
    complete_deferred_huggingface_upload,
    execute_complete_compression,
    resolve_compression_export_recipe,
)
from nanoquant.config.codec import to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.infrastructure.huggingface_upload import huggingface_upload_summary
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.publication import (
    PublishableArtifact,
    PublishableArtifactKind,
    publish_experiment_artifacts,
)
from nanoquant.quality_evaluation import (
    DEFAULT_QUALITY_TASK_BATCH_SIZE,
    DEFAULT_QUALITY_WIKITEXT_BATCH_SIZE,
    QualityEvaluationRequest,
    execute_quality_evaluation,
)
from nanoquant.resident_workflow import (
    ResolvedResidentInputs,
    resolve_resident_experiment_inputs,
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

    export: CompressionExportRecipe
    benchmark_output: Path
    expected_blocks: int = 26
    wikitext_samples: int = 64
    wikitext_sequence_length: int = 128
    wikitext_batch_size: int = DEFAULT_QUALITY_WIKITEXT_BATCH_SIZE
    task_names: tuple[str, ...] = LEGACY_007_TASKS
    task_limit: int = 200
    task_batch_size: int = DEFAULT_QUALITY_TASK_BATCH_SIZE
    local_files_only: bool = False

    def __post_init__(self) -> None:
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
    export: ResolvedCompressionExportRecipe
    benchmark_output: Path


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
        resolve_compression_export_recipe(experiment.export, repository_root),
        _repository_path(experiment.benchmark_output, repository_root),
    )


def execute_compression_benchmark_experiment(
    config: RunConfig,
    experiment: CompressionBenchmarkExperiment,
    resolved: ResolvedCompressionBenchmarkExperiment,
) -> dict[str, Any]:
    """Compress Gemma, export its committed state to GGUF, then compare quality."""

    experiment_number = config.intent.experiment_number
    if experiment_number is None:
        raise ValueError("compression benchmark requires an experiment number")
    launcher = resolved.inputs.launcher_path
    if launcher is None:
        raise ValueError("compression benchmark requires launcher provenance")
    repository_root = launcher.resolve().parent.parent
    complete = execute_complete_compression(
        config,
        resolved.inputs,
        experiment.export,
        expected_blocks=experiment.expected_blocks,
    )
    workflow = complete.workflow
    exports = complete.exports
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
            packed_artifact=resolved.export.packed_output,
        )
    )
    quality_output = resolved.benchmark_output.with_suffix(".quality.json")
    atomic_write_json(quality_output, quality)
    exports = complete_deferred_huggingface_upload(
        exports,
        experiment.export.huggingface,
        ((quality_output, "quality.json"),),
    )
    publication_directory = repository_root / "Results" / f"{experiment_number:03d}"
    payload = {
        "schema_version": 2,
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
            "logical": exports.logical,
            "packed": exports.packed,
            "gguf": {
                "output": str(exports.gguf.output),
                "checkpoint": str(exports.gguf.checkpoint),
                "converter": str(exports.gguf.converter),
                "bytes": exports.gguf.bytes,
                "sha256": exports.gguf.sha256,
                "reused": exports.gguf.reused,
            },
            "mmproj": (
                None
                if exports.gguf.mmproj is None
                else {
                    "output": str(exports.gguf.mmproj.output),
                    "converter": str(exports.gguf.mmproj.converter),
                    "bytes": exports.gguf.mmproj.bytes,
                    "sha256": exports.gguf.mmproj.sha256,
                    "tensor_count": exports.gguf.mmproj.tensor_count,
                    "tensor_types": exports.gguf.mmproj.tensor_types,
                    "reused": exports.gguf.mmproj.reused,
                }
            ),
            "huggingface": (
                None
                if exports.huggingface is None
                else huggingface_upload_summary(exports.huggingface)
            ),
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
            PublishableArtifact(exports.gguf.output, PublishableArtifactKind.MODEL),
            PublishableArtifact(exports.summary_output, PublishableArtifactKind.STATISTICS),
            PublishableArtifact(
                exports.gguf.output.with_suffix(exports.gguf.output.suffix + ".export.json"),
                PublishableArtifactKind.STATISTICS,
            ),
            *(
                ()
                if exports.gguf.mmproj is None
                else (
                    PublishableArtifact(exports.gguf.mmproj.output, PublishableArtifactKind.MODEL),
                    PublishableArtifact(
                        exports.gguf.mmproj.output.with_suffix(
                            exports.gguf.mmproj.output.suffix + ".export.json"
                        ),
                        PublishableArtifactKind.STATISTICS,
                    ),
                )
            ),
            *(
                ()
                if exports.huggingface is None
                else (
                    PublishableArtifact(
                        exports.huggingface.receipt_output,
                        PublishableArtifactKind.STATISTICS,
                    ),
                )
            ),
            PublishableArtifact(resolved.benchmark_output, PublishableArtifactKind.STATISTICS),
            PublishableArtifact(quality_output, PublishableArtifactKind.STATISTICS),
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
