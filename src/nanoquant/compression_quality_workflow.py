"""End-to-end compression and quality-proof experiment composition."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoquant.compression_export_workflow import CompressionExportRecipe, execute_complete_compression
from nanoquant.config.codec import config_hash, to_dict
from nanoquant.config.schema import ExecutorKind, RunConfig
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text
from nanoquant.infrastructure.publication import (
    PublishableArtifact,
    PublishableArtifactKind,
    publish_experiment_artifacts,
)
from nanoquant.infrastructure.runs import launcher_provenance, validate_launcher_number
from nanoquant.quality_evaluation import QualityEvaluationRequest, execute_quality_evaluation
from nanoquant.quality_evaluation_workflow import render_quality_evaluation_markdown
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    resolve_resident_experiment_inputs,
)


@dataclass(frozen=True, slots=True)
class CompressionQualityExperiment:
    export: CompressionExportRecipe
    summary_output: Path
    quality_output: Path
    quality_markdown_output: Path
    expected_blocks: int
    wikitext_samples: int = 64
    wikitext_sequence_length: int = 128
    wikitext_batch_size: int = 1
    task_names: tuple[str, ...] = (
        "piqa",
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "boolq",
    )
    task_limit: int = 200
    task_batch_size: int = 1
    local_files_only: bool = True
    maximum_wddm_shared_gib: float | None = None
    restore_completed_blocks: bool = True
    quality_backend: str = "factorized"

    def __post_init__(self) -> None:
        if self.expected_blocks <= 0:
            raise ValueError("expected block count must be positive")
        if self.wikitext_samples <= 0 or self.wikitext_sequence_length < 2:
            raise ValueError("quality dimensions are invalid")
        if self.wikitext_batch_size <= 0 or self.task_batch_size <= 0 or self.task_limit <= 0:
            raise ValueError("quality batch sizes and task limit must be positive")
        if self.maximum_wddm_shared_gib is not None and (
            not math.isfinite(self.maximum_wddm_shared_gib) or self.maximum_wddm_shared_gib < 0
        ):
            raise ValueError("maximum WDDM shared memory must be finite and non-negative")
        if self.quality_backend not in {"factorized", "dense"}:
            raise ValueError("quality backend must be factorized or dense")


@dataclass(frozen=True, slots=True)
class ResolvedCompressionQualityExperiment:
    inputs: ResolvedResidentInputs
    summary_output: Path
    quality_output: Path
    quality_markdown_output: Path


def _repository_path(path: Path, repository_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def resolve_compression_quality_experiment(
    config: RunConfig,
    experiment: CompressionQualityExperiment,
    *,
    launcher_path: str | Path,
) -> ResolvedCompressionQualityExperiment:
    launcher = Path(launcher_path).resolve()
    root = launcher.parent.parent
    return ResolvedCompressionQualityExperiment(
        resolve_resident_experiment_inputs(config, launcher_path=launcher),
        _repository_path(experiment.summary_output, root),
        _repository_path(experiment.quality_output, root),
        _repository_path(experiment.quality_markdown_output, root),
    )


def execute_compression_quality_experiment(
    config: RunConfig,
    experiment: CompressionQualityExperiment,
    resolved: ResolvedCompressionQualityExperiment,
) -> dict[str, Any]:
    """Compress the pinned model, then compare BF16 and frozen quality."""

    wall_started = time.perf_counter()
    compression_started = time.perf_counter()
    maximum_shared = experiment.maximum_wddm_shared_gib
    maximum_shared_bytes = None if maximum_shared is None else int(maximum_shared * 2**30)
    complete = execute_complete_compression(
        config,
        resolved.inputs,
        experiment.export,
        expected_blocks=experiment.expected_blocks,
        options=ResidentExecutionOptions(
            restore_completed_blocks=experiment.restore_completed_blocks,
            maximum_wddm_shared_bytes=maximum_shared_bytes,
        ),
    )
    workflow = complete.workflow
    exports = complete.exports
    compression_seconds = time.perf_counter() - compression_started
    block_count = len(workflow.quantization.inventory.blocks)
    experiment_number = config.intent.experiment_number
    if experiment_number is None:
        raise ValueError("compression-quality experiment requires an experiment number")
    if resolved.inputs.launcher_path is None:
        raise ValueError("compression-quality experiment requires launcher provenance")
    repository_root = resolved.inputs.launcher_path.resolve().parent.parent
    quality_started = time.perf_counter()
    quality = execute_quality_evaluation(
        QualityEvaluationRequest(
            snapshot=resolved.inputs.snapshot,
            source=config.model.source,
            revision=str(config.model.revision),
            run_output=resolved.inputs.output,
            device=config.runtime.compute_device,
            backend=experiment.quality_backend,
            use_global_tuning=config.distillation.enabled,
            wikitext_samples=experiment.wikitext_samples,
            wikitext_sequence_length=experiment.wikitext_sequence_length,
            wikitext_batch_size=experiment.wikitext_batch_size,
            task_names=experiment.task_names,
            task_limit=experiment.task_limit,
            task_batch_size=experiment.task_batch_size,
            local_files_only=experiment.local_files_only,
            maximum_wddm_shared_bytes=maximum_shared_bytes,
            packed_artifact=_repository_path(experiment.export.packed_output, repository_root),
            stream_base_model=config.runtime.executor in {ExecutorKind.CPU_OFFLOAD, ExecutorKind.STREAMING},
        )
    )
    quality_seconds = time.perf_counter() - quality_started
    provenance = to_dict(launcher_provenance(resolved.inputs.launcher_path, config.intent.experiment_number))
    quality_payload = {
        **quality,
        "experiment": {
            "config_hash": config_hash(config),
            "resolved_config": to_dict(config),
            "launcher": provenance,
        },
    }
    atomic_write_json(resolved.quality_output, quality_payload)
    atomic_write_text(resolved.quality_markdown_output, render_quality_evaluation_markdown(quality_payload))
    profiles = tuple(
        str(path.resolve())
        for path in sorted(resolved.inputs.output.glob("profile*.json"))
    )
    publication_directory = repository_root / "Results" / f"{experiment_number:03d}"
    payload = {
        "schema_version": 1,
        "passed": bool(quality.get("passed")),
        "experiment": quality_payload["experiment"],
        "compression": {
            "run_output": str(resolved.inputs.output.resolve()),
            "commit_identity": to_dict(workflow.quantization.identity),
            "blocks": block_count,
            "effective_bpw": workflow.quantization.frozen_model.effective_bpw,
            "peak_device_bytes": workflow.quantization.peak_device_bytes,
            "peak_host_bytes": workflow.quantization.peak_host_bytes,
            "artifact_bytes": workflow.quantization.artifact_bytes,
            "reused_commit_count": workflow.quantization.reused_commit_count,
            "profile_artifacts": profiles,
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
        },
        "stage_measurements": {
            "compression_seconds": compression_seconds,
            "resident_quantization_seconds": workflow.quantization.elapsed_seconds,
            "global_distillation_seconds": (
                None if workflow.distillation is None else workflow.distillation.result.wall_seconds
            ),
            "quality_seconds": quality_seconds,
            "wall_seconds": time.perf_counter() - wall_started,
        },
        "quality": {
            "json": str(resolved.quality_output),
            "markdown": str(resolved.quality_markdown_output),
            "comparison": quality["comparison"],
            "resource_limits": quality["resource_limits"],
        },
        "publication": {
            "directory": str(publication_directory),
            "manifest": str(publication_directory / "publication.json"),
        },
    }
    atomic_write_json(resolved.summary_output, payload)
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
            PublishableArtifact(resolved.summary_output, PublishableArtifactKind.STATISTICS),
            PublishableArtifact(resolved.quality_output, PublishableArtifactKind.STATISTICS),
            PublishableArtifact(resolved.quality_markdown_output, PublishableArtifactKind.REPORT),
            *(PublishableArtifact(path, PublishableArtifactKind.STATISTICS) for path in profile_json),
            *(PublishableArtifact(path, PublishableArtifactKind.REPORT) for path in profile_markdown),
        ),
    )
    return payload


def run_compression_quality_experiment(
    config: RunConfig,
    experiment: CompressionQualityExperiment,
    *,
    launcher_path: str | Path,
) -> int:
    raise_for_issues(validate(config, ValidationPhase.RESOLVED))
    validate_launcher_number(config, launcher_path)
    resolved = resolve_compression_quality_experiment(config, experiment, launcher_path=launcher_path)
    execute_compression_quality_experiment(config, experiment, resolved)
    return 0


__all__ = [
    "CompressionQualityExperiment",
    "ResolvedCompressionQualityExperiment",
    "execute_compression_quality_experiment",
    "resolve_compression_quality_experiment",
    "run_compression_quality_experiment",
]
