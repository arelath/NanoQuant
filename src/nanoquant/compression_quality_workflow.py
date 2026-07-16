"""End-to-end compression and quality-proof experiment composition."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoquant.config.codec import config_hash, to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text
from nanoquant.infrastructure.runs import launcher_provenance, validate_launcher_number
from nanoquant.quality_evaluation import QualityEvaluationRequest, execute_quality_evaluation
from nanoquant.quality_evaluation_workflow import render_quality_evaluation_markdown
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    execute_resident_workflow,
    resolve_resident_experiment_inputs,
)


@dataclass(frozen=True, slots=True)
class CompressionQualityExperiment:
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
    workflow = execute_resident_workflow(
        config,
        resolved.inputs,
        ResidentExecutionOptions(
            restore_completed_blocks=experiment.restore_completed_blocks,
            maximum_wddm_shared_bytes=maximum_shared_bytes,
        ),
    )
    compression_seconds = time.perf_counter() - compression_started
    block_count = len(workflow.quantization.inventory.blocks)
    if block_count != experiment.expected_blocks:
        raise ValueError(
            f"resolved model block count differs from experiment: {block_count} != {experiment.expected_blocks}"
        )
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
        )
    )
    quality_seconds = time.perf_counter() - quality_started
    if resolved.inputs.launcher_path is None:
        raise ValueError("compression-quality experiment requires launcher provenance")
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
    }
    atomic_write_json(resolved.summary_output, payload)
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
