"""End-to-end compression and quality-proof experiment composition."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoquant.compression_export_workflow import (
    CompressionExportRecipe,
    complete_deferred_huggingface_upload,
    execute_complete_compression,
)
from nanoquant.config.codec import config_hash, to_dict
from nanoquant.config.schema import ExecutorKind, RunConfig
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.infrastructure.huggingface_model_card import load_huggingface_model_card_metadata
from nanoquant.infrastructure.huggingface_upload import (
    ensure_huggingface_model_repository,
    huggingface_upload_summary,
)
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text
from nanoquant.infrastructure.publication import (
    PublishableArtifact,
    PublishableArtifactKind,
    publish_experiment_artifacts,
)
from nanoquant.infrastructure.run_session import open_run_event_append_session
from nanoquant.infrastructure.runs import launcher_provenance, validate_launcher_number
from nanoquant.llamacpp_quality import (
    LlamaCppQualityRequest,
    execute_llamacpp_quality_evaluation,
    render_llamacpp_quality_markdown,
)
from nanoquant.quality_evaluation import (
    DEFAULT_QUALITY_TASK_BATCH_SIZE,
    DEFAULT_QUALITY_WIKITEXT_BATCH_SIZE,
    QualityEvaluationRequest,
    execute_quality_evaluation,
    prepare_quality_inputs,
)
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
    wikitext_batch_size: int = DEFAULT_QUALITY_WIKITEXT_BATCH_SIZE
    task_names: tuple[str, ...] = (
        "piqa",
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "boolq",
    )
    task_limit: int = 200
    task_batch_size: int = DEFAULT_QUALITY_TASK_BATCH_SIZE
    local_files_only: bool = False
    maximum_wddm_shared_gib: float | None = None
    restore_completed_blocks: bool = True
    quality_backend: str | None = "factorized"
    large_model_guards: bool = False
    llamacpp_quality: bool = False
    llama_cpp_root: Path | None = None
    llamacpp_quality_parallel: int = 4

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
        if self.quality_backend not in {None, "factorized", "dense"}:
            raise ValueError("quality backend must be factorized, dense, or disabled")
        if self.quality_backend is None and not self.llamacpp_quality:
            raise ValueError("disabled PyTorch candidate quality requires llama.cpp quality")
        if self.large_model_guards and self.restore_completed_blocks:
            raise ValueError("large-model quality experiments must disable completed-block restoration")
        if self.llamacpp_quality and self.llama_cpp_root is None:
            raise ValueError("llama.cpp quality requires a llama.cpp repository root")
        if self.llamacpp_quality_parallel <= 0:
            raise ValueError("llama.cpp quality parallel sequence count must be positive")


@dataclass(frozen=True, slots=True)
class ResolvedCompressionQualityExperiment:
    inputs: ResolvedResidentInputs
    summary_output: Path
    quality_output: Path
    quality_markdown_output: Path
    llamacpp_quality_output: Path | None = None


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
    quality_output = _repository_path(experiment.quality_output, root)
    return ResolvedCompressionQualityExperiment(
        resolve_resident_experiment_inputs(config, launcher_path=launcher),
        _repository_path(experiment.summary_output, root),
        quality_output,
        _repository_path(experiment.quality_markdown_output, root),
        (
            quality_output.with_name(
                quality_output.stem.removesuffix("-quality") + "-gguf-quality.json"
            )
            if experiment.llamacpp_quality
            else None
        ),
    )


def execute_compression_quality_experiment(
    config: RunConfig,
    experiment: CompressionQualityExperiment,
    resolved: ResolvedCompressionQualityExperiment,
) -> dict[str, Any]:
    """Compress the pinned model, then compare BF16 and frozen quality."""

    if experiment.large_model_guards:
        if config.runtime.executor not in {ExecutorKind.CPU_OFFLOAD, ExecutorKind.STREAMING}:
            raise ValueError("large-model compression requires cpu_offload or streaming execution")
        if config.evaluation.inline_quality:
            raise ValueError("large-model compression requires inline quality to be disabled")
        if config.distillation.enabled:
            raise ValueError("large-model compression requires distillation to remain disabled until teacher streaming")

    if experiment.export.huggingface is not None:
        destination = experiment.export.huggingface
        print(
            f"Hugging Face repository preflight started: repo_id={destination.repo_id}",
            flush=True,
        )
        resolved_repo_id = ensure_huggingface_model_repository(destination)
        print(
            f"Hugging Face repository preflight completed: repo_id={resolved_repo_id}",
            flush=True,
        )

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
    quality_request = QualityEvaluationRequest(
        snapshot=resolved.inputs.snapshot,
        source=config.model.source,
        revision=str(config.model.revision),
        run_output=resolved.inputs.output,
        device=config.runtime.compute_device,
        backend=experiment.quality_backend or "factorized",
        use_global_tuning=config.distillation.enabled,
        wikitext_samples=experiment.wikitext_samples,
        wikitext_sequence_length=experiment.wikitext_sequence_length,
        wikitext_batch_size=experiment.wikitext_batch_size,
        task_names=experiment.task_names,
        task_limit=experiment.task_limit,
        task_batch_size=experiment.task_batch_size,
        local_files_only=experiment.local_files_only,
        maximum_wddm_shared_bytes=maximum_shared_bytes,
        packed_artifact=(
            None
            if experiment.quality_backend is None
            else _repository_path(experiment.export.packed_output, repository_root)
        ),
        stream_base_model=(
            experiment.large_model_guards
            or config.runtime.executor in {ExecutorKind.CPU_OFFLOAD, ExecutorKind.STREAMING}
        ),
    )
    prepared_quality = (
        prepare_quality_inputs(quality_request) if experiment.llamacpp_quality else None
    )
    quality = execute_quality_evaluation(
        quality_request,
        prepared=prepared_quality,
        evaluate_candidate=experiment.quality_backend is not None,
    )
    quality_seconds = time.perf_counter() - quality_started
    llamacpp_quality_started = time.perf_counter()
    llamacpp_quality = None
    if experiment.llamacpp_quality:
        if resolved.llamacpp_quality_output is None or experiment.llama_cpp_root is None:
            raise ValueError("resolved llama.cpp quality paths are incomplete")
        if prepared_quality is None:
            raise RuntimeError("llama.cpp quality inputs were not prepared")
        with open_run_event_append_session(
            resolved.inputs.output,
            observability=config.observability,
        ) as quality_events:
            quality_events.emit(
                "quality",
                "info",
                "quality.llamacpp.started",
                gguf=str(exports.gguf.output),
                gguf_sha256=exports.gguf.sha256,
                parallel=experiment.llamacpp_quality_parallel,
            )
            try:
                llamacpp_quality = execute_llamacpp_quality_evaluation(
                    LlamaCppQualityRequest(
                        gguf=exports.gguf.output,
                        output=resolved.llamacpp_quality_output,
                        llama_cpp_root=experiment.llama_cpp_root,
                        device=config.runtime.compute_device,
                        gpu_layers=(
                            -1
                            if config.runtime.compute_device.startswith("cuda")
                            else 0
                        ),
                        parallel=experiment.llamacpp_quality_parallel,
                    ),
                    quality_request,
                    prepared_quality,
                    quality["results"]["base"],
                    quality["protocol"],
                )
            except BaseException as exc:
                quality_events.emit(
                    "quality",
                    "error",
                    "quality.llamacpp.failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
            quality_events.emit(
                "quality",
                "info",
                "quality.llamacpp.completed",
                output=str(resolved.llamacpp_quality_output),
                reused=bool(llamacpp_quality.get("reused")),
                wall_seconds=llamacpp_quality.get("wall_seconds"),
            )
    llamacpp_quality_seconds = (
        0.0 if llamacpp_quality is None else time.perf_counter() - llamacpp_quality_started
    )
    if experiment.quality_backend is None:
        if llamacpp_quality is None:
            raise RuntimeError("llama.cpp-only quality did not produce a GGUF result")
        quality = {
            **quality,
            "schema_version": 2,
            "passed": bool(quality.get("passed")) and bool(llamacpp_quality.get("passed")),
            "candidate": {
                "run_output": str(resolved.inputs.output.resolve()),
                "commit_identity": to_dict(workflow.quantization.identity),
                "global_tuning": None,
                "backend": "llama.cpp",
                "packed_artifact": None,
                "packed_descriptor_sha256": None,
                "gguf": llamacpp_quality["gguf"],
                "runtime": llamacpp_quality["runtime"],
            },
            "results": {
                "base": quality["results"]["base"],
                "frozen": llamacpp_quality["results"]["gguf"],
            },
            "comparison": llamacpp_quality["comparison"],
            "wall_seconds": float(quality["wall_seconds"])
            + float(llamacpp_quality["wall_seconds"]),
        }
    provenance = to_dict(launcher_provenance(resolved.inputs.launcher_path, config.intent.experiment_number))
    quality_payload = {
        **quality,
        **(
            {}
            if llamacpp_quality is None
            else {"deployment_quality": llamacpp_quality}
        ),
        "deployment_storage": {
            "bf16_checkpoint_bytes": workflow.quantization.inventory.total_source_bytes,
            "packed_quantized_layer_bytes": exports.packed["packed_weight_bytes"],
            "gguf_bytes": exports.gguf.bytes,
        },
        "experiment": {
            "config_hash": config_hash(config),
            "resolved_config": to_dict(config),
            "launcher": provenance,
        },
    }
    atomic_write_json(resolved.quality_output, quality_payload)
    rendered_quality = render_quality_evaluation_markdown(quality_payload)
    if llamacpp_quality is not None and experiment.quality_backend is not None:
        rendered_quality = (
            rendered_quality.rstrip()
            + "\n\n"
            + render_llamacpp_quality_markdown(llamacpp_quality)
        )
    atomic_write_text(resolved.quality_markdown_output, rendered_quality)
    supplemental = (
        (resolved.quality_markdown_output, "README.md"),
        (resolved.quality_output, "quality.json"),
        *(
            ()
            if resolved.llamacpp_quality_output is None
            else ((resolved.llamacpp_quality_output, "gguf-quality.json"),)
        ),
    )
    model_card_metadata = (
        None
        if experiment.export.huggingface is None
        else load_huggingface_model_card_metadata(
            config.model.source,
            str(config.model.revision),
            resolved.inputs.snapshot,
        )
    )
    if experiment.export.huggingface is None:
        exports = complete_deferred_huggingface_upload(
            exports,
            None,
            supplemental,
            model_card_metadata=model_card_metadata,
        )
    else:
        with open_run_event_append_session(
            resolved.inputs.output,
            observability=config.observability,
        ) as upload_events:
            exports = complete_deferred_huggingface_upload(
                exports,
                experiment.export.huggingface,
                supplemental,
                model_card_metadata=model_card_metadata,
                events=upload_events,
            )
    profiles = tuple(
        str(path.resolve())
        for path in sorted(resolved.inputs.output.glob("profile*.json"))
    )
    publication_directory = repository_root / "Results" / f"{experiment_number:03d}"
    payload = {
        "schema_version": 2,
        "passed": bool(quality.get("passed")) and (
            llamacpp_quality is None or bool(llamacpp_quality.get("passed"))
        ),
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
            "huggingface": (
                None
                if exports.huggingface is None
                else huggingface_upload_summary(exports.huggingface)
            ),
        },
        "stage_measurements": {
            "compression_seconds": compression_seconds,
            "resident_quantization_seconds": workflow.quantization.elapsed_seconds,
            "global_distillation_seconds": (
                None if workflow.distillation is None else workflow.distillation.result.wall_seconds
            ),
            "quality_seconds": quality_seconds,
            "llamacpp_quality_seconds": llamacpp_quality_seconds,
            "wall_seconds": time.perf_counter() - wall_started,
        },
        "quality": {
            "json": str(resolved.quality_output),
            "markdown": str(resolved.quality_markdown_output),
            "comparison": quality["comparison"],
            "gguf_json": (
                None
                if resolved.llamacpp_quality_output is None
                else str(resolved.llamacpp_quality_output)
            ),
            "gguf_comparison": (
                None if llamacpp_quality is None else llamacpp_quality["comparison"]
            ),
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
            PublishableArtifact(resolved.summary_output, PublishableArtifactKind.STATISTICS),
            PublishableArtifact(resolved.quality_output, PublishableArtifactKind.STATISTICS),
            *(
                ()
                if resolved.llamacpp_quality_output is None
                else (
                    PublishableArtifact(
                        resolved.llamacpp_quality_output,
                        PublishableArtifactKind.STATISTICS,
                    ),
                )
            ),
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
