"""Numbered selective-rank experiment, export, and matched quality comparison."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from huggingface_hub import snapshot_download

from nanoquant.config.codec import config_hash, to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.device_memory import SharedDeviceMemoryMonitor
from nanoquant.infrastructure.gguf_export import export_llamacpp_gguf
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text
from nanoquant.infrastructure.publication import (
    PublishableArtifact,
    PublishableArtifactKind,
    publish_experiment_artifacts,
)
from nanoquant.infrastructure.runs import launcher_provenance, validate_launcher_number
from nanoquant.quality_evaluation import QualityEvaluationRequest, execute_quality_evaluation
from nanoquant.quality_evaluation_workflow import render_quality_evaluation_markdown
from nanoquant.rank_expansion_workflow import RankExpansionRequest, execute_rank_expansion


@dataclass(frozen=True, slots=True)
class RankExpansionExperiment:
    parent_run: Path
    source_packed: Path
    output_packed: Path
    checkpoint_output: Path
    gguf_output: Path
    expansion_report: Path
    quality_output: Path
    quality_markdown_output: Path
    summary_output: Path
    baseline_quality: Path
    llama_cpp_root: Path
    expected_blocks: int = 34
    layer_suffix: str = "self_attn.v_proj"
    bit_multiplier: float = 1.30
    maximum_wddm_shared_gib: float = 0.75
    wikitext_samples: int = 64
    wikitext_sequence_length: int = 128
    task_names: tuple[str, ...] = (
        "piqa",
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "boolq",
    )
    task_limit: int = 200

    def __post_init__(self) -> None:
        if self.expected_blocks <= 0 or self.bit_multiplier <= 1:
            raise ValueError("rank expansion experiment dimensions are invalid")
        if not math.isfinite(self.maximum_wddm_shared_gib) or self.maximum_wddm_shared_gib < 0:
            raise ValueError("rank expansion shared-memory limit is invalid")


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _snapshot(config: RunConfig) -> Path:
    configured = Path(config.model.source)
    return (
        configured.resolve()
        if configured.exists()
        else Path(
            snapshot_download(
                repo_id=config.model.source,
                revision=str(config.model.revision),
            )
        ).resolve()
    )


def _candidate_comparison(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    baseline = cast(dict[str, Any], baseline_payload["comparison"])
    candidate = cast(dict[str, Any], candidate_payload["comparison"])
    baseline_wikitext = cast(dict[str, Any], baseline["wikitext"])
    candidate_wikitext = cast(dict[str, Any], candidate["wikitext"])
    baseline_ppl = float(baseline_wikitext["frozen_perplexity"])
    candidate_ppl = float(candidate_wikitext["frozen_perplexity"])
    baseline_tasks = {
        str(item["task_name"]): item for item in cast(list[dict[str, Any]], baseline["tasks"])
    }
    task_rows = []
    for item in cast(list[dict[str, Any]], candidate["tasks"]):
        name = str(item["task_name"])
        old = float(baseline_tasks[name]["frozen"])
        new = float(item["frozen"])
        task_rows.append(
            {
                "task_name": name,
                "metric": str(item["metric"]),
                "experiment_003": old,
                "candidate": new,
                "delta": new - old,
            }
        )
    return {
        "wikitext": {
            "experiment_003_perplexity": baseline_ppl,
            "candidate_perplexity": candidate_ppl,
            "absolute_delta": candidate_ppl - baseline_ppl,
            "relative_change": candidate_ppl / baseline_ppl - 1.0,
        },
        "tasks": task_rows,
    }


def run_rank_expansion_experiment(
    config: RunConfig,
    experiment: RankExpansionExperiment,
    *,
    launcher_path: str | Path,
) -> int:
    """Run or resume the derivative, export it, then evaluate the matched protocol."""

    raise_for_issues(validate(config, ValidationPhase.RESOLVED))
    validate_launcher_number(config, launcher_path)
    launcher = Path(launcher_path).resolve()
    root = launcher.parent.parent
    snapshot = _snapshot(config)
    resolved = replace(
        experiment,
        parent_run=_resolve(root, experiment.parent_run),
        source_packed=_resolve(root, experiment.source_packed),
        output_packed=_resolve(root, experiment.output_packed),
        checkpoint_output=_resolve(root, experiment.checkpoint_output),
        gguf_output=_resolve(root, experiment.gguf_output),
        expansion_report=_resolve(root, experiment.expansion_report),
        quality_output=_resolve(root, experiment.quality_output),
        quality_markdown_output=_resolve(root, experiment.quality_markdown_output),
        summary_output=_resolve(root, experiment.summary_output),
        baseline_quality=_resolve(root, experiment.baseline_quality),
        llama_cpp_root=_resolve(root, experiment.llama_cpp_root),
    )
    maximum_shared_bytes = int(resolved.maximum_wddm_shared_gib * 2**30)
    wall_started = time.perf_counter()
    expansion_started = time.perf_counter()
    with acquire_device_lease(config.runtime.compute_device), SharedDeviceMemoryMonitor(
        maximum_shared_bytes
    ) as expansion_monitor:
        expansion = execute_rank_expansion(
            RankExpansionRequest(
                parent_run=resolved.parent_run,
                source_packed=resolved.source_packed,
                snapshot=snapshot,
                output_packed=resolved.output_packed,
                report_output=resolved.expansion_report,
                source=config.model.source,
                revision=str(config.model.revision),
                expected_blocks=resolved.expected_blocks,
                layer_suffix=resolved.layer_suffix,
                bit_multiplier=resolved.bit_multiplier,
                rank_multiple=config.allocation.bounds.multiple,
                device=config.runtime.compute_device,
                seed=config.reproducibility.seed,
                outer_iterations=config.factorization.admm.outer_iterations,
                inner_iterations=config.factorization.admm.inner_iterations,
                regularization=config.factorization.admm.regularization,
                penalty_schedule=config.factorization.admm.penalty_schedule,
                convergence_check_interval=config.factorization.admm.convergence_check_interval,
                early_stop_tolerance=config.factorization.admm.early_stop_tolerance,
            ),
            safe_point=expansion_monitor.check,
        )
    expansion_seconds = time.perf_counter() - expansion_started
    export_started = time.perf_counter()
    gguf = export_llamacpp_gguf(
        resolved.output_packed,
        snapshot,
        resolved.checkpoint_output,
        resolved.gguf_output,
        resolved.llama_cpp_root,
    )
    export_seconds = time.perf_counter() - export_started
    quality_started = time.perf_counter()
    quality = execute_quality_evaluation(
        QualityEvaluationRequest(
            snapshot=snapshot,
            source=config.model.source,
            revision=str(config.model.revision),
            run_output=resolved.parent_run,
            device=config.runtime.compute_device,
            backend="dense",
            use_global_tuning=True,
            wikitext_samples=resolved.wikitext_samples,
            wikitext_sequence_length=resolved.wikitext_sequence_length,
            task_names=resolved.task_names,
            task_limit=resolved.task_limit,
            maximum_wddm_shared_bytes=maximum_shared_bytes,
            packed_artifact=resolved.output_packed,
        )
    )
    quality_seconds = time.perf_counter() - quality_started
    provenance = to_dict(launcher_provenance(launcher, config.intent.experiment_number))
    quality_payload = {
        **quality,
        "experiment": {
            "config_hash": config_hash(config),
            "resolved_config": to_dict(config),
            "launcher": provenance,
        },
    }
    atomic_write_json(resolved.quality_output, quality_payload)
    atomic_write_text(
        resolved.quality_markdown_output,
        render_quality_evaluation_markdown(quality_payload),
    )
    baseline_payload = cast(
        dict[str, Any],
        json.loads(resolved.baseline_quality.read_text(encoding="utf-8")),
    )
    derivative_comparison = _candidate_comparison(baseline_payload, quality_payload)
    experiment_number = config.intent.experiment_number
    if experiment_number is None:
        raise ValueError("rank expansion requires an experiment number")
    summary = {
        "schema_version": 1,
        "passed": bool(quality["passed"]),
        "experiment": quality_payload["experiment"],
        "hypothesis": {
            "parent_experiment": 3,
            "layer_suffix": resolved.layer_suffix,
            "requested_bit_multiplier": resolved.bit_multiplier,
            "non_target_layers_must_be_exact": True,
        },
        "expansion": expansion,
        "gguf": {
            "output": str(gguf.output),
            "bytes": gguf.bytes,
            "sha256": gguf.sha256,
            "token_embedding_type": gguf.token_embedding_type,
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
            }
        ),
        "quality": {
            "output": str(resolved.quality_output),
            "markdown": str(resolved.quality_markdown_output),
            "comparison_to_bf16": quality["comparison"],
            "comparison_to_experiment_003": derivative_comparison,
            "resource_limits": quality["resource_limits"],
        },
        "stage_measurements": {
            "rank_expansion_seconds": expansion_seconds,
            "gguf_export_seconds": export_seconds,
            "quality_seconds": quality_seconds,
            "wall_seconds": time.perf_counter() - wall_started,
        },
    }
    atomic_write_json(resolved.summary_output, summary)
    publish_experiment_artifacts(
        root,
        experiment_number,
        (
            PublishableArtifact(gguf.output, PublishableArtifactKind.MODEL),
            PublishableArtifact(
                gguf.output.with_suffix(gguf.output.suffix + ".export.json"),
                PublishableArtifactKind.STATISTICS,
            ),
            *(
                ()
                if gguf.mmproj is None
                else (
                    PublishableArtifact(gguf.mmproj.output, PublishableArtifactKind.MODEL),
                    PublishableArtifact(
                        gguf.mmproj.output.with_suffix(gguf.mmproj.output.suffix + ".export.json"),
                        PublishableArtifactKind.STATISTICS,
                    ),
                )
            ),
            PublishableArtifact(resolved.expansion_report, PublishableArtifactKind.STATISTICS),
            PublishableArtifact(resolved.quality_output, PublishableArtifactKind.STATISTICS),
            PublishableArtifact(resolved.quality_markdown_output, PublishableArtifactKind.REPORT),
            PublishableArtifact(resolved.summary_output, PublishableArtifactKind.STATISTICS),
        ),
    )
    return 0


__all__ = ["RankExpansionExperiment", "run_rank_expansion_experiment"]
