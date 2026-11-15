"""Export a completed run's pre-global-tuning (pre-KD) weights to GGUF and benchmark them.

The run's block commits are exported with global distillation disabled into new
logical, packed, llama.cpp checkpoint and GGUF paths, so the published post-KD
exports are never reused or replaced. Quality uses the run's own protocol on the
pre-KD packed artifact and reuses the published BF16 base result when its
protocol matches. Results are added to ``Results/NNN`` under ``-pre-kd`` names.

Inspect with ``--plan-only``; it reads the launcher and paths and starts nothing.
"""

from __future__ import annotations

import argparse
import json
import runpy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
from huggingface_hub import snapshot_download
from validate_resident_run import validate_resident_run

from nanoquant.compression_export_workflow import CompressionExportRecipe, execute_compression_export
from nanoquant.compression_quality_workflow import CompressionQualityExperiment
from nanoquant.config.codec import config_hash, to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.infrastructure.global_tuning import active_global_tuning
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text
from nanoquant.infrastructure.publication import (
    PublishableArtifact,
    PublishableArtifactKind,
    publish_experiment_artifacts,
)
from nanoquant.infrastructure.quality_progress import QualityProgressLogger
from nanoquant.infrastructure.runs import launcher_provenance
from nanoquant.quality_evaluation import QualityEvaluationRequest, execute_quality_evaluation
from nanoquant.quality_evaluation_workflow import render_quality_evaluation_markdown
from nanoquant.resident_workflow import _reject_incompatible_run_before_preparation
from nanoquant.self_measured_d2_workflow import _resolved_model_block_count, _uniform_control_config

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_BASE_PROTOCOL_FIELDS = (
    "wikitext_samples",
    "wikitext_sequence_length",
    "wikitext_batch_size",
    "task_names",
    "task_limit",
    "task_batch_size",
)


@dataclass(frozen=True, slots=True)
class PreGlobalTuningPlan:
    experiment_number: int
    config: RunConfig
    workflow: CompressionQualityExperiment
    run_output: Path
    export: CompressionExportRecipe
    quality_output: Path
    quality_markdown_output: Path
    published_quality: Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("launcher", type=Path, help="numbered experiment launcher whose run is complete")
    parser.add_argument(
        "--uniform-control",
        action="store_true",
        help="the launcher published its self-measured uniform control (reuse_maximum_rank_control=True)",
    )
    parser.add_argument("--suffix", default="pre-kd", help="name suffix for the new outputs")
    parser.add_argument("--plan-only", action="store_true", help="print the resolved paths and exit")
    parser.add_argument(
        "--refresh-quality", action="store_true",
        help="validate/reuse existing pre-KD exports and recompute an existing pre-KD quality report",
    )
    return parser


def _repository_path(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _with_suffix(path: Path, suffix: str, marker: str = "") -> Path:
    stem = path.stem.removesuffix(marker) if marker else path.stem
    return path.with_name(f"{stem}-{suffix}{marker}{path.suffix}")


def build_plan(
    launcher: Path,
    *,
    uniform_control: bool,
    suffix: str = "pre-kd",
    repository_root: Path = REPOSITORY_ROOT,
) -> PreGlobalTuningPlan:
    if not suffix or Path(suffix).name != suffix or suffix.startswith("-"):
        raise ValueError("pre-global-tuning output suffix must be a plain name")
    namespace = runpy.run_path(str(launcher))
    experiment = namespace["EXPERIMENT"]
    workflow = cast(CompressionQualityExperiment, experiment.workflow)
    number = experiment.config.intent.experiment_number
    if number is None:
        raise ValueError("pre-global-tuning export requires a numbered experiment launcher")
    if not experiment.config.distillation.enabled:
        raise ValueError("the experiment has no global distillation, so its export is already pre-KD")
    config = (
        _uniform_control_config(
            experiment.config,
            (repository_root / "evidence" / f"{number:03d}").resolve(),
            tuned_operating_point=True,
        )
        if uniform_control
        else experiment.config
    )
    registry = Path(config.output.run_root)
    run_output = (registry if registry.is_absolute() else repository_root / registry) / config.intent.name
    outputs = Path("outputs") / f"{number:03d}" / suffix
    gguf = workflow.export.gguf_output
    export = replace(
        workflow.export,
        logical_output=outputs / "logical",
        packed_output=outputs / "packed",
        checkpoint_output=outputs / "llamacpp-checkpoint",
        gguf_output=gguf.with_name(f"{gguf.stem}-{suffix}{gguf.suffix}"),
        huggingface=None,
        publication_experiment_number=number,
        use_active_global_tuning=False,
    )
    return PreGlobalTuningPlan(
        number,
        config,
        workflow,
        run_output.resolve(),
        export,
        _repository_path(_with_suffix(workflow.quality_output, suffix, "-quality"), repository_root),
        _repository_path(_with_suffix(workflow.quality_markdown_output, suffix, "-quality"), repository_root),
        _repository_path(workflow.quality_output, repository_root),
    )


def _new_paths(plan: PreGlobalTuningPlan, repository_root: Path) -> tuple[Path, ...]:
    export = plan.export
    return tuple(
        _repository_path(path, repository_root)
        for path in (
            export.logical_output,
            export.packed_output,
            export.checkpoint_output,
            export.gguf_output,
        )
    ) + (plan.quality_output, plan.quality_markdown_output)


def _require_separate_outputs(plan: PreGlobalTuningPlan, repository_root: Path) -> None:
    original = plan.workflow.export
    published = {
        _repository_path(path, repository_root)
        for path in (
            original.logical_output,
            original.packed_output,
            original.checkpoint_output,
            original.gguf_output,
            plan.workflow.quality_output,
            plan.workflow.quality_markdown_output,
            plan.workflow.summary_output,
        )
    }
    overlap = published.intersection(_new_paths(plan, repository_root))
    if overlap:
        raise ValueError(f"pre-global-tuning outputs would replace published outputs: {sorted(map(str, overlap))}")


def reusable_base_result(published_quality: Path, request: QualityEvaluationRequest) -> dict[str, Any] | None:
    """Return the published BF16 result when its model and protocol match the request."""

    if not published_quality.is_file():
        return None
    payload = json.loads(published_quality.read_text(encoding="utf-8"))
    model = payload.get("model", {})
    protocol = payload.get("protocol", {})
    if model.get("source") != request.source or model.get("revision") != request.revision:
        return None
    for field in _BASE_PROTOCOL_FIELDS:
        expected = getattr(request, field)
        if (list(expected) if isinstance(expected, tuple) else expected) != protocol.get(field):
            return None
    base = payload.get("results", {}).get("base")
    return base if isinstance(base, dict) else None


def _comparison_markdown(pre: dict[str, Any], published_quality: Path) -> str:
    published = json.loads(published_quality.read_text(encoding="utf-8"))
    final = published["comparison"]
    current = pre["comparison"]
    lines = [
        "## Pre-KD versus published post-KD result",
        "",
        "Both rows evaluate the same block commits with the same protocol and BF16 reference; the",
        "post-KD row additionally applies the run's global distillation result.",
        "",
        "| Benchmark | Metric | BF16 | Pre-KD | Post-KD | Post − pre |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        (
            f"| WikiText-2 | perplexity ↓ | {current['wikitext']['base_perplexity']:.4f} | "
            f"{current['wikitext']['frozen_perplexity']:.4f} | {final['wikitext']['frozen_perplexity']:.4f} | "
            f"{final['wikitext']['frozen_perplexity'] - current['wikitext']['frozen_perplexity']:+.4f} |"
        ),
    ]
    final_tasks = {item["task_name"]: item for item in final["tasks"]}
    for item in current["tasks"]:
        after = final_tasks.get(item["task_name"])
        if after is None:
            continue
        lines.append(
            f"| {item['task_name']} | {item['metric']} ↑ | {item['base']:.4f} | {item['frozen']:.4f} | "
            f"{after['frozen']:.4f} | {after['frozen'] - item['frozen']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace, *, repository_root: Path = REPOSITORY_ROOT) -> int:
    launcher = args.launcher.resolve()
    plan = build_plan(launcher, uniform_control=args.uniform_control, suffix=args.suffix,
                      repository_root=repository_root)
    _require_separate_outputs(plan, repository_root)
    if args.plan_only:
        print(json.dumps({
            "experiment_number": plan.experiment_number,
            "run_output": str(plan.run_output),
            "config_hash": config_hash(plan.config),
            "export": to_dict(plan.export),
            "quality_output": str(plan.quality_output),
            "quality_markdown_output": str(plan.quality_markdown_output),
            "base_quality": str(plan.published_quality),
        }, indent=2, default=str))
        return 0
    if not (plan.run_output / "state" / "journal.jsonl").is_file():
        raise FileNotFoundError(f"completed run journal is missing: {plan.run_output}")
    _reject_incompatible_run_before_preparation(plan.run_output, config_hash(plan.config))
    if active_global_tuning(plan.run_output) is None:
        raise ValueError("the run has no active global tuning, so its published export is already pre-KD")
    if plan.quality_output.exists() and not getattr(args, "refresh_quality", False):
        raise FileExistsError(f"pre-global-tuning quality output already exists: {plan.quality_output}")
    config = plan.config
    workflow = plan.workflow
    snapshot = Path(snapshot_download(
        repo_id=config.model.source,
        revision=str(config.model.revision),
        local_files_only=workflow.local_files_only,
    )).resolve()
    validate_resident_run(
        plan.run_output, expected_blocks=_resolved_model_block_count(config), require_complete=True,
    )

    print(f"[pre-kd] exporting {plan.run_output} without global tuning", flush=True)
    exports = execute_compression_export(
        config,
        plan.export,
        repository_root=repository_root,
        run_output=plan.run_output,
        snapshot=snapshot,
    )
    if exports.logical.get("global_tuning") is not None:
        raise RuntimeError("pre-global-tuning logical export unexpectedly applied global tuning")
    print(f"[pre-kd] GGUF {exports.gguf.output} ({exports.gguf.bytes:,} bytes, sha256 {exports.gguf.sha256})",
          flush=True)

    maximum_shared = workflow.maximum_wddm_shared_gib
    request = QualityEvaluationRequest(
        snapshot=snapshot,
        source=config.model.source,
        revision=str(config.model.revision),
        run_output=plan.run_output,
        device=config.runtime.compute_device,
        backend=workflow.quality_backend or "factorized",
        use_global_tuning=False,
        wikitext_samples=workflow.wikitext_samples,
        wikitext_sequence_length=workflow.wikitext_sequence_length,
        wikitext_batch_size=workflow.wikitext_batch_size,
        task_names=workflow.task_names,
        task_limit=workflow.task_limit,
        task_batch_size=workflow.task_batch_size,
        local_files_only=workflow.local_files_only,
        maximum_wddm_shared_bytes=None if maximum_shared is None else int(maximum_shared * 2**30),
        packed_artifact=_repository_path(plan.export.packed_output, repository_root),
    )
    base_result = reusable_base_result(plan.published_quality, request)
    print(
        "[pre-kd] "
        + ("reusing the published BF16 result" if base_result is not None else "evaluating BF16 afresh"),
        flush=True,
    )
    quality = execute_quality_evaluation(
        request,
        progress=QualityProgressLogger(),
        base_result=base_result,
    )
    if quality["candidate"]["global_tuning"] is not None:
        raise RuntimeError("pre-global-tuning quality unexpectedly applied global tuning")
    if base_result is not None:
        published = json.loads(plan.published_quality.read_text(encoding="utf-8"))
        stored = {key: value for key, value in published["protocol"].items() if key != "base_execution"}
        current = {key: value for key, value in quality["protocol"].items() if key != "base_execution"}
        if json.loads(json.dumps(stored)) != json.loads(json.dumps(current)):
            raise RuntimeError("reused BF16 result protocol differs from the pre-KD evaluation protocol")
    payload = {
        **quality,
        "variant": {
            "name": args.suffix,
            "description": "block commits exported and evaluated without the run's global distillation result",
            "published_post_kd_quality": str(plan.published_quality),
            "base_result_reused": base_result is not None,
        },
        "deployment_storage": {
            "bf16_checkpoint_bytes": (
                json.loads(plan.published_quality.read_text(encoding="utf-8"))
                .get("deployment_storage", {})
                .get("bf16_checkpoint_bytes")
                if plan.published_quality.is_file()
                else None
            ),
            "packed_quantized_layer_bytes": exports.packed["packed_weight_bytes"],
            "gguf_bytes": exports.gguf.bytes,
        },
        "exports": {
            "summary": str(exports.summary_output),
            "gguf": str(exports.gguf.output),
            "gguf_sha256": exports.gguf.sha256,
            "packed": str(_repository_path(plan.export.packed_output, repository_root)),
        },
        "experiment": {
            "config_hash": config_hash(config),
            "resolved_config": to_dict(config),
            "launcher": to_dict(launcher_provenance(launcher, plan.experiment_number)),
        },
    }
    atomic_write_json(plan.quality_output, payload)
    markdown = render_quality_evaluation_markdown(payload).rstrip() + "\n"
    markdown = markdown.replace(
        " quality benchmark\n",
        f" quality benchmark ({args.suffix}, no global distillation)\n",
        1,
    )
    if plan.published_quality.is_file():
        markdown += "\n" + _comparison_markdown(payload, plan.published_quality)
    atomic_write_text(plan.quality_markdown_output, markdown)
    publish_experiment_artifacts(
        repository_root,
        plan.experiment_number,
        (
            PublishableArtifact(exports.gguf.output, PublishableArtifactKind.MODEL),
            PublishableArtifact(exports.summary_output, PublishableArtifactKind.STATISTICS),
            PublishableArtifact(
                exports.gguf.output.with_suffix(exports.gguf.output.suffix + ".export.json"),
                PublishableArtifactKind.STATISTICS,
            ),
            PublishableArtifact(plan.quality_output, PublishableArtifactKind.STATISTICS),
            PublishableArtifact(plan.quality_markdown_output, PublishableArtifactKind.REPORT),
        ),
    )
    print(f"[pre-kd] quality report {plan.quality_markdown_output}", flush=True)
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
