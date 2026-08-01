"""Run the retained quality benchmark directly from a durable KD checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from probe_distillation_checkpoint_tail_mass import (
    _apply_gemma_final_norm_scale,
    discover_checkpoints,
)
from run_packed_quality_benchmark import DEFAULT_TASKS

from nanoquant.config.codec import to_dict
from nanoquant.global_distillation import _selected_parameters, _thaw_frozen_layers
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.distillation_checkpoint import load_distillation_checkpoint
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.memory_cleanup import release_memory
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.quality_evaluation import (
    QualityEvaluationRequest,
    _evaluate_model,
    compare_quality_results,
    prepare_quality_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--frozen-run-output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--base-quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wikitext-samples", type=int, default=64)
    parser.add_argument("--wikitext-sequence-length", type=int, default=128)
    parser.add_argument("--wikitext-batch-size", type=int, default=8)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--task-limit", type=int, default=200)
    parser.add_argument("--task-batch-size", type=int, default=4)
    parser.add_argument("--fold-final-norm-scale", type=float, default=1.0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _progress(event: str, fields: dict[str, object]) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _matched_base_result(
    path: Path,
    request: QualityEvaluationRequest,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    protocol = report.get("protocol", {})
    if (
        report.get("model", {}).get("source") != request.source
        or report.get("model", {}).get("revision") != request.revision
        or protocol.get("wikitext_samples") != request.wikitext_samples
        or protocol.get("wikitext_sequence_length") != request.wikitext_sequence_length
        or protocol.get("task_names") != list(request.task_names)
        or protocol.get("task_limit") != request.task_limit
    ):
        raise ValueError("reused base quality report does not match the requested protocol")
    base = report.get("results", {}).get("base")
    if not isinstance(base, dict):
        raise ValueError("reused base quality report has no base result")
    return base, protocol


def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ValueError(f"quality benchmark output already exists: {args.output}")
    if not torch.isfinite(torch.tensor(args.fold_final_norm_scale)) or args.fold_final_norm_scale <= 0:
        raise ValueError("folded final RMSNorm scale must be positive and finite")
    request = QualityEvaluationRequest(
        snapshot=args.snapshot,
        source=args.source,
        revision=args.revision,
        run_output=args.frozen_run_output,
        device=args.device,
        backend="factorized",
        use_global_tuning=False,
        wikitext_samples=args.wikitext_samples,
        wikitext_sequence_length=args.wikitext_sequence_length,
        wikitext_batch_size=args.wikitext_batch_size,
        task_names=tuple(args.task) if args.task else DEFAULT_TASKS,
        task_limit=args.task_limit,
        task_batch_size=args.task_batch_size,
        local_files_only=args.local_files_only,
    )
    base_result, base_protocol = _matched_base_result(args.base_quality, request)
    candidate = discover_checkpoints(args.checkpoint_output, {args.epoch})[0]
    checkpoint_artifacts = LocalArtifactStore(args.checkpoint_output / "artifacts")
    checkpoint = load_distillation_checkpoint(
        candidate.reference,
        candidate.identity,
        checkpoint_artifacts,
    )
    inputs = prepare_quality_inputs(request, _progress)
    started = time.perf_counter()
    with acquire_device_lease(args.device):
        loaded = load_frozen_run(
            args.frozen_run_output,
            args.snapshot,
            source_name=args.source,
            revision=args.revision,
            device="cpu",
            verify_hashes=False,
            backend="factorized",
            use_global_tuning=False,
        )
        frozen_artifacts = LocalArtifactStore(args.frozen_run_output / "artifacts")
        trainable = _thaw_frozen_layers(loaded, LocalTensorStore(frozen_artifacts))
        selected_ids, _auxiliary = _selected_parameters(loaded.model, trainable)
        selected = {
            name: parameter
            for name, parameter in loaded.model.named_parameters()
            if id(parameter) in selected_ids
        }
        values = dict(checkpoint.state.parameter_values)
        if set(values) != set(selected):
            raise ValueError("checkpoint parameters differ from the retained pre-KD selector")
        with torch.no_grad():
            for name, parameter in selected.items():
                parameter.copy_(values[name].to(dtype=parameter.dtype))
        final_norm = selected.get("model.norm.weight")
        if final_norm is None:
            raise ValueError("KD selector does not contain Gemma's final RMSNorm")
        _apply_gemma_final_norm_scale(
            final_norm,
            values["model.norm.weight"],
            args.fold_final_norm_scale,
        )
        model = loaded.model.to(args.device)
        frozen_result = _evaluate_model(
            "frozen",
            model,
            request,
            inputs,
            progress=_progress,
        )
        model.cpu()
        del model, loaded, trainable
        gc.collect()
        release_memory("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "schema_version": 1,
        "passed": True,
        "model": {
            "source": request.source,
            "revision": request.revision,
            "snapshot": str(request.snapshot.resolve()),
        },
        "candidate": {
            "frozen_run_output": str(args.frozen_run_output.resolve()),
            "checkpoint_output": str(args.checkpoint_output.resolve()),
            "checkpoint": to_dict(candidate.reference),
            "checkpoint_epoch": candidate.epoch,
            "checkpoint_steps": candidate.steps,
            "checkpoint_protocol_hash": candidate.identity.protocol_hash,
            "folded_final_norm_scale": args.fold_final_norm_scale,
            "backend": "factorized",
        },
        "protocol": base_protocol,
        "results": {"base": base_result, "frozen": frozen_result},
        "comparison": compare_quality_results(base_result, frozen_result),
        "base_result_source": str(args.base_quality.resolve()),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_write_json(args.output, report)
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
