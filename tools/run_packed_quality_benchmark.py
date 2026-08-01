"""Run the complete retained BF16-versus-frozen or packed quality benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401

from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.quality_evaluation import QualityEvaluationRequest, execute_quality_evaluation

DEFAULT_TASKS = (
    "piqa",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "boolq",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packed-artifact",
        type=Path,
        help="optional packed artifact; omit it to evaluate the committed factorized run",
    )
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("dense", "factorized"), default="factorized")
    parser.add_argument("--wikitext-samples", type=int, default=64)
    parser.add_argument("--wikitext-sequence-length", type=int, default=128)
    parser.add_argument("--wikitext-batch-size", type=int, default=8)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--task-limit", type=int, default=200)
    parser.add_argument("--task-batch-size", type=int, default=4)
    parser.add_argument("--maximum-wddm-shared-bytes", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-global-tuning", action="store_true")
    return parser


def _progress(event: str, fields: dict[str, object]) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ValueError(f"quality benchmark output already exists: {args.output}")
    request = QualityEvaluationRequest(
        snapshot=args.snapshot,
        source=args.source,
        revision=args.revision,
        run_output=args.run_output,
        device=args.device,
        backend=args.backend,
        use_global_tuning=not args.no_global_tuning,
        wikitext_samples=args.wikitext_samples,
        wikitext_sequence_length=args.wikitext_sequence_length,
        wikitext_batch_size=args.wikitext_batch_size,
        task_names=tuple(args.task) if args.task else DEFAULT_TASKS,
        task_limit=args.task_limit,
        task_batch_size=args.task_batch_size,
        local_files_only=args.local_files_only,
        maximum_wddm_shared_bytes=args.maximum_wddm_shared_bytes,
        packed_artifact=args.packed_artifact,
    )
    result: dict[str, Any] = execute_quality_evaluation(request, progress=_progress)
    atomic_write_json(args.output, result)
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
