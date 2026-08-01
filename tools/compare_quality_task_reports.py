"""Compare two quality reports with a paired task-stratified bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401

from nanoquant.infrastructure.io_utils import atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-result", default="frozen")
    parser.add_argument("--candidate-result", default="frozen")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _task_rows(report: dict[str, Any], result_name: str) -> tuple[dict[str, Any], ...]:
    result = report.get("results", {}).get(result_name)
    if not isinstance(result, dict) or not isinstance(result.get("tasks"), list):
        raise ValueError(f"quality report has no task result named {result_name!r}")
    return tuple(cast(dict[str, Any], item["result"]) for item in result["tasks"])


def _example_value(example: dict[str, Any], metric: str) -> float:
    field = {"acc": "raw_correct", "acc_norm": "normalized_correct"}.get(metric)
    if field is None or not isinstance(example.get(field), bool):
        raise ValueError(f"unsupported or absent task metric: {metric}")
    return float(example[field])


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_result: str,
    candidate_result: str,
    confidence: float,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if not 0.0 < confidence < 1.0 or resamples <= 0:
        raise ValueError("paired task comparison confidence and resamples are invalid")
    protocol_fields = (
        "task_names",
        "task_limit",
        "task_batch_size",
        "tokenizer_hash",
    )
    baseline_protocol = baseline.get("protocol", {})
    candidate_protocol = candidate.get("protocol", {})
    if any(baseline_protocol.get(field) != candidate_protocol.get(field) for field in protocol_fields):
        raise ValueError("quality report task protocols differ")
    baseline_tasks = _task_rows(baseline, baseline_result)
    candidate_tasks = _task_rows(candidate, candidate_result)
    if len(baseline_tasks) != len(candidate_tasks) or not baseline_tasks:
        raise ValueError("quality report task inventories differ")

    task_deltas: list[tuple[float, ...]] = []
    task_receipts: list[dict[str, Any]] = []
    for before, after in zip(baseline_tasks, candidate_tasks, strict=True):
        identity_fields = ("task_name", "primary_metric", "prompt_hash", "task_semantic_key")
        if any(before.get(field) != after.get(field) for field in identity_fields):
            raise ValueError("quality report task identities differ")
        metric = str(before["primary_metric"])
        before_examples = cast(list[dict[str, Any]], before.get("examples"))
        after_examples = cast(list[dict[str, Any]], after.get("examples"))
        if len(before_examples) != len(after_examples) or not before_examples:
            raise ValueError("quality report task example inventories differ")
        deltas = []
        for before_example, after_example in zip(before_examples, after_examples, strict=True):
            if before_example.get("sample_id") != after_example.get("sample_id"):
                raise ValueError("quality report task sample IDs differ")
            deltas.append(
                _example_value(after_example, metric) - _example_value(before_example, metric)
            )
        observed_before = statistics.fmean(_example_value(item, metric) for item in before_examples)
        observed_after = statistics.fmean(_example_value(item, metric) for item in after_examples)
        if not math.isclose(observed_before, float(before["primary_value"]), abs_tol=1e-12):
            raise ValueError("baseline primary task value differs from its examples")
        if not math.isclose(observed_after, float(after["primary_value"]), abs_tol=1e-12):
            raise ValueError("candidate primary task value differs from its examples")
        task_deltas.append(tuple(deltas))
        task_receipts.append(
            {
                "task_name": before["task_name"],
                "metric": metric,
                "sample_count": len(deltas),
                "baseline": observed_before,
                "candidate": observed_after,
                "delta": observed_after - observed_before,
            }
        )

    point_delta = statistics.fmean(statistics.fmean(values) for values in task_deltas)
    generator = random.Random(seed)
    bootstrap = []
    for _ in range(resamples):
        bootstrap.append(
            statistics.fmean(
                statistics.fmean(values[generator.randrange(len(values))] for _ in values)
                for values in task_deltas
            )
        )
    tail = (1.0 - confidence) / 2.0
    lower = _percentile(bootstrap, tail)
    upper = _percentile(bootstrap, 1.0 - tail)
    baseline_mean = statistics.fmean(float(item["baseline"]) for item in task_receipts)
    candidate_mean = statistics.fmean(float(item["candidate"]) for item in task_receipts)
    return {
        "schema_version": 1,
        "protocol": {
            **{field: baseline_protocol.get(field) for field in protocol_fields},
            "comparison": "paired-task-stratified-bootstrap-v1",
            "confidence": confidence,
            "resamples": resamples,
            "seed": seed,
        },
        "tasks": task_receipts,
        "aggregate": {
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "candidate_minus_baseline": point_delta,
            "lower_delta": lower,
            "upper_delta": upper,
            "regression_established": upper < 0.0,
            "improvement_established": lower > 0.0,
        },
    }


def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ValueError(f"task comparison output already exists: {args.output}")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare_reports(
        baseline,
        candidate,
        baseline_result=args.baseline_result,
        candidate_result=args.candidate_result,
        confidence=args.confidence,
        resamples=args.resamples,
        seed=args.seed,
    )
    result["inputs"] = {
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": _sha256(args.baseline),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": _sha256(args.candidate),
        "baseline_result": args.baseline_result,
        "candidate_result": args.candidate_result,
    }
    atomic_write_json(args.output, result)
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
