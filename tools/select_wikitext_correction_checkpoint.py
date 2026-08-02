"""Apply Experiment 046's three-metric correction-checkpoint rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
from probe_wikitext_kd_quality import _paired_interval
from select_wikitext_kd_checkpoint import METRICS, _sequences

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--ordered-arm", action="append", required=True)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def select_correction_checkpoint(
    quality: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    baseline: str,
    ordered_arms: tuple[str, ...],
    tolerance: float,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    results = quality.get("results")
    inventories = checkpoint.get("sequences")
    if (
        quality.get("status") != "completed"
        or checkpoint.get("status") != "completed"
        or quality.get("protocol") != checkpoint.get("protocol")
        or not isinstance(results, dict)
        or not isinstance(inventories, dict)
        or baseline in ordered_arms
        or not ordered_arms
        or len(set(ordered_arms)) != len(ordered_arms)
        or tolerance < 0.0
        or resamples <= 0
    ):
        raise ValueError("correction checkpoint selection protocol is invalid")
    names = {baseline, *ordered_arms}
    if not names.issubset(results) or not names.issubset(inventories):
        raise ValueError("correction checkpoint arm inventory is incomplete")

    sequences = {name: _sequences(inventories[name]) for name in names}
    means = {
        name: {metric: float(results[name]["means"][metric]) for metric in METRICS}
        for name in names
    }
    comparisons: dict[str, object] = {}
    eligible: list[str] = []
    for arm in ordered_arms:
        arm_comparisons = {
            metric: _paired_interval(
                sequences[baseline],
                sequences[arm],
                metric,
                resamples=resamples,
                seed=seed,
            )
            for metric in METRICS
        }
        comparisons[arm] = arm_comparisons
        if all(float(arm_comparisons[metric]["upper_delta"]) < 0.0 for metric in METRICS):
            eligible.append(arm)

    minima = (
        {
            metric: min(means[arm][metric] for arm in eligible)
            for metric in METRICS
        }
        if eligible
        else None
    )
    plateau = (
        [
            arm
            for arm in eligible
            if all(
                means[arm][metric] <= minima[metric] + tolerance
                for metric in METRICS
            )
        ]
        if minima is not None
        else []
    )
    selected = plateau[0] if plateau else None
    return {
        "baseline": baseline,
        "ordered_arms": list(ordered_arms),
        "eligibility_metrics": list(METRICS),
        "tolerance": tolerance,
        "means": means,
        "eligible_arms": eligible,
        "eligible_minima": minima,
        "plateau_arms": plateau,
        "selected_arm": selected,
        "baseline_comparisons": comparisons,
        "decision": "no survivor" if selected is None else f"select {selected}",
    }


def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ValueError(f"selection output already exists: {args.output}")
    quality = json.loads(args.quality_output.read_text(encoding="utf-8"))
    checkpoint = json.loads(args.checkpoint_output.read_text(encoding="utf-8"))
    result = select_correction_checkpoint(
        quality,
        checkpoint,
        baseline=args.baseline,
        ordered_arms=tuple(args.ordered_arm),
        tolerance=args.tolerance,
        resamples=args.resamples,
        seed=args.seed,
    )
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "protocol": {
                "rule": "earliest-three-metric-correction-plateau-v1",
                "quality_output": str(args.quality_output.resolve()),
                "quality_sha256": hash_file(args.quality_output),
                "checkpoint_output": str(args.checkpoint_output.resolve()),
                "checkpoint_sha256": hash_file(args.checkpoint_output),
                "resamples": args.resamples,
                "seed": args.seed,
            },
            **result,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
