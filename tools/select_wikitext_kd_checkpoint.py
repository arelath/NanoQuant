"""Apply an uncertainty-aware checkpoint rule to a WikiText KD curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
from probe_wikitext_kd_quality import SequenceMetrics, _paired_interval

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file

METRICS = ("negative_log_likelihood", "full_kl", "topk_plus_tail_kl")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--ordered-arm", action="append", required=True)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _sequences(payload: object) -> tuple[SequenceMetrics, ...]:
    if not isinstance(payload, list):
        raise ValueError("checkpoint sequence inventory is invalid")
    return tuple(SequenceMetrics(**item) for item in payload)


def select_checkpoint(
    quality: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    baseline: str,
    incumbent: str,
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
        or incumbent not in ordered_arms
        or len(set(ordered_arms)) != len(ordered_arms)
        or tolerance < 0.0
        or resamples <= 0
    ):
        raise ValueError("checkpoint selection protocol is invalid")
    names = {baseline, *ordered_arms}
    if not names.issubset(results) or not names.issubset(inventories):
        raise ValueError("checkpoint selection arm inventory is incomplete")

    sequences = {name: _sequences(inventories[name]) for name in names}
    means = {
        name: {
            metric: float(results[name]["means"][metric])
            for metric in METRICS
        }
        for name in names
    }
    baseline_pairs: dict[str, object] = {}
    eligible: list[str] = []
    for arm in ordered_arms:
        comparisons = {
            metric: _paired_interval(
                sequences[baseline],
                sequences[arm],
                metric,
                resamples=resamples,
                seed=seed,
            )
            for metric in METRICS
        }
        baseline_pairs[arm] = comparisons
        if all(
            float(comparisons[metric]["upper_delta"]) < 0.0
            for metric in ("negative_log_likelihood", "full_kl")
        ):
            eligible.append(arm)

    minima = {
        metric: min(means[arm][metric] for arm in ordered_arms)
        for metric in METRICS
    }
    plateau = [
        arm
        for arm in eligible
        if all(means[arm][metric] <= minima[metric] + tolerance for metric in METRICS)
    ]
    selected = plateau[0] if plateau else None
    replacement_comparison = None
    replace_incumbent = False
    if selected is not None:
        replacement_comparison = {
            metric: _paired_interval(
                sequences[incumbent],
                sequences[selected],
                metric,
                resamples=resamples,
                seed=seed,
            )
            for metric in METRICS
        }
        replace_incumbent = selected != incumbent and (
            float(replacement_comparison["negative_log_likelihood"]["upper_delta"])
            < 0.0
            and float(replacement_comparison["full_kl"]["upper_delta"]) <= tolerance
            and float(replacement_comparison["topk_plus_tail_kl"]["upper_delta"])
            <= tolerance
        )
    return {
        "baseline": baseline,
        "incumbent": incumbent,
        "ordered_arms": list(ordered_arms),
        "tolerance": tolerance,
        "means": means,
        "minima": minima,
        "eligible_arms": eligible,
        "plateau_arms": plateau,
        "selected_arm": selected,
        "baseline_comparisons": baseline_pairs,
        "replacement_comparison": replacement_comparison,
        "replace_incumbent": replace_incumbent,
        "decision": (
            f"replace {incumbent} with {selected}"
            if replace_incumbent
            else f"retain {incumbent}"
        ),
    }


def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ValueError(f"selection output already exists: {args.output}")
    quality = json.loads(args.quality_output.read_text(encoding="utf-8"))
    checkpoint = json.loads(args.checkpoint_output.read_text(encoding="utf-8"))
    result = select_checkpoint(
        quality,
        checkpoint,
        baseline=args.baseline,
        incumbent=args.incumbent,
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
                "rule": "earliest-three-metric-plateau-with-paired-replacement-v1",
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
