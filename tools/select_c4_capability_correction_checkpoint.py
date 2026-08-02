"""Select one adaptive correction checkpoint from a paired C4 curve."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
from probe_composed_context_mlp_refit import _paired_metric_payload

from nanoquant.application.kl_budget import KlBudgetArmResult, KlSequenceResult
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file

METRICS = ("negative_log_likelihood", "full_kl")
SOURCE_KEYS = {
    "negative_log_likelihood": "negative_log_likelihood",
    "full_kl": "kl_nats_per_token",
}
RULE = "c4-capability-correction-earliest-joint-plateau-v1"


def _arm_steps(value: str) -> tuple[str, int]:
    name, separator, raw_steps = value.partition("=")
    try:
        steps = int(raw_steps)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("arm must use name=positive-steps") from exc
    if not separator or not name.strip() or steps <= 0:
        raise argparse.ArgumentTypeError("arm must use name=positive-steps")
    return name.strip(), steps


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=_arm_steps, required=True)
    parser.add_argument("--ordered-arm", type=_arm_steps, action="append", required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _sequences(payload: object) -> tuple[KlSequenceResult, ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("C4 sequence inventory is invalid")
    sequences = tuple(
        KlSequenceResult(
            float(item["negative_log_likelihood"]),
            float(item["kl_nats_per_token"]),
            int(item["token_count"]),
            (
                None
                if item.get("teacher_top1_agreement") is None
                else float(item["teacher_top1_agreement"])
            ),
        )
        for item in payload
    )
    if any(sequence.token_count <= 0 for sequence in sequences):
        raise ValueError("C4 sequence token count must be positive")
    return sequences


def _arm_result(
    name: str,
    reported: object,
    sequences: tuple[KlSequenceResult, ...],
) -> KlBudgetArmResult:
    if not isinstance(reported, dict):
        raise ValueError(f"C4 correction arm {name} result is invalid")
    tokens = sum(item.token_count for item in sequences)
    result = KlBudgetArmResult(
        arm=name,
        negative_log_likelihood=float(reported["negative_log_likelihood"]),
        kl_nats_per_token=float(reported["kl_nats_per_token"]),
        token_count=tokens,
        sequences=sequences,
    )
    if not math.isfinite(result.negative_log_likelihood):
        raise ValueError(f"C4 correction arm {name} NLL must be finite")
    return result


def _validate_arm_inventory(
    quality: dict[str, Any],
    *,
    baseline: tuple[str, int],
    ordered_arms: tuple[tuple[str, int], ...],
) -> None:
    expected = (baseline, *ordered_arms)
    expected_names = tuple(name for name, _steps in expected)
    protocol_arms = quality.get("protocol", {}).get("arms")
    observed_arms = quality.get("arms")
    if (
        not isinstance(protocol_arms, list)
        or not isinstance(observed_arms, dict)
        or tuple(item.get("name") for item in protocol_arms) != expected_names
        or set(observed_arms) != set(expected_names)
    ):
        raise ValueError("C4 correction arm inventory differs from the frozen policy")
    for (name, steps), protocol_arm in zip(expected, protocol_arms, strict=True):
        observed = observed_arms[name]
        is_baseline = name == baseline[0]
        checkpoint_identity = observed.get("checkpoint")
        observed_mode = observed.get("mode")
        protocol_mode = protocol_arm.get("mode")
        if (
            int(protocol_arm.get("expected_steps", -1)) != steps
            or int(observed.get("steps_completed", -1)) != steps
            or protocol_mode != observed_mode
            or (
                is_baseline
                and protocol_mode not in {"postkd", "tuning"}
            )
            or (not is_baseline and protocol_mode != "checkpoint")
            or (
                not is_baseline
                and (
                    not isinstance(checkpoint_identity, dict)
                    or int(checkpoint_identity.get("steps", -1)) != steps
                )
            )
        ):
            raise ValueError(f"C4 correction arm {name} has an unexpected step identity")


def select_c4_capability_checkpoint(
    quality: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    baseline: tuple[str, int],
    ordered_arms: tuple[tuple[str, int], ...],
    tolerance: float,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    baseline_name = baseline[0]
    ordered_names = tuple(name for name, _steps in ordered_arms)
    expected_names = {baseline_name, *ordered_names}
    results = quality.get("results")
    inventories = checkpoint.get("sequences")
    if (
        quality.get("schema_version") != 1
        or checkpoint.get("schema_version") != 1
        or quality.get("status") != "completed"
        or checkpoint.get("status") != "completed"
        or quality.get("protocol") != checkpoint.get("protocol")
        or not isinstance(results, dict)
        or not isinstance(inventories, dict)
        or set(results) != expected_names
        or set(inventories) != expected_names
        or baseline_name in ordered_names
        or not ordered_names
        or len(set(ordered_names)) != len(ordered_names)
        or tolerance < 0.0
        or resamples <= 0
    ):
        raise ValueError("adaptive C4 correction selection protocol is invalid")
    _validate_arm_inventory(quality, baseline=baseline, ordered_arms=ordered_arms)

    sequences = {name: _sequences(inventories[name]) for name in expected_names}
    sequence_count = len(sequences[baseline_name])
    if any(len(items) != sequence_count for items in sequences.values()):
        raise ValueError("adaptive C4 correction arms are not sequence aligned")
    arm_results = {
        name: _arm_result(name, results[name], sequences[name])
        for name in expected_names
    }
    means = {
        name: {
            metric: float(getattr(arm_results[name], SOURCE_KEYS[metric]))
            for metric in METRICS
        }
        for name in expected_names
    }
    comparisons: dict[str, object] = {}
    eligible: list[str] = []
    for arm in ordered_names:
        arm_comparisons = {
            metric: _paired_metric_payload(
                arm_results[baseline_name],
                arm_results[arm],
                SOURCE_KEYS[metric],
                resamples=resamples,
                seed=seed,
            )
            for metric in METRICS
        }
        comparisons[arm] = arm_comparisons
        if all(
            float(arm_comparisons[metric]["point_delta"]) < 0.0
            and float(arm_comparisons[metric]["upper_delta"]) < 0.0
            for metric in METRICS
        ):
            eligible.append(arm)

    minima = (
        {metric: min(means[arm][metric] for arm in eligible) for metric in METRICS}
        if eligible
        else None
    )
    plateau = (
        [
            arm
            for arm in eligible
            if all(means[arm][metric] <= minima[metric] + tolerance for metric in METRICS)
        ]
        if minima is not None
        else []
    )
    selected = plateau[0] if plateau else baseline_name
    steps_by_name = dict((baseline, *ordered_arms))
    selected_identity = quality["arms"][selected]
    return {
        "rule": RULE,
        "baseline": {"name": baseline_name, "steps": baseline[1]},
        "ordered_arms": [
            {"name": name, "steps": steps} for name, steps in ordered_arms
        ],
        "eligibility_metrics": list(METRICS),
        "tolerance": tolerance,
        "means": means,
        "comparisons": comparisons,
        "eligible_arms": eligible,
        "eligible_minima": minima,
        "plateau_arms": plateau,
        "selected_arm": selected,
        "selected_steps": steps_by_name[selected],
        "selected_identity": selected_identity,
        "correction_applied": selected != baseline_name,
        "decision": (
            f"select {selected}" if selected != baseline_name else "select baseline"
        ),
    }


def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ValueError(f"selection output already exists: {args.output}")
    quality = json.loads(args.quality_output.read_text(encoding="utf-8"))
    checkpoint = json.loads(args.checkpoint_output.read_text(encoding="utf-8"))
    result = select_c4_capability_checkpoint(
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
                "rule": RULE,
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
