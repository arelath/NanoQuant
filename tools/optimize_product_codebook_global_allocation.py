"""Allocate one global product-codebook bit budget across all model linears."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
from optimize_product_codebook_mixed_allocation import (
    BLOCK_COUNT,
    DEFAULT_MODEL_REVISION,
    DEFAULT_MODEL_SOURCE,
    ELEMENTS_PER_BLOCK,
    AllocationGroup,
    AllocationOption,
    _load_probe_options,
    _pareto_allocate,
)

from nanoquant.application.kl_budget import (
    load_kl_budget_profile,
    measured_unit_kl_anchors,
    validate_kl_budget_profile,
)
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.io_utils import atomic_write_json

PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "down")
UNIT_PATHS = {
    "q": "self_attn.attn_qkv",
    "k": "self_attn.attn_qkv",
    "v": "self_attn.attn_qkv",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}


def _probe_path(
    block: int,
    projection: str,
    *,
    attention_dir: Path,
    mlp_dir: Path,
    down_dir: Path,
) -> Path:
    if projection == "down":
        return down_dir / f"block-{block:02d}.json"
    if projection in {"gate", "up"}:
        return mlp_dir / f"block-{block:02d}-{projection}.json"
    return attention_dir / f"block-{block:02d}-{projection}.json"


def _load_global_groups(
    attention_dir: Path,
    mlp_dir: Path,
    down_dir: Path,
) -> tuple[AllocationGroup, ...]:
    groups = []
    for block in range(BLOCK_COUNT):
        for projection in PROJECTIONS:
            path = _probe_path(
                block,
                projection,
                attention_dir=attention_dir,
                mlp_dir=mlp_dir,
                down_dir=down_dir,
            )
            if not path.is_file():
                raise ValueError(f"missing global product-codebook probe receipt: {path}")
            groups.append(
                AllocationGroup(
                    key=f"block-{block:02d}:{projection}",
                    projection=projection,
                    block=block,
                    options=_load_probe_options(path),
                )
            )
    return tuple(groups)


def _constrain_grouped_qkv(
    groups: tuple[AllocationGroup, ...],
) -> tuple[AllocationGroup, ...]:
    """Keep the packed base's shared QKV entry indivisible.

    The current product-codebook overlay replaces complete packed entries.  Gemma's
    Q/K/V projections are one shared ``attn_qkv`` entry, so independently selected
    member options would retain the group and duplicate replacement payloads.
    """

    constrained = []
    for group in groups:
        options = group.options
        if group.projection in {"q", "k", "v"}:
            options = (_baseline(group),)
        constrained.append(
            AllocationGroup(group.key, group.projection, group.block, options)
        )
    return tuple(constrained)


def _baseline(group: AllocationGroup) -> AllocationOption:
    return next(option for option in group.options if option.name == "free_words")


def _global_kl_objective_calibration(
    groups: tuple[AllocationGroup, ...],
    profile_path: Path,
    *,
    model_source: str,
    model_revision: str,
    expected_profile_key: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    profile = load_kl_budget_profile(profile_path)
    validate_kl_budget_profile(
        profile,
        model_source=model_source,
        model_revision=model_revision,
        expected_profile_key=expected_profile_key,
    )
    by_unit_groups: dict[str, list[AllocationGroup]] = defaultdict(list)
    for group in groups:
        by_unit_groups[f"{group.block}:{UNIT_PATHS[group.projection]}"].append(group)
    unit_anchors = dict(
        measured_unit_kl_anchors(profile, tuple(by_unit_groups))
    )
    multipliers: dict[str, float] = {}
    baseline_contributions: dict[str, float] = {}
    for unit_id, unit_groups in by_unit_groups.items():
        total_baseline_error = sum(
            _baseline(group).weighted_error_energy for group in unit_groups
        )
        if total_baseline_error <= 0:
            raise ValueError(f"global allocation unit has no baseline error: {unit_id}")
        multiplier = unit_anchors[unit_id] / total_baseline_error
        for group in unit_groups:
            multipliers[group.key] = multiplier
            baseline_contributions[group.key] = (
                multiplier * _baseline(group).weighted_error_energy
            )
    return baseline_contributions, multipliers, {
        "mode": "measured_unit_kl_response",
        "formula": (
            "unit_kl_anchor * option_weighted_error / "
            "sum_unit_free_control_weighted_error"
        ),
        "shared_qkv_policy": (
            "one exact QKV unit anchor is conserved across q/k/v in proportion "
            "to their measured free-control error energy"
        ),
        "profile": str(profile_path.resolve()),
        "profile_key": profile.profile_key,
        "provenance": to_dict(profile.provenance),
    }


def run(args: argparse.Namespace) -> int:
    if not 0 < args.target_bpw <= 2:
        raise ValueError("target BPW must be in (0, 2]")
    groups = _constrain_grouped_qkv(
        _load_global_groups(args.attention_dir, args.mlp_dir, args.down_dir)
    )
    anchors, multipliers, calibration = _global_kl_objective_calibration(
        groups,
        args.kl_profile,
        model_source=args.model_source,
        model_revision=args.model_revision,
        expected_profile_key=args.expected_kl_profile_key,
    )
    total_elements = BLOCK_COUNT * ELEMENTS_PER_BLOCK
    budget_bits = math.floor(args.target_bpw * total_elements)
    selected_bits, selected, frontier = _pareto_allocate(
        groups,
        budget_bits,
        objective_multipliers=multipliers,
    )
    if len(selected.choices) != len(groups):
        raise AssertionError("global allocator returned an incomplete policy")

    selections = []
    option_counts: dict[str, dict[str, int]] = {}
    baseline_bits = 0
    baseline_objective = 0.0
    selected_raw_error = 0.0
    baseline_raw_error = 0.0
    for group, choice in zip(groups, selected.choices, strict=True):
        by_name = {option.name: option for option in group.options}
        option = by_name[choice]
        baseline = _baseline(group)
        objective = multipliers[group.key] * option.weighted_error_energy
        baseline_bits += baseline.bits
        baseline_objective += anchors[group.key]
        selected_raw_error += option.weighted_error_energy
        baseline_raw_error += baseline.weighted_error_energy
        selections.append(
            {
                "key": group.key,
                "block": group.block,
                "projection": group.projection,
                "unit_path": UNIT_PATHS[group.projection],
                "option": choice,
                "bits": option.bits,
                "actual_bpw": option.actual_bpw,
                "weighted_error_energy": option.weighted_error_energy,
                "weighted_error_change_fraction": (
                    option.weighted_error_energy / baseline.weighted_error_energy - 1
                ),
                "baseline_unit_kl_contribution": anchors[group.key],
                "kl_calibrated_objective": objective,
                "marginal_bits_from_cheapest": option.bits
                - min(item.bits for item in group.options),
            }
        )
        counts = option_counts.setdefault(group.projection, {})
        counts[choice] = counts.get(choice, 0) + 1

    result = {
        "schema_version": 2,
        "status": "completed",
        "objective": "minimum global measured-unit-KL-calibrated response error",
        "target_bpw": args.target_bpw,
        "total_elements": total_elements,
        "budget_bits": budget_bits,
        "total_bits": selected_bits,
        "effective_bpw": selected_bits / total_elements,
        "slack_bits": budget_bits - selected_bits,
        "slack_bpw": (budget_bits - selected_bits) / total_elements,
        "group_count": len(groups),
        "packed_representation": {
            "base": "Experiment 056 grouped attn_qkv packed entries",
            "qkv_policy": "fixed_grouped_base",
            "replacement_granularity": "complete packed entry",
            "independent_qkv_product_options": False,
        },
        "objective_calibration": calibration,
        "global_response": {
            "selected_kl_calibrated_objective": selected.objective_value,
            "baseline_kl_calibrated_objective": baseline_objective,
            "objective_change_fraction": selected.objective_value / baseline_objective - 1,
            "selected_weighted_error_energy": selected_raw_error,
            "baseline_weighted_error_energy": baseline_raw_error,
            "baseline_bits": baseline_bits,
            "baseline_bpw": baseline_bits / total_elements,
        },
        "option_counts": option_counts,
        "selections": selections,
        "pareto_frontier": [
            {"total_bits": bits, "kl_calibrated_objective": objective}
            for bits, objective in frontier
        ],
        "inputs": {
            "attention_dir": str(args.attention_dir.resolve()),
            "mlp_dir": str(args.mlp_dir.resolve()),
            "down_dir": str(args.down_dir.resolve()),
            "kl_profile": str(args.kl_profile.resolve()),
        },
        "limitations": [
            "the additive response objective requires packed whole-model validation",
            "Q/K/V share one exact functional anchor and remain fixed because the packed base groups them",
            "all selected product-code options use retained fixed outlier identities",
        ],
    }
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "total_bits": selected_bits,
                "effective_bpw": result["effective_bpw"],
                "slack_bits": result["slack_bits"],
                "objective_change_fraction": result["global_response"][
                    "objective_change_fraction"
                ],
                "option_counts": option_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--mlp-dir", type=Path, required=True)
    parser.add_argument("--down-dir", type=Path, required=True)
    parser.add_argument("--kl-profile", type=Path, required=True)
    parser.add_argument("--expected-kl-profile-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-bpw", type=float, default=1.0)
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
