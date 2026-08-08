"""Choose an exact-bit mixed MLP product-code allocation at a BPW ceiling."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _paths  # noqa: F401

from nanoquant.application.kl_budget import (
    load_kl_budget_profile,
    measured_unit_kl_anchors,
    validate_kl_budget_profile,
)
from nanoquant.config.codec import to_dict
from nanoquant.domain.planning import factor_bit_cost, outlier_bit_cost
from nanoquant.infrastructure.io_utils import atomic_write_json

BLOCK_COUNT = 26
MLP_ELEMENTS = 1152 * 6912
QO_ELEMENTS = 1024 * 1152
KV_ELEMENTS = 256 * 1152
ELEMENTS_PER_BLOCK = 3 * MLP_ELEMENTS + 2 * QO_ELEMENTS + 2 * KV_ELEMENTS
DEFAULT_MODEL_SOURCE = "google/gemma-3-1b-it"
DEFAULT_MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
PROFILE_UNIT_PATHS = {
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}


@dataclass(frozen=True, slots=True)
class AllocationOption:
    name: str
    bits: int
    weighted_error_energy: float
    weighted_target_energy: float
    actual_bpw: float


@dataclass(frozen=True, slots=True)
class AllocationGroup:
    key: str
    projection: str
    block: int
    options: tuple[AllocationOption, ...]


@dataclass(frozen=True, slots=True)
class FrontierState:
    objective_value: float
    choices: tuple[str, ...]


def _load_probe_options(path: Path) -> tuple[AllocationOption, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, dict) or "free_words" not in results:
        raise ValueError(f"probe receipt is incomplete: {path}")
    options: list[AllocationOption] = []
    for name, result in results.items():
        metrics = result["metrics"]
        option = AllocationOption(
            name=name,
            bits=int(result["total_bits"]),
            weighted_error_energy=float(metrics["weighted_error_energy"]),
            weighted_target_energy=float(metrics["weighted_target_energy"]),
            actual_bpw=float(result["actual_bpw"]),
        )
        if not math.isfinite(option.weighted_error_energy) or not math.isfinite(
            option.weighted_target_energy
        ):
            raise ValueError(f"probe receipt has non-finite metrics: {path}:{name}")
        options.append(option)
    target = options[0].weighted_target_energy
    if any(
        not math.isclose(item.weighted_target_energy, target, rel_tol=1e-6)
        for item in options[1:]
    ):
        raise ValueError(f"candidate target energies do not match: {path}")

    best_by_bits: dict[int, AllocationOption] = {}
    for option in options:
        existing = best_by_bits.get(option.bits)
        if existing is None or option.weighted_error_energy < existing.weighted_error_energy:
            best_by_bits[option.bits] = option
    frontier: list[AllocationOption] = []
    best_error = math.inf
    for option in sorted(best_by_bits.values(), key=lambda item: item.bits):
        if option.weighted_error_energy < best_error:
            frontier.append(option)
            best_error = option.weighted_error_energy
    baseline = next(item for item in options if item.name == "free_words")
    if all(item.name != "free_words" for item in frontier):
        frontier.append(baseline)
        frontier.sort(key=lambda item: item.bits)
    if not frontier:
        raise ValueError(f"probe receipt has no usable options: {path}")
    return tuple(frontier)


def _pareto_allocate(
    groups: tuple[AllocationGroup, ...],
    budget_bits: int,
    *,
    objective_multipliers: dict[str, float] | None = None,
) -> tuple[int, FrontierState, tuple[tuple[int, float], ...]]:
    multipliers = (
        {group.key: 1.0 for group in groups}
        if objective_multipliers is None
        else objective_multipliers
    )
    if set(multipliers) != {group.key for group in groups} or any(
        not math.isfinite(value) or value <= 0 for value in multipliers.values()
    ):
        raise ValueError("allocation objective multipliers must positively cover every group")
    frontier: dict[int, FrontierState] = {
        0: FrontierState(objective_value=0.0, choices=())
    }
    for group in groups:
        expanded: dict[int, FrontierState] = {}
        for current_bits, state in frontier.items():
            for option in group.options:
                bits = current_bits + option.bits
                if bits > budget_bits:
                    continue
                objective = (
                    state.objective_value
                    + multipliers[group.key] * option.weighted_error_energy
                )
                existing = expanded.get(bits)
                if existing is None or objective < existing.objective_value:
                    expanded[bits] = FrontierState(
                        objective_value=objective,
                        choices=state.choices + (option.name,),
                    )
        if not expanded:
            raise ValueError(
                f"bit budget cannot represent every group; failed at {group.key}"
            )
        frontier = {}
        best_objective = math.inf
        for bits, state in sorted(expanded.items()):
            if state.objective_value < best_objective:
                frontier[bits] = state
                best_objective = state.objective_value
    selected_bits, selected = min(
        frontier.items(),
        key=lambda item: (item[1].objective_value, -item[0]),
    )
    compact_frontier = tuple(
        (bits, state.objective_value)
        for bits, state in sorted(frontier.items())
    )
    return selected_bits, selected, compact_frontier


def _kl_objective_calibration(
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
    unit_ids = tuple(
        f"{group.block}:{PROFILE_UNIT_PATHS[group.projection]}"
        for group in groups
    )
    by_unit = dict(measured_unit_kl_anchors(profile, unit_ids))
    anchors = {
        group.key: by_unit[unit_id]
        for group, unit_id in zip(groups, unit_ids, strict=True)
    }
    multipliers: dict[str, float] = {}
    for group in groups:
        baseline = next(
            option for option in group.options if option.name == "free_words"
        )
        multipliers[group.key] = (
            anchors[group.key] / baseline.weighted_error_energy
        )
    return anchors, multipliers, {
        "mode": "measured_unit_kl",
        "formula": "unit_kl_anchor * option_weighted_error / free_control_weighted_error",
        "profile": str(profile_path.resolve()),
        "profile_key": profile.profile_key,
        "provenance": to_dict(profile.provenance),
    }


def _attention_control_bits() -> dict[str, int]:
    q_outliers = outlier_bit_cost(1024, 2, value_bits=16, index_bits=11).total
    kv_outliers = outlier_bit_cost(256, 2, value_bits=16, index_bits=11).total
    o_outliers = outlier_bit_cost(1152, 2, value_bits=16, index_bits=10).total
    return {
        "q": factor_bit_cost(1024, 1152, 522, scale_bits=16).total + q_outliers,
        "k": factor_bit_cost(256, 1152, 191, scale_bits=16).total + kv_outliers,
        "v": factor_bit_cost(256, 1152, 191, scale_bits=16).total + kv_outliers,
        "o": factor_bit_cost(1024, 1152, 522, scale_bits=16).total + o_outliers,
    }


def _load_groups(down_dir: Path, mlp_dir: Path) -> tuple[AllocationGroup, ...]:
    groups: list[AllocationGroup] = []
    for block in range(BLOCK_COUNT):
        paths = {
            "gate": mlp_dir / f"block-{block:02d}-gate.json",
            "up": mlp_dir / f"block-{block:02d}-up.json",
            "down": down_dir / f"block-{block:02d}.json",
        }
        for projection, path in paths.items():
            if not path.is_file():
                raise ValueError(f"missing all-layer probe receipt: {path}")
            groups.append(
                AllocationGroup(
                    key=f"block-{block:02d}:{projection}",
                    projection=projection,
                    block=block,
                    options=_load_probe_options(path),
                )
            )
    return tuple(groups)


def _limit_option_regression(
    groups: tuple[AllocationGroup, ...],
    maximum_regression_fraction: float | None,
) -> tuple[AllocationGroup, ...]:
    if maximum_regression_fraction is None:
        return groups
    limited: list[AllocationGroup] = []
    for group in groups:
        baseline = next(
            option for option in group.options if option.name == "free_words"
        )
        maximum_error = baseline.weighted_error_energy * (
            1 + maximum_regression_fraction
        )
        options = tuple(
            option
            for option in group.options
            if option.weighted_error_energy <= maximum_error
        )
        if not options:
            raise AssertionError(f"free control was removed from {group.key}")
        limited.append(
            AllocationGroup(
                key=group.key,
                projection=group.projection,
                block=group.block,
                options=options,
            )
        )
    return tuple(limited)


def _parse_group_free_row_floors(value: str) -> dict[str, int]:
    floors: dict[str, int] = {}
    for item in value.split(","):
        parts = item.strip().split("=", maxsplit=1)
        if len(parts) != 2 or not parts[0]:
            raise argparse.ArgumentTypeError(
                "group free-row floors must use group=count entries"
            )
        try:
            free_rows = int(parts[1])
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "group free-row floors must be integers"
            ) from error
        if free_rows < 0 or parts[0] in floors:
            raise argparse.ArgumentTypeError(
                "group free-row floors must be nonnegative and unique"
            )
        floors[parts[0]] = free_rows
    if not floors:
        raise argparse.ArgumentTypeError("group free-row floors must not be empty")
    return floors


def _option_free_rows(option: AllocationOption) -> int | None:
    marker = "_free"
    if marker not in option.name:
        return None
    suffix = option.name.split(marker, maxsplit=1)[1]
    digits = suffix.split("_", maxsplit=1)[0]
    return int(digits)


def _limit_group_free_rows(
    groups: tuple[AllocationGroup, ...],
    minimum_by_group: dict[str, int] | None,
) -> tuple[AllocationGroup, ...]:
    if minimum_by_group is None:
        return groups
    known = {group.key for group in groups}
    if not set(minimum_by_group) <= known:
        unknown = sorted(set(minimum_by_group) - known)
        raise ValueError(f"unknown allocation groups in free-row floors: {unknown}")
    limited: list[AllocationGroup] = []
    for group in groups:
        minimum = minimum_by_group.get(group.key)
        options = group.options
        if minimum is not None:
            allowed_options = []
            for option in options:
                free_rows = _option_free_rows(option)
                if option.name == "free_words" or (
                    free_rows is not None and free_rows >= minimum
                ):
                    allowed_options.append(option)
            options = tuple(allowed_options)
        if not options:
            raise ValueError(f"free-row floor removes every option for {group.key}")
        limited.append(
            AllocationGroup(group.key, group.projection, group.block, options)
        )
    return tuple(limited)


def run(args: argparse.Namespace) -> int:
    if not 0 < args.target_bpw <= 2:
        raise ValueError("target BPW must be in (0, 2]")
    if (
        args.maximum_matrix_error_regression_fraction is not None
        and args.maximum_matrix_error_regression_fraction < 0
    ):
        raise ValueError("maximum matrix error regression must not be negative")
    if (args.kl_profile is None) != (args.expected_kl_profile_key is None):
        raise ValueError(
            "KL calibration requires both a profile and its expected profile key"
        )
    groups = _limit_group_free_rows(
        _limit_option_regression(
            _load_groups(args.down_dir, args.mlp_dir),
            args.maximum_matrix_error_regression_fraction,
        ),
        args.minimum_free_rows_by_group,
    )
    total_elements = BLOCK_COUNT * ELEMENTS_PER_BLOCK
    budget_bits = math.floor(args.target_bpw * total_elements)
    attention_by_projection = _attention_control_bits()
    fixed_attention_bits = BLOCK_COUNT * sum(attention_by_projection.values())
    variable_budget_bits = budget_bits - fixed_attention_bits
    if variable_budget_bits <= 0:
        raise ValueError("attention controls exhaust the requested BPW budget")

    kl_anchors: dict[str, float] | None = None
    objective_multipliers: dict[str, float] | None = None
    objective_calibration: dict[str, Any] | None = None
    if args.kl_profile is not None:
        assert args.expected_kl_profile_key is not None
        kl_anchors, objective_multipliers, objective_calibration = (
            _kl_objective_calibration(
                groups,
                args.kl_profile,
                model_source=args.model_source,
                model_revision=args.model_revision,
                expected_profile_key=args.expected_kl_profile_key,
            )
        )
    selected_bits, selected, frontier = _pareto_allocate(
        groups,
        variable_budget_bits,
        objective_multipliers=objective_multipliers,
    )
    if len(selected.choices) != len(groups):
        raise AssertionError("allocator returned an incomplete policy")

    selections: list[dict[str, Any]] = []
    option_counts: dict[str, dict[str, int]] = {}
    selected_target_energy = 0.0
    selected_error = 0.0
    baseline_bits = 0
    baseline_error = 0.0
    selected_kl_proxy = 0.0
    baseline_kl_proxy = 0.0
    for group, choice in zip(groups, selected.choices, strict=True):
        by_name = {option.name: option for option in group.options}
        option = by_name[choice]
        baseline = next(
            item for item in group.options if item.name == "free_words"
        )
        selected_target_energy += option.weighted_target_energy
        selected_error += option.weighted_error_energy
        baseline_bits += baseline.bits
        baseline_error += baseline.weighted_error_energy
        anchor = None if kl_anchors is None else kl_anchors[group.key]
        option_kl_proxy = (
            None
            if anchor is None
            else anchor
            * option.weighted_error_energy
            / baseline.weighted_error_energy
        )
        if anchor is not None and option_kl_proxy is not None:
            selected_kl_proxy += option_kl_proxy
            baseline_kl_proxy += anchor
        selections.append(
            {
                "key": group.key,
                "block": group.block,
                "projection": group.projection,
                "option": choice,
                "bits": option.bits,
                "actual_bpw": option.actual_bpw,
                "weighted_error_energy": option.weighted_error_energy,
                "weighted_error_change_fraction": (
                    option.weighted_error_energy / baseline.weighted_error_energy - 1
                ),
                "measured_unit_kl_anchor": anchor,
                "kl_calibrated_objective": option_kl_proxy,
            }
        )
        projection_counts = option_counts.setdefault(group.projection, {})
        projection_counts[choice] = projection_counts.get(choice, 0) + 1

    total_bits = fixed_attention_bits + selected_bits
    kl_calibrated = objective_calibration is not None
    result = {
        "schema_version": 4 if kl_calibrated else 3,
        "status": "completed",
        "objective": (
            "minimum summed measured-unit-KL-calibrated relative MLP error"
            if kl_calibrated
            else "minimum summed calibration-weighted MLP error energy"
        ),
        "objective_calibration": objective_calibration,
        "target_bpw": args.target_bpw,
        "total_elements": total_elements,
        "budget_bits": budget_bits,
        "total_bits": total_bits,
        "effective_bpw": total_bits / total_elements,
        "slack_bits": budget_bits - total_bits,
        "slack_bpw": (budget_bits - total_bits) / total_elements,
        "constraints": {
            "maximum_matrix_weighted_error_regression_fraction": (
                args.maximum_matrix_error_regression_fraction
            ),
            "minimum_free_rows_by_group": args.minimum_free_rows_by_group,
        },
        "fixed_attention": {
            "policy": "matched free-factor controls",
            "bits_per_block_by_projection": attention_by_projection,
            "total_bits": fixed_attention_bits,
            "effective_bpw_over_attention_weights": fixed_attention_bits
            / (BLOCK_COUNT * (2 * QO_ELEMENTS + 2 * KV_ELEMENTS)),
        },
        "variable_mlp": {
            "selected_bits": selected_bits,
            "baseline_bits": baseline_bits,
            "selected_weighted_error_energy": selected_error,
            "baseline_weighted_error_energy": baseline_error,
            "weighted_error_change_fraction": (
                selected_error / baseline_error - 1
            ),
            "selected_weighted_normalized_rmse": math.sqrt(
                selected_error / selected_target_energy
            ),
            "baseline_weighted_normalized_rmse": math.sqrt(
                baseline_error / selected_target_energy
            ),
            "selected_kl_calibrated_objective": (
                selected_kl_proxy if kl_calibrated else None
            ),
            "baseline_kl_calibrated_objective": (
                baseline_kl_proxy if kl_calibrated else None
            ),
            "kl_calibrated_objective_change_fraction": (
                selected_kl_proxy / baseline_kl_proxy - 1
                if kl_calibrated
                else None
            ),
        },
        "option_counts": option_counts,
        "selections": selections,
        "pareto_frontier": [
            {
                "variable_bits": bits,
                (
                    "kl_calibrated_objective"
                    if kl_calibrated
                    else "weighted_error_energy"
                ): objective,
            }
            for bits, objective in frontier
        ],
        "inputs": {
            "down_dir": str(args.down_dir.resolve()),
            "mlp_dir": str(args.mlp_dir.resolve()),
            "kl_profile": (
                None if args.kl_profile is None else str(args.kl_profile.resolve())
            ),
        },
        "limitations": [
            "attention projections are fixed to matched free-factor controls",
            (
                "KL anchors transfer from the retained rank-allocation operating point; "
                "option response is approximated by its weighted-error ratio"
                if kl_calibrated
                else "the objective is an additive calibration-weighted matrix proxy"
            ),
            "functional splice and resident-tuning validation remain required",
        ],
    }
    atomic_write_json(args.output, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--down-dir", type=Path, required=True)
    parser.add_argument("--mlp-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-bpw", type=float, default=1.0)
    parser.add_argument(
        "--kl-profile",
        type=Path,
        help="completed exact-unit KL profile used to calibrate option error",
    )
    parser.add_argument(
        "--expected-kl-profile-key",
        help="required semantic profile key when --kl-profile is supplied",
    )
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--maximum-matrix-error-regression-fraction",
        type=float,
        help=(
            "discard options whose weighted error energy exceeds their "
            "matrix's free control by more than this fraction"
        ),
    )
    parser.add_argument(
        "--minimum-free-rows-by-group",
        type=_parse_group_free_row_floors,
        help=(
            "comma-separated group=count floors, for example "
            "block-12:gate=672,block-12:up=672"
        ),
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
