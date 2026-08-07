"""Choose an exact-bit mixed MLP product-code allocation at a BPW ceiling."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _paths  # noqa: F401

from nanoquant.domain.planning import factor_bit_cost, outlier_bit_cost
from nanoquant.infrastructure.io_utils import atomic_write_json

BLOCK_COUNT = 26
MLP_ELEMENTS = 1152 * 6912
QO_ELEMENTS = 1024 * 1152
KV_ELEMENTS = 256 * 1152
ELEMENTS_PER_BLOCK = 3 * MLP_ELEMENTS + 2 * QO_ELEMENTS + 2 * KV_ELEMENTS


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
    weighted_error_energy: float
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
) -> tuple[int, FrontierState, tuple[tuple[int, float], ...]]:
    frontier: dict[int, FrontierState] = {
        0: FrontierState(weighted_error_energy=0.0, choices=())
    }
    for group in groups:
        expanded: dict[int, FrontierState] = {}
        for current_bits, state in frontier.items():
            for option in group.options:
                bits = current_bits + option.bits
                if bits > budget_bits:
                    continue
                error = state.weighted_error_energy + option.weighted_error_energy
                existing = expanded.get(bits)
                if existing is None or error < existing.weighted_error_energy:
                    expanded[bits] = FrontierState(
                        weighted_error_energy=error,
                        choices=state.choices + (option.name,),
                    )
        if not expanded:
            raise ValueError(
                f"bit budget cannot represent every group; failed at {group.key}"
            )
        frontier = {}
        best_error = math.inf
        for bits, state in sorted(expanded.items()):
            if state.weighted_error_energy < best_error:
                frontier[bits] = state
                best_error = state.weighted_error_energy
    selected_bits, selected = min(
        frontier.items(),
        key=lambda item: (item[1].weighted_error_energy, -item[0]),
    )
    compact_frontier = tuple(
        (bits, state.weighted_error_energy)
        for bits, state in sorted(frontier.items())
    )
    return selected_bits, selected, compact_frontier


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


def run(args: argparse.Namespace) -> int:
    if not 0 < args.target_bpw <= 2:
        raise ValueError("target BPW must be in (0, 2]")
    if (
        args.maximum_matrix_error_regression_fraction is not None
        and args.maximum_matrix_error_regression_fraction < 0
    ):
        raise ValueError("maximum matrix error regression must not be negative")
    groups = _limit_option_regression(
        _load_groups(args.down_dir, args.mlp_dir),
        args.maximum_matrix_error_regression_fraction,
    )
    total_elements = BLOCK_COUNT * ELEMENTS_PER_BLOCK
    budget_bits = math.floor(args.target_bpw * total_elements)
    attention_by_projection = _attention_control_bits()
    fixed_attention_bits = BLOCK_COUNT * sum(attention_by_projection.values())
    variable_budget_bits = budget_bits - fixed_attention_bits
    if variable_budget_bits <= 0:
        raise ValueError("attention controls exhaust the requested BPW budget")

    selected_bits, selected, frontier = _pareto_allocate(
        groups,
        variable_budget_bits,
    )
    if len(selected.choices) != len(groups):
        raise AssertionError("allocator returned an incomplete policy")

    selections: list[dict[str, Any]] = []
    option_counts: dict[str, dict[str, int]] = {}
    selected_target_energy = 0.0
    baseline_bits = 0
    baseline_error = 0.0
    for group, choice in zip(groups, selected.choices, strict=True):
        by_name = {option.name: option for option in group.options}
        option = by_name[choice]
        baseline = next(
            item for item in group.options if item.name == "free_words"
        )
        selected_target_energy += option.weighted_target_energy
        baseline_bits += baseline.bits
        baseline_error += baseline.weighted_error_energy
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
            }
        )
        projection_counts = option_counts.setdefault(group.projection, {})
        projection_counts[choice] = projection_counts.get(choice, 0) + 1

    total_bits = fixed_attention_bits + selected_bits
    result = {
        "schema_version": 2,
        "status": "completed",
        "objective": "minimum summed calibration-weighted MLP error energy",
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
            "selected_weighted_error_energy": selected.weighted_error_energy,
            "baseline_weighted_error_energy": baseline_error,
            "weighted_error_change_fraction": (
                selected.weighted_error_energy / baseline_error - 1
            ),
            "selected_weighted_normalized_rmse": math.sqrt(
                selected.weighted_error_energy / selected_target_energy
            ),
            "baseline_weighted_normalized_rmse": math.sqrt(
                baseline_error / selected_target_energy
            ),
        },
        "option_counts": option_counts,
        "selections": selections,
        "pareto_frontier": [
            {"variable_bits": bits, "weighted_error_energy": error}
            for bits, error in frontier
        ],
        "inputs": {
            "down_dir": str(args.down_dir.resolve()),
            "mlp_dir": str(args.mlp_dir.resolve()),
        },
        "limitations": [
            "attention projections are fixed to matched free-factor controls",
            "the objective is an additive calibration-weighted matrix proxy",
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
        "--maximum-matrix-error-regression-fraction",
        type=float,
        help=(
            "discard options whose weighted error energy exceeds their "
            "matrix's free control by more than this fraction"
        ),
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
