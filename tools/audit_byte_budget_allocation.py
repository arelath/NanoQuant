"""Audit the greedy byte-budget pilot with a discrete solver and measured error curves.

Embeddings stay fixed. No hypothetical third stage is offered. Retain all discrete
options (not just the lower convex hull), and report the solver's certified gap.
Install the optional allocation dependencies to use this CPU-only tool.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from fit_kl_sensitivity import fit_dataset
from measure_gguf_kl import PROJECTIONS
from probe_byte_budget_allocation import Option, allocate, estimated_kl, unit_options

from nanoquant.infrastructure.discrete_allocation import DiscreteOption, solve_discrete_allocation
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file


def integer_allocate(units: dict[str, list[Option]], budget: int, *, seconds: float = 60.0) -> dict:
    if any(option.bytes != int(option.bytes) for options in units.values() for option in options):
        raise ValueError("discrete options require integer byte costs")
    result = solve_discrete_allocation(
        {key: [DiscreteOption(option.label, int(option.bytes), option.kl) for option in options]
         for key, options in units.items()},
        budget, seconds=seconds,
    )
    return {
        "choices": {name: option.label for name, option in result.choices.items()},
        "projection_bytes": result.cost, "kl": result.error, "lower_bound": result.lower_bound,
        "relative_gap": result.relative_gap, "status": result.status, "message": result.message,
    }


def refreshed_menu(menu: dict, sweep: Path) -> dict:
    menu = copy.deepcopy(menu)
    curves = fit_dataset(sweep)["curves"]
    for entry in menu["units"].values():
        for index, name in enumerate(entry["members"]):
            _, block, kind, _ = name.split(".")
            curve = curves[f"{block}:{PROJECTIONS[kind]}"]
            options = [stage["members"][index] for stage in entry["staged"]]
            options.extend(entry["standard"][name].values())
            for option in options:
                error = option["noise_equivalent_error"]
                option["kl"] = estimated_kl(curve, error)
                low, high = curve["measured_error_range"]
                option["in_measured_range"] = low <= error <= high
    return menu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("menu", type=Path)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--budget", type=int, action="append", required=True)
    parser.add_argument("--minimum-stages", type=int, default=1, choices=(1, 2))
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    menu = refreshed_menu(json.loads(args.menu.read_text(encoding="utf-8")), args.sweep)
    units = {
        name: [option for option in unit_options(entry, third_stage=False)
               if args.minimum_stages == 1 or option.label != "1 stage"]
        for name, entry in menu["units"].items()
    }
    solutions = []
    for budget in args.budget:
        remaining = budget - menu["fixed_bytes"]
        greedy = allocate(units, remaining)
        exact = integer_allocate(units, remaining, seconds=args.seconds)
        solutions.append({
            "budget": budget, "bytes": exact["projection_bytes"] + menu["fixed_bytes"], **exact,
            "greedy_kl": math.fsum(option.kl for option in greedy.values()),
            "greedy_bytes": menu["fixed_bytes"] + sum(int(option.bytes) for option in greedy.values()),
        })
        print(f"{budget}: greedy={solutions[-1]['greedy_kl']:.7f}, "
              f"integer={exact['kl']:.7f}, gap={exact['relative_gap']:.3g}", flush=True)
    report = {
        "menu": str(args.menu.resolve()), "menu_sha256": hash_file(args.menu),
        "sweep": str(args.sweep.resolve()), "sweep_sha256": hash_file(args.sweep),
        "fixed_bytes": menu["fixed_bytes"], "minimum_stages": args.minimum_stages,
        "third_stage": "excluded: no measured 1B source", "solutions": solutions,
    }
    atomic_write_json(args.output.with_suffix(".json"), report)
    lines = ["# Integer allocation audit", "", "Embeddings fixed; updated sensitivity curves; no assumed third stage.",
             "All scores are additive predictions, not measured model KL.", "",
             "| Budget | Used bytes | Greedy KL | Integer KL | Lower bound | Relative solver gap |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in solutions:
        lines.append(f"| {row['budget']} | {row['bytes']} | {row['greedy_kl']:.7f} | {row['kl']:.7f} | "
                     f"{row['lower_bound']:.7f} | {row['relative_gap']:.3g} |")
    atomic_write_text(args.output.with_suffix(".md"), "\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
