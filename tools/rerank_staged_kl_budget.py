"""Allocate staged NanoQuant ranks using measured isolated KL and a fixed factor budget."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from audit_byte_budget_allocation import integer_allocate
from probe_byte_budget_allocation import Option, allocate

from nanoquant.domain.planning import factor_bit_cost
from nanoquant.infrastructure.allocation_profile_dataset import (
    load_allocation_profile_dataset,
    publish_allocation_profile_dataset,
)
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def component_options(curve: dict, ranks: np.ndarray, old_rank: int, fit: dict, anchor: dict) -> list[dict]:
    observed_ranks = curve["ranks"]
    if not observed_ranks[0] <= min(ranks) <= max(ranks) <= observed_ranks[-1]:
        raise ValueError("candidate ranks lie outside the measured curve")
    if old_rank not in ranks or anchor["measured_kl"] <= 0:
        raise ValueError("a positive measured KL and an in-range reference rank are required")
    log_errors = np.log(curve["fitted_nmse"])
    relative_error = np.exp(0.5 * (
        np.interp(ranks, observed_ranks, log_errors) - np.interp(old_rank, observed_ranks, log_errors)
    ))
    shifted_error = anchor["noise_equivalent_error"] * relative_error
    ratio = relative_error ** fit["exponent"]
    low, high = fit["measured_error_range"]
    return [
        {"kl": float(anchor["measured_kl"] * gain), "fit_anchored_kl": float(anchor["weighted_kl"] * gain),
         "noise_equivalent_error": float(error), "outside_sweep": bool(not low <= error <= high)}
        for error, gain in zip(shifted_error, ratio, strict=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="reference rank plan and factor budget")
    parser.add_argument("attribution", type=Path)
    parser.add_argument("measured", type=Path, help="isolated KL validation JSON")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output Markdown report")
    parser.add_argument("--publish-in-range-dataset", type=Path, help="publish the in-range choice as a pinned dataset")
    parser.add_argument("--publish-kl-options", type=Path, help="publish in-range options for the resident allocator")
    args = parser.parse_args()
    plan, attribution, measured = map(load_json, (args.plan, args.attribution, args.measured))
    identity = measured["identity"]
    if (not measured["complete"] or hash_file(args.attribution) != identity["attribution_sha256"]
            or measured["protocol"] != attribution["protocol"]
            or identity["measurement_sha256"] != attribution["measurement_sha256"]
            or not any(item["sha256"] == identity["gguf_sha256"] for item in attribution["files"])):
        raise ValueError("isolated measurements do not match the attribution")
    fits_path = Path(attribution["measurements"]).with_name("kl-response-fits.json")
    if hash_file(fits_path) != attribution["fits_sha256"]:
        raise ValueError("sensitivity fits do not match the attribution")
    fits = load_json(fits_path)["curves"]
    dataset = load_allocation_profile_dataset(args.dataset, plan["dataset_key"])
    bounds = plan["config"]["allocation"]["bounds"]
    multiple = bounds["multiple"]
    budget, remainder = divmod(plan["budget_bits"], 8)
    if remainder:
        raise ValueError("factor budget must be byte aligned")
    decisions = {d["unit_id"]: d for d in plan["allocation"]["decisions"]}
    anchors = measured["projections"]
    options: dict[str, list[Option]] = {}
    prices: dict[str, dict[int, dict]] = {}
    accounted_members: set[str] = set()
    for unit in dataset.payload["units"]:
        key = unit["evidence"]["unit_id"]
        record = unit["staged_measurements"]
        reference = decisions[key]
        old, base = reference["planned_rank"], reference["baseline_rank"]
        grid = record["curves"]["objective_nmse"]["ranks"]
        lower = max(grid[0], math.ceil(base * bounds["floor_fraction_of_uniform"] / multiple) * multiple)
        upper = min(grid[-1], math.floor(base * bounds["ceiling_fraction_of_uniform"] / multiple) * multiple,
                    math.floor(record["full_rank"] * bounds["overcomplete_rank_ceiling_fraction"] / multiple)
                    * multiple)
        ranks = np.arange(lower, upper + 1, multiple)
        members = record["member_curves"]
        if accounted_members.intersection(members):
            raise ValueError("the dataset repeats a projection")
        accounted_members.update(members)
        components = {
            target: component_options(curves["imatrix_noise_nmse"], ranks, old, fits[target], anchors[target])
            for target, curves in members.items()
        }
        options[key], prices[key] = [], {}
        for index, rank in enumerate(map(int, ranks)):
            bits = factor_bit_cost(*record["identity"]["shape"], rank, residual_stages=True,
                                   rank_alignment=multiple).total
            if bits % 8:
                raise ValueError("factor option must be byte aligned")
            scores = {target: values[index] for target, values in components.items()}
            kl = math.fsum(score["kl"] for score in scores.values())
            options[key].append(Option(str(rank), bits // 8, kl))
            prices[key][rank] = {"rank": rank, "bytes": bits // 8, "kl": kl, "components": scores}
    if set(options) != set(decisions) or accounted_members != set(anchors):
        raise ValueError("rank plan and KL measurements must cover the complete dataset")
    greedy = allocate(options, budget)
    solution = integer_allocate(options, budget)
    in_range_options = {
        key: [option for option in choices if not any(
            member["outside_sweep"] for member in prices[key][int(option.label)]["components"].values()
        )]
        for key, choices in options.items()
    }
    in_range_solution = integer_allocate(in_range_options, budget)
    rows = []
    for key, choice in solution["choices"].items():
        before = prices[key][decisions[key]["planned_rank"]]
        after = prices[key][int(choice)]
        rows.append({"unit": key, "floor": min(prices[key]), "cap": max(prices[key]),
                     "before": before, "after": after, "kl_saved": before["kl"] - after["kl"]})
    rows.sort(key=lambda row: -row["kl_saved"])
    if sum(row["before"]["bytes"] for row in rows) * 8 != plan["spent_bits"]:
        raise ValueError("reference factor bytes differ from the rank plan")
    before_kl = math.fsum(row["before"]["kl"] for row in rows)
    outside_count = sum(member["outside_sweep"] for row in rows for member in row["after"]["components"].values())
    at_cap = sum(row["after"]["rank"] == row["cap"] for row in rows)
    report = {
        "dataset_key": dataset.reference.artifact_id, "source_identity": identity,
        "rank_plan_sha256": hash_file(args.plan), "fits_sha256": hash_file(fits_path), "budget_bytes": budget,
        "before_kl": before_kl, "after_kl": solution["kl"], "greedy_kl": math.fsum(o.kl for o in greedy.values()),
        "solver": solution, "solver_within_sensitivity_range": in_range_solution,
        "projections_outside_sensitivity_range": outside_count, "units_at_cap": at_cap, "units": rows,
    }
    lines = [
        "# Staged rank allocation by isolated KL", "",
        "The model prices another rank increment by the isolated KL it is expected to save per factor byte.",
        "Each projection starts at its measured 082 isolated KL. The saved rank curve supplies the relative",
        "change in IMatrix-weighted squared error, which becomes relative KL through the sensitivity exponent.",
        "A shared QKV option sums the three member KL values. The discrete solver picks one rank per unit.",
        "", f"Factor budget: {budget:,} bytes. Used: {solution['projection_bytes']:,} bytes.",
        f"Measured 082 isolated sum: {before_kl:.8f}. Reranked estimate: {solution['kl']:.8f} nats/token.",
        f"Greedy estimate: {report['greedy_kl']:.8f}. Solver relative gap: {solution['relative_gap']:.3g}.",
        f"The rerank places {at_cap} units at their rank caps and {outside_count} projections outside the",
        "sensitivity sweep's error range. Rank selection stays within the measured rank range.",
        f"Restricting every projection to the sensitivity sweep's range gives {in_range_solution['kl']:.8f} "
        f"nats/token using {in_range_solution['projection_bytes']:,} factor bytes.",
        "", "| Unit | 082 rank | Reranked rank | KL saved | Byte change |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        before, after = row["before"], row["after"]
        lines.append(f"| {row['unit']} | {before['rank']} | {after['rank']} | "
                     f"{row['kl_saved']:+.8f} | {after['bytes'] - before['bytes']:+,} |")
    atomic_write_json(args.output.with_suffix(".json"), report)
    atomic_write_text(args.output, "\n".join([*lines, ""]))
    if args.publish_in_range_dataset is not None:
        pinned = publish_allocation_profile_dataset(args.publish_in_range_dataset, {
            **dataset.payload,
            "pinned_rank_plan": {
                "objective": "measured-isolated-kl-anchored-in-range",
                "source_dataset": dataset.reference.artifact_id,
                "allocation_sha256": hash_file(args.output.with_suffix(".json")),
                "budget_bits": budget * 8,
                "spent_bits": in_range_solution["projection_bytes"] * 8,
                "ranks": {key: int(rank) for key, rank in in_range_solution["choices"].items()},
            },
        }, dataset.kl_profile)
        print(f"Published pinned rank plan: {pinned.reference.artifact_id}", flush=True)
    if args.publish_kl_options is not None:
        kl_dataset = publish_allocation_profile_dataset(args.publish_kl_options, {
            **dataset.payload,
            "kl_rank_source": {
                "method": "measured-isolated-kl-anchored-in-range",
                "allocation_sha256": hash_file(args.output.with_suffix(".json")),
                "source_dataset": dataset.reference.artifact_id,
            },
            "kl_rank_options": {
                key: {
                    option.label: {member: score["kl"] for member, score in
                                   prices[key][int(option.label)]["components"].items()}
                    for option in choices
                }
                for key, choices in in_range_options.items()
            },
        }, dataset.kl_profile)
        print(f"Published KL rank options: {kl_dataset.reference.artifact_id}", flush=True)
    print("\n".join(lines[2:10]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
