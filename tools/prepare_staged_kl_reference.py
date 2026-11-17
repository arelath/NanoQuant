"""Build a reranker reference from a validated two-stage control and a rank-response dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from nanoquant.domain.planning import factor_bit_cost
from nanoquant.infrastructure.allocation_profile_dataset import load_allocation_profile_dataset
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.runtime_export import load_frozen_run_planning_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_allocation_profile_dataset(args.dataset)
    units = dataset.payload["units"]
    source = Path(dataset.payload["source_run"])
    block_count = len({unit["evidence"]["block"] for unit in units})
    reference = load_frozen_run_planning_reference(source, block_count, fresh_validation=True)
    ranks = {entry.unit_id: entry for entry in reference.ranks}
    if ranks.keys() != {unit["evidence"]["unit_id"] for unit in units}:
        raise ValueError("reference physical units differ from the measured dataset")
    elements = sum(math.prod(shape) for unit in units for shape in unit["member_shapes"])
    bpw = reference.total_bits / elements
    if math.floor(bpw * elements) != reference.total_bits:
        raise ValueError("reference byte budget cannot be expressed exactly by its BPW")
    multiple = 32
    decisions = []
    spent_bits = 0
    for unit in units:
        measured = unit["staged_measurements"]
        key = unit["evidence"]["unit_id"]
        rank = ranks[key]
        full = measured["full_rank"]
        shape = tuple(measured["identity"]["shape"])

        def cost(candidate: int, dimensions: tuple[int, ...] = shape) -> int:
            return factor_bit_cost(*dimensions, candidate, residual_stages=True, rank_alignment=multiple).total

        if rank.rank != 2 * full or rank.factor_bits != sum(shape) * rank.rank:
            raise ValueError(f"{key} does not use two full stages with the expected cost")
        spent_bits += cost(rank.rank)
        funded = sum(math.floor(math.prod(member) * bpw) for member in unit["member_shapes"])
        baseline_rank = max(r for r in range(multiple, 3 * full + 1, multiple) if cost(r) <= funded)
        decisions.append({"unit_id": key, "baseline_rank": baseline_rank, "planned_rank": rank.rank})
    if spent_bits != reference.total_bits:
        raise ValueError("reference signs and scales differ from the committed bit cost")
    atomic_write_json(args.output, {
        "reference_label": args.reference_label, "dataset_key": dataset.reference.artifact_id,
        "budget_bits": reference.total_bits, "spent_bits": reference.total_bits,
        "allocation": {"decisions": decisions}, "config": {"allocation": {"bounds": {
            "multiple": multiple, "floor_fraction_of_uniform": 0.25, "ceiling_fraction_of_uniform": 3.0,
            "overcomplete_rank_ceiling_fraction": 3.0,
        }}},
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
