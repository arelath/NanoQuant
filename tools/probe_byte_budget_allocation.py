"""Pilot a byte-budget allocation across staged binary factors and llama.cpp formats, on the CPU.

For every decoder projection this builds a menu of storage options, each with its stored bytes and
an estimated KL: the option's imatrix-weighted weight error put through the projection's fitted
sensitivity curve. Staged options come from a NanoQuant GGUF (its first term alone, and all of its
terms), standard options from ordinary GGUFs of the same checkpoint. A greedy pass over each unit's
efficient options then picks one option per unit under a total byte budget.

This is the measurement behind ``Docs/115-byte-budget-mixed-format-allocation.md``. It estimates
isolated, additive effects only and runs no model.

Usage:
    .venv\\Scripts\\python.exe tools\\probe_byte_budget_allocation.py ^
        Datasets\\<dataset>\\measurements\\<sweep-id>\\results.json --imatrix imatrix.dat ^
        --staged nanoquant.gguf --standard Q8_0.gguf --standard Q4_0.gguf --budget 547391648
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np
from gguf.quants import dequantize
from materialize_nanoquant_gguf import _gemma_qkv_sizes
from predict_imatrix_weighted_kl import noise_equivalent_error, prediction_inputs
from weight_image_common import reconstruct_nanoquant_weight

from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text

MEMBERS = {"attn_qkv": ("attn_q", "attn_k", "attn_v")}
TERM_SUFFIXES = ("v", "u", "scale_pre", "scale_mid", "scale_post")


@dataclass(frozen=True)
class Option:
    label: str
    bytes: float
    kl: float


def estimated_kl(curve: Mapping, error: float) -> float:
    """Fitted curve inside and above the measured range, quadratic below its lowest error."""
    low = curve["measured_error_range"][0]
    if error >= low:
        return float(curve["coefficient"] * error ** curve["exponent"])
    return float(curve["coefficient"] * low ** curve["exponent"] * (error / low) ** 2)


def _saving(cheaper: Option, larger: Option) -> float:
    return (cheaper.kl - larger.kl) / (larger.bytes - cheaper.bytes)


def efficient(options: Sequence[Option]) -> list[Option]:
    """Return the options on the lower convex hull of (bytes, KL), cheapest first."""
    hull: list[Option] = []
    for option in sorted(options, key=lambda item: (item.bytes, item.kl)):
        if hull and option.kl >= hull[-1].kl:
            continue
        while len(hull) >= 2 and _saving(hull[-2], hull[-1]) <= _saving(hull[-1], option):
            hull.pop()
        hull.append(option)
    return hull


def allocate(units: Mapping[str, Sequence[Option]], budget: float) -> dict[str, Option]:
    """Pick one option per unit to minimise summed KL within a byte budget.

    Every unit starts at its cheapest efficient option. Upgrades along each unit's hull are taken
    in order of KL saved per byte while they fit, so the result is optimal up to one upgrade.
    """
    hulls = {name: efficient(options) for name, options in units.items()}
    chosen = dict.fromkeys(hulls, 0)
    spent = math.fsum(hull[0].bytes for hull in hulls.values())
    if spent > budget:
        raise ValueError(f"the cheapest options need {spent:.0f} bytes, over the budget of {budget:.0f}")
    steps = sorted(
        ((_saving(hull[index - 1], hull[index]), name, index)
         for name, hull in hulls.items() for index in range(1, len(hull))),
        reverse=True,
    )
    for _, name, index in steps:
        hull = hulls[name]
        extra = hull[index].bytes - hull[index - 1].bytes
        if chosen[name] == index - 1 and spent + extra <= budget:
            chosen[name] = index
            spent += extra
    return {name: hulls[name][index] for name, index in chosen.items()}


def combine(members: Sequence[Sequence[Option]]) -> list[Option]:
    """Every way to pick one option per member, as options for the group."""
    return [
        Option(" ".join(option.label for option in choice), math.fsum(option.bytes for option in choice),
               math.fsum(option.kl for option in choice))
        for choice in itertools.product(*members)
    ]


def staged_terms(reader: gguf.GGUFReader) -> dict[str, list[tuple[np.ndarray, int]]]:
    """Return each stored factor group's terms as (dense weight, stored bytes), first term first."""
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    groups = {}
    for base in sorted(name.removesuffix(".nq_stage0_v") for name in tensors if name.endswith(".nq_stage0_v")):
        count = int(reader.fields[f"quantization.nanoquant.{base}.residual_stage_count"].contents())
        terms = []
        for term in ("nq_stage0", *(f"nq_residual_{index}" for index in range(count))):
            stored = sum(int(tensors[f"{base}.{term}_{suffix}"].n_bytes) for suffix in TERM_SUFFIXES)
            terms.append((reconstruct_nanoquant_weight(tensors, base, gguf, term=term), stored))
        groups[base] = terms
    return groups


def build_menu(args: argparse.Namespace) -> dict:
    identity, curves, columns, _, source, keys = prediction_inputs(
        args.sweep.resolve(), args.imatrix.resolve(), None, None, snapshot=args.snapshot,
    )
    by_name = {name: (key, layer) for key, (name, layer) in keys.items()}
    staged = gguf.GGUFReader(args.staged)
    standard = [gguf.GGUFReader(path) for path in args.standard]
    menu: dict[str, dict] = {}

    def measure(name: str, quantized: np.ndarray, original: np.ndarray) -> dict:
        key, layer = by_name[name]
        error, weighted = noise_equivalent_error(original, quantized, columns[layer.layer].numpy())
        low, high = curves[key]["measured_error_range"]
        return {"error": error, "noise_equivalent_error": weighted, "kl": estimated_kl(curves[key], weighted),
                "in_measured_range": low <= weighted <= high}

    def original_weight(name: str) -> np.ndarray:
        with source.read_tensor(by_name[name][1].weight) as tensor:
            return tensor.float().numpy()

    for base, terms in staged_terms(staged).items():
        block, kind = base.split(".")[1:]
        names = [f"blk.{block}.{member}.weight" for member in MEMBERS.get(kind, (kind,))]
        originals = [original_weight(name) for name in names]
        sizes = np.cumsum(_gemma_qkv_sizes(staged.fields, int(block)))[:-1] if kind in MEMBERS else []
        entry = menu[base] = {"members": names, "elements": [int(item.size) for item in originals],
                              "staged": [], "standard": {name: {} for name in names}}
        dense, stored = np.zeros_like(terms[0][0]), 0
        for weight, size in terms:
            dense, stored = dense + weight, stored + size
            parts = np.split(dense, sizes, axis=0) if kind in MEMBERS else [dense]
            entry["staged"].append({"bytes": stored, "members": [
                measure(name, part, original) for name, part, original in zip(names, parts, originals, strict=True)
            ]})
        for reader in standard:
            stored_tensors = {tensor.name: tensor for tensor in reader.tensors}
            for name, original in zip(names, originals, strict=True):
                tensor = stored_tensors[name]
                kind_name = tensor.tensor_type.name
                if kind_name in entry["standard"][name] or kind_name in ("F32", "F16", "BF16"):
                    continue
                values = np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32)
                entry["standard"][name][kind_name] = {
                    "bytes": int(tensor.n_bytes), **measure(name, values.reshape(original.shape), original),
                }
        print(f"{base}: {len(terms)} terms, {sorted(set().union(*entry['standard'].values()))}", flush=True)
    staged_bytes = sum(entry["staged"][-1]["bytes"] for entry in menu.values())
    embedding = {}
    for reader in standard:
        tensor = next(tensor for tensor in reader.tensors if tensor.name == "token_embd.weight")
        embedding[tensor.tensor_type.name] = int(tensor.n_bytes)
    return {
        "schema_version": 1, **identity, "staged": str(args.staged), "standard": [str(path) for path in args.standard],
        "staged_file_bytes": args.staged.stat().st_size, "staged_projection_bytes": staged_bytes,
        "fixed_bytes": args.staged.stat().st_size - staged_bytes, "embedding_bytes": embedding, "units": menu,
    }


def unit_options(entry: Mapping, *, third_stage: bool, formats: Sequence[str] | None = None) -> list[Option]:
    """List a unit's options: each stage count, and every mix of standard formats over its members."""
    options = [
        Option(f"{count} stage{'s' * (count > 1)}", stage["bytes"], math.fsum(item["kl"] for item in stage["members"]))
        for count, stage in enumerate(entry["staged"], start=1)
    ]
    if third_stage and len(entry["staged"]) == 2:
        first, second = entry["staged"]
        kl = math.fsum(
            two["kl"] * (two["kl"] / one["kl"]) for one, two in zip(first["members"], second["members"], strict=True)
        )
        options.append(Option("3 stages (assumed)", second["bytes"] * 1.5, kl))
    members = [
        [Option(kind, item["bytes"], item["kl"]) for kind, item in entry["standard"][name].items()
         if formats is None or kind in formats]
        for name in entry["members"]
    ]
    if all(members):
        options.extend(combine(members))
    return options


def solve(menu: Mapping, budget: float, *, third_stage: bool, formats: Sequence[str] | None = None) -> dict:
    units = {name: unit_options(entry, third_stage=third_stage, formats=formats)
             for name, entry in menu["units"].items()}
    chosen = allocate(units, budget - menu["fixed_bytes"])
    counts: Counter[str] = Counter()
    for name, option in chosen.items():
        kind = name.split(".")[2]
        for label in (option.label.split(" ") if kind in MEMBERS and "stage" not in option.label else [option.label]):
            counts[f"{kind}: {label}"] += 1
    by_block: Counter[int] = Counter()
    for name, option in chosen.items():
        by_block[int(name.split(".")[1])] += option.bytes
    return {
        "budget": budget, "bytes": menu["fixed_bytes"] + math.fsum(option.bytes for option in chosen.values()),
        "kl": math.fsum(option.kl for option in chosen.values()), "third_stage": third_stage,
        "counts": dict(sorted(counts.items())), "block_bytes": [by_block[block] for block in sorted(by_block)],
        "choices": {name: option.label for name, option in chosen.items()},
    }


def render_markdown(menu: Mapping, solutions: Sequence[Mapping]) -> str:
    lines = [
        "# Byte-budget allocation pilot", "",
        f"Staged file `{Path(menu['staged']).name}`: {menu['staged_file_bytes']:,} bytes, of which "
        f"{menu['staged_projection_bytes']:,} are decoder projections and {menu['fixed_bytes']:,} are fixed.",
        "KL is the additive estimate in nats/token: each option's imatrix-weighted error through the",
        "projection's sensitivity curve, quadratic below the lowest measured error. No model was run.", "",
        "## Options by projection type", "",
        "Summed over all blocks, as if every projection of the type used the option.", "",
        "| Projection | Option | Bits/weight | Median raw error | Median weighted error | Estimated KL |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    rows: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
    for entry in menu["units"].values():
        group_elements = sum(entry["elements"])
        for count, stage in enumerate(entry["staged"], start=1):
            for name, item in zip(entry["members"], stage["members"], strict=True):
                rows.setdefault((name.split(".")[2], f"{count} stage{'s' * (count > 1)}"), []).append(
                    (stage["bytes"] * 8 / group_elements, item["error"], item["noise_equivalent_error"], item["kl"]))
        for (name, kinds), elements in zip(entry["standard"].items(), entry["elements"], strict=True):
            for kind_name, item in kinds.items():
                rows.setdefault((name.split(".")[2], kind_name), []).append(
                    (item["bytes"] * 8 / elements, item["error"], item["noise_equivalent_error"], item["kl"]))
    for (kind, label), values in sorted(rows.items(), key=lambda row: (row[0][0], -sum(v[3] for v in row[1]))):
        bits, error, weighted, kl = (np.array(column) for column in zip(*values, strict=True))
        lines.append(f"| {kind} | {label} | {np.median(bits):.2f} | {np.median(error):.2%} | "
                     f"{np.median(weighted):.2%} | {kl.sum():.4f} |")
    lines += ["", "## Allocations", "",
              "| Budget (bytes) | Third stage offered | Used bytes | Estimated KL | Choices |",
              "| ---: | --- | ---: | ---: | --- |"]
    for item in solutions:
        choices = "; ".join(f"{label} × {count}" for label, count in item["counts"].items())
        lines.append(f"| {item['budget']:,.0f} | {'yes' if item['third_stage'] else 'no'} | {item['bytes']:,.0f} | "
                     f"{item['kl']:.4f} | {choices} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", type=Path, help="projection sensitivity sweep results.json")
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, help="source checkpoint (default: the hub cache)")
    parser.add_argument("--staged", type=Path, required=True, help="staged NanoQuant GGUF")
    parser.add_argument("--standard", type=Path, action="append", default=[], help="standard GGUF; repeatable")
    parser.add_argument("--budget", type=float, action="append", default=[], help="total file bytes; repeatable")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report; the menu is saved beside it")
    args = parser.parse_args()
    output = args.output.resolve()
    menu_path = output.with_suffix(".menu.json")
    if menu_path.exists():
        menu = json.loads(menu_path.read_text(encoding="utf-8"))
        print(f"Reusing {menu_path}", flush=True)
    else:
        menu = build_menu(args)
        atomic_write_json(menu_path, menu)
    solutions = [solve(menu, budget, third_stage=third) for budget in args.budget for third in (False, True)]
    atomic_write_json(output.with_suffix(".json"), {"menu": menu_path.name, "solutions": solutions})
    atomic_write_text(output, render_markdown(menu, solutions))
    print(f"Report written to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
