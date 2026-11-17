"""Compare sensitivity curves from the curated 160 windows, a nested 16, and the old corpus."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from fit_kl_sensitivity import fit_curve, fit_dataset
from predict_imatrix_weighted_kl import spearman

from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file


def nested_curves(report: dict, indices: list[int]) -> dict:
    def mean_kl(score: dict) -> float:
        return statistics.mean(score["sequences"][i]["kl_nats_per_token"] for i in indices)

    baseline = mean_kl(report["clean_baseline"])
    grouped = defaultdict(list)
    for trial in report["trials"]:
        grouped[trial["target"]].append({**trial, "kl_nats_per_token": mean_kl(trial)})
    return {target: fit_curve(rows, baseline) for target, rows in grouped.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("reference", type=Path, help="attribution JSON binding the old sweep and fits")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.measurements.read_text(encoding="utf-8"))
    full = fit_dataset(args.measurements)["curves"]
    count = raw["protocol"]["teacher"]["samples"]
    indices = sorted(random.Random(83).sample(range(count), 16))
    nested = nested_curves(raw, indices)
    attribution = json.loads(args.reference.read_text(encoding="utf-8"))
    old_source = Path(attribution["measurements"])
    if hash_file(old_source) != attribution["measurement_sha256"]:
        raise ValueError("the reference sensitivity sweep changed")
    old = fit_dataset(old_source)["curves"]
    if full.keys() != old.keys():
        raise ValueError("the sensitivity sweeps cover different projections")
    rows = {}
    for key in full:
        rows[key] = {
            label: {name: curve[name] for name in ("coefficient", "exponent", "fit_quality")}
            for label, curve in (("curated_160", full[key]), ("curated_16", nested[key]), ("old_16", old[key]))
        }
        rows[key]["kl_at_10pct"] = {
            label: curve["coefficient"] * 0.1 ** curve["exponent"]
            for label, curve in (("curated_160", full[key]), ("curated_16", nested[key]), ("old_16", old[key]))
        }
    agreement = {}
    for label in ("curated_16", "old_16"):
        a = [row["kl_at_10pct"]["curated_160"] for row in rows.values()]
        b = [row["kl_at_10pct"][label] for row in rows.values()]
        agreement[label] = {
            "spearman": spearman(a, b),
            "median_fold_difference": math.exp(
                statistics.median(abs(math.log(x / y)) for x, y in zip(a, b, strict=True))
            ),
        }
    output = {"measurement_sha256": hash_file(args.measurements), "reference_sha256": hash_file(old_source),
              "nested_window_indices": indices, "agreement_at_10pct_error": agreement, "projections": rows}
    atomic_write_json(args.output.with_suffix(".json"), output)
    lines = [
        "# Sensitivity sampling comparison", "",
        "The same noise draws on the curated corpus are aggregated over 160 windows and a nested 16-window subset.",
        "The previous corpus contributes its saved 16-window sensitivity curves.", "",
        "| Comparison to curated 160 windows at 10% weight error | Spearman | Median fold difference |",
        "| --- | ---: | ---: |",
    ]
    for label, values in agreement.items():
        lines.append(f"| {label} | {values['spearman']:.4f} | {values['median_fold_difference']:.4f} |")
    atomic_write_text(args.output, "\n".join([*lines, ""]))
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
