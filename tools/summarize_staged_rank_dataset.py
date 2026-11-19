"""Summarize a completed staged rank dataset without accessing its source run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import _paths  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nanoquant.infrastructure.allocation_profile_dataset import load_allocation_profile_dataset
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text


def full_points(record):
    full = record["full_rank"]
    return [next(p for p in record["points"] if p["rank"] == full * count) for count in (1, 2, 3)]


def interpolation_checks(records, metric):
    """Interior leave-one-knot-out interpolation, not independent-data validation."""
    folds = []
    for record in records:
        points = record["points"]
        for left, middle, right in zip(points, points[1:], points[2:], strict=False):
            fraction = (middle["rank"] - left["rank"]) / (right["rank"] - left["rank"])
            predicted = math.exp((1 - fraction) * math.log(left["stored"][metric])
                                 + fraction * math.log(right["stored"][metric]))
            actual = middle["stored"][metric]
            folds.append(max(predicted / actual, actual / predicted))
    return {"count": len(folds), "median_fold_error": statistics.median(folds), "maximum_fold_error": max(folds)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    dataset = load_allocation_profile_dataset(args.dataset)
    records = [unit["staged_measurements"] for unit in dataset.payload["units"]]
    if len(records) != 130 or any(not r["complete"] or len(r["points"]) != 17 for r in records):
        raise ValueError("requires complete 130-unit, 17-point dataset")
    names = list(dict.fromkeys(r["identity"]["unit_id"].split(":", 1)[1] for r in records))
    summary = {"dataset_key": dataset.reference.artifact_id, "units": len(records),
               "points": sum(len(r["points"]) for r in records),
               "member_points": sum(len(p["members"]) for r in records for p in r["points"]),
               "measured_seconds": sum(p["seconds"] for r in records for p in r["points"]),
               "kl_measured": False, "embeddings_modified": False, "by_owner": {}, "interpolation": {}}
    for metric in ("raw_nmse", "objective_nmse", "imatrix_noise_nmse"):
        summary["interpolation"][metric] = interpolation_checks(records, metric)
    summary["full_stage_totals"] = []
    for stage in range(3):
        points = [full_points(r)[stage] for r in records]
        norm = sum(p["stored"]["weight_norm_squared"] for p in points)
        error = sum(p["stored"]["raw_squared_error"] for p in points)
        summary["full_stage_totals"].append({"stages": stage + 1,
                                             "projection_payload_bytes": sum(p["payload_bits"] for p in points) // 8,
                                             "global_raw_nrmse": math.sqrt(error / norm)})
    for name in names:
        selected = [r for r in records if r["identity"]["unit_id"].endswith(":" + name)]
        points = [full_points(r) for r in selected]
        summary["by_owner"][name] = {
            "units": len(selected),
            "median_objective_nmse": [statistics.median(p[i]["stored"]["objective_nmse"] for p in points)
                                      for i in range(3)],
            "median_ratio_2_to_1": statistics.median(p[1]["stored"]["objective_nmse"]
                                                     / p[0]["stored"]["objective_nmse"] for p in points),
            "median_ratio_3_to_2": statistics.median(p[2]["stored"]["objective_nmse"]
                                                     / p[1]["stored"]["objective_nmse"] for p in points),
        }
    summary["nonmonotone_intervals"] = {
        metric: sum(b["stored"][metric] > a["stored"][metric]
                    for r in records for a, b in zip(r["points"], r["points"][1:], strict=False))
        for metric in ("raw_nmse", "objective_nmse", "imatrix_noise_nmse")
    }
    root = args.dataset
    atomic_write_json(root / "summary.json", summary)
    fig, axes = plt.subplots(1, len(names), figsize=(17, 4.5), sharey=True, constrained_layout=True)
    for axis, name in zip(axes, names, strict=True):
        selected = [r for r in records if r["identity"]["unit_id"].endswith(":" + name)]
        all_values = []
        for record in selected:
            baseline = full_points(record)[0]["stored"]["objective_nmse"]
            fractions = [p["rank_fraction"] for p in record["points"]]
            values = [p["stored"]["objective_nmse"] / baseline for p in record["points"]]
            all_values.append(values)
            axis.plot(fractions, values, color="#2d74b4", alpha=.2, linewidth=.8)
        axis.plot(fractions, [statistics.median(v[i] for v in all_values) for i in range(17)],
                  color="#142f4d", linewidth=2.4, label="26-block median")
        axis.set_title(name.replace("mlp.", "").replace("self_attn.", ""))
        axis.set_yscale("log")
        axis.set_xticks([.25, 1, 2, 3])
        axis.set_xlabel("Total rank / full stage rank")
        axis.grid(alpha=.2)
        for boundary in (1, 2, 3):
            axis.axvline(boundary, color="gray", alpha=.3, linewidth=.8)
    axes[0].set_ylabel("Weighted NMSE / one-stage weighted NMSE")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Gemma 3 1B: fresh staged fits, 600×2 ADMM + scale fit; BF16 stored scales\n"
                 "Faint curves: individual blocks. Reconstruction response only; no model-quality claim.")
    fig.savefig(root / "rank-response.png", dpi=180)
    plt.close(fig)
    lines = ["# Gemma 3 1B reusable staged rank responses", "",
             f"Dataset: `{dataset.reference.artifact_id}`.", "",
             f"Complete: {summary['units']} physical units, {summary['points']} rank options, "
             f"{summary['member_points']} per-projection observations. Embeddings untouched.", "",
             "Original source weights; 080 IMatrix objectives; 600×2 ADMM and two scale-fit passes.",
             "One fixed seed per unit. This is pre-activation-tuning reconstruction data, not measured rank-to-KL.", "",
             "| Owner | Median weighted NMSE, 1 stage | 2 stages | 3 stages | Median 2/1 ratio | Median 3/2 ratio |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in summary["by_owner"].items():
        errors = values["median_objective_nmse"]
        lines.append(f"| {name} | {errors[0]:.6f} | {errors[1]:.6f} | {errors[2]:.6f} | "
                     f"{values['median_ratio_2_to_1']:.4f} | {values['median_ratio_3_to_2']:.4f} |")
    lines += ["", "![Measured rank responses](rank-response.png)", "",
              "Each option preserves explicit stage ranks, exact packed factor/scale payload cost, raw NMSE,",
              "input/output-weighted objective NMSE, and mean-one-column-weighted noise-equivalent NMSE.",
              "Shared QKV has individual Q/K/V metrics and one shared storage cost.", "",
              "All 17 raw points and 16 intervals are retained. Decreasing isotonic regression acts on log NMSE;",
              "piecewise log-linear interpolation is restricted to measured coverage in the query helper.", "",
              "| Interior leave-one-knot-out check | Median fold error | Maximum fold error |",
              "| --- | ---: | ---: |"]
    for metric, values in summary["interpolation"].items():
        lines.append(f"| {metric} | {values['median_fold_error']:.4f} | {values['maximum_fold_error']:.4f} |")
    lines += ["", "These checks quantify grid interpolation, not new-seed or new-corpus generalization.",
              "No extrapolated points are labeled as measurements. See `summary.json` for counts and costs.", "",
              "Load through `nanoquant.infrastructure.allocation_profile_dataset.load_allocation_profile_dataset`.",
              "Read `unit['staged_measurements']` for metrics and curves; standard allocation evidence is present.",
              "No older KL profile is included. KL-calibrated D2 launchers require a separately measured profile.",
              "The dataset is portable after source-run retirement; rebuilding requires pinned source inputs.",
              "Completed scalar checkpoints remain under `measurements/`; transient fitting tensors are retired.", "",
              "Protocol: [experiment document](../../Docs/Experiments/080-staged-rank-response.md).", ""]
    atomic_write_text(root / "README.md", "\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
