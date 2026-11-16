"""Analyze a completed matched-noise campaign without fitting any KL correction."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from compare_imatrix_shaped_noise import MODES, prediction_metrics

from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file

METHODS = ("curve_raw", "curve_weighted", *MODES)


def analyze(report):
    plan = report["identity"]["plan"]
    if not report.get("complete") or not report.get("embedding_verified_unchanged"):
        raise ValueError("requires completed measurements and verified fixed embedding")
    rows = []
    norm_mismatch = []
    for case in report["cases"].values():
        probes = case["probes"]
        expected = {"actual", *(f"{mode}:{seed}" for mode in MODES for seed in plan["seeds"])}
        if set(probes) != expected:
            raise ValueError("missing or unexpected probes")
        actual = probes["actual"]["kl_above_clean"]
        row = {key: case[key] for key in ("source", "target", "storage_type", "actual_residual",
                                        "importance_floored_fraction", "curve_raw_extrapolated",
                                        "curve_weighted_extrapolated")}
        row["block"] = int(case["target"].split(":")[0])
        row["actual_kl"] = actual
        row["actual_window_kl"] = [s["kl_nats_per_token"] for s in probes["actual"]["score"]["sequences"]]
        row["predictions"] = {name: case[name] for name in METHODS if name.startswith("curve_")}
        row["noise"] = {}
        for mode in MODES:
            samples = [probes[f"{mode}:{seed}"] for seed in plan["seeds"]]
            kls = [probe["kl_above_clean"] for probe in samples]
            row["predictions"][mode] = float(np.mean(kls))
            # Paired window uncertainty, averaging over the noise draws first.
            clean_windows = report["clean"]["sequences"]
            actual_windows = probes["actual"]["score"]["sequences"]
            differences = [
                float(np.mean([probe["score"]["sequences"][i]["kl_nats_per_token"]
                               for probe in samples])) - actual_windows[i]["kl_nats_per_token"]
                for i in range(len(clean_windows))
            ]
            row["noise"][mode] = {
                "kl_seed_sd": float(np.std(kls, ddof=1)) if len(kls) > 1 else None,
                "paired_window_se": float(np.std(differences, ddof=1) / math.sqrt(len(differences))),
                "mean_weighted_to_raw_energy": float(np.mean([p["weighted_to_raw_energy"] for p in samples])),
            }
            norm_mismatch.extend(abs(p["raw_error"] / probes["actual"]["raw_error"] - 1) for p in samples)
        rows.append(row)
    summaries = {}
    for subset, blocks in (("all", None), ("prior_blocks", plan["previously_sampled_blocks"]),
                           ("confirmation_blocks", plan["confirmation_blocks"])):
        selected = [row for row in rows if blocks is None or row["block"] in blocks]
        summaries[subset] = {}
        for source in plan["sources"]:
            group = [row for row in selected if row["source"] == source]
            summaries[subset][source] = {
                "count": len(group),
                "actual_kl_sum": sum(row["actual_kl"] for row in group),
                "median_actual_weighted_to_raw_energy": float(np.median([
                    row["actual_residual"]["weighted_to_raw_energy"] for row in group])),
                "predictors": {method: prediction_metrics([row["predictions"][method] for row in group],
                                                           [row["actual_kl"] for row in group])
                               for method in METHODS},
            }
    # Direct cross-family decisions on the same target, avoiding aggregate/rank confounding.
    pairs = {method: {"correct": 0, "total": 0, "resolved_correct": 0, "resolved_total": 0}
             for method in METHODS}
    for target in dict.fromkeys(row["target"] for row in rows):
        for a, b in itertools.combinations([row for row in rows if row["target"] == target], 2):
            truth = a["actual_kl"] < b["actual_kl"]
            differences = np.subtract(a["actual_window_kl"], b["actual_window_kl"])
            window_se = float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
            resolved = abs(a["actual_kl"] - b["actual_kl"]) > 2 * window_se
            for method in METHODS:
                pairs[method]["total"] += 1
                correct = (a["predictions"][method] < b["predictions"][method]) == truth
                pairs[method]["correct"] += correct
                pairs[method]["resolved_total"] += resolved
                pairs[method]["resolved_correct"] += correct and resolved
    common_targets = set.intersection(*[set(spec["targets"]) for spec in plan["sources"].values()])
    common_energy = {}
    for source in plan["sources"]:
        group = [row for row in rows if row["source"] == source and row["target"] in common_targets]
        common_energy[source] = {"count": len(group), "median_actual_weighted_to_raw_energy": float(np.median([
            row["actual_residual"]["weighted_to_raw_energy"] for row in group])),
            "median_imatrix_noise_weighted_to_raw_energy": float(np.median([
                row["noise"]["imatrix"]["mean_weighted_to_raw_energy"] for row in group]))}
    paired_energy = []
    for first, second in itertools.combinations(plan["sources"], 2):
        by_source = {name: {row["target"]: row for row in rows if row["source"] == name}
                     for name in (first, second)}
        ratios = [by_source[first][target]["actual_residual"]["weighted_to_raw_energy"] /
                  by_source[second][target]["actual_residual"]["weighted_to_raw_energy"]
                  for target in sorted(common_targets)]
        paired_energy.append({"first": first, "second": second, "count": len(ratios),
                              "first_lower_count": sum(ratio < 1 for ratio in ratios),
                              "median_first_over_second": float(np.median(ratios))})
    return {"rows": rows, "summaries": summaries, "cross_format_ordering": pairs,
            "common_projection_energy_ratios": common_energy,
            "paired_energy_ratios": paired_energy,
            "max_post_cast_relative_norm_mismatch": max(norm_mismatch),
            "raw_curve_extrapolated_cases": sum(row["curve_raw_extrapolated"] for row in rows),
            "weighted_curve_extrapolated_cases": sum(row["curve_weighted_extrapolated"] for row in rows)}


def render(analysis):
    lines = ["# Matched IMatrix noise analysis", "",
             "Median fold error is exp(median(abs(log(predicted / actual)))); lower is better, ideal 1.",
             "Predictions are uncorrected: no KL-fitted coefficients or family multipliers.", ""]
    for subset, sources in analysis["summaries"].items():
        lines += [f"## {subset}", "",
                  "| Source | Cases | Method | Predicted / actual sum | Median fold error | Spearman |",
                  "| --- | ---: | --- | ---: | ---: | ---: |"]
        for source, summary in sources.items():
            for method, stats in summary["predictors"].items():
                lines.append(f"| {source} | {summary['count']} | {method} | {stats['sum_ratio']:.3f} | "
                             f"{stats['median_fold_error']:.3f} | {stats['spearman']:.3f} |")
        lines.append("")
    lines += ["## Same-projection format ordering", "",
              "Resolved pairs have an actual KL gap greater than two paired-window standard errors.",
              "This is a descriptive filter on one text sample, not an independent-corpus significance claim.", "",
              "| Method | Correct pairs | Total | Resolved correct | Resolved total |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for method, result in analysis["cross_format_ordering"].items():
        lines.append(f"| {method} | {result['correct']} | {result['total']} | "
                     f"{result['resolved_correct']} | {result['resolved_total']} |")
    lines += ["", "## Error placement on projections shared by all formats", "",
              "The ratio is IMatrix-weighted residual energy / raw residual energy; smaller means error is",
              "more concentrated in low-importance columns. These medians use exactly the same projections.", "",
              "| Source | Projections | Actual residual ratio | IMatrix noise ratio |",
              "| --- | ---: | ---: | ---: |"]
    for source, entry in analysis["common_projection_energy_ratios"].items():
        lines.append(f"| {source} | {entry['count']} | {entry['median_actual_weighted_to_raw_energy']:.3f} | "
                     f"{entry['median_imatrix_noise_weighted_to_raw_energy']:.3f} |")
    lines += ["", f"Maximum post-cast norm mismatch: {analysis['max_post_cast_relative_norm_mismatch']:.4%}.",
              f"Raw/weighted curve extrapolations: {analysis['raw_curve_extrapolated_cases']}/"
              f"{analysis['weighted_curve_extrapolated_cases']} cases.", "",
              "## Individual comparisons", "",
              "| Source | Projection | Actual KL | Isotropic | Block | IMatrix | Block + IMatrix | Weighted curve |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in analysis["rows"]:
        values = " | ".join(f"{row['predictions'][method]:.6f}" for method in (*MODES, "curve_weighted"))
        lines.append(f"| {row['source']} | {row['target']} | {row['actual_kl']:.6f} | {values} |")
    return "\n".join([*lines, ""])


def plot(analysis, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = list(analysis["summaries"]["all"])
    methods = ("isotropic", "imatrix", "block_imatrix", "curve_weighted")
    names = ("Isotropic noise", "IMatrix noise", "Block + IMatrix noise", "Existing weighted curve")
    fig, axes = plt.subplots(len(sources), len(methods), figsize=(14, 10), sharex=True, sharey=True)
    for r, source in enumerate(sources):
        group = [row for row in analysis["rows"] if row["source"] == source]
        for c, (method, name) in enumerate(zip(methods, names, strict=True)):
            ax = axes[r, c]
            for row in group:
                confirmation = row["block"] in (6, 19)
                ax.scatter(row["actual_kl"], row["predictions"][method],
                           marker="s" if confirmation else "o", color="#2463a6", s=28, alpha=.8)
            ax.plot([1e-5, 1], [1e-5, 1], color="gray", lw=1, linestyle="--")
            ax.set(xscale="log", yscale="log", xlim=(1e-5, 1), ylim=(1e-5, 1))
            ax.grid(alpha=.15)
            if r == 0:
                ax.set_title(name)
            if c == 0:
                ax.set_ylabel(f"{source}\nPredicted KL (nats/token)")
            if r == len(sources)-1:
                ax.set_xlabel("Actual isolated residual KL")
    fig.suptitle("Gemma 3 1B: matched residual norms, fixed embeddings\n"
                 "Three noise seeds; squares = confirmation blocks 6 and 19", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.measurements.read_text(encoding="utf-8"))
    analysis = {"measurement_sha256": hash_file(args.measurements), "analysis_code_sha256": hash_file(Path(__file__)),
                **analyze(report)}
    atomic_write_json(args.output.with_suffix(".json"), analysis)
    atomic_write_text(args.output.with_suffix(".md"), render(analysis))
    plot(analysis, args.output.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
