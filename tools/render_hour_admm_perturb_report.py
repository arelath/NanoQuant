"""Render Experiment 068's verified results without altering any fit."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path

import _paths  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file

ROOT = Path(__file__).resolve().parents[1]


def render(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    report = json.loads((source / "report.json").read_text())
    if report["status"] not in {"complete", "cancelled"}:
        raise ValueError("search must be finished or cancelled before publication")
    for filename, expected in {**report["inputs"], **report["provenance"]}.items():
        snapshot_name = report.get("provenance_snapshots", {}).get(filename)
        checked = source / snapshot_name if snapshot_name else Path(filename)
        if hash_file(checked) != expected:
            raise ValueError(f"source hash mismatch: {filename}")
    for row in report["records"]:
        for filename, expected in ((row["state_file"], row["state_sha256"]),
                                   (row["initial_file"], row["initial_sha256"])):
            if hash_file(source / filename) != expected:
                raise ValueError(f"checkpoint hash mismatch: {filename}")
    if hash_file(source / "best.safetensors") != report["best_sha256"]:
        raise ValueError("best checkpoint hash mismatch")
    output.mkdir(parents=True, exist_ok=True)
    rows = [r for r in report["records"] if r["kind"] == "10_percent" and r["within_budget"]]
    if not rows:
        raise ValueError("no completed eligible perturbations")
    comp = report["comparison"]
    best = comp["best"]["weighted_error"]
    new = comp["best_perturbed"]["weighted_error"]
    baseline, inherited = (comp[k]["weighted_error"] for k in ("baseline", "inherited"))
    flips = [sum(r["initial_flips"].values()) for r in rows]
    distances = [sum(r["final_distance_from_parent"].values()) for r in rows]
    summary = {
        "experiment": 68, "status": report["status"], "source_report_sha256": hash_file(source / "report.json"),
        "search_seconds": report["search_elapsed_seconds"], "completed_perturbations": len(rows),
        "completed_admm_fits": len(report["records"]),
        "completed_admm_iterations": sum(r["iterations_completed"] for r in report["records"]),
        "distinct_canonical_sign_hashes": len({r["canonical_hash"] for r in rows}),
        "perturbed_candidates_beating_inherited": sum(r["weighted_error"] < inherited for r in rows),
        "perturbed_candidates_beating_baseline": sum(r["weighted_error"] < baseline for r in rows),
        "overall_gain_vs_baseline_percent": 100 * (1 - best / baseline),
        "overall_gain_vs_inherited_percent": 100 * (1 - best / inherited),
        "best_perturbed_gain_vs_baseline_percent": 100 * (1 - new / baseline),
        "best_perturbed_regression_vs_inherited_percent": 100 * (new / inherited - 1),
        "initial_flips_min_max": [min(flips), max(flips)],
        "final_distance_min_median_max": [min(distances), statistics.median(distances), max(distances)],
        "median_attempt_seconds": statistics.median(r["seconds"] for r in rows),
        "comparison": comp, "neighborhood_audit": report["neighborhood_audit"],
        "saved_state_bytes": report["saved_state_bytes"],
        "peak_cuda_allocated": report["peak_cuda_allocated"], "peak_cuda_reserved": report["peak_cuda_reserved"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    times = [r["elapsed_seconds"] / 60 for r in rows]
    errors = [r["weighted_error"] for r in rows]
    axes[0].scatter(times, errors, s=10, color="#1f77b4", alpha=0.55, label="10% + 800-step ADMM")
    axes[0].axhline(baseline, color="#8b5e34", linestyle="--", label="Original 800-step baseline")
    axes[0].axhline(inherited, color="#15734b", label="Inherited winner")
    axes[0].set(xlabel="Additional search time (minutes)", ylabel="Weighted error energy",
                title="Completed candidate fits (lower is better)")
    axes[0].legend(fontsize=8)
    keys = ["baseline", "inherited", "control", "best_perturbed"]
    labels = ["Original\nbaseline", "Inherited\nwinner", "No flips +\n800 ADMM", "Best 10% +\n800 ADMM"]
    values = [comp[k]["weighted_nrmse"] for k in keys]
    bars = axes[1].bar(labels, values, color=["#8b5e34", "#15734b", "#666666", "#1f77b4"])
    axes[1].bar_label(bars, labels=[f"{v:.6f}" for v in values], padding=4, fontsize=9)
    axes[1].set(ylim=(0, max(values) * 1.13), ylabel="Weighted NRMSE",
                title="Independent float64 comparison")
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"admm-perturb.{suffix}", dpi=160)
    plt.close(fig)
    table = "\n".join(
        f"| {label} | {comp[k]['weighted_error']:.10f} | {comp[k]['weighted_nrmse']:.10f} "
        f"| {comp[k]['raw_nrmse']:.10f} |" for k, label in zip(keys + ["best"],
            ["Original 800-step baseline", "Inherited Experiment 067 winner", "Unperturbed 800-step ADMM control",
             "Best 10% + 800-step ADMM candidate", "Overall retained best"], strict=True)
    )
    audit = "\n".join(
        f"| {k} | {v['bits_examined']:,} | {v['improving_bits']:,} | "
        f"{v['best_single_flip_energy_change']:.12g} |" for k, v in report["neighborhood_audit"].items()
    )
    gained = summary["overall_gain_vs_inherited_percent"]
    conclusion = ("The inherited winner remains best; the additional search found no improvement."
                  if report["best"]["state_file"] == "inherited.safetensors" else
                  f"The hour improved the inherited winner's weighted error by {gained:.6f}%.")
    if report["status"] == "cancelled":
        conclusion = "**Cancelled at the user's request before the hour finished.** " + conclusion
    memory = (f"Peak CUDA allocation was {report['peak_cuda_allocated'] / 2**30:.3f} GiB; "
              f"peak reservation was {report['peak_cuda_reserved'] / 2**30:.3f} GiB, measured during search."
              if report["peak_cuda_allocated"] is not None else
              "CUDA allocator peaks are unavailable because the worker was stopped.")
    audit_note = ("The neighborhood audit uses FP64 and absolute energy tolerance 1e-10. It only "
                  "tests individual flips with all scales fixed; it does not prove joint, scale-refitted, "
                  "or global optimality. Audit operations do not optimize the factors."
                  if report["neighborhood_audit"] else
                  "No new single-bit neighborhood audit was run after cancellation. Closeout only "
                  "verified completed checkpoint hashes and rescored retained comparisons in FP64 on CPU.")
    audit_table = ("| Fit | Single bits checked | Improving bits | Best fixed-scale energy delta |\n"
                   "| --- | ---: | ---: | ---: |\n" + audit if audit else "")
    text = f"""# Experiment 068: 10% sign perturbations and 800-step ADMM

{conclusion}

The search ran for **{summary['search_seconds'] / 60:.4f} minutes**, completed
**{len(rows)} perturbations** plus one unperturbed control, and performed
{summary['completed_admm_iterations']:,} ADMM outer iterations. Every completed
candidate received 800 outer iterations and two ordinary scale-fit passes.

| Fit | Weighted error energy | Weighted NRMSE | Raw NRMSE |
| --- | ---: | ---: | ---: |
{table}

The best newly perturbed candidate improves baseline weighted error by
{summary['best_perturbed_gain_vs_baseline_percent']:.6f}%, and has
{summary['best_perturbed_regression_vs_inherited_percent']:.6f}% higher error
than the inherited winner. The overall retained best improves baseline error
by {summary['overall_gain_vs_baseline_percent']:.6f}%. All table values use
independent FP64 reconstruction and reduction.

![Search trajectory and audited comparison](admm-perturb.png)

## What was searched

Each proposal flipped 10% of each factor's signs independently, then initialized
ADMM from that perturbed fit's signs and scales in Fisher-weighted coordinates.
The schedule restarted from rho zero, and duals were reset to the initial
projection residual. All fits used BF16 factors, IEEE FP32 solves, cubic rho,
regularization 0.03, five inner power iterations, and native matrix orientation.
No direct binary descent or extra finalist polishing was added.

There were {summary['distinct_canonical_sign_hashes']} distinct canonical final
sign hashes, {summary['perturbed_candidates_beating_inherited']} perturbations
below the inherited winner, and {summary['perturbed_candidates_beating_baseline']}
below the original baseline. Initial changes ranged from {min(flips):,} to
{max(flips):,} signs. Final raw sign distance from each parent ranged from
{min(distances):,} to {max(distances):,}, with median {statistics.median(distances):,.0f},
out of 9,289,728 signs. Raw distances do not remove sign gauges. Canonical hashes
remove sign gauges but do not establish independent attraction basins.

These are correlated candidate-initialized runs, not fresh independent random
starts across the full factor space. The measured median complete proposal time
was {summary['median_attempt_seconds']:.3f} seconds, including perturbation,
ADMM, scoring, hashing, and checkpoints. The 60-second lifecycle canary was
excluded from the hour and its candidates were not inherited.

The unperturbed control applies the same 800-step restart to the inherited
factors without sign perturbations. Its result distinguishes loss caused by an
800-step ADMM restart from loss caused by perturbing signs. Restarting the
standard schedule is not guaranteed to improve an existing solution. This
experiment tests that specific restart policy, not every possible warm-start
ADMM policy.

Experiment 067's best 5% candidate had error **4.8505008966** after direct binary
refinement. That earlier phase completed 41 proposals. Here both perturbation
radius and refinement method changed, so the comparison cannot isolate the
effect of increasing the radius alone.

## Audit and scope

{audit_table}

{audit_note}

Every recorded initial/final checkpoint and all bound numerical sources passed
fresh hash checks. Evidence stores {report['saved_state_bytes'] / 2**30:.3f} GiB
of states. {memory} Scope: {report['cuda_memory_scope']}.

This reuses the exact Experiment 066 Gemma block-12 1152-by-6912, rank-1152
fixture and seven unchanged outlier columns. The objective is retained
Experiment 041 diagonal Fisher, not missing Experiment 060 calibration or a
full-model quality result. Production math and model exports remain unchanged.

Best factors: `{source / 'best.safetensors'}`

Best SHA-256: `{report['best_sha256']}`

Best newly perturbed factors: `{source / comp['best_perturbed']['state_file']}`

See [frozen protocol](protocol.md), [machine-readable summary](summary.json),
[complete search report](search-report.json), and [validation](validation.json).
"""
    if (output / "residual-patterns/report.md").exists():
        text += ("\nThe user's mid-run error-location question is answered in the "
                 "[CPU residual concentration analysis](residual-patterns/report.md), "
                 "which preserves its own search snapshot and factor hashes.\n")
    (output / "report.md").write_text(text, encoding="utf-8")
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "validation.json", {
        "source_report_sha256": summary["source_report_sha256"],
        "all_recorded_checkpoint_hashes_verified": True, "input_and_source_hashes_verified": True,
        "fp64_comparison": comp, "fixed_scale_one_bit_audit": report["neighborhood_audit"],
    })
    for filename, dest in (("report.json", "search-report.json"), ("fixture.json", "fixture.json"),
                           ("protocol.md", "protocol.md"), ("experiment-receipt.json", "experiment-receipt.json"),
                           ("verification.json", "verification.json")):
        if (source / filename).exists():
            shutil.copy2(source / filename, output / dest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "evidence/068/admm-perturb")
    parser.add_argument("--output", type=Path, default=ROOT / "Results/068")
    args = parser.parse_args()
    render(args.source, args.output)
