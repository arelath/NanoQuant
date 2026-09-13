"""Render hash-verified Experiment 067 results and a standalone scientific figure."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
from safetensors.numpy import load_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from render_hour_factor_fit_report import sha256  # noqa: E402


def render(source: Path, output: Path):
    report = json.loads((source / "report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("completed experiment required")
    for row in report["records"]:
        if sha256(source / row["state_file"]) != row["state_sha256"]:
            raise ValueError(f"state hash mismatch: {row['name']}")
    if sha256(source / "best.safetensors") != report["best_sha256"]:
        raise ValueError("best hash mismatch")
    for name, expected in report["provenance"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"numerical code mismatch: {name}")
    for change in report.get("protocol_changes", []):
        if sha256(source / "pre-large-flip-runner.py") != change["previous_runner_sha256"]:
            raise ValueError("pre-amendment source hash mismatch")
        if sha256(source / "pre-large-flip-report.json") != change["previous_report_sha256"]:
            raise ValueError("pre-amendment report hash mismatch")
    output.mkdir(parents=True, exist_ok=True)
    comparison, records, attempts = report["comparison"], report["records"], report["attempts"]
    large_audit_path = source / "large-flip-audit.json"
    large_audit = None
    if large_audit_path.exists():
        large_audit = json.loads(large_audit_path.read_text())
        if (sha256(source / large_audit["state_file"]) != large_audit["state_sha256"]
                or sha256(source / "report.json") != large_audit["search_report_sha256"]):
            raise ValueError("large-flip audit identity mismatch")
        comparison = {**comparison, "large_flip_best": large_audit["metrics"]}
    previous = comparison["previous_best"]["weighted_error"]
    baseline = comparison["baseline"]["weighted_error"]
    control = comparison["continued_local"]["weighted_error"]
    best = comparison["best"]["weighted_error"]
    extra = {
        "search_minutes": report["search_seconds"] / 60,
        "completed_attempts": len(attempts),
        "completed_refinements": len(records),
        "within_budget_refinements": sum(r["within_budget"] for r in records),
        "uphill_walker_accepts": sum(a["walker_uphill"] for a in attempts),
        "distinct_final_sign_hashes": len({a["final_hash"] for a in attempts}),
        "attempts_better_than_continued_local": sum(a["final_error"] < control for a in attempts),
        "proposal_counts": dict(Counter(a["proposal"] for a in attempts)),
        "saved_state_bytes": sum((source / r["state_file"]).stat().st_size for r in records),
        "best_state_bytes": (source / "best.safetensors").stat().st_size,
        "best_name": report["best"]["name"],
        "best_sha256": report["best_sha256"],
        "peak_cuda_allocated_bytes": report["peak_allocated_bytes"],
        "peak_cuda_reserved_bytes": report["peak_reserved_bytes"],
        "fresh_admm_starts_in_this_hour": 0,
        "largest_uniform_flip_probability": 0.05 if report.get("large_flips") else 0.005,
        "coverage_scope": "Correlated local proposals around retained fits; distinct hashes are not independent basins",
    }
    large = [a for a in attempts if a.get("campaign") == "large-5pct"]
    if large:
        indexed = {r["name"]: r for r in records}
        distances = []
        for attempt in large:
            parent_row, final_row = indexed[attempt["parent"]], indexed[attempt["final"]]
            parent_state = load_file(str(source / parent_row["state_file"]))
            final_state = load_file(str(source / final_row["state_file"]))
            distances.append(sum(int((parent_state[k] != final_state[k]).sum())
                                 for k in ("left_binary", "right_binary")))
        change = report["protocol_changes"][0]
        extra["large_flip_results"] = {
            "completed_attempts": len(large),
            "minimum_final_error": min(a["final_error"] for a in large),
            "best_before_amendment": change["best_before_change"]["weighted_error"],
            "amendment_at_minutes": change["elapsed_seconds"] / 60,
            "initial_flip_count_min": min(a["left_flips"] + a["right_flips"] for a in large),
            "initial_flip_count_max": max(a["left_flips"] + a["right_flips"] for a in large),
            "final_raw_sign_distance_min": min(distances),
            "final_raw_sign_distance_median": statistics.median(distances),
            "final_raw_sign_distance_max": max(distances),
            "raw_distance_scope": "Corresponding factor entries before versus after refinement, without gauge removal",
        }
        if large_audit:
            extra["large_flip_results"]["float64_audit"] = large_audit
    host_samples = source / "host-memory-samples.jsonl"
    if host_samples.exists():
        extra["observed_host_peak_working_set_bytes"] = max(
            json.loads(line)["peak_working_set_bytes"]
            for line in host_samples.read_text(encoding="utf-8-sig").splitlines() if line
        )
    for fraction in (0.9, 0.99):
        threshold = previous - fraction * (previous - best)
        matching = [r["elapsed_seconds"] / 60 for r in records
                    if r["within_budget"] and r["weighted_error"] <= threshold]
        extra[f"minutes_to_{int(fraction * 100)}pct_new_gain"] = min(matching, default=0.0)
    summary = {**comparison, **extra, "neighborhood_audits": report["neighborhood_audits"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    xs, ys, incumbent = [0.0], [0.0], previous
    for row in records:
        if row["within_budget"]:
            incumbent = min(incumbent, row["weighted_error"])
            xs.append(row["elapsed_seconds"] / 60)
            ys.append(100 * (1 - incumbent / previous))
    xs.append(report["search_seconds"] / 60)
    ys.append(ys[-1])
    axes[0].step(xs, ys, where="post", color="#146b67", linewidth=2)
    axes[0].axhline(100 * (1 - control / previous), linestyle="--", color="#a06529",
                    label="Continued local control")
    if large:
        axes[0].axvline(extra["large_flip_results"]["amendment_at_minutes"], linestyle=":", color="#8460a2",
                       label="Switch to 5% flips")
    axes[0].set(xlabel="Additional search time (minutes)", ylabel="Error reduction vs previous hour (%)",
                title="Improvement during the second hour", xlim=(0, 60))
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    labels = ("800-step\nbaseline", "Previous\nhour", "Continued\nlocal control", "Second-hour\nbest")
    values = [comparison[k]["weighted_nrmse"] for k in ("baseline", "previous_best", "continued_local", "best")]
    axes[1].bar(labels, values, color=["#777783", "#9cb6b1", "#c8a681", "#146b67"], width=0.6)
    for i, value in enumerate(values):
        axes[1].text(i, value + 0.002, f"{value:.6f}", ha="center", fontsize=9)
    axes[1].set(ylabel="Weighted NRMSE (lower is better)", title="Common float64 reconstruction audit",
                ylim=(0, max(values) * 1.13))
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Gemma 3 1B · block 12 MLP down · rank 1152 · fixed Fisher objective", fontsize=13)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"basin-search.{suffix}", dpi=180)
    plt.close(fig)
    lines = [
        "# Experiment 067: second-hour basin search", "",
        f"The best fit has **{100 * (1 - best / baseline):.5f}% lower weighted error energy** than the ordinary "
        f"800-step baseline, and **{100 * (1 - best / previous):.5f}% lower energy** than the previous hour's winner. "
        f"The additional search ran for **{extra['search_minutes']:.3f} minutes**.", "",
        "| Fit | Weighted error energy | Weighted NMSE | Weighted NRMSE | Raw NRMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("baseline", "Ordinary 800-step baseline"), ("previous_best", "Previous hour winner"),
                       ("continued_local", "Continued local refinement control"), ("best", "Second-hour best"),
                       ("large_flip_best", "Best completed 5% proposal")):
        if key not in comparison:
            continue
        m = comparison[key]
        lines.append(f"| {label} | {m['weighted_error']:.10f} | {m['weighted_nmse']:.10f} | "
                     f"{m['weighted_nrmse']:.10f} | {m['raw_nrmse']:.10f} |")
    lines += ["", f"Weighted NRMSE is **{comparison['nrmse_reduction_vs_baseline_percent']:.5f}% lower** than "
              f"baseline and **{comparison['nrmse_reduction_vs_previous_best_percent']:.5f}% lower** than the "
              "previous winner. Energy and root-error reductions are different quantities.", "",
              "![Search trajectory and final fits](basin-search.png)", "", "## What the search tested", "",
              f"Completed {len(attempts)} proposal attempts and {len(records)} bounded refinement passes. "
              f"The walker accepted {extra['uphill_walker_accepts']} worse final fits to maintain exploration. "
              f"There were {extra['distinct_final_sign_hashes']} distinct gauge-canonical final sign hashes; "
              "these do not certify distinct attraction basins.", "",
              "**Random-start coverage:** This hour launches zero fresh independent ADMM starts. It uses "
              "local proposals and retained first-hour states. Experiment 066 used seeds 0, 1, and 2, "
              "with seed 0 alone at the deepest 51,200-step horizon. The original uniform kicks changed "
              "at most 0.5% of signs, increased to 5% at the user's request; independent random sign configurations "
              "differ in approximately "
              "50%. The small extra gain therefore measures local headroom, not global search saturation.", "",
              "Earlier measured runtimes suggest roughly 300-370 fresh 800-step starts per hour, "
              "85-95 at 3,200 steps, 43-47 at 6,400 steps, or 4-5 at 51,200 steps, before extra local "
              "polishing and checkpoint overhead. These are throughput estimates, not completed random-start "
              "experiments. A broader test should use fresh seeds and staged deepening, retaining some "
              "randomly selected starts to test whether early scores predict eventual quality.", "",
              "**ADMM coverage:** The inherited winner came from a 51,200-iteration ADMM fit. Candidates "
              "in this hour receive scale fitting and direct binary refinement, without another ADMM fit. "
              "The current factorizer always initializes random latents and has no candidate warm-start "
              "API, so rerunning it unchanged would discard the candidate. ADMM relaxation initialized "
              "from the perturbed candidates remains untested.", "",
              f"The best result is {comparison['energy_reduction_vs_continued_local_percent']:.5f}% below the "
              f"continued-local control. {extra['attempts_better_than_continued_local']} completed attempts "
              "ended below that control. This comparison includes additional compute after the frozen control; "
              "it does not isolate the causal contribution of perturbations from extra refinement time.", "",
              f"90% of the new hour's final gain was available at {extra['minutes_to_90pct_new_gain']:.2f} minutes; "
              f"99% at {extra['minutes_to_99pct_new_gain']:.2f} minutes.", "",
              "| Proposal | Completed attempts | Best final weighted error |",
              "| --- | ---: | ---: |"]
    for kind, count in sorted(extra["proposal_counts"].items()):
        minimum = min(a["final_error"] for a in attempts if a["proposal"] == kind)
        lines.append(f"| {kind} | {count} | {minimum:.10f} |")
    if large:
        info = extra["large_flip_results"]
        lines += ["", "## User-requested switch to 5% sign flips", "",
                  f"At {info['amendment_at_minutes']:.2f} minutes, the search resumed with 5% uniform flips "
                  "and up to 64 refinement passes per proposal, with at least 12 before pruning for slow "
                  "recovery. The original deadline was preserved; the pause to implement and validate the "
                  "change counts against the same one-hour wall-clock budget. The original report and "
                  "runner were frozen and hash-verified before amendment.", "",
                  f"Completed {len(large)} larger proposals. They initially changed "
                  f"{info['initial_flip_count_min']:,}-{info['initial_flip_count_max']:,} signs. "
                  f"Their best final error was {info['minimum_final_error']:.10f}, versus "
                  f"{info['best_before_amendment']:.10f} immediately before the switch.", "",
                  f"After refinement, the number of signs differing from each proposal's parent ranged from "
                  f"{info['final_raw_sign_distance_min']:,} to {info['final_raw_sign_distance_max']:,}, "
                  f"with median {info['final_raw_sign_distance_median']:,.0f}. These are raw corresponding-entry "
                  "distances, not a measure of independent basins or a gauge-invariant operator distance."]
        if large_audit:
            lines += ["", f"The independent float64 audit measured error "
                      f"{large_audit['metrics']['weighted_error']:.10f} for the best completed 5% proposal, "
                      f"{large_audit['energy_gap_above_overall_best_percent']:.5f}% above the overall best. "
                      f"Its fixed-scale one-bit audit found {large_audit['neighborhood']['improving_bits']} "
                      "improving flips. No factor was changed by this audit."]
    lines += ["", "## Local-minimum audit", "",
              "Independent float64 reconstruction was used for all reported comparison scores. The read-only "
              "audit examines every individual sign flip with all scales and the other factor fixed, "
              "using absolute energy tolerance 1e-10. It makes no changes outside the search budget.", "",
              "| Fit | Bits examined | Improving flips | Best one-flip energy change |",
              "| --- | ---: | ---: | ---: |"]
    for key, audit in report["neighborhood_audits"].items():
        lines.append(f"| {key} | {audit['bits_examined']:,} | {audit['improving_bits']:,} | "
                     f"{audit['best_single_flip_energy_change']:.12g} |")
    lines += ["", "A zero improving-flip count establishes only fixed-scale one-bit stationarity. It does "
              "not establish stationarity under scale refits, joint flips, or a global optimum. Nonzero "
              "counts mean that even this restricted neighborhood is not fully exhausted.", "",
              "## Artifacts and scope", "",
              f"Best checkpoint: `{source / 'best.safetensors'}`; SHA-256 `{report['best_sha256']}`. "
              f"All {len(records)} saved refinement states passed fresh hash checks. Numerical source "
              "hashes and preserved pre-amendment snapshots passed verification.", "",
              f"Peak CUDA allocator memory: {report['peak_allocated_bytes'] / 2**30:.3f} GiB allocated / "
              f"{report['peak_reserved_bytes'] / 2**30:.3f} GiB reserved. Saved refinement states occupy "
              f"{extra['saved_state_bytes'] / 2**30:.3f} GiB. Each factor checkpoint uses "
              f"{extra['best_state_bytes']:,} bytes with INT8 containers for signs; its equivalent packed "
              "factor payload remains 1,198,080 bytes, or 1.203704 BPW excluding unchanged outlier sidecars.", "",
              report.get("cuda_memory_scope", "CUDA allocator counters cover the uninterrupted search worker."), "",
              (f"Observed Windows process peak working set: "
               f"{extra['observed_host_peak_working_set_bytes'] / 2**30:.3f} GiB."
               if "observed_host_peak_working_set_bytes" in extra else "Host working set was not sampled."), "",
              "This inherits Experiment 066's hour and searches for one additional hour. The full matrix, "
              "rank 1152, seven outlier columns, and retained Experiment 041 Fisher objective are unchanged. "
              "The old 800-step baseline is hash-verified and rescored, not refitted with another seed. "
              "Experiment 060 calibration is absent; this is not its final tuned-model quality comparison. "
              "No held-out examples or perplexity evaluation were used, and no production policy changed.", "",
              "Protocol and command: [Docs/102-hour-basin-search.md](../../Docs/102-hour-basin-search.md).", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(source / "report.json", output / "search-report.json")
    shutil.copy2(source / "fixture.json", output / "fixture.json")
    if large_audit:
        shutil.copy2(large_audit_path, output / "large-flip-audit.json")
    (output / "validation.json").write_text(json.dumps({"status": "passed", "states_checked": len(records),
        "best_sha256": report["best_sha256"], "numerical_source_hashes_checked": True}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("evidence/067/basin-search"))
    parser.add_argument("--output", type=Path, default=Path("Results/067"))
    args = parser.parse_args()
    render(args.source.resolve(), args.output.resolve())
