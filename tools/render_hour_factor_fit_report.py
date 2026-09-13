"""Render completed Experiment 066 evidence (matplotlib==3.10.3)."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source: Path, output: Path) -> None:
    report = json.loads((source / "report.json").read_text())
    fixture = json.loads((source / "fixture.json").read_text())
    if report["status"] != "complete":
        raise ValueError("a completed search is required")
    for row in report["records"]:
        if sha256(source / row["state_file"]) != row["state_sha256"]:
            raise ValueError(f"checkpoint mismatch: {row['name']}")
    if sha256(source / "best.safetensors") != report["best_sha256"]:
        raise ValueError("best checkpoint mismatch")
    output.mkdir(parents=True, exist_ok=True)
    baseline = report["comparison"]["baseline"]
    best = report["comparison"]["best"]
    berror = baseline["weighted_error"]
    controls = [
        row for row in report["records"] if row.get("mode") == "bf16-ieee" and row.get("outer_iterations") == 800
    ]
    local = min(
        (row for row in report["records"] if row.get("family") == "local-search"),
        key=lambda row: row["weighted_error"],
    )
    joint = [row for row in report["records"] if row.get("family") == "joint-admm"]
    best_control = min(controls, key=lambda row: row["weighted_error"])
    extra = {
        "control_seed_count": len(controls),
        "control_error_mean": statistics.mean(row["weighted_error"] for row in controls),
        "control_error_min": best_control["weighted_error"],
        "control_error_max": max(row["weighted_error"] for row in controls),
        "best_vs_best_control_energy_reduction_percent": 100
        * (1 - best["weighted_error"] / best_control["weighted_error"]),
        "complete_joint_fits_in_search": len(joint),
        "complete_joint_iterations_in_search": sum(row["observed_outer_iterations"] for row in joint),
        "saved_factor_states": len(report["records"]),
        "checkpoint_bytes": sum((source / row["state_file"]).stat().st_size for row in report["records"]),
        "best_checkpoint_bytes": (source / "best.safetensors").stat().st_size,
    }
    for fraction in (0.9, 0.99):
        threshold = berror - fraction * (berror - best["weighted_error"])
        first = min(row["completed_utc"] for row in report["records"] if row["weighted_error"] <= threshold)
        extra[f"minutes_to_{int(fraction * 100)}pct_final_gain"] = max(0.0, (first - report["search_started_utc"]) / 60)
    host_path = source / "host-memory-samples.jsonl"
    host_summary = "not sampled"
    if host_path.exists():
        extra["observed_host_peak_working_set_bytes"] = max(
            json.loads(line)["peak_working_set_bytes"] for line in host_path.read_text().splitlines() if line
        )
        host_summary = (
            f"{extra['observed_host_peak_working_set_bytes'] / 2**30:.3f} GiB "
            "(Windows process peak observed during the search)"
        )
    provenance = source / "code-provenance.json"
    if provenance.exists():
        code = json.loads(provenance.read_text())
        for name, expected in code["sha256"].items():
            if sha256(Path(name)) != expected:
                raise ValueError(f"numerical code changed during search: {name}")
    (output / "summary.json").write_text(json.dumps({**report["comparison"], **extra}, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    xs, ys, incumbent = [0.0], [0.0], berror
    for row in sorted(report["records"], key=lambda row: row["completed_utc"]):
        elapsed = (row["completed_utc"] - report["search_started_utc"]) / 60
        if elapsed < 0:
            continue
        incumbent = min(incumbent, row["weighted_error"])
        xs.append(elapsed)
        ys.append(100 * (1 - incumbent / berror))
    xs.append(report["search_wall_seconds"] / 60)
    ys.append(ys[-1])
    axes[0].step(xs, ys, where="post", color="#146b67", linewidth=2.3)
    axes[0].set(
        xlabel="Additional search time (minutes)",
        ylabel="Error energy reduction vs current fit (%)",
        title="Best completed fit within the hour",
        xlim=(0, 60),
    )
    axes[0].grid(axis="y", alpha=0.2)
    colors = {"bf16-tf32": "#c16829", "fp32-tf32": "#146b67", "bf16-ieee": "#6c6c76"}
    for mode in sorted({row["mode"] for row in joint}):
        points = sorted(
            (row for row in joint if row["mode"] == mode and row["seed"] == 0),
            key=lambda row: row["outer_iterations"],
        )
        if points:
            axes[1].plot(
                [row["outer_iterations"] for row in points],
                [row["weighted_nrmse"] for row in points],
                "o-",
                label=mode,
                color=colors.get(mode),
                linewidth=1.8,
            )
    axes[1].axhline(baseline["weighted_nrmse"], color="#6c6c76", linestyle="--", label="Current 800-step fit")
    axes[1].set(
        xscale="log",
        xlabel="ADMM outer iterations (seed 0)",
        ylabel="Weighted NRMSE (lower is better)",
        title="Joint-fit duration and arithmetic",
    )
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Gemma 3 1B · block 12 MLP down · rank 1152 · fixed Fisher objective", fontsize=13)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"fit-search.{suffix}", dpi=180)
    plt.close(fig)
    rows, columns = fixture["shape"]
    rank = fixture["rank"]
    logical_bits = rank * (rows + columns) + 32 * (rows + columns + rank)
    lines = [
        "# Experiment 066: one-hour binary-factor fit",
        "",
        "The best fit reduces weighted residual error energy by "
        f"**{report['comparison']['error_reduction_percent']:.3f}%** "
        f"and weighted NRMSE by **{report['comparison']['nrmse_reduction_percent']:.3f}%** "
        "versus the fresh current-method fit. "
        f"The search used **{report['search_wall_seconds'] / 60:.2f} minutes** after a "
        f"{report['baseline']['wall_seconds']:.2f}-second baseline and speed pilots.",
        "",
        f"90% of the final error reduction was first available after "
        f"{extra['minutes_to_90pct_final_gain']:.2f} minutes; 99% after "
        f"{extra['minutes_to_99pct_final_gain']:.2f} minutes.",
        "",
        "![Search trajectory and matched horizon results](fit-search.svg)",
        "",
        "## Matched reconstruction comparison",
        "",
        "| Method | Weighted error energy | Weighted NRMSE | Raw NRMSE | Energy reduction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, row in [
        ("Current method, seed 0", baseline),
        ("Best of 3 current-method seeds", best_control),
        ("Current seed 0 + local search", local),
        ("Best fit found within the hour", best),
    ]:
        change = 100 * (1 - row["weighted_error"] / berror)
        if abs(change) < 0.00005:
            change = 0.0
        lines.append(
            f"| {label} | {row['weighted_error']:.9f} | {row['weighted_nrmse']:.9f} | "
            f"{row['raw_nrmse']:.9f} | {change:.4f}% |"
        )
    lines.extend(
        [
            "",
            f"The three current-method seeds span error energies {extra['control_error_min']:.9f}–"
            f"{extra['control_error_max']:.9f}, mean {extra['control_error_mean']:.9f}. The winner also beats the "
            f"best control seed by {extra['best_vs_best_control_energy_reduction_percent']:.3f}% error energy. "
            "This seed range is descriptive; it is not a confidence interval across layers or datasets.",
            "",
            f"Winner: `{report['incumbent']['name']}`. {len(joint)} complete joint fits consumed "
            f"{extra['complete_joint_iterations_in_search']:,} ADMM outer iterations inside the search. "
            "Five power iterations were used per sign projection throughout. The incumbent includes completed "
            "local refinements; unfinished fits cannot win.",
            "",
            "## Seed-zero joint-fit curve",
            "",
            "Every row uses the same target/rank/objective and two IEEE scale passes. Equal numerical seed across "
            "BF16 and FP32 does not guarantee identical random initial matrices, so these are arithmetic/trajectory "
            "alternatives, not an isolated causal test of rounding alone.",
            "",
            "| Mode | Outer iterations | ADMM seconds | Weighted NRMSE | Energy reduction |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in joint:
        if row["seed"] == 0:
            lines.append(
                f"| {row['mode']} | {row['outer_iterations']:,} | {row['admm_seconds']:.2f} | "
                f"{row['weighted_nrmse']:.9f} | {100 * (1 - row['weighted_error'] / berror):.4f}% |"
            )
    lines.extend(
        [
            "",
            "## Speed pilot",
            "",
            "| Arithmetic | Seconds / 64 ADMM iterations | Weighted NRMSE |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in report["pilots"]:
        lines.append(f"| {row['mode']} | {row['admm_seconds']:.4f} | {row['weighted_nrmse']:.9f} |")
    lines.extend(
        [
            "",
            f"Selected mode: `{report['fastest_mode']}`. Warmups precede each measured pilot. "
            "All pilots are rescored with the same IEEE arithmetic; FP32 `medium` and `high` happened to produce "
            "identical pilot scores in this environment. The full suite briefly competed for CPU at the beginning "
            "of the hour and was stopped; that time remains charged to the search.",
            "",
            "## Artifact and audit",
            "",
            f"- Best artifact: `{source / 'best.safetensors'}`.",
            f"- SHA-256: `{report['best_sha256']}`.",
            f"- Actual checkpoint bytes: {extra['best_checkpoint_bytes']:,}; sign tensors use INT8 containers.",
            f"- Equivalent bit-packed factor payload: {logical_bits // 8:,} bytes using 1-bit signs and FP32 scales, "
            f"{logical_bits / (rows * columns):.6f} BPW for the factor part; fixed outlier sidecars excluded.",
            "- Baseline and winner have identical dimensions, rank, sign alphabet, scale count, and storage precision.",
            f"- Peak CUDA allocated/reserved: {report['peak_allocated_bytes'] / 2**30:.3f} / "
            f"{report['peak_reserved_bytes'] / 2**30:.3f} GiB; these are fitting workspace, not deployment VRAM.",
            f"- Host peak working set: {host_summary}.",
            f"- Every saved factor checkpoint passed a fresh SHA-256 audit. The final FP64 matrix reconstruction "
            "changed the winner's scored energy by relative "
            f"{report['comparison']['fp64_audit_relative_error_delta']:.3g}.",
            f"- {extra['saved_factor_states']} saved states total {extra['checkpoint_bytes']:,} bytes.",
            "",
            "## Scope",
            "",
            "This is the full 1152×6912 pinned Gemma block-12 down-projection at rank 1152. All arms use the "
            "same retained Experiment 041 Fisher residual objective and seven fixed removed columns. The fresh "
            "control replays today's 800×5 BF16/FP32 ADMM and two-pass scale fit. Experiment 060's calibration "
            "artifacts are missing, so its final weight-error report is not a matched numerical baseline here.",
            "",
            "The result measures attainable single-matrix fit under this search budget. It is not a proof of global "
            "optimality, a prediction of perplexity improvement, or a completed model-quality/parity gate. Prior "
            "studies found that lower diagonal reconstruction error does not reliably translate into better "
            "held-out behavior. No production policy is changed by this screen.",
            "",
            "Reproduce with `python tools/probe_hour_factor_fit.py`; render with "
            "`python tools/render_hour_factor_fit_report.py`. Plotting uses `matplotlib==3.10.3`.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({**report["comparison"], **extra}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("evidence/066/hour-fit"))
    parser.add_argument("--output", type=Path, default=Path("Results/066"))
    args = parser.parse_args()
    render(args.source.resolve(), args.output.resolve())
