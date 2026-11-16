"""Measure the KL of random noise whose size follows each small block of weights.

The sensitivity sweep adds Gaussian noise of one variance to every weight of a projection. A block
quantizer's rounding error is different in one simple way: each run of consecutive weights in a row
gets its own scale, so error is small where the weights are small. This tool isolates that one
difference. It draws uniform noise scaled by the largest magnitude in each block, rescales the whole
projection to the requested relative Frobenius error, and scores it like the sweep does.

Comparing the result with the sweep's Gaussian prediction at the same error shows how much of the
gap between real quantizers and Gaussian noise comes from block scaling alone, with no calibration
data and no rounding grid. A joint run perturbs every projection at once with independent noise.

Usage:
    .venv\\Scripts\\python.exe tools\\measure_block_scaled_noise_kl.py ^
        Datasets\\<dataset>\\measurements\\<sweep-id>\\results.json --level 0.05
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Mapping
from pathlib import Path

import torch
from measure_gguf_kl import swapped, sweep_session
from measure_joint_kl_targets import projection_weights

from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file


def block_scaled_noise(weight: torch.Tensor, level: float, block: int, generator: torch.Generator) -> torch.Tensor:
    """Return uniform noise scaled by each row block's largest magnitude, at ``level`` relative error."""
    rows, columns = weight.shape
    if columns % block:
        raise ValueError(f"{columns} input features do not divide into blocks of {block}")
    amplitude = weight.abs().reshape(rows, -1, block).amax(dim=-1, keepdim=True)
    noise = (torch.rand(rows, columns // block, block, generator=generator) * 2 - 1) * amplitude
    noise = noise.reshape(rows, columns)
    return noise * (level * torch.linalg.vector_norm(weight) / torch.linalg.vector_norm(noise))


def render_markdown(report: Mapping) -> str:
    rows = report["projections"]
    ratios = {name: row["kl_above_control"] / row["gaussian_prediction"] for name, row in rows.items()}
    joint = report["joint"]
    lines = [
        f"# Block-scaled noise KL: {report['protocol']['teacher']['source']}",
        "",
        f"Measured on {report['measured_at']} against the teacher of sweep `{report['measurement_sweep']}`.",
        f"Each projection received uniform noise scaled by the largest magnitude in every run of "
        f"{report['block']} consecutive weights in a row, at {report['level']:.1%} relative Frobenius error. "
        "KL is in nats/token",
        "above the clean model; the Gaussian prediction is the sweep's fitted KL at the same error.",
        "",
        "| Projection type | Median block-scaled KL / Gaussian prediction | Range |",
        "| --- | ---: | ---: |",
    ]
    for kind in report["projection_types"]:
        values = [ratio for name, ratio in ratios.items() if name.split(":", 1)[1] == kind]
        lines.append(f"| {kind} | {statistics.median(values):.2f} | {min(values):.2f}–{max(values):.2f} |")
    values = list(ratios.values())
    lines += [
        f"| all | {statistics.median(values):.2f} | {min(values):.2f}–{max(values):.2f} |",
        "",
        f"Sum of isolated block-scaled KLs: {report['sum_isolated_kl']:.4f} nats/token against a Gaussian",
        f"prediction of {report['gaussian_prediction']:.4f} nats/token "
        f"(ratio {report['sum_isolated_kl'] / report['gaussian_prediction']:.3f}).",
        "",
        f"Every projection perturbed at once with independent noise, {len(joint)} draws: joint KL / sum of",
        "isolated KLs = " + ", ".join(f"{run['joint_over_isolated']:.3f}" for run in joint) + ".",
        "",
        f"Full per-projection values: `{report['json']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("measurements", type=Path, help="a completed projection sweep's measurements/<id>/results.json")
    parser.add_argument("--level", type=float, default=0.05, help="relative Frobenius error per projection")
    parser.add_argument("--block", type=int, default=32, help="consecutive weights in a row sharing one scale")
    parser.add_argument("--joint-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Markdown report (default block-scaled-noise-kl.md)")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    measurements = args.measurements.resolve()
    output = (args.output or measurements.with_name("block-scaled-noise-kl.md")).resolve()
    with sweep_session(measurements, local_files_only=args.local_files_only) as session:
        weights = projection_weights(session.model, session.adapter, session.targets)
        clean = session.clean["kl_nats_per_token"]

        def perturbed(name: str, seed: int) -> torch.Tensor:
            weight = weights[name].detach().float().cpu()
            generator = torch.Generator().manual_seed(seed * 1000 + session.targets.index(name))
            return weight + block_scaled_noise(weight, args.level, args.block, generator)

        def kl_of(names: list[str], seed: int) -> float:
            with swapped({name: (weights[name], perturbed(name, seed)) for name in names}):
                return session.score()["kl_nats_per_token"] - clean

        projections = {}
        for index, name in enumerate(session.targets, 1):
            curve = session.curves[name]
            projections[name] = {
                "gaussian_prediction": curve["coefficient"] * args.level ** curve["exponent"],
                "kl_above_control": kl_of([name], args.seed),
            }
            if index % 21 == 0:
                print(f"isolated {index}/{len(session.targets)}", flush=True)
        isolated = math.fsum(row["kl_above_control"] for row in projections.values())
        joint = []
        for run in range(args.joint_runs):
            value = kl_of(session.targets, args.seed + run)
            joint.append({"seed": args.seed + run, "joint_kl": value, "joint_over_isolated": value / isolated})
            print(f"joint seed {args.seed + run}: {value:.4f} nats/token, {value / isolated:.3f} of isolated sum",
                  flush=True)
        report = {
            "schema_version": 1,
            "measured_at": time.strftime("%Y-%m-%d %H:%M"),
            "measurement_sweep": measurements.parent.name,
            "measurement_sha256": hash_file(measurements),
            "protocol": session.protocol,
            "level": args.level,
            "block": args.block,
            "seed": args.seed,
            "projection_types": list(dict.fromkeys(target.split(":", 1)[1] for target in session.targets)),
            "projections": projections,
            "sum_isolated_kl": isolated,
            "gaussian_prediction": math.fsum(row["gaussian_prediction"] for row in projections.values()),
            "joint": joint,
            "json": str(output.with_suffix(".json")),
        }
    atomic_write_json(output.with_suffix(".json"), report)
    atomic_write_text(output, render_markdown(report))
    print(f"Report written to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
