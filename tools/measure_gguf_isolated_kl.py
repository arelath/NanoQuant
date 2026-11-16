"""Measure each GGUF projection's KL on its own, to split real quantization error from interaction.

``measure_gguf_kl.py`` shows that a whole quantized file costs less KL than the sum of the sweep's
Gaussian-noise predictions for its measured weight errors. That gap has two possible sources, and
this tool separates them by loading one dequantized projection at a time into the clean model:

- shape: one projection's real error costs less (or more) KL than Gaussian noise of the same
  relative size. Measured as the sum of isolated KLs over the sum of Gaussian predictions.
- interaction: real errors in different projections add up to less (or more) than their isolated
  sum. Measured as the joint KL over the sum of isolated KLs.

The product of the two factors is the file's measured / predicted ratio. The joint KL is also
measured for every block's seven projections and for every projection type's column of blocks,
which shows whether interaction happens inside blocks or across them.

Usage:
    .venv\\Scripts\\python.exe tools\\measure_gguf_isolated_kl.py ^
        Datasets\\<dataset>\\measurements\\<sweep-id>\\results.json --gguf A.gguf --gguf B.gguf
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from measure_gguf_kl import SweepSession, map_tensors, read_gguf, relative_error, swapped, sweep_session
from measure_joint_kl_targets import projection_weights

from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file


def groupings(targets: Sequence[str]) -> dict[str, dict[str, list[str]]]:
    """Return the projection sets scored jointly: everything, each block and each projection type."""
    blocks: dict[str, list[str]] = {}
    types: dict[str, list[str]] = {}
    for target in targets:
        block, projection = target.split(":", 1)
        blocks.setdefault(block, []).append(target)
        types.setdefault(projection, []).append(target)
    return {"all": {"all": list(targets)}, "block": blocks, "type": types}


def isolate(
    targets: Sequence[str], errors: Mapping[str, float], curves: Mapping, kl_of: Callable[[Sequence[str]], float],
    log=print,
) -> dict:
    """Score every projection alone and every group jointly; ``kl_of`` returns KL above control."""
    projections = {}
    for index, name in enumerate(targets, 1):
        curve, error = curves[name], errors[name]
        low, high = curve["measured_error_range"]
        projections[name] = {
            "error": error,
            "gaussian_prediction": curve["coefficient"] * error ** curve["exponent"],
            "within_measured_range": low <= error <= high,
            "kl_above_control": kl_of([name]),
        }
        if index % 21 == 0 or index == len(targets):
            log(f"  isolated {index}/{len(targets)}")
    groups: dict[str, dict] = {}
    for kind, members_by_label in groupings(targets).items():
        groups[kind] = {}
        for label, members in members_by_label.items():
            groups[kind][label] = _group(members, projections, kl_of(members))
    return {"projections": projections, "groups": groups}


def _group(members: Sequence[str], projections: Mapping[str, Mapping], joint: float) -> dict:
    isolated = math.fsum(projections[name]["kl_above_control"] for name in members)
    predicted = math.fsum(projections[name]["gaussian_prediction"] for name in members)
    return {
        "members": len(members),
        "gaussian_prediction": predicted,
        "sum_isolated_kl": isolated,
        "joint_kl": joint,
        "shape_factor": isolated / predicted,
        "interaction_factor": joint / isolated,
        "measured_over_predicted": joint / predicted,
    }


def measure_file(session: SweepSession, path: Path, log=print) -> dict:
    started = time.perf_counter()
    projections, _, _ = map_tensors(read_gguf(path), session.targets)
    weights = projection_weights(session.model, session.adapter, session.targets)
    errors = {name: relative_error(weights[name], projections[name]["values"]) for name in session.targets}
    clean = session.clean["kl_nats_per_token"]

    def kl_of(names: Sequence[str]) -> float:
        with swapped({name: (weights[name], projections[name]["values"]) for name in names}):
            return session.score()["kl_nats_per_token"] - clean

    log(f"{path.name}: {len(session.targets)} isolated swaps")
    result = isolate(session.targets, errors, session.curves, kl_of, log)
    everything = result["groups"]["all"]["all"]
    log(f"{path.name}: shape {everything['shape_factor']:.3f} × interaction {everything['interaction_factor']:.3f} "
        f"= {everything['measured_over_predicted']:.3f}")
    return {
        "file": path.name, "sha256": hash_file(path), **result, "seconds": time.perf_counter() - started,
    }


def _range(values: Sequence[float]) -> str:
    return f"{statistics.median(values):.2f} ({min(values):.2f}–{max(values):.2f})"


def render_markdown(report: Mapping) -> str:
    files = report["files"]
    clean = report["clean"]
    lines = [
        f"# GGUF isolated-projection KL: {report['protocol']['teacher']['source']}",
        "",
        f"Measured on {report['measured_at']} against the teacher of sweep `{report['measurement_sweep']}`.",
        "Each dequantized GGUF projection was loaded alone into the otherwise clean model and scored on the",
        f"sweep's {clean['token_count']:,} next-token positions. KL is full-vocabulary KL(teacher ‖ quantized)",
        "in nats/token above the clean model. The Gaussian prediction is the sweep's fitted KL for Gaussian",
        "noise at the projection's measured relative weight error.",
        "",
        "## Shape and interaction by file",
        "",
        "Shape is the sum of isolated KLs over the sum of Gaussian predictions: how much one projection's",
        "real error costs compared with Gaussian noise of the same size. Interaction is the joint KL over",
        "the sum of isolated KLs: below 1, the errors cost less together than apart. Their product is",
        "measured / predicted.",
        "",
        "| GGUF | Gaussian prediction (nats/token) | Sum of isolated KLs (nats/token) | "
        "Joint KL (nats/token) | Shape factor | Interaction factor | Measured / predicted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in files:
        row = item["groups"]["all"]["all"]
        lines.append(
            f"| {item['file']} | {row['gaussian_prediction']:.4f} | {row['sum_isolated_kl']:.4f} | "
            f"{row['joint_kl']:.4f} | {row['shape_factor']:.3f} | {row['interaction_factor']:.3f} | "
            f"{row['measured_over_predicted']:.3f} |"
        )
    lines += [
        "",
        "## Isolated KL over Gaussian prediction, per projection",
        "",
        "Median (range) of each projection's isolated KL divided by its own Gaussian prediction.",
        "Projections whose error falls outside the sweep's measured range use an extrapolated fit.",
        "",
        "| GGUF | All projections | Extrapolated projections | "
        + " | ".join(report["projection_types"]) + " |",
        "| --- | ---: | ---: | " + " | ".join("---:" for _ in report["projection_types"]) + " |",
    ]
    for item in files:
        ratios = {name: row["kl_above_control"] / row["gaussian_prediction"]
                  for name, row in item["projections"].items()}
        outside = sum(not row["within_measured_range"] for row in item["projections"].values())
        cells = [
            f"{statistics.median(v for name, v in ratios.items() if name.split(':', 1)[1] == kind):.2f}"
            for kind in report["projection_types"]
        ]
        lines.append(f"| {item['file']} | {_range(list(ratios.values()))} | {outside} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Interaction inside blocks and across blocks",
        "",
        "Joint KL over the sum of isolated KLs. A block group holds one block's projections; a type",
        "group holds one projection type across every block.",
        "",
        "| GGUF | Whole file | Block groups, median (range) | Type groups, median (range) | "
        + " | ".join(report["projection_types"]) + " |",
        "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in report["projection_types"]) + " |",
    ]
    for item in files:
        groups = item["groups"]
        types = {label: row["interaction_factor"] for label, row in groups["type"].items()}
        lines.append(
            f"| {item['file']} | {groups['all']['all']['interaction_factor']:.3f} | "
            f"{_range([row['interaction_factor'] for row in groups['block'].values()])} | "
            f"{_range(list(types.values()))} | "
            + " | ".join(f"{types[kind]:.2f}" for kind in report["projection_types"]) + " |"
        )
    lines += ["", f"Full per-projection and per-group values: `{report['json']}`.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("measurements", type=Path, help="a completed projection sweep's measurements/<id>/results.json")
    parser.add_argument("--gguf", type=Path, action="append", required=True, help="GGUF file; repeatable")
    parser.add_argument("--output", type=Path, help="Markdown report (default gguf-isolated-kl.md beside the sweep)")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    measurements = args.measurements.resolve()
    output = (args.output or measurements.with_name("gguf-isolated-kl.md")).resolve()
    data = output.with_suffix(".json")
    done = {}
    if data.exists():
        prior_report = json.loads(data.read_text(encoding="utf-8"))
        if prior_report["measurement_sha256"] == hash_file(measurements):
            done = {item["sha256"]: item for item in prior_report["files"]}
    with sweep_session(measurements, local_files_only=args.local_files_only) as session:
        report = {
            "schema_version": 1,
            "measured_at": time.strftime("%Y-%m-%d %H:%M"),
            "measurement_sweep": measurements.parent.name,
            "measurement_sha256": hash_file(measurements),
            "protocol": session.protocol,
            "clean": {key: value for key, value in session.clean.items() if key != "sequences"},
            "projection_types": list(dict.fromkeys(target.split(":", 1)[1] for target in session.targets)),
            "files": [],
            "json": str(data),
        }
        for path in args.gguf:
            path = path.resolve()
            prior = done.get(hash_file(path))
            if prior is not None:
                print(f"{path.name}: reusing the measurement already in {data.name}", flush=True)
            report["files"].append(prior or measure_file(session, path, lambda text: print(text, flush=True)))
            atomic_write_json(data, report)
            atomic_write_text(output, render_markdown(report))
    print(f"Report written to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
