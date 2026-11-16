"""Render the figures for the quantization-error-and-KL article from a completed sweep.

Reads a projection sweep's ``results.json`` and the ``joint-kl-targets*.json`` and ``gguf-kl.json``
reports beside it. Writes the sweep's own sensitivity and equal-KL charts plus the article
figures: joint additivity, where each quantized file puts its weight error, measured KL against
the additive prediction and, when ``gguf-isolated-kl.json`` exists, how each file's saving splits
into per-matrix error shape and interaction between matrices. A
``gguf-isolated-kl-imatrix-ablation.json`` report adds the same split for files quantized with and
without calibration data, and ``gguf-kl-imatrix-ablation.json`` adds a plain bar chart of their KL.

Usage:
    .venv\\Scripts\\python.exe tools\\plot_kl_article_figures.py ^
        Datasets\\<dataset>\\measurements\\<sweep-id>\\results.json --output <figure directory>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from fit_kl_sensitivity import fit_dataset, plot_measurements, plot_targets, require_plotting

ROWS = {
    "self_attn.q_proj": "Attn q",
    "self_attn.k_proj": "Attn k",
    "self_attn.v_proj": "Attn v",
    "self_attn.o_proj": "Attn o",
    "mlp.gate_proj": "MLP gate",
    "mlp.up_proj": "MLP up",
    "mlp.down_proj": "MLP down",
}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def short_name(file: str) -> str:
    name = file.removesuffix(".gguf")
    return name.split("-it-", 1)[1] if "-it-" in name else name


def display_name(file: str) -> str:
    name = short_name(file)
    for suffix, label in (("-noimatrix", ", no calibration"), ("-imatrix", ", calibrated")):
        if name.endswith(suffix):
            return name.removesuffix(suffix) + label
    return {"nanoquant": "Binary factors", "nanoquant-pre-kd": "Binary factors, before distillation"}.get(name, name)


def error_grid(errors: Mapping[str, float]) -> np.ndarray:
    """Arrange ``block:projection`` errors as a projection-by-block grid in ``ROWS`` order."""
    blocks = sorted({int(name.split(":", 1)[0]) for name in errors})
    grid = np.full((len(ROWS), len(blocks)), np.nan)
    for name, error in errors.items():
        block, projection = name.split(":", 1)
        grid[list(ROWS).index(projection), blocks.index(int(block))] = error
    return grid


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def plot_additivity(joint: Mapping, output: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = [row for row in joint["results"] if row["kl_target"] > 0]
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    generator = np.random.default_rng(0)
    for index, row in enumerate(rows):
        ratios = [run["measured_over_additive"] for run in row["runs"]]
        ax.scatter(index + generator.uniform(-0.16, 0.16, len(ratios)), ratios, s=22, color=BLUE, alpha=0.35,
                   linewidths=0)
        mean = row["mean_measured_over_additive"]
        low, high = row["ratio_confidence_interval_95"]
        ax.errorbar(index + 0.27, mean, yerr=[[mean - low], [high - mean]], fmt="o", color=INK, markersize=6,
                    capsize=4, linewidth=1.5)
        ax.annotate(f"mean {mean:.3f}\n95% CI\n{low:.3f}–{high:.3f}", (index + 0.27, mean), xytext=(10, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9, color=INK)
    ax.axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("perfectly additive", (-0.42, 1), xytext=(0, 3), textcoords="offset points", fontsize=9, color=MUTED)
    ax.set_xticks(range(len(rows)), [
        f"{row['kl_target']:g} nats/token per projection\nadditive prediction {row['mean_additive_prediction']:.3f}"
        for row in rows
    ])
    ax.set_xlim(-0.45, len(rows) - 0.35)
    ax.set_ylabel("Measured joint KL / additive prediction", color=MUTED)
    ax.set_title(
        f"All {rows[0]['projection_count']} projections perturbed at once: joint KL lands on the sum of the parts",
        loc="left", fontsize=12, color=INK, fontweight="bold",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    path = output / "joint-additivity.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def plot_error_maps(report: Mapping, output: Path, skip: tuple[str, ...] = ("nanoquant-pre-kd",)) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter

    files = [item for item in report["files"] if short_name(item["file"]) not in skip]
    files.sort(key=lambda item: item["projection_bits_per_weight"])
    low = min(min(item["projection_errors"].values()) for item in files)
    high = max(max(item["projection_errors"].values()) for item in files)
    norm = LogNorm(vmin=low, vmax=high)
    height = -(-len(files) // 2)
    fig, axes = plt.subplots(height, 2, figsize=(12, 2.9 * height), layout="constrained", squeeze=False)
    for ax, item in zip(axes.flat, files, strict=False):
        grid = error_grid(item["projection_errors"])
        image = ax.imshow(grid, aspect="auto", interpolation="nearest", cmap="Blues", norm=norm)
        ax.set_title(
            f"{display_name(item['file'])} · {item['projection_bits_per_weight']:.2f} bits/weight · "
            f"KL {item['variants']['projections']['kl_above_control']:.3f} nats/token",
            loc="left", fontsize=10, color=INK,
        )
        ax.set_xticks(range(grid.shape[1]), range(grid.shape[1]), fontsize=7)
        ax.set_yticks(range(len(ROWS)), ROWS.values(), fontsize=8)
        ax.tick_params(colors=MUTED, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in list(axes.flat)[len(files):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Decoder block (zero-based)", color=MUTED)
    fig.suptitle("Where each quantized file puts its weight error", x=0.01, ha="left", fontsize=13, color=INK,
                 fontweight="bold")
    bar = fig.colorbar(image, ax=axes, label="Relative weight error (log scale)", pad=0.01,
                       format=FuncFormatter(lambda value, _: f"{value:.0%}" if value >= 0.01 else f"{value:.1%}"))
    bar.set_ticks([tick for tick in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4) if low <= tick <= high])
    bar.outline.set_visible(False)
    path = output / "quantizer-weight-error-maps.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def plot_predicted_vs_measured(report: Mapping, joint: Mapping, output: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6.4), layout="constrained")
    ax.plot([0.003, 60], [0.003, 60], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("measured = predicted", (0.012, 0.012), xytext=(4, -12), textcoords="offset points", fontsize=9,
                color=MUTED, rotation=38)
    gaussian = [(row["mean_additive_prediction"], row["mean_kl_above_control"])
                for row in joint["results"] if row["kl_target"] > 0]
    ax.scatter(*zip(*gaussian, strict=True), s=60, color=BLUE, label="Gaussian noise, every projection at once",
               zorder=3, edgecolors="white", linewidths=1.5)
    ax.annotate("Gaussian noise\n0.98–0.99×", gaussian[0], xytext=(-10, 6), textcoords="offset points", ha="right",
                fontsize=9, color=INK)
    series = {
        ORANGE: ("llama.cpp quantizations", [f for f in report["files"] if "nanoquant" not in f["file"]]),
        AQUA: ("Binary-factor format (experimental)", [f for f in report["files"] if "nanoquant" in f["file"]]),
    }
    for color, (label, files) in series.items():
        points = [(item["additive_prediction"], item["variants"]["projections"]["kl_above_control"]) for item in files]
        ax.scatter(*zip(*points, strict=True), s=60, color=color, label=label, zorder=3, edgecolors="white",
                   linewidths=1.5)
        for item, (x, y) in zip(files, points, strict=True):
            name = short_name(item["file"])
            offset, align = {
                "nanoquant": ((9, -13), "left"), "nanoquant-pre-kd": ((0, 9), "center"),
                "UD-IQ2_XXS": ((-9, -3), "right"),
            }.get(name, ((8, -4), "left"))
            name = {"nanoquant": "after distillation", "nanoquant-pre-kd": "before distillation"}.get(name, name)
            ax.annotate(f"{name}  {y / x:.2f}×", (x, y), xytext=offset, textcoords="offset points", ha=align,
                        fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.003, 60)
    ax.set_ylim(0.003, 20)
    ax.set_xlabel("Additive prediction from Gaussian-noise sensitivity (nats/token)", color=MUTED)
    ax.set_ylabel("Measured KL, projections only (nats/token)", color=MUTED)
    ax.set_title("Real quantization error costs less KL than Gaussian noise of the same size", loc="left",
                 fontsize=12, color=INK, fontweight="bold")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    _style(ax)
    path = output / "predicted-vs-measured-kl.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def decomposition_rows(
    isolated: Mapping, joint: Mapping, block_noise: list[Mapping], *, report_order: bool = False,
) -> list[dict]:
    """Return one row per noise model or file with its shape and interaction factors.

    Files are listed most bits first, or in the report's own order with each uncalibrated file
    ahead of its calibrated twin when ``report_order`` is set.
    """
    ratios = [row["mean_measured_over_additive"] for row in joint["results"] if row["kl_target"] > 0]
    rows = [{"label": "Gaussian noise", "shape": 1.0, "interaction": float(np.mean(ratios))}]
    if block_noise:
        rows.append({
            "label": "Block-scaled random noise",
            "shape": float(np.mean([r["sum_isolated_kl"] / r["gaussian_prediction"] for r in block_noise])),
            # Only each report's first joint run shares its noise with the isolated measurements.
            "interaction": float(np.mean([r["joint"][0]["joint_over_isolated"] for r in block_noise])),
        })
    files = list(reversed(isolated["files"]))
    if report_order:
        names = [short_name(item["file"]).rsplit("-", 1)[0] for item in isolated["files"]]
        files = sorted(isolated["files"], key=lambda item: (
            names.index(short_name(item["file"]).rsplit("-", 1)[0]), not item["file"].endswith("-noimatrix.gguf"),
        ))
    for item in files:
        everything = item["groups"]["all"]["all"]
        label = display_name(item["file"])
        label = "Binary factors, after distillation" if label == "Binary factors" else label
        rows.append({"label": label, "shape": everything["shape_factor"],
                     "interaction": everything["interaction_factor"]})
    return rows


def plot_decomposition(
    rows: list[dict], output: Path, *, name: str = "shape-and-interaction.png",
    title: str = "Where the predicted damage goes: most of the saving is inside each matrix",
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 0.56 * len(rows) + 1.9), layout="constrained")
    positions = np.arange(len(rows))[::-1]
    parts = {
        "KL the weights actually cost": (BLUE, [row["shape"] * row["interaction"] for row in rows]),
        "Saved: the errors cost less together than apart": (
            AQUA, [row["shape"] * (1 - row["interaction"]) for row in rows]),
        "Saved: each matrix's error costs less than Gaussian noise": (ORANGE, [1 - row["shape"] for row in rows]),
    }
    left = np.zeros(len(rows))
    for label, (color, values) in parts.items():
        widths = np.clip(values, 0, None)
        ax.barh(positions, widths, left=left, height=0.62, color=color, label=label, edgecolor="white", linewidth=2)
        left += widths
    for position, row in zip(positions, rows, strict=True):
        ax.annotate(
            f"{row['shape']:.2f} × {row['interaction']:.2f} = {row['shape'] * row['interaction']:.2f}",
            (1, position), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK,
        )
    ax.annotate("per-matrix × interaction", (1, positions[0] + 0.62), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=MUTED)
    ax.set_yticks(positions, [row["label"] for row in rows], fontsize=9)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of the KL predicted from Gaussian-noise sensitivity", color=MUTED)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.legend(loc="outside lower left", ncols=1, frameon=False, fontsize=9, labelcolor=INK)
    _style(ax)
    ax.spines["left"].set_visible(False)
    path = output / name
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def calibration_pairs(report: Mapping) -> list[tuple[str, float, float]]:
    """Return (format, calibrated KL, uncalibrated KL) for each format quantized both ways."""
    kl = {short_name(item["file"]): item["variants"]["projections"]["kl_above_control"] for item in report["files"]}
    formats = dict.fromkeys(name.rsplit("-", 1)[0] for name in kl if name.endswith(("-imatrix", "-noimatrix")))
    return [(name, kl[f"{name}-imatrix"], kl[f"{name}-noimatrix"]) for name in formats]


def plot_calibration_bars(report: Mapping, output: Path) -> Path:
    import matplotlib.pyplot as plt

    pairs = calibration_pairs(report)
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    positions = np.arange(len(pairs))
    series = (("Rounded with calibration", BLUE, -0.2, [pair[1] for pair in pairs]),
              ("Rounded without calibration", ORANGE, 0.2, [pair[2] for pair in pairs]))
    for label, color, offset, values in series:
        bars = ax.bar(positions + offset, values, width=0.36, color=color, label=label, edgecolor="white",
                      linewidth=2)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=9, color=INK)
    for position, (_, calibrated, plain) in zip(positions, pairs, strict=True):
        ax.annotate(f"{plain / calibrated - 1:+.0%}", (position + 0.2, plain), xytext=(0, 18),
                    textcoords="offset points", ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(positions, [f"{name} recipe" for name, _, _ in pairs])
    ax.set_ylim(0, max(pair[2] for pair in pairs) * 1.2)
    ax.set_ylabel("Drift from the original model (KL, nats/token)", color=MUTED)
    ax.set_title("Same file size, same rounding error: calibration cuts the drift", loc="left", fontsize=12,
                 color=INK, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    _style(ax)
    path = output / "calibration-drift-bars.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("measurements", type=Path, help="a completed projection sweep's measurements/<id>/results.json")
    parser.add_argument("--output", type=Path, required=True, help="directory for the PNG figures")
    parser.add_argument("--joint-report", type=Path, help="joint JSON (default joint-kl-targets-100-runs.json)")
    parser.add_argument("--gguf-report", type=Path, help="GGUF KL JSON (default gguf-kl.json)")
    parser.add_argument("--isolated-report", type=Path, help="default gguf-isolated-kl.json, skipped if absent")
    args = parser.parse_args()
    measurements = args.measurements.resolve()
    require_plotting()
    args.output.mkdir(parents=True, exist_ok=True)
    joint_path = args.joint_report or measurements.with_name("joint-kl-targets-100-runs.json")
    joint = json.loads(joint_path.read_text(encoding="utf-8"))
    gguf = json.loads((args.gguf_report or measurements.with_name("gguf-kl.json")).read_text(encoding="utf-8"))
    plot_measurements(json.loads(measurements.read_text(encoding="utf-8")), args.output)
    paths = [
        args.output / "kl-sensitivity-heatmaps.png", args.output / "kl-error-response.png",
        *plot_targets(fit_dataset(measurements), args.output),
        plot_additivity(joint, args.output), plot_error_maps(gguf, args.output),
        plot_predicted_vs_measured(gguf, joint, args.output),
    ]
    isolated_path = args.isolated_report or measurements.with_name("gguf-isolated-kl.json")
    if isolated_path.exists():
        block_noise = [json.loads(path.read_text(encoding="utf-8"))
                       for path in sorted(measurements.parent.glob("block-scaled-noise-kl*.json"))]
        rows = decomposition_rows(json.loads(isolated_path.read_text(encoding="utf-8")), joint, block_noise)
        paths.append(plot_decomposition(rows, args.output))
        ablation_path = measurements.with_name("gguf-isolated-kl-imatrix-ablation.json")
        if ablation_path.exists():
            ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
            paths.append(plot_decomposition(
                decomposition_rows(ablation, joint, block_noise, report_order=True), args.output,
                name="calibration-ablation.png",
                title="Same format, same weight error: calibration is the per-matrix saving",
            ))
    bars_path = measurements.with_name("gguf-kl-imatrix-ablation.json")
    if bars_path.exists():
        paths.append(plot_calibration_bars(json.loads(bars_path.read_text(encoding="utf-8")), args.output))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
