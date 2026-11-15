"""Fit and invert per-projection KL response curves from a completed sensitivity dataset."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file


def fit_curve(rows: list[dict], baseline: float) -> dict:
    """Fit KL above the clean control as coefficient × relative weight error ** exponent."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["requested_error"]].append(row)
    points = []
    for level, trials in sorted(grouped.items()):
        kl = [row["kl_nats_per_token"] - baseline for row in trials]
        points.append({
            "requested_error": level,
            "actual_error": float(np.mean([row["actual_error"] for row in trials])),
            "kl_above_control": float(np.mean(kl)),
            "kl_seed_sd": float(np.std(kl, ddof=1)) if len(kl) > 1 else None,
            "seed_count": len(trials),
        })
    x = np.array([point["actual_error"] for point in points])
    y = np.array([point["kl_above_control"] for point in points])
    if len(points) < 4 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("curve fitting requires at least four finite measured error levels")
    if min(x) <= 0 or min(y) <= 0 or np.any(np.diff(x) <= 0) or np.ptp(y) == 0:
        raise ValueError("power-law fitting requires increasing positive weight errors and positive, varying KL")
    log_x, log_y = np.log(x), np.log(y)
    exponent, log_coefficient = np.polyfit(log_x, log_y, 1)
    if exponent <= 0:
        raise ValueError("power-law fit is not increasing and cannot be inverted for equal-KL targets")
    fitted = np.exp(log_coefficient + exponent * log_x)
    for i, point in enumerate(points):
        mask = np.arange(len(points)) != i
        slope, intercept = np.polyfit(log_x[mask], log_y[mask], 1)
        point["fitted_kl"] = float(fitted[i])
        point["held_out_kl"] = math.exp(intercept + slope * log_x[i])
    held_out = np.array([point["held_out_kl"] for point in points])
    return {
        "coefficient": math.exp(log_coefficient),
        "exponent": float(exponent),
        "measured_error_range": [float(x[0]), float(x[-1])],
        "measured_kl_range": [float(min(y)), float(max(y))],
        "points": points,
        "fit_quality": {
            "r_squared_kl": float(1 - np.sum((fitted - y)**2) / np.sum((y - np.mean(y))**2)),
            "rmse_kl": float(np.sqrt(np.mean((fitted - y)**2))),
            "rmse_log_kl": float(np.sqrt(np.mean((np.log(fitted) - log_y)**2))),
            "leave_one_level_out_rmse_kl": float(np.sqrt(np.mean((held_out - y)**2))),
            "leave_one_level_out_max_relative_error": float(np.max(np.abs(held_out / y - 1))),
        },
    }


DEFAULT_KL_TARGETS = (0.001, 0.0025, 0.005)


def _target_error(curve: dict, kl: float) -> dict:
    """Invert one fitted curve at a KL above the clean control and flag extrapolation."""
    error = (kl / curve["coefficient"]) ** (1 / curve["exponent"])
    low, high = curve["measured_error_range"]
    # Ignore tiny differences from FP16 rounding at a measured endpoint.
    status = (
        "below_measured" if error < low and not math.isclose(error, low, rel_tol=1e-4)
        else "above_measured" if error > high and not math.isclose(error, high, rel_tol=1e-4)
        else "within_measured"
    )
    return {"kl_above_control": kl, "target_error": error, "range_status": status}


def heatmap_name(kl: float) -> str:
    return f"equal-kl-target-error-heatmap-{kl:g}-nats.png"


def fit_dataset(source: Path, kl_targets: tuple[float, ...] = DEFAULT_KL_TARGETS) -> dict:
    if not kl_targets or any(not math.isfinite(kl) or kl <= 0 for kl in kl_targets):
        raise ValueError("KL targets must be finite and positive nats per token")
    if len(set(kl_targets)) != len(kl_targets):
        raise ValueError("KL targets must be unique")
    kl_targets = tuple(sorted(kl_targets))
    digest = hash_file(source)
    data = json.loads(source.read_text(encoding="utf-8"))
    if hash_file(source) != digest:
        raise ValueError("measurement dataset changed while reading it")
    protocol = data["protocol"]
    baseline = data["clean_baseline"]["kl_nats_per_token"]
    expected = {
        (target, level, seed) for target in protocol["targets"]
        for level in protocol["levels"] for seed in protocol["seeds"]
    }
    keys = [(row["target"], row["requested_error"], row["seed"]) for row in data["trials"]]
    if set(keys) != expected or len(keys) != len(expected) or len(keys) != data["expected_trials"]:
        raise ValueError("measurement dataset is incomplete or has duplicate trials")
    grouped = defaultdict(list)
    for row in data["trials"]:
        if row["requested_error"] > 0:
            grouped[row["target"]].append(row)
    curves = {}
    for target in protocol["targets"]:
        block, projection = target.split(":", 1)
        curve = {"block": int(block), "projection": projection, **fit_curve(grouped[target], baseline)}
        curve["kl_targets"] = [_target_error(curve, kl) for kl in kl_targets]
        curves[target] = curve
    return {
        "schema_version": 2,
        "measurement_file": source.name,
        "measurement_sha256": digest,
        "teacher_artifact": protocol["teacher_artifact"],
        "model": "kl_above_control = coefficient * actual_relative_frobenius_error ** exponent",
        "fit_method": "ordinary least squares on log of seed means at each measured error level",
        "clean_kl_nats_per_token": baseline,
        "kl_targets": list(kl_targets),
        "curves": curves,
    }


def require_plotting() -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError("install the NanoQuant visualization extra to render sensitivity plots") from exc

    matplotlib.use("Agg")


def plot_measurements(report: dict, output: Path) -> None:
    """Render measured seed means, retaining signed KL deltas near the clean control."""
    require_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.ticker import PercentFormatter

    protocol = report["protocol"]
    levels = sorted(protocol["levels"])
    groups = {}
    for target in protocol["targets"]:
        block, separator, projection = target.partition(":")
        if not separator:
            block, projection = block.removeprefix("block "), "All projections"
        groups[target] = (int(block), projection)
    blocks = sorted({b for b, _ in groups.values()})
    projections = list(dict.fromkeys(p for _, p in groups.values()))
    baseline = report["clean_baseline"]["kl_nats_per_token"]
    trials = defaultdict(list)
    for row in report["trials"]:
        trials[row["target"], row["requested_error"]].append(row)
    means = {
        key: (float(np.mean([r["actual_error"] for r in rows])),
              float(np.mean([r["kl_nats_per_token"] - baseline for r in rows])))
        for key, rows in trials.items()
    }
    grids = np.full((len(levels), len(projections), len(blocks)), np.nan)
    for (target, level), (_, kl) in means.items():
        b, p = groups[target]
        grids[levels.index(level), projections.index(p), blocks.index(b)] = kl
    vmin = min(0.0, float(np.nanmin(grids)))
    vmax = max(1e-8, float(np.nanmax(grids)))
    norm = SymLogNorm(linthresh=max(1e-8, vmax * 1e-4), vmin=vmin, vmax=vmax)
    output.mkdir(parents=True, exist_ok=True)
    height = math.ceil(len(levels) / 2)
    fig, axes = plt.subplots(height, 2, figsize=(max(12, 0.39 * len(blocks)), 3.1 * height),
                             layout="constrained", squeeze=False)
    for ax, grid, level in zip(axes.flat, grids, levels, strict=False):
        im = ax.imshow(grid, aspect="auto", interpolation="nearest", cmap="magma", norm=norm)
        ax.set_title(f"{level:.0%} requested weight error")
        ax.set_xticks(range(len(blocks)), blocks, fontsize=7)
        ax.set_yticks(range(len(projections)), projections, fontsize=8)
        ax.set_xlabel("Decoder block")
    for ax in list(axes.flat)[len(levels):]:
        ax.set_visible(False)
    fig.suptitle(f"{protocol['teacher']['source']}: measured KL sensitivity")
    fig.colorbar(im, ax=axes, label="KL above clean control (nats/token, symlog)")
    fig.savefig(output / "kl-sensitivity-heatmaps.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(math.ceil(len(projections) / 2), 2, figsize=(12, 3 * math.ceil(len(projections) / 2)),
                             layout="constrained", squeeze=False, sharex=True, sharey=True)
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(blocks)))
    for ax, projection in zip(axes.flat, projections, strict=False):
        for target, (block, name) in groups.items():
            if name == projection:
                points = sorted(means[target, level] for level in levels if (target, level) in means)
                if points:
                    x, y = zip(*points, strict=True)
                    ax.plot(x, y, marker=".", color=colors[blocks.index(block)], linewidth=0.9)
        ax.set_title(projection)
        ax.set_yscale("symlog", linthresh=norm.linthresh)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(projections):]:
        ax.set_visible(False)
    block_norm = plt.Normalize(min(blocks), max(max(blocks), min(blocks) + 1))
    fig.colorbar(plt.cm.ScalarMappable(norm=block_norm, cmap="viridis"), ax=axes, label="Decoder block")
    fig.suptitle(f"{protocol['teacher']['source']}: KL response by projection")
    fig.supxlabel("Measured relative Frobenius error")
    fig.supylabel("KL above clean control (nats/token)")
    fig.savefig(output / "kl-error-response.png", dpi=160)
    plt.close(fig)


def plot_targets(fit: dict, directory: Path) -> list[Path]:
    """Render one weight-error target heatmap per KL target, sharing one colour scale."""
    require_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import PercentFormatter

    curves = fit["curves"]
    blocks = sorted({curve["block"] for curve in curves.values()})
    projections = list(dict.fromkeys(curve["projection"] for curve in curves.values()))
    labels = [p.replace("self_attn.", "Attn ").replace("mlp.", "MLP ").removesuffix("_proj") for p in projections]
    values = [item["target_error"] for curve in curves.values() for item in curve["kl_targets"]]
    norm = LogNorm(vmin=min(values), vmax=max(max(values), min(values) * 1.001))
    cmap = plt.get_cmap("viridis")
    low = min(curve["measured_error_range"][0] for curve in curves.values())
    high = max(curve["measured_error_range"][1] for curve in curves.values())
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, kl in enumerate(fit["kl_targets"]):
        grid = np.full((len(projections), len(blocks)), np.nan)
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(blocks) + 2), 5.3), layout="constrained")
        extrapolated = 0
        for curve in curves.values():
            item = curve["kl_targets"][index]
            if not math.isclose(item["kl_above_control"], kl):
                raise ValueError("curve KL targets do not match the fit's KL targets")
            col, row = blocks.index(curve["block"]), projections.index(curve["projection"])
            error = item["target_error"]
            grid[row, col] = error
            suffix = {"below_measured": "↓", "above_measured": "↑", "within_measured": ""}[item["range_status"]]
            extrapolated += bool(suffix)
            color = "#12202a" if norm(error) > 0.57 else "white"
            ax.text(col, row, f"{error:.1%}{suffix}", ha="center", va="center", color=color, fontsize=8)
        image = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        fig.suptitle("Weight-error targets for equal predicted output KL", fontsize=17, fontweight="bold")
        ax.set_title(
            f"Target {kl:g} nats/token above control · "
            f"{extrapolated} of {len(curves)} targets outside the measured {low:.0%}–{high:.0%} error range",
            loc="left", fontsize=11,
        )
        ax.set_xticks(range(len(blocks)), blocks)
        ax.set_yticks(range(len(projections)), labels)
        ax.tick_params(length=0)
        ax.set_xlabel("Decoder block (zero-based)")
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, pad=0.02)
        ticks = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0)
        bar.set_ticks([tick for tick in ticks if norm.vmin <= tick <= norm.vmax])
        bar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        bar.set_label("Target relative weight error (log scale, shared across KL targets)")
        fig.supxlabel(
            "↓ Below measured range (extrapolated)   ·   ↑ Above measured range (extrapolated)", fontsize=10,
        )
        path = directory / heatmap_name(kl)
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, help="the teacher dataset's measurements/<sweep-id>/results.json")
    parser.add_argument(
        "--kl-target", type=float, action="append",
        help=(
            "KL above the clean control in nats/token; repeatable "
            f"(default {', '.join(map(str, DEFAULT_KL_TARGETS))})"
        ),
    )
    parser.add_argument("--heatmap-directory", type=Path, help="optional directory for one heatmap PNG per KL target")
    args = parser.parse_args()
    try:
        fit = fit_dataset(args.measurements, tuple(args.kl_target or DEFAULT_KL_TARGETS))
    except ValueError as exc:
        parser.error(str(exc))
    output = args.measurements.with_name("kl-response-fits.json")
    atomic_write_json(output, fit)
    print(f"Saved {len(fit['curves'])} curves to {output}")
    if args.heatmap_directory is not None:
        for path in plot_targets(fit, args.heatmap_directory):
            print(f"Saved heatmap to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
