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


def fit_dataset(source: Path, anchor: str | None, anchor_error: float) -> dict:
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
        grouped[row["target"]].append(row)
    curves = {}
    for target in protocol["targets"]:
        block, projection = target.split(":", 1)
        curves[target] = {"block": int(block), "projection": projection, **fit_curve(grouped[target], baseline)}
    if anchor is None:
        ranked = sorted(
            curves,
            key=lambda target: curves[target]["coefficient"] * anchor_error ** curves[target]["exponent"],
        )
        anchor = ranked[(len(ranked) - 1) // 2]
    if anchor not in curves:
        raise ValueError(f"unknown anchor projection {anchor!r}; use a block:projection key from the measurements")
    reference = curves[anchor]
    common_kl = reference["coefficient"] * anchor_error ** reference["exponent"]
    for target, curve in curves.items():
        error = anchor_error if target == anchor else (common_kl / curve["coefficient"]) ** (1 / curve["exponent"])
        low, high = curve["measured_error_range"]
        # Ignore tiny differences from FP16 rounding at a measured endpoint.
        status = (
            "below_measured" if error < low and not math.isclose(error, low, rel_tol=1e-4)
            else "above_measured" if error > high and not math.isclose(error, high, rel_tol=1e-4)
            else "within_measured"
        )
        curve["equal_kl"] = {"target_error": error, "range_status": status}
    return {
        "schema_version": 1,
        "measurement_file": source.name,
        "measurement_sha256": digest,
        "teacher_artifact": protocol["teacher_artifact"],
        "model": "kl_above_control = coefficient * actual_relative_frobenius_error ** exponent",
        "fit_method": "ordinary least squares on log of seed means at each measured error level",
        "clean_kl_nats_per_token": baseline,
        "equal_kl": {"anchor": anchor, "anchor_error": anchor_error, "kl_above_control": common_kl},
        "curves": curves,
    }


def plot_targets(fit: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import PercentFormatter

    curves = fit["curves"]
    blocks = sorted({curve["block"] for curve in curves.values()})
    projections = list(dict.fromkeys(curve["projection"] for curve in curves.values()))
    grid = np.full((len(projections), len(blocks)), np.nan)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(blocks) + 2), 5.3), layout="constrained")
    values = [curve["equal_kl"]["target_error"] for curve in curves.values()]
    norm = LogNorm(vmin=min(values), vmax=max(values))
    cmap = plt.get_cmap("viridis")
    for target, curve in curves.items():
        col = blocks.index(curve["block"])
        row = projections.index(curve["projection"])
        error = curve["equal_kl"]["target_error"]
        grid[row, col] = error
        status = curve["equal_kl"]["range_status"]
        suffix = {"below_measured": "↓", "above_measured": "↑", "within_measured": ""}[status]
        color = "#12202a" if norm(error) > 0.57 else "white"
        ax.text(col, row, f"{error:.1%}{suffix}", ha="center", va="center", color=color, fontsize=8)
        if target == fit["equal_kl"]["anchor"]:
            ax.add_patch(Rectangle((col - 0.48, row - 0.48), 0.96, 0.96, fill=False,
                                   edgecolor="#ffb454", linewidth=2.4))
    image = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    reference = fit["equal_kl"]
    fig.suptitle("Weight-error targets for equal predicted output KL", fontsize=17, fontweight="bold")
    ax.set_title(
        f"{reference['kl_above_control']:.5f} nats/token above control · "
        f"Anchor {reference['anchor']} at {reference['anchor_error']:.0%}",
        loc="left", fontsize=11,
    )
    ax.set_xticks(range(len(blocks)), blocks)
    labels = [p.replace("self_attn.", "Attn ").replace("mlp.", "MLP ").removesuffix("_proj") for p in projections]
    ax.set_yticks(range(len(projections)), labels)
    ax.tick_params(length=0)
    ax.set_xlabel("Decoder block (zero-based)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, pad=0.02)
    ticks = (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0)
    bar.set_ticks([tick for tick in ticks if norm.vmin <= tick <= norm.vmax])
    bar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    bar.set_label("Target relative weight error (log scale)")
    fig.supxlabel(
        "↓ Below measured range   ·   ↑ Above measured range   ·   Gold outline marks the anchor", fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, help="the teacher dataset's measurements/<sweep-id>/results.json")
    parser.add_argument("--anchor", help="block:projection key; defaults to the lower median fitted response")
    parser.add_argument("--anchor-error", type=float, default=0.10)
    parser.add_argument("--heatmap", type=Path, help="optional path for a target-error heatmap PNG")
    args = parser.parse_args()
    if not math.isfinite(args.anchor_error) or args.anchor_error <= 0:
        parser.error("--anchor-error must be finite and positive")
    fit = fit_dataset(args.measurements, args.anchor, args.anchor_error)
    output = args.measurements.with_name("kl-response-fits.json")
    atomic_write_json(output, fit)
    print(f"Saved {len(fit['curves'])} curves to {output}")
    if args.heatmap is not None:
        plot_targets(fit, args.heatmap)
        print(f"Saved heatmap to {args.heatmap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
