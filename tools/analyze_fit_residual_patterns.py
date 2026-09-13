"""CPU-only row/column concentration diagnostics for retained factor fits."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import _paths  # noqa: E402, F401
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def concentration(energy, target_energy):
    total = float(energy.sum())
    order = np.argsort(energy)[::-1]
    ranked = energy[order]
    shares = {str(p): float(ranked[:max(1, math.ceil(len(energy) * p / 100))].sum() / total)
              for p in (1, 5, 10, 25)}
    valid = target_energy > max(float(target_energy.max()) * 1e-12, 1e-30)
    nrmse = np.sqrt(energy[valid] / target_energy[valid])
    return {
        "total_energy": total, "axis_size": len(energy), "top_percent_energy_share": shares,
        "top10_indices": order[:10].tolist(),
        "top10_target_energy_share": float(target_energy[order[:10]].sum() / target_energy.sum()),
        "mean_energy_cv": float(energy.std() / energy.mean()),
        "nrmse_quantiles_10_50_90_99": np.quantile(nrmse, [0.1, 0.5, 0.9, 0.99]).tolist(),
        "count_for_50pct_energy": int(np.searchsorted(np.cumsum(ranked), total * 0.5) + 1),
    }


def describe(energy, target_energy):
    row, col = energy.sum(axis=1), energy.sum(axis=0)
    mean = float(energy.mean())
    denominator = float(np.square(energy - mean).sum())
    return {
        "rows": concentration(row, target_energy.sum(axis=1)),
        "columns": concentration(col, target_energy.sum(axis=0)),
        "row_effect_fraction_of_cell_energy_variance": float(
            energy.shape[1] * np.square(energy.mean(axis=1) - mean).sum() / denominator),
        "column_effect_fraction_of_cell_energy_variance": float(
            energy.shape[0] * np.square(energy.mean(axis=0) - mean).sum() / denominator),
    }


def analyze(output, snapshot_path=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "evidence/068/admm-perturb"
    snapshot = json.loads((snapshot_path or source / "report.json").read_text())
    atomic_write_json(output / "search-snapshot.json", snapshot)
    fixture_path = ROOT / "evidence/066/hour-fit/fixture.safetensors"
    if hash_file(fixture_path) != snapshot["inputs"][str(fixture_path)]:
        raise ValueError("fixture hash mismatch")
    fixture = load_file(str(fixture_path))
    columns = np.arange(fixture["target"].shape[1])
    active = ~np.isin(columns, fixture["outlier_indices"])
    target = fixture["target"].astype(np.float64)[:, active]
    weights = (fixture["output_importance"].astype(np.float64)[:, None]
               * fixture["input_importance"].astype(np.float64)[None, active])
    models = {
        "baseline": {"state_file": "baseline.safetensors",
                     "state_sha256": next(v for k, v in snapshot["inputs"].items()
                                          if Path(k).name == "baseline.safetensors")},
        "inherited": snapshot["inherited"], "best_perturbed_so_far": snapshot["best_perturbed"],
    }
    result = {"fixture_sha256": hash_file(fixture_path), "excluded_outlier_columns": columns[~active].tolist(),
              "search_elapsed_at_snapshot": snapshot["records"][-1]["elapsed_seconds"], "fits": {}}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    heatmaps = []
    deviations = []
    for model_index, (name, row) in enumerate(models.items()):
        path = source / row["state_file"]
        if hash_file(path) != row["state_sha256"]:
            raise ValueError("factor hash mismatch")
        state = load_file(str(path))
        a = state["left_binary"].astype(np.float64) * state["scale_post"].astype(np.float64)[:, None]
        b = state["right_binary"].astype(np.float64)[:, active]
        b *= state["scale_mid"].astype(np.float64)[:, None]
        b *= state["scale_pre"].astype(np.float64)[None, active]
        residual = target - a @ b
        energy = np.square(residual)
        omitted_b = state["right_binary"].astype(np.float64)[:, ~active]
        omitted_b *= state["scale_mid"].astype(np.float64)[:, None]
        omitted_b *= state["scale_pre"].astype(np.float64)[None, ~active]
        omitted_energy = np.square(fixture["target"].astype(np.float64)[:, ~active] - a @ omitted_b)
        omitted_weight = (fixture["output_importance"].astype(np.float64)[:, None]
                          * fixture["input_importance"].astype(np.float64)[None, ~active])
        metrics = {"state_file": str(path), "state_sha256": row["state_sha256"],
                   "excluded_outlier_raw_error": float(omitted_energy.sum()),
                   "excluded_outlier_weighted_error": float((omitted_energy * omitted_weight).sum()),
                   "raw": describe(energy, np.square(target)),
                   "weighted": describe(energy * weights, np.square(target) * weights)}
        for kind in ("raw", "weighted"):
            positions = metrics[kind]["columns"]["top10_indices"]
            metrics[kind]["columns"]["top10_original_columns"] = columns[active][positions].tolist()
        result["fits"][name] = metrics
        for kind, values, axis_index in (("raw", energy, 0), ("weighted", energy * weights, 1)):
            ax = axes[axis_index, model_index]
            row_splits = np.array_split(np.arange(values.shape[0]), 48)
            col_splits = np.array_split(np.arange(values.shape[1]), 96)
            coarse = np.array([[values[np.ix_(r, c)].mean() for c in col_splits] for r in row_splits])
            deviation = 100 * (coarse / values.mean() - 1)
            deviations.append(deviation)
            im = ax.imshow(deviation, aspect="auto", cmap="RdBu_r", extent=(0, 6912, 1152, 0))
            heatmaps.append(im)
            ax.set(title=f"{name.replace('_', ' ')}: {kind}", xlabel="Input column (binned)", ylabel="Output row")
    color_limit = math.ceil(max(float(np.abs(x).max()) for x in deviations))
    color_limit = max(1, color_limit)
    for heatmap in heatmaps:
        heatmap.set_clim(-color_limit, color_limit)
    result["heatmap_scale"] = {"units": "percent deviation from panel mean squared error",
                               "shared_symmetric_limit_percent": color_limit, "clipped_bins": 0}
    fig.colorbar(im, ax=axes, label="Local mean squared error: % above / below panel average", shrink=0.8)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"residual-heatmaps.{extension}", dpi=150)
    plt.close(fig)
    atomic_write_json(output / "summary.json", result)
    lines = []
    for name, metrics in result["fits"].items():
        for kind in ("raw", "weighted"):
            for axis in ("rows", "columns"):
                values = metrics[kind][axis]
                top = values["top_percent_energy_share"]
                lines.append(f"| {name} | {kind} | {axis} | {100*top['1']:.2f}% | {100*top['5']:.2f}% "
                             f"| {100*top['10']:.2f}% | {values['count_for_50pct_energy']:,} |")
    (output / "report.md").write_text(
        "# Residual concentration diagnostics\n\n"
        "CPU FP64 reconstruction. Seven fixed outlier columns are excluded from this residual-only analysis. "
        f"The active search snapshot was taken at {result['search_elapsed_at_snapshot']/60:.2f} minutes.\n\n"
        "| Fit | Error | Axis | Worst 1% share | Worst 5% share | Worst 10% share | Count for half the error |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: |\n" + "\n".join(lines)
        + "\n\n![Residual heatmaps](residual-heatmaps.png)\n\n"
        "Each heatmap cell averages a roughly 24-by-72 region. Colors show percent deviation from that panel's "
        f"overall mean on a shared scale of -{color_limit}% to +{color_limit}%; no bins are clipped. "
        "Red means more error, blue means less. Bins use retained column order with seven omitted outlier columns. "
        "This is squared residual energy, not signed residuals. Row/column concentration can reflect target "
        "magnitude and Fisher importance, so the JSON also includes per-axis target-normalized NRMSE quantiles "
        "and the row/column contributions to cell-energy variance. All measured factor files are immutable "
        "and hash-verified; this diagnostic does not modify the search.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Results/068/residual-patterns")
    parser.add_argument("--snapshot", type=Path, help="Reuse a saved search snapshot when changing plot presentation")
    args = parser.parse_args()
    analyze(args.output, args.snapshot)
