"""CPU-only exact-entry row/column error coverage for retained factor fits."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import _paths  # noqa: E402, F401
import numpy as np  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.25, 0.5, 0.75)


def coverage_curve(energy, mode):
    """Exact single-axis ordering or greedy mixed gain per newly covered entry.

    Mixed coverage is a heuristic, not a globally optimal selection. Its costs
    count the union of stored entries, so intersections are paid for only once.
    """
    energy = np.asarray(energy, dtype=np.float64)
    if (energy.ndim != 2 or not energy.size or not np.isfinite(energy).all()
            or np.any(energy < 0) or energy.sum() <= 0):
        raise ValueError("energy must be a finite nonnegative matrix with positive total")
    if mode not in ("rows", "columns", "mixed"):
        raise ValueError("unknown selection mode")
    m, n = energy.shape
    row_energy, col_energy = energy.sum(axis=1), energy.sum(axis=0)
    rows, columns = [], []
    used_rows, used_columns = np.zeros(m, dtype=bool), np.zeros(n, dtype=bool)
    counts, removed = [(0, 0)], [0.0]
    while len(rows) < m and len(columns) < n:
        row_score = np.where(used_rows, -np.inf, row_energy / (n-len(columns)))
        col_score = np.where(used_columns, -np.inf, col_energy / (m-len(rows)))
        r, c = int(row_score.argmax()), int(col_score.argmax())
        choose_row = mode == "rows" or (mode == "mixed" and row_score[r] > col_score[c])
        if choose_row:
            gain = float(row_energy[r])
            rows.append(r)
            used_rows[r] = True
            col_energy = np.maximum(0, col_energy - energy[r, :])
        else:
            gain = float(col_energy[c])
            columns.append(c)
            used_columns[c] = True
            row_energy = np.maximum(0, row_energy - energy[:, c])
        counts.append((len(rows), len(columns)))
        removed.append(removed[-1] + gain)
    # Cross-check total conservation before eliminating roundoff at the endpoint.
    assert math.isclose(removed[-1], float(energy.sum()), rel_tol=1e-12, abs_tol=1e-12)
    removed[-1] = float(energy.sum())
    counts = np.asarray(counts)
    entries = counts[:, 0]*n + counts[:, 1]*m - counts[:, 0]*counts[:, 1]
    return {"rows": rows, "columns": columns, "counts": counts,
            "entries": entries, "removed": np.asarray(removed)}


def summarize_thresholds(energy, curve, original_columns, full_column_count):
    m, n = energy.shape
    total = float(energy.sum())
    result = []
    for fraction in THRESHOLDS:
        step = int(np.searchsorted(curve["removed"], fraction * total))
        r, c = (int(x) for x in curve["counts"][step])
        rows, columns = curve["rows"][:r], curve["columns"][:c]
        remaining_rows, remaining_columns = np.ones(m, dtype=bool), np.ones(n, dtype=bool)
        remaining_rows[rows] = False
        remaining_columns[columns] = False
        # Independent complement reduction verifies union coverage without counting overlaps.
        remaining = float(energy[np.ix_(remaining_rows, remaining_columns)].sum())
        recovered = total-remaining
        assert math.isclose(recovered, float(curve["removed"][step]), rel_tol=1e-11, abs_tol=1e-12)
        assert recovered >= fraction*total - 1e-12
        assert curve["removed"][step-1] < fraction*total
        entries = int(curve["entries"][step])
        result.append({
            "target_error_fraction": fraction, "rows": r, "columns": c,
            "row_indices": rows, "original_column_indices": original_columns[columns].tolist(),
            "error_removed_fraction": recovered/total, "remaining_error": remaining,
            "nrmse_reduction_fraction": 1-math.sqrt(remaining/total),
            "separate_entries": entries, "active_entry_fraction": entries/(m*n),
            "full_matrix_entry_fraction": entries/(m*full_column_count),
            "bf16_payload_bytes": 2*entries, "bf16_added_bpw_if_base_retained": 16*entries/(m*full_column_count),
        })
    return result


def analyze(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "evidence/068/admm-perturb"
    report_path = source / "report.json"
    report = json.loads(report_path.read_text())
    fixture_path = Path(next(k for k in report["inputs"] if Path(k).name == "fixture.safetensors"))
    if hash_file(fixture_path) != report["inputs"][str(fixture_path)]:
        raise ValueError("fixture hash mismatch")
    fixture = load_file(str(fixture_path))
    m, n = fixture["target"].shape
    active = ~np.isin(np.arange(n), fixture["outlier_indices"])
    original_columns = np.flatnonzero(active)
    target = fixture["target"].astype(np.float64)[:, active]
    weights = (fixture["output_importance"].astype(np.float64)[:, None]
               * fixture["input_importance"].astype(np.float64)[None, active])
    result = {"source_report_sha256": hash_file(report_path), "fixture_sha256": hash_file(fixture_path),
              "analysis_source_sha256": hash_file(Path(__file__)), "shape": [m, n],
              "active_columns": int(active.sum()), "protected_columns": fixture["outlier_indices"].tolist(),
              "assumption": "Exact selected entries; remaining signs/scales fixed; no refit or encoder rounding",
              "mixed_method": "Greedy marginal error per newly stored entry; not a global optimum", "fits": {}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    table = []
    arrays = {}
    for name in ("best", "baseline"):
        record = report["comparison"][name]
        path = source / record["state_file"]
        if hash_file(path) != record["state_sha256"]:
            raise ValueError("factor hash mismatch")
        state = load_file(str(path))
        a = state["left_binary"].astype(np.float64) * state["scale_post"].astype(np.float64)[:, None]
        b = state["right_binary"].astype(np.float64)[:, active]
        b *= state["scale_mid"].astype(np.float64)[:, None]
        b *= state["scale_pre"].astype(np.float64)[None, active]
        raw_energy = np.square(target-a@b)
        fit = {"state_file": str(path), "state_sha256": record["state_sha256"], "metrics": {}}
        for panel, (kind, energy) in enumerate((("weighted", raw_energy*weights), ("raw", raw_energy))):
            total = float(energy.sum())
            assert math.isclose(total, record[f"{kind}_error"], rel_tol=1e-12)
            metric = {"initial_error": total, "methods": {}}
            for mode in ("rows", "columns", "mixed"):
                curve = coverage_curve(energy, mode)
                summaries = summarize_thresholds(energy, curve, original_columns, n)
                metric["methods"][mode] = summaries
                key = f"{name}_{kind}_{mode}"
                for field, value in curve.items():
                    arrays[f"{key}_{field}"] = np.asarray(value)
                if name == "best":
                    axes[panel].plot(100*curve["entries"]/(m*active.sum()), 100*curve["removed"]/total,
                                     label=mode if mode != "mixed" else "Mixed (greedy)",
                                     linestyle="--" if mode == "mixed" else "-")
                for item in summaries:
                    table.append(f"| {name} | {kind} | {mode} | {100*item['target_error_fraction']:.0f}% | "
                                 f"{item['rows']:,} | {item['columns']:,} | "
                                 f"{100*item['active_entry_fraction']:.2f}% | "
                                 f"{item['bf16_payload_bytes']/1e6:.3f} |")
            fit["metrics"][kind] = metric
        result["fits"][name] = fit
    for ax, label in zip(axes, ("Fisher-weighted squared error", "Ordinary squared error"), strict=True):
        ax.plot([0, 100], [0, 100], color="0.6", linestyle=":", label="Uniform error reference")
        for percent in (25, 50, 75):
            ax.axhline(percent, color="0.9", linewidth=0.8)
        ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Active entries encoded separately (%)",
               ylabel="Current error removed (%)", title=label)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Current best fit: exact row/column storage, no refit")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"coverage.{suffix}", dpi=160)
    plt.close(fig)
    np.savez_compressed(output / "coverage-curves.npz", **arrays)
    result["curves_sha256"] = hash_file(output / "coverage-curves.npz")
    atomic_write_json(output / "summary.json", result)
    (output / "report.md").write_text(
        "# Separately encoding rows and columns\n\n"
        "Exact selected-entry replacement on the retained best fit and original baseline, evaluated in CPU FP64. "
        "The fixed matrix is 1,152 by 6,912; seven already protected columns are excluded, leaving 6,905. "
        "All percentages of error refer to current residual squared energy, not NRMSE. "
        "No factors were refitted and no search was restarted.\n\n"
        "Rows-only and columns-only counts are exact minima for that axis: sort by total residual energy "
        "and take the smallest prefix crossing the target. Mixed selection is a greedy heuristic choosing "
        "the greatest marginal error reduction per newly stored entry. Intersections count once, both "
        "for error and storage. It is not a proof of the best possible mixed selection.\n\n"
        "| Fit | Objective | Method | Error removed, at least | Rows | Columns | Active entries | BF16 MB |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n" + "\n".join(table)
        + "\n\nThe BF16 column is a payload-size estimate, not a measured BF16 encoder result. "
        "The ideal coverage assumes selected entries are reconstructed exactly; lossy encoding recovers less. "
        "It excludes index metadata and any savings from dropping unused factors. The original base representation "
        "is retained for the additional-BPW figures in the JSON. Refitting a smaller matrix could change these results "
        "and is not measured here. This is full selected-row/column storage, not additive bias vectors.\n\n"
        "A 25%, 50%, or 75% reduction in squared error corresponds to approximately 13.4%, 29.3%, or 50% "
        "lower NRMSE when normalized to the same original target. Each threshold is independently checked by "
        "summing the unselected row/column complement, and each preceding prefix remains below the target.\n\n"
        "![Coverage curves](coverage.png)\n\n"
        "All selected original row/column IDs are saved in summary.json. Full selection orders and coverage curves "
        "are in coverage-curves.npz. Input/state/source hashes are recorded in summary.json.\n",
        encoding="utf-8")
    print(json.dumps({name: {kind: {mode: [{k: v for k, v in item.items()
                                          if k not in ("row_indices", "original_column_indices")}
                                         for item in items]
                                  for mode, items in metric["methods"].items()}
                            for kind, metric in fit["metrics"].items()}
                      for name, fit in result["fits"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Results/068/axis-excision")
    analyze(parser.parse_args().output.resolve())
