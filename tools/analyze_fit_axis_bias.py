"""Exact CPU least-squares row/column offsets for retained binary-factor fits."""

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
import torch  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def fit_biases(residual, row_weight, column_weight):
    """Exact weighted additive projection: row_mean + column_mean - grand_mean."""
    if residual.shape != (len(row_weight), len(column_weight)):
        raise ValueError("weight dimensions do not match residual")
    if np.any(row_weight <= 0) or np.any(column_weight <= 0):
        raise ValueError("weights must be positive")
    row_mean = residual @ column_weight / column_weight.sum()
    column_mean = row_weight @ residual / row_weight.sum()
    grand_mean = float(row_weight @ row_mean / row_weight.sum())
    return row_mean, column_mean, grand_mean


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
    active = ~np.isin(np.arange(fixture["target"].shape[1]), fixture["outlier_indices"])
    target = fixture["target"].astype(np.float64)[:, active]
    u = fixture["output_importance"].astype(np.float64)
    v = fixture["input_importance"].astype(np.float64)[active]
    weights = u[:, None] * v[None, :]
    target_energy = float((np.square(target) * weights).sum())
    raw_target_energy = float(np.square(target).sum())
    result = {
        "source_report_sha256": hash_file(report_path), "fixture_sha256": hash_file(fixture_path),
        "analysis_source_sha256": hash_file(Path(__file__)), "shape": list(fixture["target"].shape),
        "active_columns": int(active.sum()), "protected_columns": fixture["outlier_indices"].tolist(),
        "scope": "Fixed signs and scales, exact additive weight-matrix offsets; no new ADMM or model tuning",
        "fits": {},
    }
    best_residual = best_corrected = None
    for name in ("baseline", "best", "best_perturbed"):
        record = report["comparison"][name]
        path = source / record["state_file"]
        if hash_file(path) != record["state_sha256"]:
            raise ValueError("factor hash mismatch")
        state = load_file(str(path))
        a = state["left_binary"].astype(np.float64) * state["scale_post"].astype(np.float64)[:, None]
        b = state["right_binary"].astype(np.float64)[:, active]
        b *= state["scale_mid"].astype(np.float64)[:, None]
        b *= state["scale_pre"].astype(np.float64)[None, active]
        residual = target - a @ b
        row, column, grand = fit_biases(residual, u, v)
        patches = {"none": 0.0, "row": row[:, None], "column": column[None, :],
                   "both": (row - grand)[:, None] + column[None, :]}
        quant_row = torch.from_numpy(row - grand).bfloat16().double().numpy()
        quant_column = torch.from_numpy(column).bfloat16().double().numpy()
        patches["both_bf16"] = quant_row[:, None] + quant_column[None, :]
        raw_row, raw_column, raw_grand = fit_biases(residual, np.ones_like(u), np.ones_like(v))
        patches["both_raw_optimal"] = (raw_row - raw_grand)[:, None] + raw_column[None, :]
        metrics = {}
        for method, patch in patches.items():
            corrected = residual - patch
            weighted = float((np.square(corrected) * weights).sum())
            raw = float(np.square(corrected).sum())
            metrics[method] = {"weighted_error": weighted, "raw_error": raw,
                               "weighted_nrmse": math.sqrt(weighted / target_energy),
                               "raw_nrmse": math.sqrt(raw / raw_target_energy)}
            if name == "best" and method == "both":
                best_residual, best_corrected = residual.copy(), corrected.copy()
        for item in metrics.values():
            item["weighted_error_reduction_percent"] = 100 * (1 - item["weighted_error"] / metrics["none"][
                "weighted_error"])
            item["weighted_nrmse_reduction_percent"] = 100 * (1 - item["weighted_nrmse"] / metrics["none"][
                "weighted_nrmse"])
            item["raw_error_reduction_percent"] = 100 * (1 - item["raw_error"] / metrics["none"]["raw_error"])
        corrected = residual - patches["both"]
        correction_energy = float((np.square(patches["both"]) * weights).sum())
        actual_gain = metrics["none"]["weighted_error"] - metrics["both"]["weighted_error"]
        assert math.isclose(correction_energy, actual_gain, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(metrics["none"]["weighted_error"], record["weighted_error"], rel_tol=1e-12)
        full_column = np.zeros(fixture["target"].shape[1])
        full_column[active] = column
        np.savez(output / f"{name}-biases.npz", row_bias=row-grand, column_bias=full_column,
                 active_column_mask=active, row_only=row, column_only=column, grand_mean=grand)
        result["fits"][name] = {
            "state_file": str(path), "state_sha256": record["state_sha256"], "metrics": metrics,
            "bias_file_sha256": hash_file(output / f"{name}-biases.npz"),
            "grand_signed_residual": grand,
            "normal_equation_row_max_abs": float(np.abs(corrected @ v / v.sum()).max()),
            "normal_equation_column_max_abs": float(np.abs(u @ corrected / u.sum()).max()),
            "orthogonal_correction_energy": correction_energy,
        }
    m, n = fixture["target"].shape
    result["storage"] = {"full_bias_vector_values": m+n, "bf16_bytes": 2*(m+n), "fp32_bytes": 4*(m+n),
                         "bf16_additional_bpw": 16*(m+n)/(m*n), "fp32_additional_bpw": 32*(m+n)/(m*n),
                         "note": "Payload estimate only; gate row correction off on the existing protected columns"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    images = []
    means = []
    rms = float(np.sqrt(np.square(best_residual).mean()))
    titles = ("Current best: signed residual", "After optimal row + column offsets")
    for ax, residual, title in zip(axes, (best_residual, best_corrected), titles, strict=True):
        coarse = np.array([[residual[np.ix_(r, c)].mean() for c in np.array_split(np.arange(target.shape[1]), 96)]
                           for r in np.array_split(np.arange(m), 48)])
        values = 100 * coarse / rms
        means.append(values)
        images.append(ax.imshow(values, aspect="auto", cmap="RdBu_r", extent=(0, n, m, 0)))
        ax.set(title=title, xlabel="Input column (binned)", ylabel="Output row")
    limit = max(1, math.ceil(max(float(np.abs(x).max()) for x in means)))
    for im in images:
        im.set_clim(-limit, limit)
    fig.colorbar(images[0], ax=axes, label="Mean signed residual (% of original residual RMS)", shrink=0.85)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"signed-residual.{suffix}", dpi=160)
    plt.close(fig)
    result["plot"] = {"symmetric_color_limit_percent_of_residual_rms": limit,
                      "meaning": "Signed target minus reconstruction, not squared-error deviation"}
    atomic_write_json(output / "summary.json", result)
    lines = []
    for name, fit in result["fits"].items():
        for method in ("none", "row", "column", "both", "both_bf16", "both_raw_optimal"):
            x = fit["metrics"][method]
            lines.append(f"| {name} | {method} | {x['weighted_error']:.10f} | "
                         f"{x['weighted_error_reduction_percent']:.6f}% | {x['weighted_nrmse']:.10f} | "
                         f"{x['raw_error_reduction_percent']:.6f}% |")
    (output / "report.md").write_text(
        "# Additive row and column bias diagnostic\n\n"
        "The previous heatmap showed squared error relative to its panel mean. Its negative colors were "
        "below-average error magnitude, not negative residuals. This diagnostic uses signed residuals.\n\n"
        "Fit additive weight corrections `row_bias[i] + column_bias[j]` while holding all factor signs and "
        "scales fixed. On retained columns, the exact weighted least-squares solution is the weighted row "
        "mean plus weighted column mean minus the weighted grand mean of `target - prediction`. "
        "The row correction is gated off on seven protected outlier columns and their column correction "
        "is zero. Original factors, outlier sidecars, and the cancelled search remain unchanged.\n\n"
        "| Fit | Correction | Weighted error | Weighted error reduction | Weighted NRMSE | Raw error reduction |\n"
        "| --- | --- | ---: | ---: | ---: | ---: |\n" + "\n".join(lines)
        + "\n\n`both_raw_optimal` instead minimizes unweighted error; all other corrections minimize the "
        "retained diagonal-Fisher objective. The BF16 row/column biases are fitted in FP64 and then rounded "
        "to BF16 before the corrected fit is rescored.\n\n"
        "![True signed residual before and after correction](signed-residual.png)\n\n"
        "Each bin averages approximately 24 by 72 entries. Both panels share a scale relative to the "
        "uncorrected residual RMS; no bins are clipped. Normal-equation checks establish the exact optimum "
        "within this additive subspace. This does not measure gains from jointly refitting the binary factors.\n\n"
        f"Two full vectors require {2*(m+n):,} BF16 payload bytes, adding {16*(m+n)/(m*n):.6f} BPW "
        "to this matrix, excluding metadata. This changes the representation; current results do not apply "
        "at exactly the original storage budget. These are weight-matrix offsets, not a normal constant "
        "output bias. Runtime evaluation needs the corresponding input-dependent rank-one terms.\n\n"
        "CPU FP64 reconstruction and reduction reproduce the existing comparison errors. "
        "The JSON records input hashes, coefficient hashes, normal-equation residuals, and the independent "
        "correction-energy identity. Bias coefficients are saved as NPZ files alongside this report.\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v["metrics"] for k, v in result["fits"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Results/068/axis-bias")
    analyze(parser.parse_args().output.resolve())
