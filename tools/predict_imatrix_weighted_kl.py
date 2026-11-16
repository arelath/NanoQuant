"""Predict each GGUF projection's KL from its imatrix-weighted weight error, with no model run.

The sensitivity sweep's Gaussian prediction assumes a projection's error is spread evenly over
its input columns. An importance matrix records how strongly each input column is driven, so
weighting the squared residual of each column by its mean-one importance gives the error that
evenly spread noise would need to cost the same. That noise-equivalent error goes through the
sweep's fitted curve unchanged; nothing is fitted to the measured GGUF KL.

The result is compared with ``measure_gguf_isolated_kl.py``'s measurements: each projection's
isolated KL, their sum and the file's joint KL. Runs on the CPU from the model's safetensors.

Usage:
    .venv\\Scripts\\python.exe tools\\predict_imatrix_weighted_kl.py ^
        Datasets\\<dataset>\\measurements\\<sweep-id>\\gguf-isolated-kl.json ^
        --imatrix imatrix.gguf --gguf A.gguf --gguf B.gguf
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import gguf
import numpy as np
from huggingface_hub import snapshot_download
from kl_sensitivity_dataset import verify_local_weights
from materialize_nanoquant_gguf import dense_nanoquant_projections
from measure_gguf_kl import PROJECTIONS, map_tensors, read_gguf
from measure_joint_kl_targets import spec_for_protocol
from safetensors import safe_open

from nanoquant.infrastructure.imatrix import load_imatrix
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file

GGUF_NAMES = {path: kind for kind, path in PROJECTIONS.items()}


def prediction_inputs(measurements: Path, imatrix: Path, imatrix_sha256: str | None, fits_sha256: str | None):
    """Validate a projection sweep, matching source weights and imatrix without inference."""
    sweep = json.loads(measurements.read_bytes())
    sweep_sha = hash_file(measurements)
    fits_path = measurements.with_name("kl-response-fits.json")
    fit = json.loads(fits_path.read_bytes())
    fit_sha = hash_file(fits_path)
    protocol = sweep["protocol"]
    curves = fit["curves"]
    targets = protocol["targets"]
    if (fit["measurement_sha256"] != sweep_sha or fit["teacher_artifact"] != protocol["teacher_artifact"]
            or len(sweep["trials"]) != sweep["expected_trials"] or set(targets) != curves.keys()
            or (fits_sha256 is not None and fits_sha256 != fit_sha)):
        raise ValueError("sweep and projection sensitivity fits do not match")
    spec = spec_for_protocol(protocol)
    if protocol["model_weights_sha256"] != spec.weights_identity or len(targets) != 7 * spec.blocks:
        raise ValueError("sensitivity sweep does not cover the pinned decoder projection inventory")
    if len(spec.weight_files) != 1:
        raise ValueError("prediction-only currently requires a single safetensors checkpoint")
    snapshot = Path(snapshot_download(repo_id=spec.source, revision=spec.revision, local_files_only=True))
    verify_local_weights(snapshot, spec)
    matrix = load_imatrix(imatrix, imatrix_sha256)
    keys = {}
    ref_path = snapshot / spec.weight_files[0][0]
    with safe_open(ref_path, framework="pt", device="cpu") as reference:
        for target in targets:
            block, projection = target.split(":", 1)
            name = f"blk.{block}.{GGUF_NAMES[projection]}.weight"
            source_key = f"model.layers.{block}.{projection}.weight"
            shape = reference.get_slice(source_key).get_shape()
            value = matrix.entries.get(name)
            if value is None or tuple(value.shape) != (1, shape[1]):
                raise ValueError(f"imatrix projection missing or wrong width: {name}")
            keys[target] = (name, source_key)
    identity = {
        "protocol": protocol, "measurement_sha256": sweep_sha, "fits_sha256": fit_sha,
        "measurements": str(measurements), "imatrix": {"path": str(imatrix), "sha256": matrix.sha256},
        "checkpoint": str(ref_path), "projection_count": len(keys),
    }
    return identity, curves, matrix, ref_path, keys


def predict_components(path: Path, curves: Mapping, importance: Mapping, reference, keys: Mapping) -> dict:
    """Compute export-weight residuals and additive component estimates."""
    reader = gguf.GGUFReader(path)
    dense = dense_nanoquant_projections(reader.fields, {tensor.name: tensor for tensor in reader.tensors}, gguf)
    if dense.keys() != {name for name, _ in keys.values()}:
        raise ValueError("GGUF projection coverage does not match the sensitivity sweep")
    rows = []
    for key, (name, source_key) in keys.items():
        original = reference.get_tensor(source_key).float().numpy()
        packed = dense[name].weight
        if packed.shape != original.shape:
            raise ValueError(f"exported projection does not match the pinned source shape: {name}")
        error, weighted = noise_equivalent_error(original, packed, importance[name].numpy().ravel())
        curve = curves[key]
        low, high = curve["measured_error_range"]
        if not math.isfinite(weighted):
            raise ValueError(f"non-finite weight reconstruction error: {name}")
        block, projection = key.split(":", 1)
        rows.append({
            "block": int(block), "projection": projection, "relative_error": error,
            "noise_equivalent_error": weighted, "unweighted_kl": curve["coefficient"] * error ** curve["exponent"],
            "weighted_kl": curve["coefficient"] * weighted ** curve["exponent"],
            "weighted_extrapolated": not low <= weighted <= high,
        })
    rows.sort(key=lambda row: -row["weighted_kl"])
    total = math.fsum(row["weighted_kl"] for row in rows)
    for row in rows:
        row["share"] = row["weighted_kl"] / total if total else 0.0

    def summarize_by(field: str) -> list[dict]:
        grouped = []
        for key in sorted({row[field] for row in rows}):
            group = [row for row in rows if row[field] == key]
            grouped.append({field: key, "weighted_kl": math.fsum(row["weighted_kl"] for row in group),
                            "share": math.fsum(row["share"] for row in group)})
        return sorted(grouped, key=lambda row: -row["weighted_kl"])

    return {
        "file": str(path), "sha256": hash_file(path), "weighted_kl_sum": total,
        "unweighted_kl_sum": math.fsum(row["unweighted_kl"] for row in rows),
        "blocks": summarize_by("block"), "types": summarize_by("projection"), "projections": rows,
    }


def render_component_predictions(report: Mapping) -> str:
    lines = [
        f"# Imatrix error attribution | {report['protocol']['teacher']['source']}", "",
        "Each fitted sensitivity curve takes the exported projection's imatrix-weighted residual.",
        "KL estimates are in nats/token. Shares divide by the sum of the weighted projection estimates.",
        "These are additive estimates of isolated effects. Joint model KL depends on interactions between layers.", "",
        f"Imatrix SHA-256 `{report['imatrix']['sha256']}` and sweep SHA-256 `{report['measurement_sha256']}`.", "",
    ]
    for result in report["files"]:
        lines.extend([
            f"## {Path(result['file']).name}", "", f"GGUF SHA-256 `{result['sha256']}`.", "",
            f"Weighted sum **{result['weighted_kl_sum']:.6f}**, unweighted sum {result['unweighted_kl_sum']:.6f}.",
            f"The five largest projections account for {sum(row['share'] for row in result['projections'][:5]):.2%}.",
            "Block indices are zero-based.", "",
        ])
        for group, key, heading in (("blocks", "block", "Blocks"), ("types", "projection", "Projection types")):
            lines.extend([f"### {heading}", "", "| Component | Estimated KL | Share |", "| --- | ---: | ---: |"])
            lines.extend(f"| {row[key]} | {row['weighted_kl']:.6f} | {row['share']:.2%} |" for row in result[group])
            lines.append("")
        lines.extend([
            "### Individual projections", "", "An asterisk marks a weighted error outside the measured sweep range.",
            "", "| Block | Projection | Raw weight error | Noise-equivalent error | Estimated KL | Share |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ])
        for row in result["projections"]:
            marker = "*" if row["weighted_extrapolated"] else ""
            lines.append(
                f"| {row['block']} | {row['projection']} | {row['relative_error']:.3%} | "
                f"{row['noise_equivalent_error']:.3%} | {row['weighted_kl']:.6f}{marker} | {row['share']:.2%} |"
            )
        lines.append("")
    return "\n".join(lines)


def noise_equivalent_error(original: np.ndarray, quantized: np.ndarray, importance: np.ndarray) -> tuple[float, float]:
    """Return the relative weight error and its imatrix-weighted, evenly-spread-noise equivalent."""
    if importance.shape != (original.shape[1],):
        raise ValueError(f"importance has {importance.shape} entries for {original.shape[1]} input columns")
    if not importance.mean() > 0:
        raise ValueError("importance must have a positive mean")
    energy = np.square(quantized.astype(np.float64) - original).sum(axis=0)
    norm = float(np.square(original, dtype=np.float64).sum())
    weights = importance.astype(np.float64) / importance.mean()
    return math.sqrt(energy.sum() / norm), math.sqrt(float(energy @ weights) / norm)


def _ranks(values: Sequence[float]) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def spearman(predicted: Sequence[float], measured: Sequence[float]) -> float:
    return float(np.corrcoef(_ranks(predicted), _ranks(measured))[0, 1])


def summarize(rows: Mapping[str, Mapping[str, float]], joint_kl: float) -> dict:
    measured = [row["measured_isolated_kl"] for row in rows.values()]
    isolated = math.fsum(measured)
    summary: dict = {"sum_isolated_kl": isolated, "joint_kl": joint_kl}
    for name in ("unweighted", "weighted"):
        predicted = [row[f"{name}_prediction"] for row in rows.values()]
        total = math.fsum(predicted)
        summary[name] = {
            "prediction": total,
            "over_isolated": total / isolated,
            "over_joint": total / joint_kl,
            "spearman": spearman(predicted, measured),
            "median_abs_log_ratio": float(np.median(np.abs(np.log(np.divide(predicted, measured))))),
        }
    return summary


def predict_file(path: Path, saved: Mapping, reference, curves: Mapping, importance: Mapping) -> dict:
    projections, _, _ = map_tensors(read_gguf(path), list(saved["projections"]))
    rows = {}
    for key, previous in saved["projections"].items():
        block, projection = key.split(":", 1)
        original = reference.get_tensor(f"model.layers.{block}.{projection}.weight").float().numpy()
        column_importance = importance[f"blk.{block}.{GGUF_NAMES[projection]}.weight"].numpy().ravel()
        error, weighted = noise_equivalent_error(original, projections[key]["values"].numpy(), column_importance)
        if not math.isclose(error, previous["error"], rel_tol=1e-3):
            raise ValueError(f"{path.name} {key}: weight error {error} differs from the measured {previous['error']}")
        curve = curves[key]
        rows[key] = {
            "error": error,
            "noise_equivalent_error": weighted,
            "unweighted_prediction": curve["coefficient"] * error ** curve["exponent"],
            "weighted_prediction": curve["coefficient"] * weighted ** curve["exponent"],
            "measured_isolated_kl": previous["kl_above_control"],
        }
    return {
        "file": path.name, "sha256": saved["sha256"],
        **summarize(rows, saved["groups"]["all"]["all"]["joint_kl"]), "projections": rows,
    }


def render_markdown(report: Mapping) -> str:
    lines = [
        f"# Imatrix-weighted KL prediction: {report['protocol']['teacher']['source']}",
        "",
        f"Computed on {report['computed_at']} from `{report['isolated_report']}` and imatrix",
        f"`{report['imatrix']['path']}`. Each projection's squared weight residual is weighted per input",
        "column by the mean-one importance, and the resulting noise-equivalent error goes through the",
        "sweep's Gaussian-noise curve. Nothing is fitted to the measured KL. Values are in nats/token.",
        "",
        "| GGUF | Unweighted prediction | Weighted prediction | Sum of isolated KLs | Joint KL | "
        "Unweighted / isolated | Weighted / isolated | Weighted / joint | Rank correlation, unweighted | "
        "Rank correlation, weighted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["files"]:
        plain, weighted = item["unweighted"], item["weighted"]
        lines.append(
            f"| {item['file']} | {plain['prediction']:.4f} | {weighted['prediction']:.4f} | "
            f"{item['sum_isolated_kl']:.4f} | {item['joint_kl']:.4f} | {plain['over_isolated']:.3f} | "
            f"{weighted['over_isolated']:.3f} | {weighted['over_joint']:.3f} | {plain['spearman']:.2f} | "
            f"{weighted['spearman']:.2f} |"
        )
    lines += [
        "",
        "Rank correlation is Spearman's, between predicted and measured isolated KL over the projections.",
        f"Full per-projection values: `{report['json']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("isolated", type=Path, help="isolated GGUF report, or sweep results.json with --predict-only")
    parser.add_argument("--imatrix", type=Path, required=True, help="llama.cpp imatrix (GGUF or legacy)")
    parser.add_argument("--gguf", type=Path, action="append", help="GGUF file; repeatable")
    parser.add_argument("--output", type=Path, help="Markdown report (default <isolated>-imatrix-weighted.md)")
    parser.add_argument("--predict-only", action="store_true", help="estimate components from a pinned sweep")
    parser.add_argument("--check-inputs", action="store_true", help="validate prediction inputs without an export")
    parser.add_argument("--imatrix-sha256")
    parser.add_argument("--fits-sha256")
    args = parser.parse_args()
    isolated = args.isolated.resolve()
    if args.check_inputs and not args.predict_only:
        parser.error("--check-inputs requires --predict-only")
    if not args.check_inputs and not args.gguf:
        parser.error("--gguf is required unless checking prediction inputs")
    if args.predict_only:
        identity, curves, matrix, ref_path, keys = prediction_inputs(
            isolated, args.imatrix.resolve(), args.imatrix_sha256, args.fits_sha256,
        )
        if args.check_inputs:
            print(json.dumps(identity, indent=2))
            return 0
        output = (args.output or isolated.with_name("imatrix-error-attribution.md")).resolve()
        with safe_open(ref_path, framework="pt", device="cpu") as reference:
            files = [predict_components(path.resolve(), curves, matrix.entries, reference, keys) for path in args.gguf]
        result = {"schema_version": 1, **identity, "files": files}
        atomic_write_json(output.with_suffix(".json"), result)
        atomic_write_text(output, render_component_predictions(result))
        print(f"Report written to {output}", flush=True)
        return 0
    if args.fits_sha256 is not None:
        parser.error("--fits-sha256 requires --predict-only")
    report = json.loads(isolated.read_text(encoding="utf-8"))
    fits = json.loads(isolated.with_name("kl-response-fits.json").read_text(encoding="utf-8"))
    if not fits["measurement_sha256"] == report["measurement_sha256"] == hash_file(isolated.with_name("results.json")):
        raise ValueError("the isolated report, the fits and the sweep do not belong together")
    output = (args.output or isolated.with_name(isolated.stem + "-imatrix-weighted.md")).resolve()
    matrix = load_imatrix(args.imatrix.resolve(), args.imatrix_sha256)
    spec = spec_for_protocol(report["protocol"])
    snapshot = Path(snapshot_download(repo_id=spec.source, revision=spec.revision, local_files_only=True))
    verify_local_weights(snapshot, spec)
    saved = {item["sha256"]: item for item in report["files"]}
    files = []
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as reference:
        for path in args.gguf:
            path = path.resolve()
            item = predict_file(path, saved[hash_file(path)], reference, fits["curves"], matrix.entries)
            files.append(item)
            print(f"{path.name}: unweighted {item['unweighted']['over_isolated']:.3f}, weighted "
                  f"{item['weighted']['over_isolated']:.3f} of the isolated sum", flush=True)
    result = {
        "schema_version": 1,
        "computed_at": time.strftime("%Y-%m-%d %H:%M"),
        "isolated_report": isolated.name,
        "measurement_sha256": report["measurement_sha256"],
        "protocol": report["protocol"],
        "imatrix": {"path": str(args.imatrix), "sha256": matrix.sha256},
        "files": files,
        "json": str(output.with_suffix(".json")),
    }
    atomic_write_json(output.with_suffix(".json"), result)
    atomic_write_text(output, render_markdown(result))
    print(f"Report written to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
