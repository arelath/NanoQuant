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

import numpy as np
from huggingface_hub import snapshot_download
from kl_sensitivity_dataset import verify_local_weights
from measure_gguf_kl import PROJECTIONS, map_tensors, read_gguf
from measure_joint_kl_targets import spec_for_protocol
from safetensors import safe_open

from nanoquant.infrastructure.imatrix import load_imatrix
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file

GGUF_NAMES = {path: kind for kind, path in PROJECTIONS.items()}


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
    parser.add_argument("isolated", type=Path, help="a gguf-isolated-kl JSON from measure_gguf_isolated_kl.py")
    parser.add_argument("--imatrix", type=Path, required=True, help="llama.cpp imatrix (GGUF or legacy)")
    parser.add_argument("--gguf", type=Path, action="append", required=True, help="GGUF file; repeatable")
    parser.add_argument("--output", type=Path, help="Markdown report (default <isolated>-imatrix-weighted.md)")
    args = parser.parse_args()
    isolated = args.isolated.resolve()
    report = json.loads(isolated.read_text(encoding="utf-8"))
    fits = json.loads(isolated.with_name("kl-response-fits.json").read_text(encoding="utf-8"))
    if not fits["measurement_sha256"] == report["measurement_sha256"] == hash_file(isolated.with_name("results.json")):
        raise ValueError("the isolated report, the fits and the sweep do not belong together")
    output = (args.output or isolated.with_name(isolated.stem + "-imatrix-weighted.md")).resolve()
    matrix = load_imatrix(args.imatrix.resolve())
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
