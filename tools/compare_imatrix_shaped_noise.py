"""Compare matched Gaussian probes with actual GGUF residuals, embeddings fixed.

A plan specifies sources (path and targets), seeds, importance_floor and block_size.
Each synthetic probe matches the actual residual's raw Frobenius norm before casting
to the evaluation dtype. Importance shaping divides noise by sqrt(mean-one column
importance), floored before inversion. Block controls multiply by local weight RMS.
No shape parameter is fitted to KL. Every probe is atomically checkpointed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from measure_gguf_kl import map_tensors, read_gguf, swapped, sweep_session
from measure_joint_kl_targets import projection_weights
from predict_imatrix_weighted_kl import GGUF_NAMES, noise_equivalent_error, spearman

from nanoquant.infrastructure.imatrix import load_imatrix
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file

MODES = ("isotropic", "block", "imatrix", "block_imatrix")


def shaped_noise(original, importance, *, mode, seed, norm, floor, block_size):
    """Construct a deterministic CPU FP32 probe with a specified raw norm."""
    original = original.detach().float().cpu()
    importance = importance.detach().double().cpu().flatten()
    if (mode not in MODES or original.ndim != 2 or importance.shape != (original.shape[1],)
            or not torch.isfinite(importance).all() or (importance < 0).any()
            or not importance.mean() > 0 or not math.isfinite(floor) or floor <= 0
            or not math.isfinite(norm) or norm <= 0 or block_size <= 0):
        raise ValueError("invalid noise shape, importance, norm or floor")
    noise = torch.randn(original.shape, generator=torch.Generator().manual_seed(seed))
    if mode in ("block", "block_imatrix"):
        if original.shape[1] % block_size:
            raise ValueError("block size must divide the input width")
        scale = original.reshape(original.shape[0], -1, block_size).square().mean(-1).sqrt()
        noise *= scale.repeat_interleave(block_size, dim=1)
    if mode in ("imatrix", "block_imatrix"):
        normalized = importance / importance.mean()
        noise *= normalized.clamp_min(floor).rsqrt().float()[None, :]
    magnitude = float(torch.linalg.vector_norm(noise.double()))
    if magnitude == 0:
        raise ValueError("noise has no nonzero entries")
    return noise * (norm / magnitude)


def residual_metrics(original, candidate, importance):
    raw, weighted = noise_equivalent_error(original.numpy(), candidate.numpy(), importance.numpy())
    return {"raw_error": raw, "weighted_error": weighted,
            "weighted_to_raw_energy": (weighted / raw) ** 2}


def prediction_metrics(predicted, measured):
    p, m = np.asarray(predicted), np.asarray(measured)
    if np.any(p <= 0) or np.any(m <= 0):
        raise ValueError("prediction comparisons require positive KL above control")
    return {"sum_ratio": float(p.sum() / m.sum()),
            "median_fold_error": float(np.exp(np.median(np.abs(np.log(p / m))))),
            "spearman": spearman(predicted, measured) if len(p) > 1 else None}


def summarize(report):
    summaries = {}
    seeds = report["identity"]["plan"]["seeds"]
    for source in report["identity"]["plan"]["sources"]:
        cases = [case for case in report["cases"].values() if case["source"] == source
                 and "actual" in case["probes"]
                 and all(f"{mode}:{seed}" in case["probes"] for mode in MODES for seed in seeds)]
        if not cases:
            continue
        measured = [case["probes"]["actual"]["kl_above_clean"] for case in cases]
        summary = {"count": len(cases), "actual_kl_sum": sum(measured), "predictors": {}}
        for method in ("curve_raw", "curve_weighted", *MODES):
            predicted = [case[method] if method.startswith("curve_") else
                         float(np.mean([case["probes"][f"{method}:{seed}"]["kl_above_clean"]
                                        for seed in seeds])) for case in cases]
            summary["predictors"][method] = prediction_metrics(predicted, measured)
        summaries[source] = summary
    return summaries


def render(report):
    lines = ["# IMatrix-shaped noise versus actual residuals", "",
             "Pinned sensitivity teacher; decoder projections only. Embeddings, tied output and norms fixed.",
             "Noise matches each candidate's raw residual norm. Metrics use residuals after evaluation-dtype casting.",
             "Three seeds are repeated noise draws, not independent text datasets. No shape parameters fitted to KL.",
             f"Completed {sum(len(c['probes']) for c in report['cases'].values())}/{report['expected_probes']} probes.",
             "", "| Source | Cases | Predictor | Predicted / actual sum | Median fold error | Spearman |",
             "| --- | ---: | --- | ---: | ---: | ---: |"]
    for source, summary in report["summary"].items():
        for method, result in summary["predictors"].items():
            rank = result["spearman"]
            rank_text = "—" if rank is None else f"{rank:.3f}"
            lines.append(f"| {source} | {summary['count']} | {method} | {result['sum_ratio']:.3f} | "
                         f"{result['median_fold_error']:.3f} | {rank_text} |")
    lines += ["", "JSON retains all per-window scores, source/code hashes, residual norms, and seed-level probes.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    sweep = json.loads(args.measurements.read_text(encoding="utf-8"))
    known = set(sweep["protocol"]["targets"])
    for spec in plan["sources"].values():
        if not spec["targets"] or not set(spec["targets"]) <= known:
            raise ValueError("invalid source targets")
        if len(set(spec["targets"])) != len(spec["targets"]):
            raise ValueError("duplicate source targets")
    if not plan["seeds"] or len(set(plan["seeds"])) != len(plan["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    matrix = load_imatrix(Path(plan["imatrix"]), plan["imatrix_sha256"])
    dependencies = [Path(__file__), *(Path(__file__).with_name(name) for name in (
        "measure_gguf_kl.py", "measure_joint_kl_targets.py", "kl_sensitivity_dataset.py",
        "materialize_nanoquant_gguf.py", "predict_imatrix_weighted_kl.py", "fit_kl_sensitivity.py")),
        Path(__file__).parents[1] / "src/nanoquant/infrastructure/imatrix.py"]
    identity = {"plan": plan, "measurement_sha256": hash_file(args.measurements),
                "source_hashes": {name: hash_file(Path(spec["path"])) for name, spec in plan["sources"].items()},
                "code": {path.name: hash_file(path) for path in dependencies}, "embedding_policy": "clean-fixed"}
    output = args.output.resolve().with_suffix(".json")
    prior = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
    if prior is not None and prior["identity"] != identity:
        raise ValueError("resume identity changed; use a new output")
    expected = sum(len(spec["targets"]) for spec in plan["sources"].values()) * (1 + len(MODES)*len(plan["seeds"]))
    with sweep_session(args.measurements, local_files_only=True) as session:
        report = {"schema_version": 1, "identity": identity, "protocol": session.protocol,
                  "clean": session.clean, "expected_probes": expected, "summary": {},
                  "cases": {} if prior is None else prior["cases"]}
        weights = projection_weights(session.model, session.adapter, session.targets)
        # Guard the shared embedding / output tensor, in addition to limiting swaps to projections.
        embedding = session.model.get_input_embeddings().weight
        embedding_before = embedding.detach().cpu().clone()

        def save():
            report["summary"] = summarize(report)
            atomic_write_json(output, report)
            atomic_write_text(output.with_suffix(".md"), render(report))

        save()
        for source, spec in plan["sources"].items():
            print(f"Loading {source}", flush=True)
            projections, _, _ = map_tensors(read_gguf(Path(spec["path"])), session.targets)
            for target in spec["targets"]:
                key = f"{source}/{target}"
                weight = weights[target]
                original = weight.detach().float().cpu()
                actual = projections[target]["values"].to(dtype=weight.dtype).float()
                block, path = target.split(":", 1)
                importance = matrix.entries[f"blk.{block}.{GGUF_NAMES[path]}.weight"].flatten()
                normalized = importance / importance.mean()
                metrics = residual_metrics(original, actual, importance)
                curve = session.curves[target]
                low, high = curve["measured_error_range"]
                case = report["cases"].setdefault(key, {
                    "source": source, "target": target, "storage_type": projections[target]["type"],
                    "actual_residual": metrics, "importance_min_mean_one": float(normalized.min()),
                    "importance_floored_fraction": float((normalized < plan["importance_floor"]).float().mean()),
                    "curve_raw": curve["coefficient"] * metrics["raw_error"] ** curve["exponent"],
                    "curve_weighted": curve["coefficient"] * metrics["weighted_error"] ** curve["exponent"],
                    "curve_raw_extrapolated": not low <= metrics["raw_error"] <= high,
                    "curve_weighted_extrapolated": not low <= metrics["weighted_error"] <= high, "probes": {},
                })
                norm = float(torch.linalg.vector_norm((actual-original).double()))
                target_seed = int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], "little")
                tasks = [("actual", None, None),
                         *[(f"{mode}:{seed}", mode, seed) for mode in MODES for seed in plan["seeds"]]]
                for label, mode, seed in tasks:
                    if label in case["probes"]:
                        continue
                    started = time.perf_counter()
                    candidate = actual if mode is None else (original + shaped_noise(
                        original, importance, mode=mode, seed=target_seed + seed, norm=norm,
                        floor=plan["importance_floor"], block_size=plan["block_size"],
                    )).to(dtype=weight.dtype).float()
                    with swapped({target: (weight, candidate)}):
                        score = session.score()
                    if not torch.equal(weight.detach().float().cpu(), original):
                        raise RuntimeError("projection was not restored exactly")
                    case["probes"][label] = {
                        **residual_metrics(original, candidate, importance), "score": score,
                        "kl_above_clean": score["kl_nats_per_token"] - session.clean["kl_nats_per_token"],
                        "seconds": time.perf_counter() - started,
                    }
                    save()
                    count = sum(len(c["probes"]) for c in report["cases"].values())
                    print(f"{count}/{expected} {key} {label}: KL={case['probes'][label]['kl_above_clean']:.7f}",
                          flush=True)
            del projections
        if not torch.equal(embedding.detach().cpu(), embedding_before):
            raise RuntimeError("embedding changed")
        report["embedding_verified_unchanged"] = True
        report["complete"] = True
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
