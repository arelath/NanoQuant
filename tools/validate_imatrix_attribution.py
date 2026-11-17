"""Measure one exported NanoQuant projection at a time on the attribution sweep's teacher."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import gguf
import torch
from materialize_nanoquant_gguf import iter_dense_nanoquant_projections
from measure_gguf_kl import swapped, sweep_session
from measure_joint_kl_targets import projection_weights
from predict_imatrix_weighted_kl import GGUF_NAMES, spearman

from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file


def summarize(rows: list[dict], estimate: str) -> dict:
    predicted = [row[estimate] for row in rows]
    measured = [row["measured_kl"] for row in rows]
    positive = [(p, m) for p, m in zip(predicted, measured, strict=True) if min(p, m) > 0]
    return {
        "predicted_sum": math.fsum(predicted),
        "measured_sum": math.fsum(measured),
        "spearman": spearman(predicted, measured),
        "median_fold_error": math.exp(statistics.median(abs(math.log(p / m)) for p, m in positive))
        if positive else None,
        "positive_pairs": len(positive),
        "top_10_overlap": len(
            set(sorted(range(len(rows)), key=predicted.__getitem__, reverse=True)[:10])
            & set(sorted(range(len(rows)), key=measured.__getitem__, reverse=True)[:10])
        ),
    }


def render(report: dict, attribution: Path) -> str:
    lines = [
        "# Measured versus estimated projection KL", "",
        f"Estimates from [{attribution.stem}]({attribution.with_suffix('.md').name}).",
        f"Each exported projection was swapped alone into the clean FP16 model on the original sweep's "
        f"{report['clean']['token_count']:,} scored token positions.",
        "Measured values are full-vocabulary KL(teacher || perturbed) above the clean control, in nats/token.",
        "Q, K and V were scored individually. Embeddings and the tied output remained at the clean reference.",
        "", "## Agreement across projections", "",
        "| Estimator | Predicted sum | Measured isolated sum | Spearman | Median fold error | Top 10 overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in report["summary"].items():
        lines.append(f"| {name} | {values['predicted_sum']:.6f} | {values['measured_sum']:.6f} | "
                     f"{values['spearman']:.3f} | {values['median_fold_error']:.3f} | "
                     f"{values['top_10_overlap']}/10 |")
    lines.extend(["", "Fold errors use positive measured and predicted values.", "",
                  "## Breakdown by projection type", "",
                  "| Projection type | Estimated weighted sum | Measured sum |",
                  "| --- | ---: | ---: |"])
    for kind, values in report["projection_types"].items():
        lines.append(f"| {kind} | {values['weighted']:.6f} | {values['measured']:.6f} |")
    lines.extend(["", "## Individual projections", "",
                  "An asterisk marks estimates outside the sweep's measured error range.", "",
                  "| Block | Projection | Estimated weighted KL | Measured KL | Estimated unweighted KL |",
                  "| ---: | --- | ---: | ---: | ---: |"])
    for row in sorted(report["projections"].values(), key=lambda row: -row["measured_kl"]):
        marker = "*" if row["weighted_extrapolated"] else ""
        lines.append(f"| {row['block']} | {row['projection']} | {row['weighted_kl']:.8f}{marker} | "
                     f"{row['measured_kl']:.8f} | {row['unweighted_kl']:.8f} |")
    return "\n".join([*lines, ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attribution", type=Path, help="the published attribution JSON")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path")
    args = parser.parse_args()
    attribution = args.attribution.resolve()
    output = args.output.resolve()
    data = output.with_suffix(".json")
    if data == attribution:
        raise ValueError("output would overwrite the attribution")
    source = json.loads(attribution.read_text(encoding="utf-8"))
    measurements = Path(source["measurements"])
    gguf_hash = hash_file(args.gguf)
    matches = [item for item in source["files"] if item["sha256"] == gguf_hash]
    if len(matches) != 1 or hash_file(measurements) != source["measurement_sha256"]:
        raise ValueError("GGUF or sensitivity sweep differs from the attribution inputs")
    estimates = {f"{r['block']}:{r['projection']}": r for r in matches[0]["projections"]}
    targets = source["protocol"]["targets"]
    if len(estimates) != len(matches[0]["projections"]) or set(estimates) != set(targets):
        raise ValueError("attribution estimates must cover each sweep projection exactly once")
    names = {}
    for target in targets:
        block, projection = target.split(":", 1)
        names[f"blk.{block}.{GGUF_NAMES[projection]}.weight"] = target
    dependencies = [Path(__file__), *(Path(__file__).with_name(name) for name in (
        "measure_gguf_kl.py", "kl_sensitivity_dataset.py", "measure_joint_kl_targets.py",
        "materialize_nanoquant_gguf.py", "weight_image_common.py",
    ))]
    identity = {
        "gguf_sha256": gguf_hash, "attribution_sha256": hash_file(attribution),
        "measurement_sha256": source["measurement_sha256"], "torch_version": torch.__version__,
        "code": {path.name: hash_file(path) for path in dependencies},
    }
    prior = json.loads(data.read_text(encoding="utf-8")) if data.exists() else None
    if prior is not None and prior["identity"] != identity:
        raise ValueError("resume inputs differ from the saved run")
    print(f"Validated attribution, sweep and GGUF hashes for {len(targets)} projections.", flush=True)
    reader = gguf.GGUFReader(args.gguf)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    with sweep_session(measurements, local_files_only=True) as session:
        if session.protocol != source["protocol"]:
            raise ValueError("the scoring protocol differs from the attribution")
        if prior is not None and prior["clean"] != session.clean:
            raise ValueError("the clean score differs from the saved run")
        report = {
            "schema_version": 1, "identity": identity, "protocol": session.protocol,
            "clean": session.clean, "expected_projections": len(targets),
            "projections": {} if prior is None else prior["projections"], "complete": False,
        }
        rows = report["projections"]
        weights = projection_weights(session.model, session.adapter, targets)
        embedding = session.model.get_input_embeddings().weight
        embedding_before = embedding.detach().cpu().clone()
        unseen = set(names)
        for name, projection in iter_dense_nanoquant_projections(reader.fields, tensors, gguf):
            unseen.remove(name)
            target = names[name]
            if target in rows:
                continue
            started = time.perf_counter()
            weight = weights[target]
            original = weight.detach().cpu().clone()
            candidate = torch.from_numpy(projection.weight)
            if candidate.shape != weight.shape:
                raise ValueError(f"reconstructed shape differs for {target}")
            with swapped({target: (weight, candidate)}):
                score = session.score()
            if not torch.equal(weight.detach().cpu(), original):
                raise RuntimeError(f"failed to restore {target}")
            kl = score["kl_nats_per_token"] - session.clean["kl_nats_per_token"]
            rows[target] = {**estimates[target], "score": score, "measured_kl": kl,
                            "seconds": time.perf_counter() - started}
            atomic_write_json(data, report)
            print(f"{len(rows)}/{len(targets)} {target}: KL={kl:.8f}", flush=True)
        if unseen or set(rows) != set(targets):
            raise RuntimeError("GGUF did not cover every projection exactly once")
        if not torch.equal(embedding.detach().cpu(), embedding_before):
            raise RuntimeError("the shared embedding changed")
        all_rows = list(rows.values())
        report["summary"] = {method: summarize(all_rows, f"{method}_kl") for method in ("weighted", "unweighted")}
        report["projection_types"] = {}
        for kind in sorted(GGUF_NAMES):
            group = [row for row in all_rows if row["projection"] == kind]
            report["projection_types"][kind] = {
                label: math.fsum(row[f"{label}_kl"] for row in group) for label in ("weighted", "measured")
            }
        report["embedding_verified_unchanged"] = True
        report["complete"] = True
        atomic_write_json(data, report)
        atomic_write_text(output, render(report, attribution))
        print(f"Report written to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
