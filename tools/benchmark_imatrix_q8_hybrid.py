"""Export a compact Q8_0 hybrid and score it with the original dense quality protocol."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch
from measure_gguf_kl import PROJECTIONS
from safetensors import safe_open

from nanoquant.application.layers import DenseWeightReferenceLinear
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.device_memory import SharedDeviceMemoryMonitor
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.packed_model_loader import load_packed_model
from nanoquant.quality_evaluation import (
    QualityEvaluationRequest,
    _evaluate_model,
    compare_quality_results,
    describe_quality_protocol,
    prepare_quality_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
Q8 = gguf.GGMLQuantizationType.Q8_0


def selected_projections(artifact: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    if not 0 <= threshold < 1:
        raise ValueError("share threshold must be in [0, 1)")
    reverses = {kind: name for name, kind in PROJECTIONS.items()}
    selected = []
    seen = set()
    for row in artifact["projections"]:
        key = (row["block"], row["projection"])
        if key in seen or not 0 <= row["share"] <= 1:
            raise ValueError(f"invalid attribution row: {key}")
        seen.add(key)
        if row["share"] > threshold:
            selected.append({**row, "tensor": f"blk.{key[0]}.{reverses[key[1]]}.weight"})
    if not selected:
        raise ValueError("no projections exceed the share threshold")
    return selected


def export_hybrid(source: Path, checkpoint: Path, output: Path, selected: list[dict[str, Any]]) -> None:
    """Replace ungrouped factor owners and keep every other stored tensor byte-identical."""
    reader = gguf.GGUFReader(source)
    tensors = {t.name: t for t in reader.tensors}
    owners = {row["tensor"].removesuffix(".weight"): row for row in selected}
    removed = set()
    for owner in owners:
        sidecars = {name for name in tensors if name.startswith(owner + ".nq_")}
        if not sidecars or owner + ".weight" in tensors:
            raise ValueError(f"expected an independent staged NanoQuant projection: {owner}")
        removed.update(sidecars)
    architecture = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(None, architecture, endianess=reader.endianess)
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        if any(name.startswith(f"quantization.nanoquant.{owner}.") for owner in owners):
            continue
        if name == "general.alignment":
            writer.add_custom_alignment(int(field.contents()))
            continue
        value = field.contents()
        if name == "quantization.nanoquant.tensor_count":
            value = int(value) - len(owners)
        kind = field.types[0]
        subtype = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(name, value, kind, sub_type=subtype)
    untouched = tensors.keys() - removed
    for name in sorted(untouched):
        tensor = tensors[name]
        writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)
    encoded = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as reference:
        for row in selected:
            key = f"model.layers.{row['block']}.{row['projection']}.weight"
            weight = reference.get_tensor(key)
            if weight.dtype != torch.bfloat16 or weight.ndim != 2 or weight.shape[-1] % 32:
                raise ValueError(f"source weight does not support BF16 to Q8_0 conversion: {key}")
            name = row["tensor"]
            encoded[name] = gguf.quantize(weight.float().numpy(), Q8)
            writer.add_tensor(name, encoded[name], raw_dtype=Q8)
    with tempfile.TemporaryDirectory(prefix=".q8-hybrid-", dir=output.parent) as tmp:
        staged = Path(tmp) / output.name
        writer.write_header_to_file(staged)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        validate_export(staged, tensors, untouched, encoded)
        staged.replace(output)


def validate_export(staged: Path, original: dict, untouched: set, encoded: dict) -> None:
    # Close all ndarray views of the staged file before its atomic rename on Windows.
    tensors = {t.name: t for t in gguf.GGUFReader(staged).tensors}
    if tensors.keys() != untouched | encoded.keys():
        raise ValueError("hybrid GGUF tensor inventory differs from the replacement plan")
    for name, tensor in tensors.items():
        if name in encoded:
            expected_kind, expected = Q8, encoded[name]
        else:
            expected_kind, expected = original[name].tensor_type, original[name].data
        if tensor.tensor_type != expected_kind or not np.array_equal(tensor.data, expected):
            raise ValueError(f"exported tensor differs from the validated payload: {name}")


def replace_dense_projections(model, gguf_path: Path, selected: list[dict[str, Any]]) -> None:
    blocks = adapter_for_config(model.config.to_dict()).get_decoder_layers(model)
    tensors = {t.name: t for t in gguf.GGUFReader(gguf_path).tensors}
    with torch.no_grad():
        for row in selected:
            tensor = tensors[row["tensor"]]
            if tensor.tensor_type != Q8:
                raise ValueError(f"replacement has incorrect type: {tensor.name}")
            target = blocks[row["block"]].get_submodule(row["projection"])
            if not isinstance(target, DenseWeightReferenceLinear):
                raise ValueError(f"replacement is not an independent dense reference projection: {tensor.name}")
            weight = target.dense_weight()
            decoded = torch.from_numpy(gguf.dequantize(tensor.data, Q8))
            if decoded.shape != weight.shape:
                raise ValueError(f"replacement has incompatible shape: {tensor.name}")
            weight.copy_(decoded)


def request_for(baseline: dict[str, Any]) -> QualityEvaluationRequest:
    protocol, model, candidate = (baseline[k] for k in ("protocol", "model", "candidate"))
    return QualityEvaluationRequest(
        snapshot=Path(model["snapshot"]), source=model["source"], revision=model["revision"],
        run_output=Path(candidate["run_output"]), packed_artifact=Path(candidate["packed_artifact"]),
        backend="dense", use_global_tuning=False, local_files_only=True, device="cuda:0",
        wikitext_samples=protocol["wikitext_samples"], wikitext_sequence_length=protocol["wikitext_sequence_length"],
        wikitext_batch_size=protocol["wikitext_batch_size"], task_names=tuple(protocol["task_names"]),
        task_limit=protocol["task_limit"], task_batch_size=protocol["task_batch_size"], include_mmlu="mmlu" in protocol,
        maximum_wddm_shared_bytes=baseline["resource_limits"]["maximum_wddm_shared_bytes"],
    )


def progress(event: str, fields) -> None:
    if event.endswith(("_started", "_completed")) and "_batch_" not in event:
        print(event, dict(fields), flush=True)
    elif event == "task_batch_completed" and fields["completed_batches"] % 50 == 0:
        print(event, dict(fields), flush=True)


def score_hybrid(baseline: dict, request: QualityEvaluationRequest, gguf_path: Path, selected: list) -> dict:
    prepared = prepare_quality_inputs(request, progress)
    protocol = json.loads(json.dumps(describe_quality_protocol(request, prepared)))
    old = dict(baseline["protocol"])
    old.pop("base_execution", None)
    # Arrow paths can change its fingerprint. Identical pinned revision and scored token digest are required.
    for entry in (protocol, old):
        entry.pop("wikitext_fingerprint", None)
    if protocol != old:
        raise ValueError("quality protocol differs from the original run")
    expected_tasks = {x["result"]["task_name"]: x["task_input_identity"] for x in baseline["results"]["base"]["tasks"]}
    prepared_tasks = {x.task.task_name: to_dict(x.cache_identity) for x in prepared.tasks}
    if prepared.mmlu is not None:
        prepared_tasks["mmlu"] = prepared.mmlu.identity
    if prepared_tasks != expected_tasks:
        raise ValueError("prepared task inputs differ from the original run")
    limit = request.maximum_wddm_shared_bytes
    if limit is None or request.packed_artifact is None:
        raise ValueError("the matched hybrid replay requires a packed candidate and its memory guard")
    with acquire_device_lease(request.device), SharedDeviceMemoryMonitor(limit) as monitor:
        loaded = load_packed_model(
            request.packed_artifact, request.run_output, request.snapshot, source_name=request.source,
            revision=request.revision, device=request.device, backend="dense", use_global_tuning=False,
        )
        candidate = baseline["candidate"]
        if (to_dict(loaded.identity) != candidate["commit_identity"]
                or loaded.packed_descriptor_sha256 != candidate["packed_descriptor_sha256"]):
            raise ValueError("packed reference identity differs from the original quality benchmark")
        replace_dense_projections(loaded.model, gguf_path, selected)
        monitor.check()
        hybrid = _evaluate_model("q8_hybrid", loaded.model, request, prepared, monitor, progress)
    baseline_scores = baseline["results"]["frozen"]
    return {
        "protocol": protocol, "gguf": str(gguf_path), "gguf_sha256": hash_file(gguf_path),
        "projection_count": len(selected), "estimated_share": math.fsum(row["share"] for row in selected),
        "replacements": selected, "hybrid": hybrid,
        "comparison_to_baseline": compare_quality_results(baseline_scores, hybrid),
        "comparison_to_bf16": compare_quality_results(baseline["results"]["base"], hybrid),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="compact hybrid GGUF output path")
    parser.add_argument("--report", type=Path, required=True, help="per-example quality JSON path")
    parser.add_argument("--share", type=float, default=0.01)
    parser.add_argument("--audit", type=Path, required=True, help="fresh complete resident audit path")
    args = parser.parse_args()
    attribution = json.loads(args.attribution.read_bytes())
    baseline = json.loads(args.quality.read_bytes())
    artifact, = attribution["files"]
    source = Path(artifact["file"])
    checkpoint = Path(attribution["checkpoint"])
    selected = selected_projections(artifact, args.share)
    if hash_file(source) != artifact["sha256"]:
        raise ValueError("attribution belongs to a different GGUF")
    if hash_file(checkpoint) != attribution["protocol"]["model_weights_sha256"]:
        raise ValueError("BF16 checkpoint differs from the pinned sensitivity source")
    request = request_for(baseline)
    if checkpoint.parent != request.snapshot or not baseline["passed"] or baseline["candidate"]["global_tuning"]:
        raise ValueError("expected a completed no-KD quality baseline from the same checkpoint")
    audit = json.loads(args.audit.read_bytes())
    if audit["identity"] != baseline["candidate"]["commit_identity"] or Path(audit["run_output"]) != request.run_output:
        raise ValueError("resident audit does not identify the quality baseline")
    if not audit["complete"] or len(artifact["projections"]) != 7 * len(audit["completed_blocks"]):
        raise ValueError("resident audit or attribution is incomplete")
    out = args.output.resolve()
    if out.exists() or out == source or args.report.exists():
        raise FileExistsError("hybrid and report need separate, unused output paths")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    export_hybrid(source, checkpoint, out, selected)
    print(f"Exported {len(selected)} Q8_0 projections to {out}", flush=True)
    result = score_hybrid(baseline, request, out, selected)
    atomic_write_json(args.report, {
        **result, "threshold": args.share, "attribution": str(args.attribution.resolve()),
        "attribution_sha256": hash_file(args.attribution), "control_gguf_sha256": artifact["sha256"],
        "original_quality_sha256": hash_file(args.quality), "gguf_bytes": out.stat().st_size,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
