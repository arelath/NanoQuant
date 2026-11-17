"""Score decoder exports on a larger teacher dataset and a nested 16-window control."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from kl_sensitivity_dataset import GEMMA_1B, score_model, verify_local_weights
from measure_gguf_kl import map_tensors, read_gguf, swapped
from measure_joint_kl_targets import projection_weights

from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file
from nanoquant.infrastructure.kl_teacher_dataset import load_kl_teacher_dataset
from nanoquant.infrastructure.model_adapters import adapter_for_config


def paired_delta(score: dict, clean: dict, selected: list[int]) -> dict:
    values = [score["sequences"][i]["kl_nats_per_token"] - clean["sequences"][i]["kl_nats_per_token"]
              for i in selected]
    return {"kl": statistics.mean(values), "window_standard_error": statistics.stdev(values) / math.sqrt(len(values))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("reference", type=Path, help="earlier joint comparison JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    dataset = load_kl_teacher_dataset(args.dataset)
    protocol = dataset.protocol
    spec = GEMMA_1B
    if (protocol.source, protocol.revision, protocol.model_dtype) != (spec.source, spec.revision, str(spec.dtype)):
        raise ValueError("teacher dataset model or precision differs from the pinned comparison")
    snapshot = Path(snapshot_download(repo_id=spec.source, revision=spec.revision, local_files_only=True))
    verify_local_weights(snapshot, spec)
    sources = {number: Path("Results") / number / "gemma-3-1b-it-nanoquant.gguf"
               for number in reference["experiments"]}
    identity = {
        "teacher": dataset.reference.artifact_id, "reference": hash_file(args.reference),
        "exports": {key: hash_file(path) for key, path in sources.items()},
        "code": hash_file(Path(__file__)), "torch": str(torch.__version__),
    }
    for key, digest in identity["exports"].items():
        if digest != reference["experiments"][key]["gguf"]["sha256"]:
            raise ValueError(f"{key} export differs from the reference")
    output = args.output.with_suffix(".json")
    prior = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
    if prior is not None and prior["identity"] != identity:
        raise ValueError("comparison inputs changed")
    selected = sorted(random.Random(83).sample(range(protocol.samples), 16))
    report = {"identity": identity, "nested_indices": selected, "results": {} if prior is None else prior["results"]}
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    targets = [f"{block}:{name}" for block in range(26) for name in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    )]
    with acquire_device_lease(protocol.device):
        model = load_causal_language_model(
            snapshot, torch_dtype=spec.dtype, attention_implementation=adapter.attention_implementation,
            local_files_only=True,
        ).to(protocol.device).eval()
        clean = score_model(model, dataset, device=protocol.device, token_chunk_size=32)
        if prior is not None and clean != prior["clean"]:
            raise ValueError("clean score changed")
        report["clean"] = clean
        weights = projection_weights(model, adapter, targets)
        for number, path in sources.items():
            if number in report["results"]:
                continue
            started = time.perf_counter()
            print(f"Scoring {number} on {protocol.samples} windows", flush=True)
            projections, _, _ = map_tensors(read_gguf(path), targets)
            with swapped({key: (weights[key], value["values"]) for key, value in projections.items()}):
                score = score_model(model, dataset, device=protocol.device, token_chunk_size=32)
            report["results"][number] = {
                "score": score, "all_windows": paired_delta(score, clean, list(range(protocol.samples))),
                "nested_16": paired_delta(score, clean, selected), "seconds": time.perf_counter() - started,
            }
            atomic_write_json(output, report)
            del projections
        report["complete"] = True
        atomic_write_json(output, report)
    lines = [
        "# Curated corpus joint decoder KL", "",
        f"The larger dataset has {protocol.samples} windows and {clean['token_count']:,} scored positions.",
        "All exports use the clean reference embeddings. The nested 16 windows are sampled with seed 83.",
        "Window standard errors describe variation within this sample.", "",
        "| Experiment | Old corpus 16 windows | Curated 16 windows | Curated full sample | Window SE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, result in report["results"].items():
        old = reference["experiments"][key]["gguf"]["variants"]["projections"]["kl_above_control"]
        full, small = result["all_windows"], result["nested_16"]
        lines.append(f"| {key} | {old:.6f} | {small['kl']:.6f} | {full['kl']:.6f} | "
                     f"{full['window_standard_error']:.6f} |")
    atomic_write_text(args.output, "\n".join([*lines, ""]))
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
