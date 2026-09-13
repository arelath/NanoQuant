"""Profile a downloaded Gemma GGUF and retained Experiment 060 on identical inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import _paths  # noqa: F401
import torch
from safetensors import safe_open

from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.llamacpp_quality import (
    LlamaCppQualityRequest,
    _quality_sequences,
    _write_input,
    execute_llamacpp_quality_evaluation,
)
from nanoquant.quality_evaluation import QualityEvaluationRequest, prepare_quality_inputs

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "Results/060/060-best-methods-2bpw-compress-and-benchmark-gemma-3-1b-it-quality.json"
REFERENCE_GGUF = ROOT / "Results/060/gemma-3-1b-it-nanoquant.gguf"
LLAMA = ROOT.parent / "llama.cpp"


def check_fields(observed, expected):
    for key, value in observed.items():
        if value != expected[key]:
            raise ValueError(f"Protocol mismatch for {key}: {value!r} != {expected[key]!r}")


def inventory(path):
    sys.path.insert(0, str(LLAMA / "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(path, "r")
    types = defaultdict(lambda: {"tensors": 0, "parameters": 0, "bytes": 0})
    for tensor in reader.tensors:
        group = types[tensor.tensor_type.name]
        group["tensors"] += 1
        group["parameters"] += int(tensor.n_elements)
        group["bytes"] += int(tensor.n_bytes)
    vocabulary = {}
    for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.bos_token_id",
                "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id"):
        contents = reader.fields[key].contents()
        vocabulary[key] = hashlib.sha256(json.dumps(contents, ensure_ascii=False).encode()).hexdigest()
    embedding = next(t for t in reader.tensors if t.name == "token_embd.weight")
    return {
        "path": str(path), "file_bytes": path.stat().st_size, "file_sha256": hash_file(path),
        "tensor_types": dict(types), "vocabulary_hashes": vocabulary,
        "architecture": {key: value.contents() for key, value in reader.fields.items()
                         if key.startswith("gemma3.") or key == "general.architecture"},
        "embedding_sha256": hashlib.sha256(embedding.data.tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-receipt", type=Path,
                        default=ROOT / "outputs/external-2bit-profile/download-receipt.json")
    parser.add_argument("--output", type=Path, default=ROOT / "Results/069")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def progress(event, fields):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        print(json.dumps(record), flush=True)
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        atomic_write_json(output / "status.json", record)

    try:
        reference = json.loads(REFERENCE.read_text())
        download = json.loads(args.download_receipt.read_text())
        external = Path(download["path"])
        if hash_file(external) != download["sha256"] or external.stat().st_size != download["bytes"]:
            raise ValueError("download no longer matches Hugging Face receipt")
        export = json.loads(REFERENCE_GGUF.with_suffix(".gguf.export.json").read_text())
        if hash_file(REFERENCE_GGUF) != export["gguf_sha256"]:
            raise ValueError("retained NanoQuant GGUF no longer matches its export receipt")
        protocol, model = reference["protocol"], reference["model"]
        progress("inventory_started", {})
        inventories = {"external_iq2_m": inventory(external), "nanoquant060": inventory(REFERENCE_GGUF)}
        check_fields(inventories["external_iq2_m"]["vocabulary_hashes"],
                     inventories["nanoquant060"]["vocabulary_hashes"])
        # The downloaded GGUF omits the SWA rope key; this pinned runtime's
        # llama-hparams.h defaults it to 10000, matching the NanoQuant GGUF.
        for item in inventories.values():
            item["architecture"].setdefault("gemma3.rope.freq_base_swa", 10000.0)
        check_fields(inventories["external_iq2_m"]["architecture"],
                     inventories["nanoquant060"]["architecture"])
        parameters = 0
        for shard in Path(model["snapshot"]).glob("*.safetensors"):
            with safe_open(str(shard), framework="pt", device="cpu") as tensors:
                for key in tensors.keys():
                    count = 1
                    for size in tensors.get_slice(key).get_shape():
                        count *= size
                    parameters += count
        external_parameters = sum(x["parameters"] for x in inventories["external_iq2_m"]["tensor_types"].values())
        if not parameters or external_parameters != parameters:
            raise ValueError(f"source and external model parameter counts differ: {parameters}, {external_parameters}")
        for item in inventories.values():
            item["original_model_parameters"] = parameters
            item["whole_file_bpw"] = 8*item["file_bytes"]/parameters
        atomic_write_json(output / "gguf-inventory.json", inventories)
        request = QualityEvaluationRequest(
            snapshot=Path(model["snapshot"]), source=model["source"], revision=model["revision"],
            run_output=output, device="cuda:0", wikitext_samples=protocol["wikitext_samples"],
            wikitext_sequence_length=protocol["wikitext_sequence_length"],
            wikitext_batch_size=protocol["wikitext_batch_size"], task_names=tuple(protocol["task_names"]),
            task_limit=protocol["task_limit"], task_batch_size=protocol["task_batch_size"], local_files_only=True)
        prepared = prepare_quality_inputs(request, progress=progress)
        checks = {
            "wikitext_token_hash": "sha256:" + hashlib.sha256(
                prepared.wikitext_tokens.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            "wikitext_fingerprint": prepared.wikitext_fingerprint, "tokenizer_hash": prepared.tokenizer_hash,
            "bos_token_id": prepared.bos_token_id, "pad_token_id": prepared.pad_token_id,
        }
        check_fields(checks, protocol)
        for task, recorded in zip(prepared.tasks, reference["results"]["base"]["tasks"], strict=True):
            if to_dict(task.cache_identity) != recorded["task_input_identity"]:
                raise ValueError(f"task input identity mismatch: {task.task.task_name}")
        sequences, _, _ = _quality_sequences(request, prepared)
        input_sha256 = _write_input(output / "prepared-inputs.bin", sequences)
        old_gguf = json.loads(next((ROOT / "Results/048").glob("*gguf-quality.json")).read_text())
        # Dataset-cache fingerprints changed between 048 and 060. The current
        # preparation must match 060's fingerprint (checked above), while 048's
        # retained scorer input must match byte-for-byte and all semantic fields.
        check_fields({k: v for k, v in old_gguf["protocol"].items() if k != "wikitext_fingerprint"}, protocol)
        if old_gguf["identity"]["input_sha256"] != input_sha256:
            raise ValueError("prepared scorer inputs differ from the retained shared GGUF protocol")
        frozen = {
            "protocol": protocol, "reference_quality_sha256": hash_file(REFERENCE),
            "source_sha256": hash_file(Path(__file__)), "download": download,
            "nanoquant_gguf_sha256": inventories["nanoquant060"]["file_sha256"],
            "prepared_input_sha256": input_sha256, "sequence_count": len(sequences),
            "external_upstream_weight_revision": "not published by quantizer; reference revision identifies tokenizer",
            "scope": "Retained diagnostic protocol comparison; no fresh held-out gate and no compression or fitting",
            "execution": {"parallel": 4, "gpu_layers": -1, "threads": 0, "batch_threads": 0},
        }
        protocol_path = output / "frozen-protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != frozen:
            raise ValueError("existing profile has a different frozen protocol")
        atomic_write_json(protocol_path, frozen)
        progress("protocol_verified", {"sequence_count": len(sequences), "input_sha256": input_sha256})
        results = {}
        for label, path in (("external_iq2_m", external), ("nanoquant060", REFERENCE_GGUF)):
            progress("evaluation_started", {"model": label})
            result = execute_llamacpp_quality_evaluation(
                LlamaCppQualityRequest(gguf=path, output=output / f"{label}-quality.json", llama_cpp_root=LLAMA,
                                      device="cuda:0", gpu_layers=-1, parallel=4),
                request, prepared, reference["results"]["base"], protocol)
            if result["identity"]["input_sha256"] != input_sha256:
                raise ValueError("evaluated input identity changed")
            results[label] = result
            progress("evaluation_completed", {"model": label, "passed": result["passed"],
                                               "perplexity": result["results"]["gguf"]["wikitext"]["perplexity"]})
        for key in ("input_sha256", "runner_sha256", "runtime_sha256", "protocol_sha256", "scoring"):
            if results["external_iq2_m"]["identity"][key] != results["nanoquant060"]["identity"][key]:
                raise ValueError(f"comparison execution mismatch: {key}")
        progress("complete", {"models": list(results)})
    except BaseException as error:
        progress("failed", {"error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
