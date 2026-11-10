"""Score packed residual pilots in a GGUF-recovered Gemma 3 1B dense context."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_non_wikitext_kd_quality import _c4_slice_reservation, _token_hash
from probe_residual_stage_budget import decode_payload, reconstruct
from safetensors.torch import load_file
from weight_image_common import _float_tensor, load_gguf_module, reconstruct_nanoquant_weight, tensor_map

from nanoquant.application.kl_budget import (
    KlBudgetArmResult,
    KlSequenceResult,
    causal_kl_nll_per_sequence_from_logits,
)
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config

REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
BASELINE_SHA = "5e445cab6c93f03492ca0c9cea4ce7977fc780b9dbd75ad60578aef609dad02c"
PROJECTIONS = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
               "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
               "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
NORMS = {"attn_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm",
         "ffn_norm": "pre_feedforward_layernorm", "post_ffw_norm": "post_feedforward_layernorm",
         "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_parameter(parameter: torch.Tensor, value: torch.Tensor, *, gemma_norm: bool = False) -> None:
    if value.shape != parameter.shape or not torch.isfinite(value).all():
        raise ValueError("GGUF parameter shape or finiteness mismatch")
    # GGUF stores effective gamma; Hugging Face Gemma applies (1 + weight).
    if gemma_norm:
        value = value.float() - 1
    with torch.no_grad():
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def restore_gguf(student: torch.nn.Module, path: Path, llama_root: Path) -> dict:
    gguf = load_gguf_module(llama_root)
    tensors = tensor_map(gguf.GGUFReader(str(path)))
    consumed = set()
    projection_count = 0
    with torch.no_grad():
        for block_index, block in enumerate(student.model.layers):
            shared_base = f"blk.{block_index}.attn_qkv"
            shared = shared_base + ".nq_v" in tensors
            if shared:
                qkv = torch.from_numpy(reconstruct_nanoquant_weight(tensors, shared_base, gguf))
                modules = [block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj]
                for module, value in zip(modules, split_qkv(qkv, [m.out_features for m in modules]), strict=True):
                    copy_parameter(module.weight, value)
                consumed.update(key for key in tensors if key.startswith(shared_base + ".nq_"))
                projection_count += 3
            for name, module_path in PROJECTIONS.items():
                if shared and name in {"attn_q", "attn_k", "attn_v"}:
                    continue
                base = f"blk.{block_index}.{name}"
                value = torch.from_numpy(reconstruct_nanoquant_weight(tensors, base, gguf))
                copy_parameter(block.get_submodule(module_path).weight, value)
                consumed.update(key for key in tensors if key.startswith(base + ".nq_"))
                projection_count += 1
            for name, module_path in NORMS.items():
                key = f"blk.{block_index}.{name}.weight"
                copy_parameter(block.get_submodule(module_path).weight,
                               torch.from_numpy(_float_tensor(tensors[key], gguf)), gemma_norm=True)
                consumed.add(key)
        copy_parameter(student.model.norm.weight,
                       torch.from_numpy(_float_tensor(tensors["output_norm.weight"], gguf)), gemma_norm=True)
        copy_parameter(student.model.embed_tokens.weight,
                       torch.from_numpy(_float_tensor(tensors["token_embd.weight"], gguf)))
        consumed.update(("output_norm.weight", "token_embd.weight"))
    if consumed != set(tensors):
        raise ValueError(f"unconsumed GGUF tensors: {sorted(set(tensors) - consumed)}")
    if student.model.embed_tokens.weight.data_ptr() != student.lm_head.weight.data_ptr():
        raise ValueError("Gemma tied embedding/output identity was lost")
    return dict(projections=projection_count, tensors=len(consumed),
                norm_conversion="HF weight = GGUF effective gamma - 1",
                execution="BF16 dense reconstruction; not native llama.cpp execution")


def split_qkv(weight: torch.Tensor, widths: list[int]) -> tuple[torch.Tensor, ...]:
    if len(widths) != 3 or sum(widths) != weight.shape[0] or min(widths) <= 0:
        raise ValueError("shared QKV output widths do not match the packed group")
    return torch.split(weight, widths, dim=0)


def aggregate(name: str, rows: list[dict]) -> KlBudgetArmResult:
    sequences = tuple(KlSequenceResult(**row[name]) for row in rows)
    count = sum(item.token_count for item in sequences)
    return KlBudgetArmResult(name,
                            sum(item.negative_log_likelihood * item.token_count for item in sequences) / count,
                            sum(item.kl_nats_per_token * item.token_count for item in sequences) / count,
                            count, sequences=sequences)


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(8)
    if file_hash(args.gguf) != BASELINE_SHA:
        raise ValueError("060 GGUF identity mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    tokens = load_file(str(args.tokens))["input_ids"].long()
    token_receipt = json.loads(args.tokens.with_suffix(".json").read_text())
    if token_receipt["file_sha256"] != file_hash(args.tokens) or not token_receipt["retired_before_evaluation"]:
        raise ValueError("evaluation token receipt is invalid")
    entry = token_receipt["entry"]
    if tokens.shape != (entry["samples"], entry["sequence_length"]) or _token_hash(tokens) != entry["token_hash"]:
        raise ValueError("evaluation token dimensions or content differ from reservation")
    _c4_slice_reservation(Path(token_receipt["registry"]), entry["id"], offset=entry["offset"],
                          samples=entry["samples"], sequence_length=entry["sequence_length"],
                          token_hash=entry["token_hash"])
    pilot = json.loads((args.pilot / "summary.json").read_text())
    if pilot["source_receipt"]["source_key"] != f"model.layers.{args.block}.mlp.down_proj.weight":
        raise ValueError("pilot source layer differs from requested splice target")
    replacements = {}
    payloads = {}
    for name in ("int8", "sign", "residual_factor"):
        path = args.pilot / (name + ".bin")
        digest = file_hash(path)
        if digest != pilot["results"][name]["sha256"] or path.stat().st_size > pilot["cap_bytes"]:
            raise ValueError("candidate digest/budget mismatch")
        replacements[name] = reconstruct(decode_payload(path.read_bytes())).to(torch.bfloat16)
        payloads[name] = dict(sha256=digest, bytes=path.stat().st_size)
    identity = dict(gguf_sha256=BASELINE_SHA, tokens=token_receipt, payloads=payloads,
                    block=args.block, torch_version=torch.__version__, model_revision=REVISION,
                    source_model_sha256=file_hash(args.snapshot / "model.safetensors"),
                    tool_sha256=file_hash(Path(__file__)),
                    primary_comparison="residual_factor minus int8, equal 2.0 BPW cap",
                    reference="unchanged060 is lower-byte reference, not the matched-budget control",
                    gate="KL paired 95% upper delta < 0 and NLL paired 95% upper delta <= 0.01 nats/token")
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != identity:
        raise ValueError("resume protocol mismatch")
    atomic_write_json(protocol_path, identity)
    progress_path = args.output / "progress.json"
    rows = json.loads(progress_path.read_text())["rows"] if progress_path.exists() else []
    config = json.loads((args.snapshot / "config.json").read_text())
    with acquire_device_lease(args.device):
        teacher = load_causal_language_model(args.snapshot, torch_dtype=torch.bfloat16,
                                            attention_implementation=adapter_for_config(config).attention_implementation,
                                            local_files_only=True).eval()
        student = copy.deepcopy(teacher).eval()
        restoration = restore_gguf(student, args.gguf, args.llama_root)
        atomic_write_json(args.output / "restoration.json", restoration)
        teacher.to(args.device)
        student.to(args.device)
        target = student.model.layers[args.block].mlp.down_proj.weight
        original = target.detach().clone()
        for name, value in replacements.items():
            if value.shape != target.shape:
                raise ValueError(f"candidate shape mismatch: {name}")
        with torch.inference_mode():
            for index in range(len(rows), len(tokens)):
                batch = tokens[index:index + 1].to(args.device)
                teacher_logits = teacher(input_ids=batch, use_cache=False).logits
                row = {}
                for name in ("unchanged060", "int8", "sign", "residual_factor"):
                    target.copy_(original if name == "unchanged060" else replacements[name].to(args.device))
                    logits = student(input_ids=batch, use_cache=False).logits
                    metric = causal_kl_nll_per_sequence_from_logits(
                        teacher_logits, logits, batch, token_chunk_size=32)[0]
                    row[name] = to_dict(metric)
                    del logits
                row["teacher"] = to_dict(causal_kl_nll_per_sequence_from_logits(
                    teacher_logits, teacher_logits, batch, token_chunk_size=32)[0])
                del teacher_logits
                rows.append(row)
                atomic_write_json(progress_path, dict(status="running", rows=rows))
                print(json.dumps(dict(completed=len(rows), total=len(tokens))), flush=True)
        peak = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None
    results = {name: aggregate(name, rows) for name in rows[0]}
    comparisons = {}
    for before, after in (("int8", "residual_factor"), ("int8", "sign"), ("sign", "residual_factor"),
                          ("unchanged060", "residual_factor"), ("unchanged060", "int8")):
        comparisons[f"{after}_minus_{before}"] = {
            key: _paired_metric_payload(results[before], results[after], attribute)
            for key, attribute in (("kl", "kl_nats_per_token"), ("nll", "negative_log_likelihood"))}
    primary = comparisons["residual_factor_minus_int8"]
    report = dict(status="completed", role="block12 pilot selection screen; confirmation and multi-block gates open",
                  results={name: to_dict(result) for name, result in results.items()}, paired=comparisons,
                  primary_gate_passed=primary["kl"]["upper_delta"] < 0 and primary["nll"]["upper_delta"] <= 0.01,
                  peak_cuda_allocated_bytes=peak)
    atomic_write_json(args.output / "summary.json", report)
    atomic_write_json(progress_path, dict(status="completed", rows=rows))
    print(json.dumps({name: {"nll": result.negative_log_likelihood, "kl": result.kl_nats_per_token}
                      for name, result in results.items()}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, default=Path("Results/060/gemma-3-1b-it-nanoquant.gguf"))
    parser.add_argument("--llama-root", type=Path, default=Path("D:/dev/research/llama.cpp"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
