"""Matched objective-only continuation from the preserved Experiment 060 GGUF."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import time
from pathlib import Path

import _paths  # noqa: F401
import torch
from objective_ablation_common import (
    install_dense,
    install_trainable,
    load_scales,
    normalized_hidden_loss,
    read_units,
    scale_state,
)
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_residual_multiblock_fit import verified_tokens
from probe_residual_multiblock_functional import full_kl_hidden_gradient
from probe_residual_stage_functional import BASELINE_SHA, REVISION, aggregate, file_hash, restore_gguf
from safetensors.torch import load_file, save_file

from nanoquant.application.distillation import topk_distillation_loss
from nanoquant.application.kl_budget import causal_kl_nll_per_sequence_from_logits
from nanoquant.application.parity_adamw import ParityAdamW
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config

ARMS = ("top_k", "full_kl", "full_kl_hidden")
SOURCE_SHA = "3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6"


def topk_hidden_gradient(hidden, teacher_hidden, head, teacher_head):
    gradient = torch.zeros_like(hidden)
    targets = hidden.shape[0] * (hidden.shape[1] - 1)
    total = 0.0
    for start in range(0, hidden.shape[1] - 1, 32):
        end = min(start + 32, hidden.shape[1] - 1)
        with torch.no_grad():
            values, indices = teacher_head(teacher_hidden[:, start:end]).float().reshape(-1, head.out_features).topk(64)
        detached = hidden[:, start:end].detach().requires_grad_(True)
        loss = topk_distillation_loss(detached.reshape(-1, detached.shape[-1]), values, indices, head,
                                     temperature=1, token_chunk_size=32) * (end - start) * hidden.shape[0] / targets
        gradient[:, start:end] = torch.autograd.grad(loss, detached)[0]
        total += float(loss.detach())
    return gradient, total


def load_models(args, *, trainable):
    config = json.loads((args.snapshot / "config.json").read_text())
    if config.get("final_logit_softcapping") is not None:
        raise ValueError("pinned uncapped Gemma output head required")
    teacher = load_causal_language_model(
        args.snapshot, torch_dtype=torch.bfloat16, local_files_only=True,
        attention_implementation=adapter_for_config(config).attention_implementation).eval().requires_grad_(False)
    student = copy.deepcopy(teacher).eval().requires_grad_(False)
    restoration = restore_gguf(student, args.output / "common060.gguf", args.llama_root)
    units = read_units(args.output / "common060.gguf", args.llama_root)
    if trainable:
        install_trainable(student, units)
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        student.train()
    teacher.to(args.device)
    student.to(args.device)
    for unit in units.values():
        unit.to(args.device)
    return teacher, student, units, restoration


def train(args, protocol):
    tokens, _ = verified_tokens(args.fit_tokens, "fit")
    with acquire_device_lease(args.device):
        teacher, student, units, restoration = load_models(args, trainable=True)
        atomic_write_json(args.output / "restoration.json", restoration)
        base = scale_state(units)
        common_path = args.output / "common-scales.safetensors"
        if common_path.exists():
            existing = load_file(str(common_path))
            if set(existing) != set(base) or any(not torch.equal(existing[key], value) for key, value in base.items()):
                raise ValueError("preserved common factor scales changed")
        else:
            save_file(base, str(common_path))
        parameters = [parameter for unit in units.values() for parameter in unit.parameters()]
        count = sum(parameter.numel() for parameter in parameters)
        atomic_write_json(args.output / "parameter-inventory.json",
                          dict(units=len(units), trainable_count=count,
                               trainable="factor pre/mid/post scales only; shared QKV parameters counted once",
                               scale_storage_bytes=sum(t.numel() * t.element_size() for t in base.values())))
        for arm in args.arms:
            torch.manual_seed(0)
            load_scales(units, base)
            folder = args.output / arm
            folder.mkdir(exist_ok=True)
            if (folder / "receipt.json").exists():
                receipt = json.loads((folder / "receipt.json").read_text())
                if receipt["sha256"] != file_hash(folder / "scales.safetensors") or receipt["steps"] != args.steps:
                    raise ValueError("completed objective arm changed")
                continue
            optimizer = ParityAdamW(parameters, lr=1e-5, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
            start_step, losses = 0, []
            if (folder / "checkpoint.json").exists():
                receipt = json.loads((folder / "checkpoint.json").read_text())
                if file_hash(folder / "checkpoint.pt") != receipt["sha256"]:
                    raise ValueError("training checkpoint changed")
                saved = torch.load(folder / "checkpoint.pt", map_location=args.device, weights_only=True)
                load_scales(units, saved["parameters"])
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                start_step, losses = saved["step"], saved["losses"]
            started = time.perf_counter()
            for step in range(start_step, min(args.steps, args.stop_after or args.steps)):
                batch = tokens[step % len(tokens):step % len(tokens) + 1].to(args.device)
                with torch.no_grad():
                    target = teacher.model(input_ids=batch, use_cache=False, output_hidden_states=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = student.model(input_ids=batch, use_cache=False, output_hidden_states=True)
                hidden = prediction.last_hidden_state
                if arm == "top_k":
                    gradient, loss = topk_hidden_gradient(hidden, target.last_hidden_state,
                                                         student.lm_head, teacher.lm_head)
                else:
                    gradient, loss = full_kl_hidden_gradient(hidden, target.last_hidden_state,
                                                            student.lm_head, teacher.lm_head)
                anchor = normalized_hidden_loss(prediction.hidden_states, target.hidden_states)
                if arm == "full_kl_hidden":
                    torch.autograd.backward((hidden, anchor), (gradient, torch.ones_like(anchor) * 0.1))
                else:
                    hidden.backward(gradient)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1)
                if not torch.isfinite(norm) or not torch.isfinite(anchor) or not torch.isfinite(torch.tensor(loss)):
                    raise ValueError("nonfinite objective arm; last durable checkpoint preserved")
                optimizer.step()
                scheduler.step()
                losses.append(dict(step=step + 1, output_loss=loss, hidden_nmse=float(anchor.detach()),
                                   gradient_norm=float(norm), lr=scheduler.get_last_lr()[0]))
                del target, prediction, hidden, gradient, anchor
                done = step + 1
                if done % 16 == 0 or done == min(args.steps, args.stop_after or args.steps):
                    temporary = folder / "checkpoint.tmp"
                    torch.save(dict(parameters=scale_state(units, persisted=False), optimizer=optimizer.state_dict(),
                                    scheduler=scheduler.state_dict(), step=done, losses=losses), temporary)
                    temporary.replace(folder / "checkpoint.pt")
                    atomic_write_json(folder / "checkpoint.json",
                                      dict(step=done, sha256=file_hash(folder / "checkpoint.pt")))
                    atomic_write_json(folder / "progress.json", dict(status="running", steps=done, losses=losses,
                                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                      seconds_this_process=time.perf_counter() - started))
                    print(json.dumps(dict(arm=arm, steps=done, loss=loss, hidden_nmse=losses[-1]["hidden_nmse"],
                                          peak_cuda_bytes=torch.cuda.max_memory_allocated())), flush=True)
            if losses[-1]["step"] == args.steps:
                path = folder / "scales.safetensors"
                save_file(scale_state(units), str(path))
                # Evaluation consumes only the storage-rounded state, never FP32 masters.
                load_scales(units, load_file(str(path)))
                atomic_write_json(folder / "receipt.json", dict(steps=args.steps, sha256=file_hash(path),
                                  bytes=path.stat().st_size, trainable_count=count, protocol=protocol,
                                  final_losses=losses[-16:]))
            del optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()


def evaluate(args, protocol):
    tokens, receipt = verified_tokens(args.tokens, args.role)
    fit_entry = protocol["fit_tokens"]["entry"]
    entry = receipt["entry"]
    if max(entry["token_start"], fit_entry["token_start"]) < min(entry["token_end"], fit_entry["token_end"]):
        raise ValueError("evaluation overlaps fit tokens")
    folder = args.output / args.role
    folder.mkdir(exist_ok=True)
    states = {"untreated060": load_file(str(args.output / "common-scales.safetensors"))}
    hashes = {}
    for arm in ARMS:
        path = args.output / arm / "scales.safetensors"
        arm_receipt = json.loads(path.with_name("receipt.json").read_text())
        if file_hash(path) != arm_receipt["sha256"] or arm_receipt["steps"] != args.steps:
            raise ValueError("evaluation objective checkpoint mismatch")
        states[arm] = load_file(str(path))
        hashes[arm] = file_hash(path)
    identity = dict(training=protocol, tokens=receipt, frozen_scales=hashes, role=args.role)
    path = folder / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("evaluation resume identity mismatch")
    atomic_write_json(path, identity)
    progress = folder / "progress.json"
    rows = json.loads(progress.read_text())["rows"] if progress.exists() else []
    with acquire_device_lease(args.device):
        teacher, student, units, _ = load_models(args, trainable=False)
        student.eval()
        with torch.inference_mode():
            # Evaluate by arm so weight materialization happens once per checkpoint.
            for arm, state in states.items():
                load_scales(units, state)
                install_dense(student, units)
                for index, sequence in enumerate(tokens):
                    if index < len(rows) and arm in rows[index]:
                        continue
                    batch = sequence[None].to(args.device)
                    teacher_logits = teacher(input_ids=batch, use_cache=False).logits
                    logits = student(input_ids=batch, use_cache=False).logits
                    metric = causal_kl_nll_per_sequence_from_logits(
                        teacher_logits, logits, batch, token_chunk_size=32)[0]
                    if index == len(rows):
                        rows.append({})
                    rows[index][arm] = to_dict(metric)
                    del teacher_logits, logits
                    atomic_write_json(progress, dict(status="running", rows=rows))
                    if (index + 1) % 16 == 0:
                        print(json.dumps(dict(evaluation=arm, sequences=index + 1, total=len(tokens))), flush=True)
        peak = torch.cuda.max_memory_allocated()
    results = {arm: aggregate(arm, rows) for arm in states}
    comparisons = {}
    pairs = [("untreated060", arm) for arm in ARMS] + [("top_k", "full_kl"), ("top_k", "full_kl_hidden"),
                                                       ("full_kl", "full_kl_hidden")]
    for before, after in pairs:
        comparisons[after + "_minus_" + before] = {
            metric: _paired_metric_payload(results[before], results[after], attribute)
            for metric, attribute in (("kl", "kl_nats_per_token"), ("nll", "negative_log_likelihood"))}
    gates = {arm: comparisons[arm + "_minus_untreated060"]["kl"]["upper_delta"] < 0
             and comparisons[arm + "_minus_untreated060"]["nll"]["upper_delta"] <= 0.01 for arm in ARMS}
    atomic_write_json(folder / "summary.json", dict(status="completed", results={arm: to_dict(result)
                      for arm, result in results.items()}, paired=comparisons, incumbent_gates=gates,
                      peak_cuda_allocated_bytes=peak))
    atomic_write_json(progress, dict(status="completed", rows=rows))
    print(json.dumps({arm: dict(kl=result.kl_nats_per_token, nll=result.negative_log_likelihood)
                      for arm, result in results.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-tokens", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--role", choices=("selection", "confirmation"), default="selection")
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--gguf", type=Path, default=Path("Results/060/gemma-3-1b-it-nanoquant.gguf"))
    parser.add_argument("--llama-root", type=Path, default=Path("D:/dev/research/llama.cpp"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps <= 0 or (args.stop_after is not None and not 0 < args.stop_after <= args.steps):
        raise ValueError("training horizons must be positive and bounded by declared total")
    torch.set_num_threads(8)
    torch.manual_seed(0)
    if file_hash(args.snapshot / "model.safetensors") != SOURCE_SHA or file_hash(args.gguf) != BASELINE_SHA:
        raise ValueError("pinned common teacher/student identity mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    common = args.output / "common060.gguf"
    if not common.exists():
        os.link(args.gguf.resolve(), common)
    if file_hash(common) != BASELINE_SHA:
        raise ValueError("preserved common checkpoint changed")
    _, token_receipt = verified_tokens(args.fit_tokens, "fit")
    protocol = dict(scope="matched post-060 scale-only objective continuation; not original pre-distillation rerun",
                    common_gguf_sha256=BASELINE_SHA, source_sha256=SOURCE_SHA, revision=REVISION,
                    fit_tokens=token_receipt, arms=list(ARMS), steps=args.steps, learning_rate=1e-5,
                    optimizer="ParityAdamW defaults, weight_decay=0", scheduler="cosine to zero over declared steps",
                    parameters="all 130 units' factor scales only; signs/outliers/norms/embedding frozen",
                    storage="BF16 scales, FP32 masters and BF16 straight-through QAT, BF16 dense weights",
                    hidden_anchor="0.1 * mean block MSE / teacher block mean square; excludes embedding state",
                    tokens="every causal next-token target, sequential cycles through fit slice, batch 1",
                    gradient_clip=1, temperature=1, top_k=64, chunk=32,
                    gate="paired KL 95% upper <0 and NLL 95% upper <=+0.01; fresh confirmation before extending",
                    torch_version=torch.__version__,
                    tool_hashes={name: file_hash(Path(__file__).with_name(name)) for name in
                                 (Path(__file__).name, "objective_ablation_common.py")})
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("objective continuation resume identity mismatch")
    atomic_write_json(path, protocol)
    if args.stage == "train":
        train(args, protocol)
    else:
        if args.tokens is None:
            raise ValueError("evaluation requires independently retired token file")
        evaluate(args, protocol)


if __name__ == "__main__":
    main()
