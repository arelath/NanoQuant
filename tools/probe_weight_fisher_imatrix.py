"""Analysis-only, equal-rate scalar-quantization screen of exact weight Fisher.

This is not a resident compression recipe or a deployable GGUF. All decoder
projections are perturbed; embeddings/norms remain FP32. WikiText is an already
retired exploratory monitor, not a fresh promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import _paths  # noqa: F401
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.imatrix import load_imatrix
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.resource_usage import peak_process_memory_bytes
from nanoquant.quality_evaluation import _wikitext_tokens

ROOT = Path(__file__).resolve().parents[1]
STEMS = dict(q_proj="attn_q", k_proj="attn_k", v_proj="attn_v", o_proj="attn_output",
             gate_proj="ffn_gate", up_proj="ffn_up", down_proj="ffn_down")
ARMS = ("uniform", "imatrix_columns", "imatrix_rows_columns", "matched_rows_columns",
        "empirical_fisher", "model_fisher")


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def tiles(value):
    rows, cols = value.shape
    if rows % 8 or cols % 8:
        raise ValueError("probe requires dimensions divisible by eight")
    return value.reshape(rows // 8, 8, cols // 8, 8).permute(0, 2, 1, 3).reshape(-1, 64)


def untile(value, shape):
    rows, cols = shape
    return value.reshape(rows // 8, cols // 8, 8, 8).permute(0, 2, 1, 3).reshape(shape)


def quantize(weight, importance, bits):
    """Same 8x8 blocks, signed codes, F16 scales and 31-point grid in every arm.

    Tiles span both axes so output-row weights do not cancel from the objective.
    Minimize sum(I * actual_quantization_error**2), not I squared a second time.
    """
    if bits not in (3, 4):
        raise ValueError("supported probe widths are 3 and 4")
    if importance.shape != weight.shape or not torch.isfinite(importance).all() or (importance < 0).any():
        raise ValueError("invalid importance")
    w, imp = tiles(weight.float()), tiles(importance.float())
    low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    base = (w.abs().amax(1) / high).clamp_min(1e-7)
    best = torch.full_like(base, float("inf"))
    result = torch.zeros_like(w)
    for ratio in torch.linspace(0.5, 1.25, 31, device=w.device):
        scale = (base * ratio).half().float().clamp_min(2**-24)
        q = (w / scale[:, None]).round().clamp(low, high) * scale[:, None]
        error = ((q - w).square() * imp).sum(1)
        use = error < best
        best = torch.where(use, error, best)
        result = torch.where(use[:, None], q, result)
    return untile(result, weight.shape)


def exact_gradient_square(objective, parameters, *, retain_graph=False):
    """Square AFTER autograd sums all contextual uses of each shared weight."""
    return tuple(g.detach().float().square() for g in torch.autograd.grad(
        objective, parameters, retain_graph=retain_graph,
    ))


def paired_interval(candidate, baseline):
    differences = torch.tensor(candidate, dtype=torch.float64) - torch.tensor(baseline, dtype=torch.float64)
    generator = torch.Generator().manual_seed(194)
    indices = torch.randint(len(differences), (5000, len(differences)), generator=generator)
    means = differences[indices].mean(1)
    return dict(mean=float(differences.mean()), low=float(means.quantile(.025)), high=float(means.quantile(.975)))


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def save_stage(root, name, tensors, metadata):
    temporary = root / f"{name}.partial.safetensors"
    path = root / f"{name}.safetensors"
    save_file({k: v.cpu().contiguous() for k, v in tensors.items()}, str(temporary))
    temporary.replace(path)
    atomic_write_json(root / f"{name}.json", dict(**metadata, sha256=hash_file(path)))


def load_stage(root, name):
    receipt = root / f"{name}.json"
    if not receipt.exists():
        return None
    meta = json.loads(receipt.read_text())
    path = root / f"{name}.safetensors"
    if hash_file(path) != meta["sha256"]:
        raise ValueError(f"corrupt {name} checkpoint")
    return load_file(str(path)), meta


def collect_activations(model, modules, tokens, device):
    sums, count, handles = {}, {}, []
    for name, module in modules.items():
        def capture(_module, inputs, output, name=name):
            for axis, value in (("in", inputs[0]), ("out", output)):
                key = f"{name}.{axis}"
                x = value.detach().float().reshape(-1, value.shape[-1])
                sums[key] = sums.get(key, 0) + x.square().sum(0)
                count[key] = count.get(key, 0) + len(x)
        handles.append(module.register_forward_hook(capture))
    sync(device)
    start = time.perf_counter()
    try:
        with torch.no_grad():
            for row in tokens:
                model.model(row[None].to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    sync(device)
    return {k: (v / count[k]).cpu() for k, v in sums.items()}, dict(seconds=time.perf_counter() - start)


def collect_fisher(model, modules, tokens, positions, device, kind, root):
    previous = load_stage(root, kind)
    if previous is not None:
        return previous
    names = list(modules)
    parameters = tuple(modules[n].weight for n in names)
    sums = {n: torch.zeros_like(p, device="cpu") for n, p in zip(names, parameters, strict=True)}
    generator = torch.Generator(device=device).manual_seed(814)
    sync(device)
    start = time.perf_counter()
    count = 0
    for index, (row, selected) in enumerate(zip(tokens, positions, strict=True)):
        row = row.to(device)
        hidden = model.model(row[None], use_cache=False).last_hidden_state
        for j, pos in enumerate(selected.tolist()):
            logits = model.lm_head(hidden[:, pos]).float()[0]
            target = (int(row[pos + 1]) if kind == "empirical_fisher" else
                      int(torch.multinomial(logits.detach().softmax(-1), 1, generator=generator)))
            loss = -logits.log_softmax(-1)[target]
            squares = exact_gradient_square(loss, parameters, retain_graph=j + 1 < len(selected))
            for name, square in zip(names, squares, strict=True):
                sums[name].add_(square.cpu())
            count += 1
        del hidden, logits, loss, squares
        if (index + 1) % 4 == 0:
            print(f"{kind}: {index + 1}/{len(tokens)} contexts, {count} targets", flush=True)
    sync(device)
    elapsed = time.perf_counter() - start
    result = {k: v / count for k, v in sums.items()}
    if any(not torch.isfinite(v).all() or float(v.sum()) <= 0 for v in result.values()):
        raise ValueError("non-finite or empty Fisher")
    metadata = dict(seconds=elapsed, targets=count,
                    statistic="mean of individually squared per-target weight gradients",
                    target_distribution="observed data" if kind == "empirical_fisher" else "teacher sampled")
    save_stage(root, kind, result, metadata)
    return result, metadata


def inputs(args, tokenizer):
    previous = load_stage(args.output, "inputs")
    if previous is not None:
        return previous
    text = (args.imatrix.parent / "calibration.txt").read_text(encoding="utf-8")
    corpus = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    width = args.length - 1
    selected = random.Random(args.seed).sample(range(len(corpus) // width), args.calibration_samples)
    calibration = torch.tensor([[tokenizer.bos_token_id, *corpus[i*width:(i+1)*width]] for i in selected])
    evaluation, fingerprint, _ = _wikitext_tokens(
        args.snapshot, samples=args.eval_samples, sequence_length=args.length, local_files_only=True,
    )
    generator = torch.Generator().manual_seed(args.seed)
    positions = torch.stack([torch.randperm(width, generator=generator)[:args.targets].sort().values
                             for _ in calibration])
    eval_positions = torch.stack([torch.randperm(width, generator=generator)[:32].sort().values
                                  for _ in evaluation])
    result = dict(calibration=calibration, evaluation=evaluation, positions=positions, eval_positions=eval_positions)
    metadata = dict(calibration_windows=selected, calibration_token_sha256=tensor_hash(calibration),
                    evaluation_token_sha256=tensor_hash(evaluation), evaluation_fingerprint=fingerprint,
                    evaluation_role="already-retired WikiText test prefix; exploratory monitor only",
                    calibration_text_sha256=hash_file(args.imatrix.parent / "calibration.txt"))
    save_stage(args.output, "inputs", result, metadata)
    return result, metadata


def evaluate(model, data, root, device, *, teacher=False):
    result = []
    with torch.no_grad():
        for i, (row, positions) in enumerate(zip(data["evaluation"], data["eval_positions"], strict=True)):
            row, positions = row.to(device), positions.to(device)
            hidden = model.model(row[None], use_cache=False).last_hidden_state[0, positions]
            logp = model.lm_head(hidden).float().log_softmax(-1)
            nll = float(-logp.gather(1, row[positions + 1, None]).mean())
            path = root / f"teacher-{i}.safetensors"
            if teacher:
                save_file({"logp": logp.cpu().contiguous()}, str(path))
                kl = 0.0
            else:
                reference = load_file(str(path))["logp"].to(device)
                kl = float((reference.exp() * (reference - logp)).sum(-1).mean())
            if not math.isfinite(nll) or not math.isfinite(kl):
                raise ValueError("non-finite quality metric")
            result.append(dict(nll=nll, kl=kl))
    return dict(nll=sum(r["nll"] for r in result) / len(result),
                kl=sum(r["kl"] for r in result) / len(result), sequences=result)


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    imatrix = load_imatrix(args.imatrix)
    request = json.loads((args.imatrix.parent / "request.json").read_text())
    if args.snapshot.name != request["revision"]:
        raise ValueError("snapshot revision differs from imatrix source")
    protocol = dict(schema=1, evidence_tier="exploratory full-model scalar quantization screen",
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    model=request["model"], revision=request["revision"], imatrix_sha256=imatrix.sha256,
                    weights_sha256=hash_file(args.snapshot / "model.safetensors"),
                    code_sha256=hash_file(Path(__file__)), arms=ARMS, bits=[3, 4],
                    quantizer="8x8 tile, signed uniform codes, one F16 scale, 31 ratios 0.5..1.25",
                    precision="FP32 model/gradient/statistics; F16 representable quantizer scales",
                    adoption_rule="no production adoption from this screen; follow-up if KL improves >=5%",
                    uncertainty="5000 paired sequence bootstrap resamples; exploratory, no multiple-test correction",
                    limitations=["short contexts and sampled targets", "scalar quantizer, not NanoQuant factors",
                                 "no GGUF or deployed BPW claim", "existing imatrix has much larger sample budget",
                                 "WikiText prefix is a reused monitor, not a fresh gate"])
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("output protocol changed; use a new directory")
    atomic_write_json(path, protocol)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    data, input_meta = inputs(args, tokenizer)
    print("Inputs ready; loading FP32 model", flush=True)
    with acquire_device_lease(args.device):
        model = load_causal_language_model(args.snapshot, torch_dtype=torch.float32,
                                          attention_implementation="eager", local_files_only=True)
        model = model.eval().to(args.device)
        model.requires_grad_(False)
        modules = {name: module for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in STEMS}
        if len(modules) != 126:
            raise ValueError("expected all 126 Gemma 270M projections")
        original = {name: module.weight.detach().cpu().clone() for name, module in modules.items()}
        # Warm up before the forward-only timer; no backward is needed here.
        with torch.no_grad():
            model.model(data["calibration"][:1].to(args.device), use_cache=False)
        activation = load_stage(args.output, "activations")
        if activation is None:
            activation = collect_activations(model, modules, data["calibration"], args.device)
            save_stage(args.output, "activations", *activation)
        for module in modules.values():
            module.weight.requires_grad_(True)
        fisher, fisher_meta = {}, {}
        for kind in ("empirical_fisher", "model_fisher"):
            fisher[kind], fisher_meta[kind] = collect_fisher(
                model, modules, data["calibration"], data["positions"], args.device, kind, args.output,
            )
        model.requires_grad_(False)
        print("Collecting held-out teacher", flush=True)
        baseline = evaluate(model, data, args.output, args.device, teacher=True)
        report = dict(status="running", protocol_sha256=hash_file(path), inputs=input_meta,
                      activation=activation[1], fisher=fisher_meta, baseline=baseline, results=[])
        for bits in (3, 4):
            for arm in ARMS:
                start = time.perf_counter()
                for name, module in modules.items():
                    weight = original[name].to(args.device)
                    block = name.split(".")[2]
                    key = f"blk.{block}.{STEMS[name.split('.')[-1]]}.weight"
                    if arm == "uniform":
                        importance = torch.ones_like(weight)
                    elif arm.startswith("imatrix"):
                        column = imatrix.entries[key].reshape(-1).to(args.device)
                        row = (imatrix.row_entries[key].reshape(-1).to(args.device) if arm.endswith("rows_columns")
                               else torch.ones(weight.shape[0], device=args.device))
                        importance = row[:, None] * column[None, :]
                    elif arm == "matched_rows_columns":
                        importance = (activation[0][name + ".out"][:, None] *
                                      activation[0][name + ".in"][None, :]).to(args.device)
                    else:
                        importance = fisher[arm][name].to(args.device)
                    module.weight.copy_(quantize(weight, importance, bits))
                sync(args.device)
                quant_seconds = time.perf_counter() - start
                metrics = evaluate(model, data, args.output, args.device)
                entry = dict(arm=arm, bits=bits, projection_logical_bpw=bits + 16 / 64,
                             projection_weights=sum(w.numel() for w in original.values()),
                             quant_seconds=quant_seconds, **metrics)
                report["results"].append(entry)
                atomic_write_json(args.output / "summary.json", report)
                print(json.dumps({k: v for k, v in entry.items() if k != "sequences"}), flush=True)
        for entry in report["results"]:
            control = next(r for r in report["results"]
                           if r["bits"] == entry["bits"] and r["arm"] == "imatrix_rows_columns")
            entry["paired_kl_vs_current"] = paired_interval(
                [s["kl"] for s in entry["sequences"]], [s["kl"] for s in control["sequences"]],
            )
        cuda_peak = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
        report.update(status="complete_exploratory_probe", peak_process_bytes=peak_process_memory_bytes(),
                      peak_cuda_allocated_bytes=cuda_peak)
        atomic_write_json(args.output / "summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path.home() / ".cache/huggingface/hub/"
                        "models--unsloth--gemma-3-270m-it/snapshots/23cf460f6bb16954176b3ddcc8d4f250501458a9")
    parser.add_argument("--imatrix", type=Path,
                        default=ROOT / "Datasets/gemma-3-270m-it-imatrix-eaddario-huge-rows/imatrix.gguf")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/probes/gemma270m-weight-fisher-imatrix")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    if min(args.calibration_samples, args.eval_samples, args.threads, args.targets) <= 0 or not 32 < args.length:
        parser.error("positive sample/thread/target counts and length >32 required")
    if args.targets >= args.length:
        parser.error("targets must be smaller than context length")
    run(args)


if __name__ == "__main__":
    main()
