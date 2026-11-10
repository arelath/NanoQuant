"""Matched full-model scale tuning and composed residual-stage selection."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import _paths  # noqa: F401
import torch
import torch.nn.functional as F
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_residual_multiblock_fit import verified_tokens
from probe_residual_stage_budget import decode_payload, encode_payload, reconstruct
from probe_residual_stage_functional import BASELINE_SHA, REVISION, aggregate, file_hash, restore_gguf

from nanoquant.application.kl_budget import causal_kl_nll_per_sequence_from_logits
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config

ARMS = ("int8", "sign", "residual_factor")


class FirstStageScaleOverlay(torch.nn.Module):
    """Freeze every sign/correction; optimize equal first-stage scale roles."""

    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        self.state = {key: value.detach().cpu().clone() for key, value in state.items()}
        for short, name in (("left", "left_binary"), ("right", "right_binary"),
                            ("pre", "scale_pre"), ("mid", "scale_mid"), ("post", "scale_post")):
            self.register_buffer(short, state["first." + name].float().clone())
        first = ((state["first.left_binary"].double() * state["first.scale_post"].double()[:, None])
                 @ (state["first.right_binary"].double() * state["first.scale_mid"].double()[:, None]
                    * state["first.scale_pre"].double()[None, :]))
        self.register_buffer("fixed", (reconstruct(state) - first).float())
        self.register_buffer("protected", state["protected.indices"].long().clone())
        self.register_buffer("protected_values", state["protected.values"].float().clone())
        self.delta_pre = torch.nn.Parameter(torch.zeros_like(self.pre))
        self.delta_mid = torch.nn.Parameter(torch.zeros_like(self.mid))
        self.delta_post = torch.nn.Parameter(torch.zeros_like(self.post))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        left = self.left * (self.post * self.delta_post.exp())[:, None]
        right = self.right * (self.mid * self.delta_mid.exp())[:, None]
        right = right * (self.pre * self.delta_pre.exp())[None, :]
        weight = left @ right + self.fixed
        # Freeze protected columns even if an imported first stage used nonzero pre-scales there.
        weight = weight.index_copy(1, self.protected, self.protected_values)
        return F.linear(value, weight.to(value.dtype))

    def persisted_state(self) -> dict[str, torch.Tensor]:
        state = dict(self.state)
        for name in ("pre", "mid", "post"):
            state["first.scale_" + name] = (getattr(self, name) * getattr(self, "delta_" + name).exp()).detach().cpu()
        return state


def full_kl_hidden_gradient(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor,
                            student_head: torch.nn.Module, teacher_head: torch.nn.Module,
                            *, chunk: int = 32) -> tuple[torch.Tensor, float]:
    """Exact full-vocabulary KL gradient, without retaining vocabulary-sized graphs."""
    gradient = torch.zeros_like(student_hidden)
    targets = student_hidden.shape[0] * (student_hidden.shape[1] - 1)
    total = 0.0
    for start in range(0, student_hidden.shape[1] - 1, chunk):
        end = min(start + chunk, student_hidden.shape[1] - 1)
        hidden = student_hidden[:, start:end].detach().requires_grad_(True)
        with torch.no_grad():
            teacher_log = F.log_softmax(teacher_head(teacher_hidden[:, start:end]).float(), dim=-1)
        student_log = F.log_softmax(student_head(hidden).float(), dim=-1)
        loss = (teacher_log.exp() * (teacher_log - student_log)).sum() / targets
        gradient[:, start:end] = torch.autograd.grad(loss, hidden)[0]
        total += float(loss.detach())
    return gradient, total


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(8)
    torch.manual_seed(0)
    fit_tokens, fit_receipt = verified_tokens(args.fit_tokens, "fit")
    evaluation_tokens, evaluation_receipt = verified_tokens(args.tokens, args.role)
    fit_entry, eval_entry = fit_receipt["entry"], evaluation_receipt["entry"]
    if max(fit_entry["token_start"], eval_entry["token_start"]) < min(fit_entry["token_end"], eval_entry["token_end"]):
        raise ValueError("fit/evaluation slices overlap")
    if file_hash(args.gguf) != BASELINE_SHA:
        raise ValueError("060 GGUF changed")
    selection = json.loads((args.fits / "selection.json").read_text())
    states = {arm: {} for arm in ARMS}
    candidates = {}
    for block, arms in selection["blocks"].items():
        for arm in ARMS:
            item = arms[arm]
            path = Path(item["path"])
            if file_hash(path) != item["sha256"] or path.stat().st_size > 1990656:
                raise ValueError("selected payload changed or exceeds cap")
            states[arm][int(block)] = decode_payload(path.read_bytes())
        candidates[block] = arms
    args.output.mkdir(parents=True, exist_ok=True)
    if args.role == "confirmation" and args.trained_from is None:
        raise ValueError("confirmation requires frozen selection payloads via --trained-from")
    training_root = args.trained_from or args.output
    frozen_receipts = ({arm: file_hash(training_root / ("tuned_" + arm) / "receipt.json") for arm in ARMS}
                       if args.trained_from is not None else None)
    identity = dict(role=args.role, gguf_sha256=BASELINE_SHA, model_revision=REVISION,
                    source_sha256=file_hash(args.snapshot / "model.safetensors"),
                    fit_tokens=fit_receipt, evaluation_tokens=evaluation_receipt, candidates=candidates,
                    steps=32, optimizer="Adam", learning_rate=1e-4, weight_decay=0,
                    gradient_norm_cap=1, log_scale_delta_bound=0.1,
                    objective="full-distribution teacher KL at temperature 1, every next-token target",
                    trainable="first-stage pre/mid/post log-scale multipliers only, same roles/count in every arm",
                    endpoint="fixed final step; no selection-data early stopping",
                    primary="tuned_residual_factor minus unchanged060",
                    frozen_training_receipts=frozen_receipts,
                    gate="KL paired 95% upper delta < 0; NLL paired 95% upper delta <= +0.01",
                    tool_sha256=file_hash(Path(__file__)), torch_version=torch.__version__)
    protocol = args.output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != identity:
        raise ValueError("functional resume protocol mismatch")
    atomic_write_json(protocol, identity)
    config = json.loads((args.snapshot / "config.json").read_text())
    if config.get("final_logit_softcapping") is not None:
        raise ValueError("this pinned-model tuner requires an uncapped output head")
    with acquire_device_lease(args.device):
        teacher = load_causal_language_model(
            args.snapshot, torch_dtype=torch.bfloat16, local_files_only=True,
            attention_implementation=adapter_for_config(config).attention_implementation).eval()
        student = copy.deepcopy(teacher).eval()
        atomic_write_json(args.output / "restoration.json", restore_gguf(student, args.gguf, args.llama_root))
        teacher.requires_grad_(False).to(args.device)
        student.requires_grad_(False).to(args.device)
        originals = {block: student.model.layers[block].mlp.down_proj for block in states["int8"]}
        replacements = {"unchanged060": {block: module.weight.detach().cpu().clone()
                                         for block, module in originals.items()}}
        for arm in ARMS:
            replacements["untuned_" + arm] = {block: reconstruct(state).bfloat16()
                                              for block, state in states[arm].items()}
            folder = training_root / ("tuned_" + arm)
            folder.mkdir(exist_ok=True)
            if not (folder / "receipt.json").exists():
                if args.trained_from is not None:
                    raise ValueError("frozen tuning receipt missing")
                modules = {block: FirstStageScaleOverlay(state).to(args.device) for block, state in states[arm].items()}
                for block, module in modules.items():
                    student.model.layers[block].mlp.down_proj = module
                parameters = [parameter for module in modules.values() for parameter in module.parameters()]
                optimizer = torch.optim.Adam(parameters, lr=1e-4)
                losses = []
                for step in range(32):
                    batch = fit_tokens[step % len(fit_tokens):step % len(fit_tokens) + 1].to(args.device)
                    with torch.no_grad():
                        teacher_hidden = teacher.model(input_ids=batch, use_cache=False).last_hidden_state
                    optimizer.zero_grad(set_to_none=True)
                    hidden = student.model(input_ids=batch, use_cache=False).last_hidden_state
                    gradient, loss = full_kl_hidden_gradient(hidden, teacher_hidden, student.lm_head, teacher.lm_head)
                    hidden.backward(gradient)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1)
                    if not torch.isfinite(norm) or not torch.isfinite(torch.tensor(loss)):
                        raise ValueError("nonfinite tuning gradient/loss")
                    optimizer.step()
                    with torch.no_grad():
                        for parameter in parameters:
                            parameter.clamp_(-0.1, 0.1)
                    losses.append(dict(step=step + 1, kl=loss, gradient_norm=float(norm)))
                    atomic_write_json(folder / "progress.json", dict(losses=losses))
                    del hidden, gradient, teacher_hidden
                    print(json.dumps(dict(tuning_arm=arm, step=step + 1, kl=loss)), flush=True)
                payloads = {}
                for block, module in modules.items():
                    payload = encode_payload(module.persisted_state())
                    if len(payload) > 1990656:
                        raise ValueError("tuned payload exceeds cap")
                    path = folder / f"block{block}.bin"
                    path.write_bytes(payload)
                    payloads[str(block)] = dict(sha256=file_hash(path), bytes=len(payload))
                atomic_write_json(folder / "receipt.json", dict(payloads=payloads, steps=32,
                                  trainable_count=sum(p.numel() for p in parameters), losses=losses))
                for block, original in originals.items():
                    student.model.layers[block].mlp.down_proj = original
                del modules, optimizer, parameters, module
                gc.collect()
                torch.cuda.empty_cache()
            tuned = json.loads((folder / "receipt.json").read_text())
            replacements["tuned_" + arm] = {}
            for block in originals:
                path = folder / f"block{block}.bin"
                if file_hash(path) != tuned["payloads"][str(block)]["sha256"]:
                    raise ValueError("tuned payload changed")
                replacements["tuned_" + arm][block] = reconstruct(decode_payload(path.read_bytes())).bfloat16()
        progress_path = args.output / "progress.json"
        rows = json.loads(progress_path.read_text())["rows"] if progress_path.exists() else []
        with torch.inference_mode():
            for index in range(len(rows), len(evaluation_tokens)):
                batch = evaluation_tokens[index:index + 1].to(args.device)
                teacher_logits = teacher(input_ids=batch, use_cache=False).logits
                row = {}
                for arm, weights in replacements.items():
                    for block, weight in weights.items():
                        originals[block].weight.copy_(weight.to(args.device))
                    logits = student(input_ids=batch, use_cache=False).logits
                    row[arm] = to_dict(causal_kl_nll_per_sequence_from_logits(
                        teacher_logits, logits, batch, token_chunk_size=32)[0])
                    del logits
                row["teacher"] = to_dict(causal_kl_nll_per_sequence_from_logits(
                    teacher_logits, teacher_logits, batch, token_chunk_size=32)[0])
                del teacher_logits
                rows.append(row)
                atomic_write_json(progress_path, dict(status="running", rows=rows))
                print(json.dumps(dict(evaluation_done=len(rows), total=len(evaluation_tokens))), flush=True)
        peak = torch.cuda.max_memory_allocated()
    results = {arm: aggregate(arm, rows) for arm in rows[0]}
    comparisons = {}
    pairs = [("unchanged060", arm) for arm in replacements if arm != "unchanged060"]
    pairs += [("tuned_" + arm, "tuned_residual_factor") for arm in ("int8", "sign")]
    pairs += [("untuned_" + arm, "tuned_" + arm) for arm in ARMS]
    for before, after in pairs:
        comparisons[after + "_minus_" + before] = {
            metric: _paired_metric_payload(results[before], results[after], attribute)
            for metric, attribute in (("kl", "kl_nats_per_token"), ("nll", "negative_log_likelihood"))}
    primary = comparisons["tuned_residual_factor_minus_unchanged060"]
    atomic_write_json(args.output / "summary.json", dict(status="completed", role=args.role,
                      results={arm: to_dict(result) for arm, result in results.items()}, paired=comparisons,
                      primary_gate_passed=primary["kl"]["upper_delta"] < 0 and primary["nll"]["upper_delta"] <= 0.01,
                      peak_cuda_allocated_bytes=peak))
    atomic_write_json(progress_path, dict(status="completed", rows=rows))
    print(json.dumps({arm: dict(nll=result.negative_log_likelihood, kl=result.kl_nats_per_token)
                      for arm, result in results.items()}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fit-tokens", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("selection", "confirmation"), default="selection")
    parser.add_argument("--trained-from", type=Path, help="reuse frozen tuned payloads; required for confirmation")
    parser.add_argument("--gguf", type=Path, default=Path("Results/060/gemma-3-1b-it-nanoquant.gguf"))
    parser.add_argument("--llama-root", type=Path, default=Path("D:/dev/research/llama.cpp"))
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
