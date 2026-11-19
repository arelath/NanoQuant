"""Hold both 071 stage ranks/signs fixed while testing shared versus independent outer scales."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_corrected_codebook_splice import _paired_payload
from probe_mlp_overlays_kl import _split_tokens
from probe_real_block_tabu import _load_cases, _state_tensors
from probe_residual_rank_split import capture, frozen_payload, metrics, payload_weight, split_cost, tune_joint
from probe_residual_stage_functional import aggregate
from safetensors.torch import load_file, save_file
from torch import nn

from nanoquant.application.kl_budget import causal_kl_nll_per_sequence_from_logits
from nanoquant.application.layers import TrainableFactorizedLinear
from nanoquant.config.codec import to_dict
from nanoquant.domain.rank_expansion import fit_residual_middle_scales
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.global_tuning import active_global_tuning, load_global_tuning
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.kl_budget_workflow import _token_hash

OWNERS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "self_attn.attn_qkv", "self_attn.o_proj")
REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"


def make_model(left, right, pre, middle, post, ranks, shared):
    l1, l2 = left.split(ranks, dim=1)
    r1, r2 = right.split(ranks, dim=0)
    m1, m2 = middle.split(ranks)
    residual = TrainableFactorizedLinear(l2, r2, pre, m2, post, immutable_binary_factors=True)
    primary = TrainableFactorizedLinear(l1, r1, pre, m1, post, immutable_binary_factors=True,
                                       residual_stages=(residual,))
    if shared:
        residual.scale_pre = primary.scale_pre
        residual.scale_post = primary.scale_post
    for term in (primary, residual):
        term.left_latent.requires_grad_(False)
        term.right_latent.requires_grad_(False)
    return primary


def encode_shared(module):
    other, = module.residual_terms
    if other.scale_pre is not module.scale_pre or other.scale_post is not module.scale_post:
        raise ValueError("cannot omit independent outer scales")
    payload = frozen_payload(module)
    del payload["stage1.scale_pre"], payload["stage1.scale_post"]
    return payload


def decode_shared(payload):
    if len(payload) != 8 or any(f"stage1.scale_{axis}" in payload for axis in ("pre", "post")):
        raise ValueError("expected one stored pair of outer scales")
    return {**payload, "stage1.scale_pre": payload["stage0.scale_pre"],
            "stage1.scale_post": payload["stage0.scale_post"]}


def paired_start(target, v, u, tensors):
    """Select a shared-scale starting point by weighted weight error, not holdout error."""
    left = torch.cat([term[0] for term in tensors], dim=1)
    right = torch.cat([term[1] for term in tensors], dim=0)
    options = []
    for _, _, pre, _, post in tensors:
        middle = fit_residual_middle_scales(target, left, right, pre, post, v, u)
        fit = fit_scales(target, left, right, pre, middle.scale_mid, post, v, u, alternating_passes=3)
        options.append(fit)
    best = min(options, key=lambda f: f.after_error)
    return left, right, best.scale_pre, best.scale_mid, best.scale_post


def study_owner(args, case, state, store):
    if any(getattr(state, name, None) is not None for name in
           ("outliers", "bias", "patch_left", "product_codebook")):
        raise ValueError("probe only supports plain factor owners")
    if hasattr(state, "members") and tuple(m.layer for m in state.members) != case.members:
        raise ValueError("distilled QKV inventory changed")
    target = case.target.to(args.device).float()
    v, u = case.input_importance.to(args.device), case.output_importance.to(args.device)
    terms = (state, *state.residual_stages)
    ranks = tuple(s.rank for s in terms)
    if ranks != (min(target.shape),) * 2:
        raise ValueError(f"expected two physical full ranks, got {ranks}")
    tensors = [tuple(t.to(args.device).float() for t in _state_tensors(term, store)) for term in terms]
    data = load_file(str(args.output / f"b{case.block}.{case.name}.activations.safetensors"))
    keys = ("left_binary", "right_binary", "scale_pre", "scale_mid", "scale_post")
    original = {f"stage{i}.{name}": t.bfloat16() for i, term in enumerate(tensors)
                for name, t in zip(keys, term, strict=True)}
    reference = metrics(payload_weight(original, args.device), target, v, u, data["validation"])
    start = paired_start(target, v, u, tensors)
    arms = {}
    for label in ("shared", "independent"):
        tied = label == "shared"
        module = make_model(*start, ranks, tied)
        before = metrics(payload_weight(frozen_payload(module), args.device), target, v, u, data["validation"])
        tune_joint(module, target, data["train"], args.epochs)
        payload = encode_shared(module) if tied else frozen_payload(module)
        for name, value in original.items():
            if name.endswith("_binary") and not torch.equal(value.cpu(), payload[name]):
                raise AssertionError("scale tuning changed a binary factor")
        file = args.output / f"b{case.block}.{case.name}.{label}.safetensors"
        save_file(payload, str(file))
        saved = load_file(str(file))
        bits = sum(t.numel() * (1 if name.endswith("_binary") else 16) for name, t in saved.items())
        if bits != split_cost(tuple(target.shape), ranks) - (16 * sum(target.shape) if tied else 0):
            raise AssertionError("unexpected bit charge")
        weight = payload_weight(decode_shared(saved) if tied else saved, args.device)
        after = metrics(weight, target, v, u, data["validation"])
        arms[label] = dict(before=before, after=after, bits=bits, sha256=hash_file(file), payload=file.name)
    return dict(block=case.block, owner=case.name, ranks=ranks, shape=tuple(target.shape),
                original_071=reference, arms=arms)


@contextmanager
def splice(replacements):
    prior = [(parent, name, getattr(parent, name)) for parent, name, _ in replacements]
    try:
        for parent, name, module in replacements:
            setattr(parent, name, module)
        yield
    finally:
        for parent, name, module in prior:
            setattr(parent, name, module)


def functional_check(args, cases, rows, identity, tuning_ref):
    tokens, fingerprint, _ = _split_tokens(args.snapshot, split="validation", samples=256,
                                           sequence_length=128, local_files_only=True)
    tokens = tokens[224:256]
    loaded = load_frozen_run(args.run_output, args.snapshot, source_name="google/gemma-3-1b-it",
                             revision=REVISION, device=args.device, verify_hashes=True,
                             global_tuning_override=tuning_ref)
    if loaded.identity != identity or loaded.global_tuning != tuning_ref:
        raise ValueError("functional replay provenance differs")
    student = loaded.model.eval()
    variants = {"original_071": [], "shared": [], "independent": []}
    for case, result in zip(cases, rows, strict=True):
        block = student.model.layers[case.block]
        for label in ("shared", "independent"):
            arm = result["arms"][label]
            file = args.output / arm["payload"]
            if hash_file(file) != arm["sha256"]:
                raise ValueError("probe payload changed")
            saved = load_file(str(file))
            weight = payload_weight(decode_shared(saved) if label == "shared" else saved, args.device)
            for member, part in zip(case.members, weight.split(case.member_rows), strict=True):
                path, name = member.path.rsplit(".", 1)
                replacement = nn.Linear(part.shape[1], part.shape[0], bias=False,
                                        dtype=torch.bfloat16, device=args.device)
                with torch.no_grad():
                    replacement.weight.copy_(part)
                variants[label].append((block.get_submodule(path), name, replacement))
    config = json.loads((args.snapshot / "config.json").read_text())
    teacher = load_causal_language_model(args.snapshot, torch_dtype=torch.bfloat16, local_files_only=True,
        attention_implementation=adapter_for_config(config).attention_implementation).to(args.device).eval()
    results = []
    print("Functional check: composed overlays on 32 disjoint validation sequences", flush=True)
    with torch.no_grad():
        for index, row in enumerate(tokens):
            x = row[None].to(args.device)
            reference = teacher(input_ids=x, use_cache=False).logits
            values = {}
            for label, replacements in variants.items():
                with splice(replacements):
                    logits = student(input_ids=x, use_cache=False).logits
                values[label] = to_dict(causal_kl_nll_per_sequence_from_logits(
                    reference, logits, x, token_chunk_size=32)[0])
            results.append(values)
            atomic_write_json(args.output / "functional-progress.json", dict(rows=results))
            if (index + 1) % 8 == 0:
                print(f"  functional: {index + 1}/{len(tokens)}", flush=True)
    arms = {label: aggregate(label, results) for label in variants}
    comparisons = {}
    for label, before, after in (("independent_minus_shared", "shared", "independent"),
                                 ("shared_minus_original", "original_071", "shared"),
                                 ("independent_minus_original", "original_071", "independent")):
        comparisons[label] = dict(kl=_paired_payload(arms[before], arms[after], seed=0),
                                  nll=_paired_metric_payload(arms[before], arms[after], "negative_log_likelihood"))
    atomic_write_json(args.output / "functional.json", dict(status="completed", offset=224,
                      samples=len(tokens), token_hash=_token_hash(tokens), fingerprint=fingerprint,
                      arms={label: to_dict(arm) for label, arm in arms.items()}, comparisons=comparisons))


def run(args):
    torch.set_num_threads(8)
    torch.manual_seed(0)
    identity, cases = _load_cases(args.run_output, args.snapshot / "model.safetensors", tuple(args.blocks), OWNERS, 26)
    artifacts = LocalArtifactStore(args.run_output / "artifacts")
    store = LocalTensorStore(artifacts)
    reference = active_global_tuning(args.run_output)
    if reference is None:
        raise ValueError("071 requires a completed distillation checkpoint")
    tuned = load_global_tuning(reference, artifacts).result
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(identity=asdict(identity), tuning=to_dict(reference), blocks=args.blocks, epochs=args.epochs,
                    source_run=str(args.run_output.resolve()), tool_sha256=hash_file(Path(__file__)),
                    model_sha256=hash_file(args.snapshot / "model.safetensors"))
    rows = []
    with acquire_device_lease(args.device):
        protocol["tokens"] = capture(args, cases)
        atomic_write_json(args.output / "protocol.json", protocol)
        for case in cases:
            block = tuned.tuned_blocks[case.block]
            state = next(s for s in (*block.quantized_layers, *block.shared_input_groups)
                         if (getattr(s, "name", None) or s.layer.path) == case.name)
            row = study_owner(args, case, state, store)
            rows.append(row)
            atomic_write_json(args.output / "summary.json", dict(status="running", rows=rows))
            print(json.dumps(dict(block=case.block, owner=case.name, ranks=row["ranks"],
                                  scores={k: a["after"]["activation_nmse"] for k, a in row["arms"].items()})),
                  flush=True)
        atomic_write_json(args.output / "summary.json", dict(status="complete_local_probe", rows=rows))
        functional_check(args, cases, rows, identity, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-output", "snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 12, 25])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    args.samples = 8
    args.sequence_length = 128
    args.validation_offset = 64
    run(args)


if __name__ == "__main__":
    main()
