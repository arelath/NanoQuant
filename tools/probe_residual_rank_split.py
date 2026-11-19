"""Matched-budget explicit residual-rank splits; diagnostic only, no resident mutation.

Refit each split from the same source target using production ADMM settings,
then jointly tune all signs/scales on teacher activations. Validation is the
WikiText validation split, not the final quality test. No production rank
policy, cache identity, frozen run, or allocator is changed.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_global_foldable_mlp_multipliers import _load_calibration
from probe_mlp_overlays_kl import _split_tokens
from probe_real_block_tabu import _load_cases
from safetensors.torch import load_file, save_file

from nanoquant.application.layers import TrainableFactorizedLinear
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.models import ArtifactRef, ArtifactTypes
from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.scale_fit import fit_scales, reconstruct
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import latest_complete_identity, load_committed_block
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config


def split_cost(shape: tuple[int, int], ranks: tuple[int, ...], multiple: int = 32) -> int:
    if not ranks or any(r <= 0 or r % multiple or r > min(shape) for r in ranks):
        raise ValueError("stage ranks must be positive, aligned, and within physical dimension")
    return sum(factor_bit_cost(*shape, r, rank_alignment=multiple).total for r in ranks)


def candidate_splits(shape: tuple[int, int], baseline: tuple[int, ...], multiple: int = 32):
    """Fixed-stage equal-cost splits; single-stage controls pay new scale overhead."""
    cap = split_cost(shape, baseline, multiple)
    total = sum(baseline)
    count = max(2, len(baseline))
    while total >= count * multiple:
        units, extra = divmod(total // multiple, count)
        balanced = tuple((units + (i < extra)) * multiple for i in range(count))
        if split_cost(shape, balanced, multiple) <= cap:
            break
        total -= multiple
    else:
        return (baseline,)
    candidates = [baseline, balanced, tuple(reversed(baseline))]
    if len(baseline) == 1:
        primary = min(min(shape) // multiple * multiple, (total * 3 // 4) // multiple * multiple)
        candidates.append((primary, total - primary))
    return tuple(dict.fromkeys(r for r in candidates if split_cost(shape, r, multiple) <= cap))


def stage_inventory(run: Path, blocks: list[int]) -> dict:
    records = [json.loads(x) for x in (run / "state/journal.jsonl").read_text().splitlines() if x.strip()]
    identity, committed = latest_complete_identity(records, 26)
    store = LocalArtifactStore(run / "artifacts")
    result = {}
    for block in blocks:
        value = load_committed_block(
            ArtifactRef(ArtifactTypes.BLOCK_RESULT, committed[block]["artifact_id"], 1), store, identity
        ).result
        for item in (*value.layers, *value.shared_input_groups):
            name = getattr(item, "name", None) or item.layer.path
            state = item.frozen_state
            result[(block, name)] = (state.rank,) + tuple(s.rank for s in state.residual_stages)
    return result


def fit_split(target, v, u, ranks, seed, parameters):
    terms = []
    prediction = torch.zeros_like(target)
    for index, rank in enumerate(ranks):
        residual = (target.float() - prediction.float()).to(target.dtype)
        fit = factorize_admm_with_parameters(
            residual, v, u, rank, torch.Generator(device=target.device).manual_seed(seed + 100003 * index),
            parameters,
        )
        scales = fit_scales(residual, fit.left_binary, fit.right_binary,
                            fit.scale_pre, fit.scale_mid, fit.scale_post, v, u, alternating_passes=2)
        term = TrainableFactorizedLinear(
            fit.left_latent.float(), fit.right_latent.float(),
            scales.scale_pre.float(), scales.scale_mid.float(), scales.scale_post.float(),
        )
        # Match persisted signs even if an ADMM latent is exactly zero.
        with torch.no_grad():
            term.left_latent.copy_(fit.left_binary.float() * fit.left_latent.float().abs().clamp_min(1e-6))
            term.right_latent.copy_(fit.right_binary.float() * fit.right_latent.float().abs().clamp_min(1e-6))
        terms.append(term)
        prediction = prediction + scales.reconstruction.to(target.dtype)
    primary = terms[0]
    primary.residual_stages = torch.nn.ModuleList(terms[1:])
    return primary


def frozen_payload(module):
    tensors = {}
    for index, term in enumerate((module, *module.residual_terms)):
        for name in ("left", "right"):
            tensors[f"stage{index}.{name}_binary"] = (
                (getattr(term, name + "_latent").detach() >= 0).to(torch.int8).mul(2).sub(1).cpu()
            )
        for name in ("pre", "mid", "post"):
            tensors[f"stage{index}.scale_{name}"] = getattr(term, "scale_" + name).detach().bfloat16().cpu()
    return tensors


def payload_weight(tensors, device):
    stages = len(tensors) // 5
    return sum(reconstruct(*(tensors[f"stage{i}.{name}"].to(device).float() for name in
                            ("left_binary", "right_binary", "scale_pre", "scale_mid", "scale_post")))
               for i in range(stages))


@torch.no_grad()
def metrics(weight, target, v, u, activations):
    difference = weight.float() - target.float()
    weighted = (difference.square() * u[:, None] * v[None, :]).sum()
    denominator = (target.float().square() * u[:, None] * v[None, :]).sum().clamp_min(1e-30)
    rows = []
    for inputs in activations:
        x = inputs.to(weight.device).float()
        error = x @ difference.T
        y = x @ target.float().T
        rows.append(float(error.square().sum() / y.square().sum().clamp_min(1e-30)))
    result = dict(weighted_nmse=float(weighted / denominator), activation_nmse=sum(rows) / len(rows),
                  activation_nmse_per_sequence=rows)
    if not all(torch.isfinite(torch.tensor(x)).all() for x in result.values()):
        raise FloatingPointError("nonfinite probe metrics")
    return result


def tune_joint(module, target, activations, epochs):
    parameters = list(module.named_parameters())
    optimizer = torch.optim.AdamW([
        dict(params=[p for n, p in parameters if "latent" in n], lr=3e-5),
        dict(params=[p for n, p in parameters if "latent" not in n], lr=1e-5),
    ], weight_decay=0.0)
    initial = {n: p.detach().clone() for n, p in parameters}
    losses = []
    for _epoch in range(epochs):
        for inputs in activations:
            x = inputs.to(target.device).bfloat16()
            # Targets are real teacher inputs through original source weights.
            y = x.float() @ target.float().T
            prediction = module(x).float()
            loss = (prediction - y).square().mean() / y.square().mean().clamp_min(1e-12)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite joint tuning loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    changes = {n: float((p.detach() - initial[n]).abs().max()) for n, p in parameters}
    return dict(steps=len(losses), first_loss=losses[0] if losses else None,
                last_loss=losses[-1] if losses else None, max_parameter_changes=changes)


def capture(args, cases):
    train = _load_calibration(args.run_output)[:args.samples, :args.sequence_length]
    validation, fingerprint, _ = _split_tokens(
        args.snapshot, split="validation", samples=args.validation_offset + args.samples,
        sequence_length=args.sequence_length, local_files_only=True,
    )
    validation = validation[args.validation_offset:]
    save_file(dict(train=train.contiguous(), validation=validation.contiguous()),
              str(args.output / "tokens.safetensors"))
    config = json.loads((args.snapshot / "config.json").read_text())
    teacher = load_causal_language_model(
        args.snapshot, torch_dtype=torch.bfloat16, local_files_only=True,
        attention_implementation=adapter_for_config(config).attention_implementation,
    ).to(args.device).eval()
    cache = {f"b{c.block}.{c.name}": [] for c in cases}
    handles = []
    for case in cases:
        key = f"b{case.block}.{case.name}"
        module = teacher.model.layers[case.block].get_submodule(case.members[0].path)
        def hook(_module, inputs, key=key):
            cache[key].append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu())
        handles.append(module.register_forward_pre_hook(hook))
    with torch.no_grad():
        for row in torch.cat((train, validation)):
            # Backbone only: no needless vocabulary-logit materialization.
            teacher.model(input_ids=row[None].to(args.device), use_cache=False)
    for handle in handles:
        handle.remove()
    del teacher
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    for key, rows in cache.items():
        if len(rows) != 2 * args.samples:
            raise ValueError("activation hook count mismatch")
        save_file(dict(train=torch.stack(rows[:args.samples]), validation=torch.stack(rows[args.samples:])),
                  str(args.output / (key + ".activations.safetensors")))
    return dict(validation_split="WikiText validation (exploratory, not a fresh final test)",
                fingerprint=fingerprint, validation_offset=args.validation_offset,
                tokens_sha256=hash_file(args.output / "tokens.safetensors"))


def run(args):
    torch.set_num_threads(8)
    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=False)
    identity, cases = _load_cases(args.run_output, args.snapshot / "model.safetensors",
                                  tuple(args.blocks), tuple(args.owners), 26)
    inventory = stage_inventory(args.run_output, args.blocks)
    config = json.loads((args.run_output / "manifest.json").read_text())["resolved_config"]["canonical_run_config"]
    parameters = AdmmParameters(**config["factorization"]["admm"])
    protocol = dict(source_run=str(args.run_output.resolve()), identity=asdict(identity),
                    source_model_sha256=hash_file(args.snapshot / "model.safetensors"),
                    tool_sha256=hash_file(Path(__file__)), admm=asdict(parameters), seeds=args.seeds,
                    blocks=args.blocks, owners=args.owners, fit_samples=args.samples,
                    sequence_length=args.sequence_length, tuning_epochs=args.epochs,
                    compute="BF16 ADMM; FP32 shadow parameters; BF16 factorized forward",
                    storage="hard signs and BF16 scales; exact production represented-bit cost",
                    scope="fresh local fit and joint activation tuning; not production block/KD or model quality",
                    tuning="all stage signs/scales; AdamW zero decay, binary 3e-5 scales 1e-5, clip 1",
                    split_policy="baseline, balanced, reversed; single-stage adds paid two-stage alternatives")
    atomic_write_json(args.output / "protocol.json", protocol)
    results = []
    with acquire_device_lease(args.device):
        token_receipt = capture(args, cases)
        protocol["tokens"] = token_receipt
        atomic_write_json(args.output / "protocol.json", protocol)
        for case in cases:
            key = f"b{case.block}.{case.name}"
            baseline = inventory[(case.block, case.name)]
            shape = tuple(case.target.shape)
            splits = candidate_splits(shape, baseline)
            target = case.target.to(args.device).bfloat16()
            v, u = case.input_importance.to(args.device), case.output_importance.to(args.device)
            data = load_file(str(args.output / (key + ".activations.safetensors")))
            for seed in args.seeds:
                for ranks in splits:
                    start = time.perf_counter()
                    label = key + f".seed{seed}." + "-".join(map(str, ranks))
                    print(json.dumps(dict(starting=label)), flush=True)
                    module = fit_split(target, v, u, ranks, seed, parameters)
                    before = metrics(payload_weight(frozen_payload(module), args.device),
                                     target, v, u, data["validation"])
                    tuning = tune_joint(module, target, data["train"], args.epochs)
                    tensors = frozen_payload(module)
                    file = args.output / (label + ".safetensors")
                    save_file(tensors, str(file))
                    # Score the read-back persisted precision, not transient trainable tensors.
                    weight = payload_weight(load_file(str(file)), args.device)
                    after = metrics(weight, target, v, u, data["validation"])
                    row = dict(block=case.block, owner=case.name, seed=seed, ranks=ranks,
                               baseline_ranks=baseline, is_baseline=ranks == baseline,
                               bits=split_cost(shape, ranks), baseline_bits=split_cost(shape, baseline),
                               shape=shape, before=before, after=after, tuning=tuning,
                               seconds=time.perf_counter()-start, payload=file.name, sha256=hash_file(file))
                    results.append(row)
                    atomic_write_json(args.output / "summary.json", dict(status="running", rows=results))
                    print(json.dumps({k: row[k] for k in
                                      ("block", "owner", "seed", "ranks", "before", "after", "seconds")}), flush=True)
                    del module, weight
    atomic_write_json(args.output / "summary.json", dict(status="complete_local_probe", rows=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[25])
    parser.add_argument("--owners", nargs="+", default=["mlp.up_proj", "mlp.down_proj", "mlp.gate_proj",
                                                        "self_attn.o_proj", "self_attn.attn_qkv"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--validation-offset", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.samples, args.sequence_length, args.epochs) <= 0 or args.validation_offset < 0:
        parser.error("invalid sample/tuning settings")
    run(args)


if __name__ == "__main__":
    main()
