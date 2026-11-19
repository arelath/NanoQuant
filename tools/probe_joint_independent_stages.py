"""Longer joint tuning of independent residual stages with matched single-stage controls."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_real_block_tabu import _load_cases
from probe_residual_rank_split import fit_split, frozen_payload, metrics, payload_weight, split_cost
from probe_residual_split_warm_start import split_module
from safetensors.torch import load_file, save_file
from torch import nn

from nanoquant.application.layers import TrainableFactorizedLinear
from nanoquant.domain.factorization import AdmmParameters
from nanoquant.domain.rank_expansion import fit_residual_middle_scales
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file

OWNERS = ("mlp.up_proj", "self_attn.attn_qkv")
CHECKPOINTS = (0, 8, 32, 128)
SPLIT_ARMS = ("warm", "simultaneous", "alternate")


@torch.no_grad()
def simultaneous_start(target, v, u, ranks, seed):
    """No primary fit: seed both sign groups, and solve their middles together."""
    rng = torch.Generator(device=target.device).manual_seed(seed)
    weight = target.float()
    post = weight.square().mean(dim=1).sqrt().clamp_min(1e-8)
    pre = weight.square().mean(dim=0).sqrt().clamp_min(1e-8)
    pre = pre / pre.mean()
    left = torch.randn((weight.shape[0], sum(ranks)), generator=rng, device=target.device) * 0.003
    right = torch.randn((sum(ranks), weight.shape[1]), generator=rng, device=target.device) * 0.003
    fit = fit_residual_middle_scales(weight, left.sign(), right.sign(), pre, post, v, u)
    if not fit.accepted or any(not torch.count_nonzero(part) for part in fit.scale_mid.split(ranks)):
        raise ValueError("joint middle fit did not initialise both groups")
    return split_module(TrainableFactorizedLinear(left, right, pre, fit.scale_mid, post), ranks[0])


@torch.no_grad()
def alternate_start(target, v, u, ranks, seed, parameters):
    """Sequential seed, then one full backfitting sweep with both stages revisable."""
    first = fit_split(target, v, u, ranks, seed, parameters)
    terms = [first, first.residual_terms[0]]
    first.residual_stages = nn.ModuleList()
    weights = [payload_weight(frozen_payload(term), target.device) for term in terms]
    importance = u[:, None] * v[None, :]

    def error(combined):
        return float(((combined - target.float()).square() * importance).sum())

    moves = []
    for index in range(2):
        other = weights[1 - index]
        candidate = fit_split((target.float() - other).to(target.dtype), v, u,
                              (ranks[index],), seed + 200003 + index, parameters)
        weight = payload_weight(frozen_payload(candidate), target.device)
        before = error(weights[index] + other)
        after = error(weight + other)
        accepted = after < before
        if accepted:
            terms[index], weights[index] = candidate, weight
        moves.append(dict(stage=index, accepted=accepted, before=before, candidate=after))
    terms[0].residual_stages = nn.ModuleList([terms[1]])
    return terms[0], moves


def tune_checkpoints(module, inputs, outputs, record, checkpoints=CHECKPOINTS):
    params = list(module.named_parameters())
    optimizer = torch.optim.AdamW([
        dict(params=[p for n, p in params if "latent" in n], lr=3e-5),
        dict(params=[p for n, p in params if "latent" not in n], lr=1e-5),
    ], weight_decay=0)
    record(0, module)
    for epoch in range(1, checkpoints[-1] + 1):
        for x, y in zip(inputs, outputs, strict=True):
            optimizer.zero_grad(set_to_none=True)
            error = (module(x).float() - y).square().mean() / y.square().mean().clamp_min(1e-12)
            error.backward()
            nn.utils.clip_grad_norm_(module.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
        if epoch in checkpoints:
            record(epoch, module)


def arm_starts(target, v, u, full, split, seed, parameters):
    yield "full", full, fit_split(target, v, u, full, seed, parameters), []
    reduced = fit_split(target, v, u, (sum(split),), seed, parameters)
    warm = split_module(reduced, split[0])
    yield "reduced", (sum(split),), reduced, []
    del reduced
    yield "warm", split, warm, []
    del warm
    yield "simultaneous", split, simultaneous_start(target, v, u, split, seed), []
    model, moves = alternate_start(target, v, u, split, seed, parameters)
    yield "alternate", split, model, moves


def select(rows, owner, candidates):
    """Use a common epoch and method across seeds; never select an individual seed."""
    selected = [r for r in rows if r["owner"] == owner and r["method"] in candidates and r["epoch"] > 0]
    seeds = {r["seed"] for r in selected}
    if not seeds:
        raise ValueError("no fitted candidates")
    groups = {(r["method"], r["epoch"]) for r in selected}
    options = []
    for method, epoch in sorted(groups):
        pairs = [r for r in selected if (r["method"], r["epoch"]) == (method, epoch)]
        if len(pairs) != len(seeds) or {r["seed"] for r in pairs} != seeds:
            raise ValueError("unmatched seed grid")
        mean = sum(r["after"]["activation_nmse"] for r in pairs) / len(seeds)
        options.append((mean, method, epoch, pairs))
    return min(options)[3]


def evaluate(args, case, reference, data, parameters, rows):
    target = case.target.to(args.device).bfloat16()
    v, u = case.input_importance.to(args.device), case.output_importance.to(args.device)
    inputs = data["train"].to(args.device)
    outputs = inputs.float() @ target.float().mT
    full = tuple(reference["full"]["ranks"])
    split = tuple(reference["warm"]["ranks"])
    seed = reference["seed"]
    for name, ranks, module, moves in arm_starts(target, v, u, full, split, seed, parameters):
        print(f"{case.name} seed {seed}: {name}", flush=True)
        initial_signs = {}

        def record(epoch, model, name=name, ranks=ranks, moves=moves, initial_signs=initial_signs):
            file = args.output / f"b25.{case.name}.seed{seed}.{name}.epoch{epoch}.safetensors"
            save_file(frozen_payload(model), str(file))
            saved = load_file(str(file))
            weight = payload_weight(saved, args.device)
            signs = {k: t for k, t in saved.items() if k.endswith("_binary")}
            if epoch == 0:
                initial_signs.update(signs)
            changes = {k: int(torch.count_nonzero(t != initial_signs[k])) for k, t in signs.items()}
            score = metrics(weight, target, v, u, data["validation"])
            train_score = metrics(weight, target, v, u, data["train"])
            bits = split_cost(tuple(target.shape), ranks)
            if bits > reference["full"]["bits"]:
                raise ValueError("candidate exceeds the full-rank control's bit cap")
            row = dict(block=25, owner=case.name, seed=seed, method=name, epoch=epoch, ranks=ranks,
                       after=score, train=train_score, bits=bits, sha256=hash_file(file),
                       payload=file.name, backfit=moves, sign_changes=changes)
            rows.append(row)
            atomic_write_json(args.output / "summary.json", dict(status="running", rows=rows))
            print(f"  epoch {epoch}: validation activation NMSE {score['activation_nmse']:.8f}", flush=True)

        tune_checkpoints(module, inputs, outputs, record)


def run(args):
    torch.set_num_threads(8)
    source = args.warm / "summary.json"
    warm = json.loads((args.warm / "protocol.json").read_text())
    prior = warm["previous"]
    previous = json.loads(source.read_text())
    model = args.snapshot / "model.safetensors"
    if (previous["status"] != "complete_local_probe"
            or hash_file(model) != prior["source_model_sha256"]
            or hash_file(Path(__file__).with_name("probe_residual_rank_split.py")) != prior["tool_sha256"]
            or hash_file(Path(__file__).with_name("probe_residual_split_warm_start.py")) != warm["tool_sha256"]):
        raise ValueError("reference probe or model provenance differs")
    identity, cases = _load_cases(Path(prior["source_run"]), model, (25,), OWNERS, 26)
    if asdict(identity) != prior["identity"]:
        raise ValueError("resident identity differs")
    args.output.mkdir(parents=True, exist_ok=False)
    parameters = AdmmParameters(**prior["admm"])
    protocol = dict(**prior, warm_summary_sha256=hash_file(source), checkpoints=CHECKPOINTS,
                    joint_tool_sha256=hash_file(Path(__file__)))
    atomic_write_json(args.output / "protocol.json", protocol)
    rows = []
    with acquire_device_lease(args.device):
        for case in cases:
            key = f"b25.{case.name}"
            capture = args.screen / f"{key}.activations.safetensors"
            if hash_file(capture) != warm["activation_hashes"][key]:
                raise ValueError("activation capture changed")
            data = load_file(str(capture))
            for reference in (r for r in previous["rows"] if r["owner"] == case.name):
                evaluate(args, case, reference, data, parameters, rows)
    atomic_write_json(args.output / "summary.json", dict(status="complete_local_probe", rows=rows))
    # Adapt the selected pairs to the existing disjoint-window model scorer.
    score_input = args.output / "confirmation-input"
    score_input.mkdir()
    selected = []
    for owner in OWNERS:
        for label, methods in ((True, ("full",)), (False, SPLIT_ARMS)):
            for row in select(rows, owner, methods):
                selected.append(dict(row, is_baseline=label, payload="../" + row["payload"]))
    atomic_write_json(score_input / "protocol.json", protocol)
    atomic_write_json(score_input / "summary.json", dict(status="complete_local_probe", rows=selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("warm", "screen", "snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
