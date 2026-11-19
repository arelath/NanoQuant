"""Separate rank cost, sequential initialization, and untied-scale tuning on 070."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_real_block_tabu import _load_cases
from probe_residual_rank_split import fit_split, frozen_payload, metrics, payload_weight, split_cost, tune_joint
from safetensors.torch import load_file, save_file

from nanoquant.application.layers import TrainableFactorizedLinear
from nanoquant.domain.factorization import AdmmParameters
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file

# Best previous split per owner, fixed before the new fits. Do not select per seed.
SPLITS = {"mlp.up_proj": (832, 288), "self_attn.o_proj": (672, 256), "self_attn.attn_qkv": (640, 224)}


def split_module(parent: TrainableFactorizedLinear, cut: int) -> TrainableFactorizedLinear:
    rank = parent.left_latent.shape[1]
    if (not 0 < cut < rank or parent.residual_terms or parent.bias is not None
            or parent.outlier_indices is not None or parent.patch_left is not None
            or parent.product_codebook_free_rows is not None):
        raise ValueError("expected an unadorned single-stage factor and an interior split")

    def term(span, residual_stages=()):
        return TrainableFactorizedLinear(
            parent.left_latent[:, span], parent.right_latent[span, :],
            parent.scale_pre, parent.scale_mid[span], parent.scale_post,
            residual_stages=residual_stages,
        )

    return term(slice(None, cut), (term(slice(cut, None)),))


@torch.no_grad()
def replay(parent, split, activations, device):
    original = payload_weight(frozen_payload(parent), device)
    divided = payload_weight(frozen_payload(split), device)
    relative = float((original - divided).norm() / original.norm().clamp_min(1e-30))
    if not relative <= 1e-5:
        raise AssertionError(f"split does not replay stored parent weights: relative Frobenius error {relative}")
    error, energy = 0.0, 0.0
    for row in activations:
        x = row.to(device=device, dtype=torch.bfloat16)
        y = parent(x).float()
        error += float((split(x).float() - y).square().sum())
        energy += float(y.square().sum())
    return dict(relative_weight_error=relative, relative_bf16_output_rmse=(error / max(energy, 1e-30)) ** 0.5)


def run(args):
    torch.set_num_threads(8)
    source = args.previous
    prior = json.loads((source / "protocol.json").read_text())
    previous = json.loads((source / "summary.json").read_text())
    model = args.snapshot / "model.safetensors"
    if (previous["status"] != "complete_local_probe"
            or hash_file(model) != prior["source_model_sha256"]
            or hash_file(source / "tokens.safetensors") != prior["tokens"]["tokens_sha256"]
            or hash_file(Path(__file__).with_name("probe_residual_rank_split.py")) != prior["tool_sha256"]):
        raise ValueError("prior probe/model provenance differs")
    identity, cases = _load_cases(Path(prior["source_run"]), model, (25,), tuple(SPLITS), 26)
    if asdict(identity) != prior["identity"]:
        raise ValueError("resident identity differs")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(previous=prior, previous_summary_sha256=hash_file(source / "summary.json"),
                    tool_sha256=hash_file(Path(__file__)), activation_hashes={}, splits=SPLITS)
    rows = []
    with acquire_device_lease(args.device):
        for case in cases:
            key = f"b25.{case.name}"
            data_path = source / f"{key}.activations.safetensors"
            protocol["activation_hashes"][key] = hash_file(data_path)
            data = load_file(str(data_path))
            atomic_write_json(args.output / "protocol.json", protocol)
            target = case.target.to(args.device).bfloat16()
            v = case.input_importance.to(args.device)
            u = case.output_importance.to(args.device)
            splits = SPLITS[case.name]
            for seed in prior["seeds"]:
                retained = [row for row in previous["rows"] if row["owner"] == case.name
                            and row["block"] == 25 and row["seed"] == seed]
                full = tuple(next(row for row in retained if row["is_baseline"])["ranks"])
                if len(full) != 1 or full[0] - sum(splits) != 32:
                    raise ValueError("unexpected reference rank")
                parameters = AdmmParameters(**prior["admm"])

                def fit(ranks, target=target, v=v, u=u, seed=seed, parameters=parameters):
                    return fit_split(target, v, u, ranks, seed, parameters)

                def evaluate(label, module, ranks, target=target, v=v, u=u, data=data, key=key, seed=seed):
                    before = metrics(payload_weight(frozen_payload(module), args.device),
                                     target, v, u, data["validation"])
                    tuning = tune_joint(module, target, data["train"], prior["tuning_epochs"])
                    file = args.output / f"{key}.seed{seed}.{label}.safetensors"
                    save_file(frozen_payload(module), str(file))
                    after = metrics(payload_weight(load_file(str(file)), args.device),
                                    target, v, u, data["validation"])
                    return dict(ranks=ranks, bits=split_cost(tuple(target.shape), ranks), before=before,
                                after=after, tuning=tuning, payload=file.name, sha256=hash_file(file))

                print(f"{key} seed {seed}: fitting full and reduced single stages", flush=True)
                parent = fit(full)
                full_replay = replay(parent, split_module(parent, splits[0]), data["validation"], args.device)
                full_arm = evaluate("full", parent, full)
                del parent
                reduced = fit((sum(splits),))
                warm = split_module(reduced, splits[0])
                reduced_replay = replay(reduced, warm, data["validation"], args.device)
                reduced_arm = evaluate("reduced", reduced, (sum(splits),))
                del reduced
                warm_arm = evaluate("warm", warm, splits)
                del warm
                scratch_arm = evaluate("scratch", fit(splits), splits)
                if not reduced_arm["bits"] < warm_arm["bits"] == scratch_arm["bits"] <= full_arm["bits"]:
                    raise AssertionError("unexpected staged bit cost")
                row = dict(owner=case.name, seed=seed, full_replay=full_replay, reduced_replay=reduced_replay,
                           full=full_arm, reduced=reduced_arm, warm=warm_arm, scratch=scratch_arm)
                rows.append(row)
                atomic_write_json(args.output / "summary.json", dict(status="running", rows=rows))
                print(json.dumps(dict(owner=case.name, seed=seed, replay=full_replay,
                                      after={k: row[k]["after"]["activation_nmse"]
                                             for k in ("full", "reduced", "warm", "scratch")})), flush=True)
    atomic_write_json(args.output / "summary.json", dict(status="complete_local_probe", rows=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
