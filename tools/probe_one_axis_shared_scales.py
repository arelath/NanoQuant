"""Warm-start two jointly tuned BF16 stages with only one untied outer scale."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_real_block_tabu import _load_cases
from probe_residual_rank_split import fit_split, frozen_payload, metrics, payload_weight, split_cost, tune_joint
from probe_residual_split_warm_start import split_module
from safetensors.torch import load_file, save_file

from nanoquant.domain.factorization import AdmmParameters
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file


def scale_name(axis):
    if axis not in ("pre", "post"):
        raise ValueError("shared axis must be pre or post")
    return f"scale_{axis}"


def shared_split(primary, cut, axis):
    module = split_module(primary, cut)
    name = scale_name(axis)
    # Alias the Parameter itself. Backprop sums both stages' gradients;
    # named_parameters() yields the shared object once for the optimiser.
    setattr(module.residual_terms[0], name, getattr(module, name))
    return module


def shared_payload(module, axis):
    name = scale_name(axis)
    if getattr(module, name) is not getattr(module.residual_terms[0], name):
        raise ValueError("cannot omit an untied scale vector")
    payload = frozen_payload(module)
    del payload[f"stage1.{name}"]
    return payload


def restore_shared_payload(payload, axis):
    name = scale_name(axis)
    key = f"stage1.{name}"
    if len(payload) != 9 or key in payload:
        raise ValueError("expected a two-stage payload with one omitted outer scale")
    return {**payload, key: payload[f"stage0.{name}"]}


def shared_cost(shape, ranks, axis):
    shared_entries = shape[1] if scale_name(axis) == "scale_pre" else shape[0]
    return split_cost(shape, ranks) - 16 * shared_entries


def evaluate(args, reference, target, v, u, data, epochs, parameters):
    seed = reference["seed"]
    cut, other = reference["warm"]["ranks"]
    full_rank, = reference["full"]["ranks"]
    shape = tuple(target.shape)
    if cut + other + 32 != full_rank:
        raise ValueError("unexpected rank inventory")
    parent = fit_split(target, v, u, (cut + other,), seed, parameters)
    arms = {}
    for axis in ("pre", "post"):
        module = shared_split(parent, cut, axis)
        before = metrics(payload_weight(restore_shared_payload(shared_payload(module, axis), axis), args.device),
                         target, v, u, data["validation"])
        if abs(before["activation_nmse"] - reference["warm"]["before"]["activation_nmse"]) > 1e-5:
            raise AssertionError("initial split differs from the retained warm-start control")
        tuning = tune_joint(module, target, data["train"], epochs)
        path = args.output / f"b25.{reference['owner']}.seed{seed}.share_{axis}.safetensors"
        save_file(shared_payload(module, axis), str(path))
        saved = load_file(str(path))
        bits = shared_cost(shape, (cut, other), axis)
        actual = sum(t.numel() * (1 if name.endswith("_binary") else 16) for name, t in saved.items())
        cap = reference["full"]["bits"]
        if not actual == bits <= cap < shared_cost(shape, (cut, other + 32), axis):
            raise AssertionError("cost mismatch or 32-aligned rank is not maximal under the cap")
        after = metrics(payload_weight(restore_shared_payload(saved, axis), args.device),
                        target, v, u, data["validation"])
        arms[axis] = dict(ranks=[cut, other], bits=bits, before=before, after=after, tuning=tuning,
                          payload=path.name, sha256=hash_file(path))
    return dict(owner=reference["owner"], seed=seed,
                controls={k: reference[k] for k in ("full", "reduced", "warm")}, arms=arms)


def run(args):
    torch.set_num_threads(8)
    protocol_path = args.warm / "protocol.json"
    warm = json.loads(protocol_path.read_text())
    prior = warm["previous"]
    summary = args.warm / "summary.json"
    previous = json.loads(summary.read_text())
    model = args.snapshot / "model.safetensors"
    if (previous["status"] != "complete_local_probe"
            or hash_file(model) != prior["source_model_sha256"]
            or hash_file(Path(__file__).with_name("probe_residual_rank_split.py")) != prior["tool_sha256"]
            or hash_file(Path(__file__).with_name("probe_residual_split_warm_start.py")) != warm["tool_sha256"]):
        raise ValueError("reference probe or model provenance differs")
    identity, cases = _load_cases(Path(prior["source_run"]), model, (25,), tuple(warm["splits"]), 26)
    if asdict(identity) != prior["identity"]:
        raise ValueError("resident identity differs")
    args.output.mkdir(parents=True, exist_ok=False)
    parameters = AdmmParameters(**prior["admm"])
    atomic_write_json(args.output / "protocol.json",
        dict(warm_summary_sha256=hash_file(summary), warm_protocol_sha256=hash_file(protocol_path),
             tool_sha256=hash_file(Path(__file__)), admm=asdict(parameters), epochs=prior["tuning_epochs"]))
    rows = []
    with acquire_device_lease(args.device):
        for case in cases:
            capture = args.screen / f"b25.{case.name}.activations.safetensors"
            if hash_file(capture) != warm["activation_hashes"][f"b25.{case.name}"]:
                raise ValueError("activation capture changed")
            data = load_file(str(capture))
            target = case.target.to(args.device).bfloat16()
            v, u = case.input_importance.to(args.device), case.output_importance.to(args.device)
            for reference in (row for row in previous["rows"] if row["owner"] == case.name):
                for key in ("full", "reduced", "warm"):
                    arm = reference[key]
                    if hash_file(args.warm / arm["payload"]) != arm["sha256"]:
                        raise ValueError("retained control payload changed")
                print(f"{case.name} seed {reference['seed']}: fitting shared start", flush=True)
                row = evaluate(args, reference, target, v, u, data, prior["tuning_epochs"], parameters)
                rows.append(row)
                atomic_write_json(args.output / "summary.json", dict(status="running", rows=rows))
                print(json.dumps(dict(owner=case.name, seed=reference["seed"],
                    after={k: a["after"]["activation_nmse"] for k, a in row["arms"].items()})), flush=True)
    atomic_write_json(args.output / "summary.json", dict(status="complete_local_probe", rows=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("warm", "screen", "snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
