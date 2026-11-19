"""Fit a residual with fixed outer scales, then jointly tune a fused shared-scale factor."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_real_block_tabu import _load_cases
from probe_residual_rank_split import fit_split, frozen_payload, metrics, payload_weight, split_cost
from safetensors.torch import load_file, save_file
from torch import nn
from torch.func import functional_call

from nanoquant.application.layers import TrainableFactorizedLinear
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.rank_expansion import fit_residual_middle_scales
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file


class JointResidualGain(nn.Module):
    """One fused factor; second-group gain folds into its middle coefficients."""

    def __init__(self, primary, left, right, mid, gain):
        super().__init__()
        self.cut = primary.scale_mid.numel()
        self.factor = TrainableFactorizedLinear(
            torch.cat((primary.left_latent, left), dim=1),
            torch.cat((primary.right_latent, right), dim=0),
            primary.scale_pre, torch.cat((primary.scale_mid, mid)), primary.scale_post,
        )
        self.gain = nn.Parameter(mid.new_tensor(gain))

    def middle(self):
        mid = self.factor.scale_mid
        return torch.cat((mid[:self.cut], mid[self.cut:] * self.gain))

    def forward(self, x):
        return functional_call(self.factor, {"scale_mid": self.middle()}, (x,))

    def folded_payload(self):
        result = frozen_payload(self.factor)
        result["stage0.scale_mid"] = self.middle().detach().bfloat16().cpu()
        return result


@torch.no_grad()
def fit_residual(primary, target, v, u, rank, seed, parameters):
    """Seed signs with ADMM, then solve residual coefficients under the fixed outer scales."""
    persisted = frozen_payload(primary)
    weight = payload_weight(persisted, target.device)
    residual = target.float() - weight
    pre = persisted["stage0.scale_pre"].to(residual)
    post = persisted["stage0.scale_post"].to(residual)
    # Put the sign search in the shared-scale orientation without dividing by tiny scales.
    oriented = (residual * pre.sign()[None, :] * post.sign()[:, None]).to(target.dtype)
    fit = factorize_admm_with_parameters(
        oriented, v, u, rank,
        torch.Generator(device=target.device).manual_seed(seed + 100003), parameters,
    )
    middle = fit_residual_middle_scales(residual, fit.left_binary, fit.right_binary, pre, post, v, u)
    if not middle.accepted:
        raise ValueError("fixed-outer residual solve rejected the correction")
    left = fit.left_binary.float() * fit.left_latent.float().abs().clamp_min(1e-6)
    right = fit.right_binary.float() * fit.right_latent.float().abs().clamp_min(1e-6)
    return left, right, middle, weight


@torch.no_grad()
def output_ratio(train, base, combined):
    x = train.reshape(-1, train.shape[-1]).float().to(base.device)
    primary = x @ base.mT
    residual = x @ (combined - base).mT
    return float(residual.norm() / primary.norm().clamp_min(1e-30))


def tune(module, train, target, epochs):
    latent = [module.factor.left_latent, module.factor.right_latent]
    scales = [module.factor.scale_pre, module.factor.scale_mid, module.factor.scale_post]
    optimizer = torch.optim.AdamW([
        {"params": latent, "lr": 3e-5}, {"params": scales, "lr": 1e-5}, {"params": [module.gain], "lr": 1e-3},
    ], weight_decay=0)
    inputs = train.to(target.device)
    outputs = inputs.float() @ target.float().mT
    for _ in range(epochs):
        for x, y in zip(inputs, outputs, strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss = (module(x).float() - y).square().mean() / y.square().mean().clamp_min(1e-12)
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()


def evaluate_arms(args, target, v, u, data, reference, cut, epochs, parameters):
    rank, = reference["full"]["ranks"]
    seed = reference["seed"]
    primary = fit_split(target, v, u, (cut,), seed, parameters)
    left, right, middle, primary_weight = fit_residual(primary, target, v, u, rank - cut, seed, parameters)
    # Normalise the actual correction magnitude, not the arbitrary scale coefficient.
    full_ratio = output_ratio(data["train"], primary_weight, primary_weight + middle.correction)
    if not 0 < full_ratio < float("inf"):
        raise ValueError("residual output must be nonzero and finite")
    arms = {}
    for name, gain in (("fitted", 1.0), ("small", 0.03 / full_ratio)):
        module = JointResidualGain(primary, left, right, middle.scale_mid, gain)
        weight = payload_weight(module.folded_payload(), args.device)
        before = metrics(weight, target, v, u, data["validation"])
        ratio = output_ratio(data["train"], primary_weight, weight)
        if name == "small" and not 0.029 < ratio < 0.031:
            raise AssertionError(f"small residual initialisation missed its RMS target: {ratio}")
        tune(module, data["train"], target, epochs)
        path = args.output / f"b25.{reference['owner']}.seed{seed}.{name}.safetensors"
        save_file(module.folded_payload(), str(path))
        saved = load_file(str(path))
        bits = split_cost(tuple(target.shape), (saved["stage0.right_binary"].shape[0],))
        if bits != reference["full"]["bits"]:
            raise AssertionError("folded residual did not preserve the control's exact bit cost")
        after = metrics(payload_weight(saved, args.device), target, v, u, data["validation"])
        arms[name] = dict(bits=bits, before=before, after=after, initial_gain=gain,
                          initial_output_ratio=ratio, final_gain=float(module.gain.detach()),
                          payload=path.name, sha256=hash_file(path))
    return dict(owner=reference["owner"], seed=seed, ranks=[cut, rank - cut],
                primary=metrics(primary_weight, target, v, u, data["validation"]),
                residual_weighted_error=dict(before=middle.before_error, after=middle.after_error),
                control=reference["full"], arms=arms)


def run(args):
    torch.set_num_threads(8)
    warm = json.loads((args.warm / "protocol.json").read_text())
    parent = warm["previous"]
    source = args.warm / "summary.json"
    results = json.loads(source.read_text())
    model = args.snapshot / "model.safetensors"
    if (results["status"] != "complete_local_probe"
            or hash_file(model) != parent["source_model_sha256"]
            or hash_file(args.screen / "summary.json") != warm["previous_summary_sha256"]
            or hash_file(Path(__file__).with_name("probe_residual_rank_split.py")) != parent["tool_sha256"]):
        raise ValueError("source probe/model provenance differs")
    identity, cases = _load_cases(Path(parent["source_run"]), model, (25,), tuple(warm["splits"]), 26)
    if asdict(identity) != parent["identity"]:
        raise ValueError("resident identity differs")
    args.output.mkdir(parents=True, exist_ok=False)
    parameters = AdmmParameters(**parent["admm"])
    atomic_write_json(args.output / "protocol.json",
        dict(warm_summary_sha256=hash_file(source), warm_protocol_sha256=hash_file(args.warm / "protocol.json"),
             tool_sha256=hash_file(Path(__file__)), admm=asdict(parameters), epochs=parent["tuning_epochs"],
             small_output_rms_ratio=0.03, tuning_lr=dict(binary=3e-5, scale=1e-5, gain=1e-3)))
    rows = []
    with acquire_device_lease(args.device):
        for case in cases:
            capture = args.screen / f"b25.{case.name}.activations.safetensors"
            if hash_file(capture) != warm["activation_hashes"][f"b25.{case.name}"]:
                raise ValueError("activation capture changed")
            data = load_file(str(capture))
            target = case.target.to(args.device).bfloat16()
            v, u = case.input_importance.to(args.device), case.output_importance.to(args.device)
            for reference in (row for row in results["rows"] if row["owner"] == case.name):
                if hash_file(args.warm / reference["full"]["payload"]) != reference["full"]["sha256"]:
                    raise ValueError("retained control payload changed")
                print(f"{case.name} seed {reference['seed']}: fitting primary and separate residual", flush=True)
                row = evaluate_arms(args, target, v, u, data, reference, warm["splits"][case.name][0],
                                    parent["tuning_epochs"], parameters)
                rows.append(row)
                atomic_write_json(args.output / "summary.json", dict(status="running", rows=rows))
                print(json.dumps(dict(owner=case.name, seed=reference["seed"],
                    control=row["control"]["after"]["activation_nmse"],
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
