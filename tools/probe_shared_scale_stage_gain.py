"""Test a foldable scalar gain on the second component group of a 070 factor."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_real_block_tabu import _load_cases
from probe_residual_rank_split import metrics, payload_weight, split_cost
from safetensors.torch import load_file, save_file
from torch import nn

from nanoquant.domain.linear_math import functional_factorized_linear
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file

SCALE_LR = 1e-5
GAIN_LR = 1e-3


class SharedStageGain(nn.Module):
    """One shared input/output scale pair, and optionally one trainable group gain."""

    def __init__(self, payload, cut, mode, device):
        super().__init__()
        left = payload["stage0.left_binary"].to(device)
        right = payload["stage0.right_binary"].to(device)
        if len(payload) != 5 or not 0 < cut < right.shape[0]:
            raise ValueError("expected a single-stage payload and an interior component split")
        self.cut = cut
        self.register_buffer("left", left)
        self.register_buffer("right", right)
        for axis in ("pre", "mid", "post"):
            value = payload[f"stage0.scale_{axis}"].to(device=device, dtype=torch.float32, copy=True)
            setattr(self, f"scale_{axis}", nn.Parameter(value, requires_grad=mode != "gain_only"))
        self.gain = nn.Parameter(torch.ones((), device=device)) if mode != "scales" else None

    def middle(self):
        if self.gain is None:
            return self.scale_mid
        return torch.cat((self.scale_mid[:self.cut], self.scale_mid[self.cut:] * self.gain))

    def forward(self, inputs):
        # Same fused operation for both arms; do not introduce stage-wise BF16 rounding.
        return functional_factorized_linear(
            inputs, self.left.to(inputs), self.right.to(inputs),
            self.scale_pre.to(inputs), self.middle().to(inputs), self.scale_post.to(inputs),
        )

    def folded_payload(self):
        return {
            "stage0.left_binary": self.left.detach().cpu().contiguous(),
            "stage0.right_binary": self.right.detach().cpu().contiguous(),
            "stage0.scale_pre": self.scale_pre.detach().bfloat16().cpu(),
            "stage0.scale_mid": self.middle().detach().bfloat16().cpu(),
            "stage0.scale_post": self.scale_post.detach().bfloat16().cpu(),
        }


def tune(module, train, epochs):
    scales = [p for name, p in module.named_parameters() if name != "gain" and p.requires_grad]
    groups = [{"params": scales, "lr": SCALE_LR}] if scales else []
    if module.gain is not None:
        groups.append({"params": [module.gain], "lr": GAIN_LR})
    optimizer = torch.optim.AdamW(groups, weight_decay=0)
    for _ in range(epochs):
        for x, y in train:
            optimizer.zero_grad(set_to_none=True)
            error = (module(x).float() - y).square().mean() / y.square().mean().clamp_min(1e-12)
            error.backward()
            nn.utils.clip_grad_norm_(module.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
    return 1.0 if module.gain is None else float(module.gain.detach())


def run(args):
    torch.set_num_threads(8)
    previous = json.loads((args.warm / "protocol.json").read_text())
    parent = previous["previous"]
    results = json.loads((args.warm / "summary.json").read_text())
    model = args.snapshot / "model.safetensors"
    if (results["status"] != "complete_local_probe"
            or hash_file(model) != parent["source_model_sha256"]
            or hash_file(args.screen / "summary.json") != previous["previous_summary_sha256"]):
        raise ValueError("source probe/model provenance differs")
    identity, cases = _load_cases(Path(parent["source_run"]), model, (25,), tuple(previous["splits"]), 26)
    if asdict(identity) != parent["identity"]:
        raise ValueError("resident identity differs")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(warm_summary_sha256=hash_file(args.warm / "summary.json"),
                    warm_protocol_sha256=hash_file(args.warm / "protocol.json"),
                    tool_sha256=hash_file(Path(__file__)), scale_lr=SCALE_LR, gain_lr=GAIN_LR,
                    epochs=parent["tuning_epochs"], gain_storage_bits=0)
    atomic_write_json(args.output / "protocol.json", protocol)
    rows = []
    with acquire_device_lease(args.device):
        for case in cases:
            key = f"b25.{case.name}"
            capture = args.screen / f"{key}.activations.safetensors"
            if hash_file(capture) != previous["activation_hashes"][key]:
                raise ValueError("activation capture changed")
            data = load_file(str(capture))
            target = case.target.to(args.device).bfloat16()
            v, u = case.input_importance.to(args.device), case.output_importance.to(args.device)
            train = [(x, x.float() @ target.float().T) for x in data["train"].to(args.device)]
            cut = previous["splits"][case.name][0]
            for reference in (r for r in results["rows"] if r["owner"] == case.name):
                seed = reference["seed"]
                full = reference["full"]
                source = args.warm / full["payload"]
                if hash_file(source) != full["sha256"]:
                    raise ValueError("parent payload changed")
                payload = load_file(str(source))
                before = metrics(payload_weight(payload, args.device), target, v, u, data["validation"])
                arms = {}
                for mode in ("scales", "gain_only", "scales_and_gain"):
                    module = SharedStageGain(payload, cut, mode, args.device)
                    if not all(torch.equal(t, payload[k]) for k, t in module.folded_payload().items()):
                        raise AssertionError("unit gain does not reproduce the persisted parent")
                    gain = tune(module, train, parent["tuning_epochs"])
                    folded = module.folded_payload()
                    if any(not torch.equal(payload[k], folded[k]) for k in
                           ("stage0.left_binary", "stage0.right_binary")):
                        raise AssertionError("signs changed")
                    path = args.output / f"{key}.seed{seed}.{mode}.safetensors"
                    save_file(folded, str(path))
                    saved = load_file(str(path))
                    bits = split_cost(tuple(target.shape), (saved["stage0.right_binary"].shape[0],))
                    if bits != full["bits"]:
                        raise AssertionError("folded payload costs a different number of bits")
                    after = metrics(payload_weight(saved, args.device), target, v, u, data["validation"])
                    arms[mode] = dict(gain=gain, after=after, bits=bits, sha256=hash_file(path), payload=path.name)
                rows.append(dict(owner=case.name, seed=seed, cut=cut, before=before, arms=arms))
                atomic_write_json(args.output / "summary.json", dict(status="running", rows=rows))
                print(json.dumps(dict(owner=case.name, seed=seed,
                    before=before["activation_nmse"],
                    after={k: a["after"]["activation_nmse"] for k, a in arms.items()},
                    gains={k: a["gain"] for k, a in arms.items()})), flush=True)
    atomic_write_json(args.output / "summary.json", dict(status="complete_local_probe", rows=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("warm", "screen", "snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
