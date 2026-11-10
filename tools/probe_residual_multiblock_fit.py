"""Fresh diagonal-Fisher objectives and multi-seed equal-byte residual screens."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_non_wikitext_kd_quality import _c4_slice_reservation, _token_hash
from probe_residual_stage_budget import run as fit_residual
from probe_residual_stage_functional import REVISION, file_hash
from safetensors.torch import load_file, save_file

from nanoquant.application.calibration import calibrate_causal_model
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config


def verified_tokens(path: Path, role: str) -> tuple[torch.Tensor, dict]:
    receipt = json.loads(path.with_suffix(".json").read_text())
    entry = receipt["entry"]
    tokens = load_file(str(path))["input_ids"].long()
    if (file_hash(path) != receipt["file_sha256"] or _token_hash(tokens) != entry["token_hash"]
            or tuple(tokens.shape) != (entry["samples"], entry["sequence_length"])
            or not receipt["retired_before_evaluation"] or entry["retirement"]["role"] != role):
        raise ValueError("token identity or purpose mismatch")
    _c4_slice_reservation(Path(receipt["registry"]), entry["id"], offset=entry["offset"],
                          samples=entry["samples"], sequence_length=entry["sequence_length"],
                          token_hash=entry["token_hash"])
    return tokens, receipt


def normalized_importance(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float().cpu()
    if not torch.isfinite(value).all() or (value < 0).any() or value.mean() <= 0:
        raise ValueError("invalid fresh importance vector")
    return (value / value.mean()).clamp_min(1e-6)


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    tokens, token_receipt = verified_tokens(args.tokens, "fit")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model_revision=REVISION, source_sha256=file_hash(args.snapshot / "model.safetensors"),
                    tokens=token_receipt, blocks=args.blocks, seeds=args.seeds, iterations=args.iterations,
                    cap_bpw=2.0, protected_columns=7, device=args.device,
                    objective="original BF16 teacher online Fisher, each vector mean-normalized and floored at 1e-6",
                    seed_selection="lowest weighted NMSE per block and arm, before model-level selection",
                    tool_hashes={name: file_hash(Path(__file__).with_name(name)) for name in
                                 (Path(__file__).name, "probe_residual_stage_budget.py")})
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("fresh-fit resume protocol mismatch")
    atomic_write_json(path, protocol)
    with acquire_device_lease(args.device):
        if not (args.output / "calibration.json").exists():
            config = json.loads((args.snapshot / "config.json").read_text())
            teacher = load_causal_language_model(
                args.snapshot, torch_dtype=torch.bfloat16, local_files_only=True,
                attention_implementation=adapter_for_config(config).attention_implementation).to(args.device).eval()
            layers = tuple((f"model.layers.{block}.mlp.down_proj",
                            teacher.model.layers[block].mlp.down_proj) for block in args.blocks)
            statistics = calibrate_causal_model(
                teacher, tuple(batch[None].to(args.device) for batch in tokens), layers,
                method="online_fisher", progress_callback=lambda done, total:
                    print(json.dumps(dict(calibration_done=done, total=total)), flush=True))
            receipts = {}
            for block, statistic in zip(args.blocks, statistics, strict=True):
                target = teacher.model.layers[block].mlp.down_proj.weight.detach().float().cpu()
                u = normalized_importance(statistic.output_importance)
                v = normalized_importance(statistic.input_importance)
                salience = (target.square() * u[:, None] * v[None, :]).sum(0)
                protected = torch.argsort(salience, descending=True, stable=True)[:7].sort().values
                folder = args.output / f"block{block}"
                folder.mkdir(exist_ok=True)
                fixture = folder / "fixture.safetensors"
                save_file(dict(target=target, input_importance=v, output_importance=u,
                               outlier_indices=protected, outlier_values=target[:, protected].contiguous()),
                          str(fixture))
                receipt = dict(source_key=statistic.path + ".weight", fixture_sha256=file_hash(fixture),
                               model_revision=REVISION, objective_scope=protocol["objective"],
                               tokens_sha256=token_receipt["file_sha256"], outlier_indices=protected.tolist())
                atomic_write_json(fixture.with_suffix(".json"), receipt)
                receipts[str(block)] = receipt
            atomic_write_json(args.output / "calibration.json", receipts)
            del teacher, layers, statistics
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        for block in args.blocks:
            fixture_path = args.output / f"block{block}" / "fixture.safetensors"
            receipt = json.loads(fixture_path.with_suffix(".json").read_text())
            if file_hash(fixture_path) != receipt["fixture_sha256"]:
                raise ValueError("fresh fixture changed")
            fixture = load_file(str(fixture_path))
            target = fixture["target"].clone().to(args.device)
            protected = fixture["outlier_indices"].long()
            target[:, protected.to(args.device)] = 0
            v = fixture["input_importance"].to(args.device)
            u = fixture["output_importance"].to(args.device)
            for seed in args.seeds:
                folder = args.output / f"block{block}" / f"seed{seed}"
                folder.mkdir(exist_ok=True)
                first_path = folder / "first.safetensors"
                if not first_path.exists():
                    start = time.perf_counter()
                    fit = factorize_admm_with_parameters(
                        target, v, u, min(target.shape), torch.Generator(device=args.device).manual_seed(seed),
                        AdmmParameters(outer_iterations=args.iterations, transpose_wide=False))
                    scales = fit_scales(target, fit.left_binary, fit.right_binary, fit.scale_pre, fit.scale_mid,
                                        fit.scale_post, v, u, alternating_passes=2)
                    state = {name: getattr(fit, name).to(device="cpu", dtype=torch.int8)
                             for name in ("left_binary", "right_binary")}
                    state.update({name: getattr(scales, name).float().cpu()
                                  for name in ("scale_pre", "scale_mid", "scale_post")})
                    state["scale_pre"][protected] = 0
                    save_file(state, str(first_path))
                    atomic_write_json(folder / "first.json", dict(seconds=time.perf_counter() - start,
                                      sha256=file_hash(first_path), seed=seed))
                    del fit, scales
                pilot = folder / "pilot"
                if (pilot / "summary.json").exists():
                    saved = json.loads((pilot / "summary.json").read_text())
                    if saved["status"] != "complete_reconstruction_only":
                        raise ValueError("incomplete pilot requires a new directory; existing evidence preserved")
                else:
                    fit_residual(argparse.Namespace(fixture=fixture_path, first=first_path, output=pilot,
                                                    bpw=2.0, iterations=args.iterations, seed=seed,
                                                    threads=args.threads, device=args.device))
                print(json.dumps(dict(completed_block=block, seed=seed)), flush=True)
    selection = {}
    for block in args.blocks:
        candidates = [(seed, args.output / f"block{block}" / f"seed{seed}" / "pilot") for seed in args.seeds]
        summaries = {seed: json.loads((folder / "summary.json").read_text()) for seed, folder in candidates}
        selection[str(block)] = {}
        for arm in ("int8", "sign", "residual_factor"):
            seed, folder = min(candidates, key=lambda item: summaries[item[0]]["results"][arm]["weighted_nmse"])
            result = summaries[seed]["results"][arm]
            selection[str(block)][arm] = dict(seed=seed, path=str((folder / (arm + ".bin")).resolve()), **result)
    atomic_write_json(args.output / "selection.json", dict(status="complete_reconstruction_only", blocks=selection))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", nargs="+", type=int, default=[0, 12, 25])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
