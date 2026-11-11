"""Compare complete ADMM schedules on retained Gemma owner ranks and objectives.

Fresh fits use original BF16 source weights, saved Fisher weights, independent
residual stages and matched seeds. This is not a replay of historical tuned
intermediate weights or an end-to-end model-quality benchmark.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_mlp_overlays_kl import _split_tokens
from probe_real_block_tabu import _load_cases
from safetensors.torch import load_file, save_file

from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.models import ArtifactRef, ArtifactTypes
from nanoquant.domain.profiling import NULL_RECORDER
from nanoquant.domain.scale_fit import fit_scales, reconstruct
from nanoquant.domain.seeds import logical_seed
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import latest_complete_identity, load_committed_block
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.profiling import Profiler
from nanoquant.resident_quantization import _legacy_cuda_numerics


def stage_inventory(run, blocks, expected_blocks):
    records = [json.loads(x) for x in (run / "state/journal.jsonl").read_text().splitlines() if x.strip()]
    identity, committed = latest_complete_identity(records, expected_blocks)
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


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def fit_owner(target, v, u, ranks, seed, block, path, parameters, scale_options, recorder=NULL_RECORDER):
    prediction = torch.zeros_like(target)
    frozen_prediction = torch.zeros_like(target, dtype=torch.float32)
    stages = []
    for index, rank in enumerate(ranks):
        residual = (target.float() - prediction.float()).to(target.dtype)
        stage_seed = seed if index == 0 else logical_seed(seed, "residual-factor-stage", block, path, index)
        generator = torch.Generator(device=target.device).manual_seed(stage_seed)
        synchronize(target.device)
        start = time.perf_counter()
        fit = factorize_admm_with_parameters(residual, v, u, rank, generator, parameters, recorder=recorder)
        synchronize(target.device)
        admm_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scales = fit_scales(residual, fit.left_binary, fit.right_binary,
                            fit.scale_pre, fit.scale_mid, fit.scale_post, v, u, **scale_options)
        # Production residual fitting reconstructs the stored FP32 fitted scales.
        prediction = prediction + reconstruct(fit.left_binary, fit.right_binary,
                                               scales.scale_pre, scales.scale_mid, scales.scale_post)
        synchronize(target.device)
        scale_seconds = time.perf_counter() - start
        frozen_prediction += reconstruct(fit.left_binary, fit.right_binary,
                                         scales.scale_pre.bfloat16(), scales.scale_mid.bfloat16(),
                                         scales.scale_post.bfloat16())
        stages.append(dict(rank=rank, seed=stage_seed, admm_seconds=admm_seconds,
                           scale_seconds=scale_seconds, iterations=fit.iterations_completed,
                           trace=[asdict(point) for point in fit.trace]))
    return frozen_prediction, stages


@torch.no_grad()
def score(prediction, target, v, u, inputs):
    difference = prediction.float() - target.float()
    denominator = (target.float().square() * u[:, None] * v[None, :]).sum().clamp_min(1e-30)
    weighted_nmse = float((difference.square() * u[:, None] * v[None, :]).sum() / denominator)
    raw_nmse = float(difference.square().sum() / target.float().square().sum().clamp_min(1e-30))
    rows = []
    for row in inputs:
        x = row.to(target.device).float()
        rows.append(float((x @ difference.T).square().sum() /
                          (x @ target.float().T).square().sum().clamp_min(1e-30)))
    if not all(math.isfinite(x) for x in [weighted_nmse, raw_nmse, *rows]):
        raise FloatingPointError("nonfinite ADMM comparison metric")
    return dict(weighted_nmse=weighted_nmse, relative_frobenius_error=math.sqrt(raw_nmse),
                activation_nmse=sum(rows) / len(rows), activation_nmse_per_sequence=rows)


def summarize(rows, baseline=(800, 5)):
    """Equal-weight paired owner/seed comparisons, never best-seed selection."""
    arms = {}
    for row in rows:
        key = (row["block"], row["owner"], row["seed"])
        arm = arms.setdefault((row["iterations"], row["inner_iterations"]), {})
        if key in arm:
            raise ValueError("duplicate owner/seed in an iteration arm")
        arm[key] = row
    if baseline not in arms:
        raise ValueError("missing baseline iteration arm")
    control = arms[baseline]
    result = []
    for budget, arm in sorted(arms.items()):
        if arm.keys() != control.keys():
            raise ValueError("unmatched owner/seed pairs")
        deltas = {metric: [100 * (arm[key]["metrics"][metric] / control[key]["metrics"][metric] - 1)
                           for key in control] for metric in ("weighted_nmse", "activation_nmse")}
        result.append(dict(iterations=budget[0], inner_iterations=budget[1], pairs=len(arm),
                           seconds=sum(r["seconds"] for r in arm.values()),
                           time_reduction_percent=100 * (1 - sum(r["seconds"] for r in arm.values()) /
                                                        sum(r["seconds"] for r in control.values())),
                           errors={name: dict(mean_percent=statistics.mean(values),
                                              median_percent=statistics.median(values), max_percent=max(values),
                                              min_percent=min(values)) for name, values in deltas.items()}))
    return result


def parse_arm(value):
    try:
        outer, inner = map(int, value.split("x"))
        if outer <= 0 or inner <= 0:
            raise ValueError
        return outer, inner
    except ValueError as error:
        raise argparse.ArgumentTypeError("arm must be positive OUTERxINNER, e.g. 800x2") from error


def capture_inputs(args, cases):
    tokens, fingerprint, _ = _split_tokens(
        args.snapshot, split="validation", samples=args.validation_offset + args.samples,
        sequence_length=args.sequence_length, local_files_only=True,
    )
    tokens = tokens[args.validation_offset:]
    save_file({"tokens": tokens.contiguous()}, args.output / "tokens.safetensors")
    teacher = load_causal_language_model(args.snapshot, torch_dtype=torch.bfloat16,
                                         local_files_only=True, attention_implementation="sdpa")
    teacher = teacher.to(args.device).eval()
    cache = {f"b{c.block}.{c.name}": [] for c in cases}
    handles = []
    for case in cases:
        key = f"b{case.block}.{case.name}"
        module = teacher.model.layers[case.block].get_submodule(case.members[0].path)

        def hook(_module, inputs, key=key):
            cache[key].append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu())

        handles.append(module.register_forward_pre_hook(hook))
    with torch.no_grad():
        for row in tokens:
            teacher.model(input_ids=row[None].to(args.device), use_cache=False)
    for handle in handles:
        handle.remove()
    del teacher
    gc.collect()
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
    for key, rows in cache.items():
        if len(rows) != args.samples:
            raise ValueError("activation capture count mismatch")
        save_file({"inputs": torch.stack(rows)}, args.output / f"{key}.inputs.safetensors")
    return dict(split="validation", fingerprint=fingerprint, offset=args.validation_offset,
                samples=args.samples, sequence_length=args.sequence_length,
                sha256=hash_file(args.output / "tokens.safetensors"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 12, 25])
    parser.add_argument("--owners", nargs="+", default=["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                                                       "self_attn.o_proj", "self_attn.attn_qkv"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--iterations", type=int, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument("--inner-iterations", type=int)
    parser.add_argument("--arms", nargs="+", type=parse_arm,
                        help="Explicit OUTERxINNER schedules, alternating order within each owner/seed")
    parser.add_argument("--expected-blocks", type=int, default=26)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--validation-offset", type=int, default=256)
    parser.add_argument("--profile", action="store_true", help="Instrument phases; do not use for A/B wall timing")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.iterations) <= 0 or args.samples <= 0 or args.sequence_length <= 1 or args.validation_offset < 0:
        parser.error("invalid iteration or validation settings")
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=False)
    identity, cases = _load_cases(args.run_output, args.snapshot / "model.safetensors",
                                  tuple(args.blocks), tuple(args.owners), args.expected_blocks)
    inventory = stage_inventory(args.run_output, args.blocks, args.expected_blocks)
    recipe = json.loads((args.run_output / "manifest.json").read_text())["resolved_config"]["canonical_run_config"]
    if (recipe["factorization"]["compute_dtype"] != "bfloat16"
            or recipe["factorization"]["implementation"] != "nanoquant_admm"):
        raise ValueError("this probe implements the BF16 NanoQuant ADMM recipe")
    parameters = AdmmParameters(**recipe["factorization"]["admm"])
    if args.inner_iterations is not None:
        parameters = replace(parameters, inner_iterations=args.inner_iterations)
    arms = args.arms or [(count, parameters.inner_iterations) for count in args.iterations]
    if len(set(arms)) != len(arms) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("duplicate arms or seeds")
    scale_options = dict(recipe["factorization"]["scale_fit"])
    if not scale_options.pop("enabled"):
        raise ValueError("this probe requires the source recipe's scale fit")
    protocol = dict(source_run=str(args.run_output.resolve()), identity=asdict(identity),
                    model_sha256=hash_file(args.snapshot / "model.safetensors"),
                    tool_sha256=hash_file(Path(__file__)), torch=torch.__version__,
                    arms=arms, seeds=args.seeds, blocks=args.blocks, owners=args.owners,
                    admm=asdict(parameters), scale_fit=scale_options, instrumented=args.profile,
                    scope="fresh original-weight fits at retained ranks/objectives; no block tuning or model quality",
                    schedule="full cubic schedule rescaled to each budget, not early truncation",
                    initialization="matched fresh seeds; production residual seed derivation; not historical RNG state",
                    precision="BF16 ADMM, FP32 scale fit/residual sum, final hard signs and BF16 scales")
    atomic_write_json(args.output / "protocol.json", protocol)
    results = []
    with acquire_device_lease(args.device), _legacy_cuda_numerics():
        protocol["cuda_numerics"] = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                          cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        protocol["device"] = torch.cuda.get_device_name(args.device) if "cuda" in args.device else "cpu"
        protocol["validation"] = capture_inputs(args, cases)
        atomic_write_json(args.output / "protocol.json", protocol)
        # Warm CUDA libraries with a tiny fit, outside all reported arm timings.
        factorize_admm_with_parameters(torch.eye(32, device=args.device), torch.ones(32, device=args.device),
                                      torch.ones(32, device=args.device), 8,
                                      torch.Generator(device=args.device).manual_seed(0),
                                      replace(parameters, outer_iterations=2))
        for case_index, case in enumerate(cases):
            key = f"b{case.block}.{case.name}"
            target = case.target.to(args.device).bfloat16()
            v, u = case.input_importance.to(args.device), case.output_importance.to(args.device)
            inputs = load_file(args.output / f"{key}.inputs.safetensors")["inputs"]
            for seed in args.seeds:
                # Alternate order across cases/seeds to reduce warmup/thermal ordering bias.
                budgets = arms if (case_index + seed) % 2 == 0 else list(reversed(arms))
                for count, inner in budgets:
                    label = f"{key}.seed{seed}.steps{count}x{inner}"
                    print(json.dumps(dict(starting=label)), flush=True)
                    recorder = (Profiler(ProfilingConfig(level=ProfilingLevel.MICRO, cuda_timing=True,
                                                         cuda_sample_every=1, raw_samples_per_phase=2048,
                                                         emit_span_events=False), run_id=label)
                                if args.profile else NULL_RECORDER)
                    synchronize(args.device)
                    start = time.perf_counter()
                    prediction, stages = fit_owner(target, v, u, inventory[(case.block, case.name)], seed,
                                                   case.block, case.members[0].path,
                                                   replace(parameters, outer_iterations=count, inner_iterations=inner),
                                                   scale_options, recorder)
                    synchronize(args.device)
                    elapsed = time.perf_counter() - start
                    row = dict(block=case.block, owner=case.name, seed=seed, iterations=count,
                               inner_iterations=inner, ranks=inventory[(case.block, case.name)],
                               seconds=elapsed, stages=stages, metrics=score(prediction, target, v, u, inputs))
                    results.append(row)
                    atomic_write_json(args.output / "results.json", results)
                    if isinstance(recorder, Profiler):
                        atomic_write_json(args.output / f"{label}.phases.json", recorder.snapshot())
                    print(json.dumps({k: value for k, value in row.items() if k != "stages"}), flush=True)
    print(f"Completed {len(results)} fits: {args.output}", flush=True)
    if (800, 5) in arms:
        atomic_write_json(args.output / "summary.json", summarize(results))


if __name__ == "__main__":
    main()
