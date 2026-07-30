"""Screen equal-bit fitted sign-word codebooks on one pinned Gemma matrix.

This is an analysis-only probe.  It does not introduce a packed schema or
runtime.  Each codebook arm is constrained throughout ADMM, includes the full
two-table decode cost, and spends the saved sign bits on aligned rank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open

from nanoquant.domain.calibration_math import shrink_importance
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.domain.sign_word_codebook import (
    codebook_index_metrics,
    factorize_sign_word_codebook_admm,
    maximum_codebook_rank_for_budget,
    sign_word_codebook_bit_cost,
)
from nanoquant.infrastructure.device_lease import acquire_device_lease

PINNED_MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
PROJECTION_PATHS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}


@dataclass(frozen=True, slots=True)
class SignWordCodebookProtocol:
    schema_version: int
    model_revision: str
    block: int
    projection: str
    baseline_rank: int
    index_widths: tuple[int, ...]
    rank_multiple: int
    scale_bits: int
    outer_iterations: int
    inner_iterations: int
    regularization: float
    penalty_schedule: str
    convergence_check_interval: int
    codebook_update_interval: int
    codebook_freeze_fraction: float
    codebook_mode: str
    assignment_batch_words: int
    scale_fit_passes: int
    calibration_shrinkage: float
    calibration_state: str
    seed: int
    device: str


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("integer list must not be empty")
    return result


def _protocol_hash(protocol: SignWordCodebookProtocol) -> str:
    encoded = json.dumps(asdict(protocol), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _logical_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _load_output(path: Path, protocol: SignWordCodebookProtocol) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "protocol_hash": _protocol_hash(protocol),
            "protocol": asdict(protocol),
            "results": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_hash") != _protocol_hash(protocol):
        raise ValueError("existing output uses a different sign-word-codebook protocol")
    if not isinstance(payload.get("results"), dict):
        raise ValueError("existing output is missing its result map")
    return cast(dict[str, Any], payload)


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_profile(
    state_directory: Path,
    path: str,
    shrinkage: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    manifest = json.loads((state_directory / "manifest.json").read_text(encoding="utf-8"))
    sample_count = int(manifest["sample_count"])
    layers = manifest.get("layers")
    if sample_count <= 0 or not isinstance(layers, list):
        raise ValueError("calibration state manifest is invalid")
    for index, layer in enumerate(layers):
        if str(layer["path"]) != path:
            continue
        with safe_open(str(state_directory / "state.safetensors"), framework="pt", device="cpu") as handle:
            return (
                shrink_importance(
                    handle.get_tensor(f"layer_{index}.inputs.total").float() / sample_count,
                    shrinkage,
                ),
                shrink_importance(
                    handle.get_tensor(f"layer_{index}.outputs.total").float() / sample_count,
                    shrinkage,
                ),
            )
    raise ValueError(f"calibration state is missing {path}")


def _metrics(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
) -> dict[str, float]:
    difference = weight.float() - reconstruction.float()
    weight32 = weight.float()
    raw_error = float(difference.square().sum())
    raw_target = float(weight32.square().sum())
    weighted_error = float(
        (
            difference.square()
            * output_importance.float().reshape(-1, 1)
            * input_importance.float().reshape(1, -1)
        ).sum()
    )
    weighted_target = float(
        (
            weight32.square()
            * output_importance.float().reshape(-1, 1)
            * input_importance.float().reshape(1, -1)
        ).sum()
    )
    return {
        "raw_error_energy": raw_error,
        "raw_target_energy": raw_target,
        "raw_normalized_rmse": math.sqrt(raw_error / max(raw_target, 1e-30)),
        "weighted_error_energy": weighted_error,
        "weighted_target_energy": weighted_target,
        "weighted_normalized_rmse": math.sqrt(weighted_error / max(weighted_target, 1e-30)),
    }


def _trace(trace: tuple[Any, ...]) -> list[dict[str, int | float]]:
    return [
        {
            "iteration": int(point.iteration),
            "rho": float(point.rho),
            "primal_residual": float(point.primal_residual),
            "dual_residual": float(point.dual_residual),
        }
        for point in trace
    ]


def _run_baseline(
    weight: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    protocol: SignWordCodebookProtocol,
) -> dict[str, Any]:
    generator = torch.Generator(device=protocol.device).manual_seed(
        _logical_seed(protocol.seed, "free-word-baseline")
    )
    parameters = AdmmParameters(
        outer_iterations=protocol.outer_iterations,
        inner_iterations=protocol.inner_iterations,
        regularization=protocol.regularization,
        penalty_schedule=protocol.penalty_schedule,
        convergence_check_interval=protocol.convergence_check_interval,
        transpose_wide=True,
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(protocol.device)
        torch.cuda.synchronize(protocol.device)
    started = time.perf_counter()
    factorized = factorize_admm_with_parameters(
        weight,
        input_importance,
        output_importance,
        protocol.baseline_rank,
        generator,
        parameters,
    )
    fitted = fit_scales(
        weight,
        factorized.left_binary,
        factorized.right_binary,
        factorized.scale_pre,
        factorized.scale_mid,
        factorized.scale_post,
        input_importance,
        output_importance,
        alternating_passes=protocol.scale_fit_passes,
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.synchronize(protocol.device)
    bit_cost = factor_bit_cost(
        weight.shape[0],
        weight.shape[1],
        protocol.baseline_rank,
        scale_bits=protocol.scale_bits,
    )
    return {
        "arm": "free_words",
        "rank": protocol.baseline_rank,
        "bit_cost": asdict(bit_cost),
        "total_bits": bit_cost.total,
        "actual_bpw": bit_cost.total / weight.numel(),
        "signed_contributions_per_weight": protocol.baseline_rank,
        "factorized_work_over_dense": (
            protocol.baseline_rank * sum(weight.shape) / weight.numel()
        ),
        "metrics": _metrics(
            weight,
            fitted.reconstruction,
            input_importance,
            output_importance,
        ),
        "scale_fit_accepted": fitted.accepted,
        "scale_fit_rollback_reason": fitted.rollback_reason,
        "trace": _trace(factorized.trace),
        "wall_seconds": time.perf_counter() - started,
        "peak_device_bytes": (
            int(torch.cuda.max_memory_allocated(protocol.device))
            if protocol.device.startswith("cuda")
            else 0
        ),
    }


def _run_codebook(
    weight: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    protocol: SignWordCodebookProtocol,
    index_width: int,
    target_bits: int,
) -> dict[str, Any]:
    rank = maximum_codebook_rank_for_budget(
        weight.shape[0],
        weight.shape[1],
        target_bits,
        index_width=index_width,
        rank_multiple=protocol.rank_multiple,
        scale_width=protocol.scale_bits,
    )
    generator = torch.Generator(device=protocol.device).manual_seed(
        _logical_seed(protocol.seed, f"codebook-{index_width}-rank-{rank}")
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(protocol.device)
        torch.cuda.synchronize(protocol.device)
    started = time.perf_counter()
    factorized = factorize_sign_word_codebook_admm(
        weight,
        input_importance,
        output_importance,
        rank,
        generator,
        index_bits=index_width,
        outer_iterations=protocol.outer_iterations,
        inner_iterations=protocol.inner_iterations,
        regularization=protocol.regularization,
        penalty_schedule=protocol.penalty_schedule,
        convergence_check_interval=protocol.convergence_check_interval,
        codebook_update_interval=protocol.codebook_update_interval,
        codebook_freeze_fraction=protocol.codebook_freeze_fraction,
        assignment_batch_words=protocol.assignment_batch_words,
        codebook_mode=protocol.codebook_mode,
    )
    fitted = fit_scales(
        weight,
        factorized.factors.left_binary,
        factorized.factors.right_binary,
        factorized.factors.scale_pre,
        factorized.factors.scale_mid,
        factorized.factors.scale_post,
        input_importance,
        output_importance,
        alternating_passes=protocol.scale_fit_passes,
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.synchronize(protocol.device)
    bit_cost = sign_word_codebook_bit_cost(
        weight.shape[0],
        weight.shape[1],
        rank,
        index_width=index_width,
        scale_width=protocol.scale_bits,
    )
    return {
        "arm": f"codebook_k{index_width}",
        "rank": rank,
        "rank_multiple_vs_baseline": rank / protocol.baseline_rank,
        "bit_cost": asdict(bit_cost),
        "total_bits": bit_cost.total,
        "unused_budget_bits": target_bits - bit_cost.total,
        "actual_bpw": bit_cost.total / weight.numel(),
        "signed_contributions_per_weight": rank,
        "factorized_work_over_dense": rank * sum(weight.shape) / weight.numel(),
        "metrics": _metrics(
            weight,
            fitted.reconstruction,
            input_importance,
            output_importance,
        ),
        "index_metrics": codebook_index_metrics(factorized),
        "scale_fit_accepted": fitted.accepted,
        "scale_fit_rollback_reason": fitted.rollback_reason,
        "trace": _trace(factorized.factors.trace),
        "wall_seconds": time.perf_counter() - started,
        "peak_device_bytes": (
            int(torch.cuda.max_memory_allocated(protocol.device))
            if protocol.device.startswith("cuda")
            else 0
        ),
    }


def _comparison(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    current = result["metrics"]
    control = baseline["metrics"]
    weighted = float(current["weighted_normalized_rmse"])
    baseline_weighted = float(control["weighted_normalized_rmse"])
    raw = float(current["raw_normalized_rmse"])
    baseline_raw = float(control["raw_normalized_rmse"])
    return {
        "weighted_rmse_change_fraction": weighted / baseline_weighted - 1,
        "weighted_error_energy_change_fraction": (
            float(current["weighted_error_energy"]) / float(control["weighted_error_energy"]) - 1
        ),
        "raw_rmse_change_fraction": raw / baseline_raw - 1,
        "raw_error_energy_change_fraction": (
            float(current["raw_error_energy"]) / float(control["raw_error_energy"]) - 1
        ),
    }


def run(args: argparse.Namespace) -> int:
    if args.projection not in PROJECTION_PATHS:
        raise ValueError(f"unknown projection: {args.projection}")
    if args.baseline_rank <= 0 or any(width <= 0 or width % 2 for width in args.index_widths):
        raise ValueError("baseline rank must be positive and index widths positive/even")
    protocol = SignWordCodebookProtocol(
        3,
        args.model_revision,
        args.block,
        args.projection,
        args.baseline_rank,
        args.index_widths,
        args.rank_multiple,
        args.scale_bits,
        args.outer_iterations,
        args.inner_iterations,
        args.regularization,
        args.penalty_schedule,
        args.convergence_check_interval,
        args.codebook_update_interval,
        args.codebook_freeze_fraction,
        args.codebook_mode,
        args.assignment_batch_words,
        args.scale_fit_passes,
        args.calibration_shrinkage,
        str(args.calibration_state.resolve()),
        args.seed,
        args.device,
    )
    output = _load_output(args.output, protocol)
    tensor_name = (
        f"model.layers.{args.block}.{PROJECTION_PATHS[args.projection]}.weight"
    )
    calibration_path = f"block.{args.block}.{PROJECTION_PATHS[args.projection]}"
    input_cpu, output_cpu = _load_profile(
        args.calibration_state,
        calibration_path,
        args.calibration_shrinkage,
    )
    lease_context = (
        acquire_device_lease(args.device)
        if args.device.startswith("cuda")
        else nullcontext()
    )
    with lease_context, safe_open(str(args.model), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(tensor_name).to(args.device)
        input_importance = input_cpu.to(args.device).float()
        output_importance = output_cpu.to(args.device).float()
        baseline = output["results"].get("free_words")
        if baseline is None:
            print("running free-word baseline", flush=True)
            baseline = _run_baseline(
                weight,
                input_importance,
                output_importance,
                protocol,
            )
            output["results"]["free_words"] = baseline
            output["matrix"] = {
                "tensor_name": tensor_name,
                "shape": list(weight.shape),
                "source_elements": weight.numel(),
            }
            _write_output(args.output, output)
            print(
                "completed free-word baseline "
                f"rank={baseline['rank']} weighted_rmse="
                f"{baseline['metrics']['weighted_normalized_rmse']:.6f}",
                flush=True,
            )
        else:
            print("reusing free-word baseline", flush=True)
        target_bits = int(baseline["total_bits"])
        for width in args.index_widths:
            key = f"codebook_k{width}"
            result = output["results"].get(key)
            if result is None:
                rank = maximum_codebook_rank_for_budget(
                    weight.shape[0],
                    weight.shape[1],
                    target_bits,
                    index_width=width,
                    rank_multiple=protocol.rank_multiple,
                    scale_width=protocol.scale_bits,
                )
                print(f"running {key} rank={rank}", flush=True)
                result = _run_codebook(
                    weight,
                    input_importance,
                    output_importance,
                    protocol,
                    width,
                    target_bits,
                )
                result["comparison_to_free_words"] = _comparison(result, baseline)
                output["results"][key] = result
                _write_output(args.output, output)
                print(
                    f"completed {key} rank={rank} weighted_rmse="
                    f"{result['metrics']['weighted_normalized_rmse']:.6f} "
                    f"change={result['comparison_to_free_words']['weighted_rmse_change_fraction'] * 100:+.2f}%",
                    flush=True,
                )
            else:
                print(f"reusing {key}", flush=True)
        output["decision_screen"] = {
            "candidate_arms": [
                key
                for key in (f"codebook_k{width}" for width in args.index_widths)
                if float(
                    output["results"][key]["comparison_to_free_words"][
                        "weighted_error_energy_change_fraction"
                    ]
                )
                < 0
            ],
            "requires_splice_kl_before_promotion": True,
            "runtime_schema_changed": False,
        }
        _write_output(args.output, output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--block", type=int, default=12)
    parser.add_argument("--projection", default="down")
    parser.add_argument("--baseline-rank", type=int, default=970)
    parser.add_argument("--index-widths", type=_parse_ints, default=(8, 12))
    parser.add_argument("--rank-multiple", type=int, default=32)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--outer-iterations", type=int, default=400)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=3e-2)
    parser.add_argument("--penalty-schedule", default="cubic")
    parser.add_argument("--convergence-check-interval", type=int, default=100)
    parser.add_argument("--codebook-update-interval", type=int, default=10)
    parser.add_argument("--codebook-freeze-fraction", type=float, default=0.5)
    parser.add_argument("--codebook-mode", choices=("product", "full"), default="product")
    parser.add_argument("--assignment-batch-words", type=int, default=65_536)
    parser.add_argument("--scale-fit-passes", type=int, default=2)
    parser.add_argument("--calibration-shrinkage", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
