"""Screen cyclic component-group scale banks at fixed and equal bit budgets.

For scale rank K, binary component r belongs to group ``r % K``. Each group
owns one input-channel and one output-channel scale vector; scale_mid remains
per binary component. K=1 is the current NanoQuant representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from nanoquant.config.codec import semantic_hash
from nanoquant.domain.calibration_math import shrink_importance
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json

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
class CyclicScaleProtocol:
    schema_version: int
    model_revision: str
    blocks: tuple[int, ...]
    projections: tuple[str, ...]
    target_bpw: float
    scale_ranks: tuple[int, ...]
    rank_alignment: int
    scale_bits: int
    outer_iterations: int
    inner_iterations: int
    regularization: float
    penalty_schedule: str
    convergence_check_interval: int
    scale_fit_passes: int
    seed: int
    device: str
    calibration_state: str
    calibration_shrinkage: float


@dataclass(frozen=True, slots=True)
class CyclicScaleFitResult:
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    reconstruction: torch.Tensor
    before_error: float
    after_error: float
    accepted: bool


def cyclic_scale_bit_cost(
    out_features: int,
    in_features: int,
    factor_rank: int,
    scale_rank: int,
    *,
    scale_bits: int = 16,
    rank_alignment: int = 32,
) -> int:
    if min(out_features, in_features, factor_rank, scale_rank, scale_bits, rank_alignment) <= 0:
        raise ValueError("cyclic scale bit-cost inputs must be positive")
    base = factor_bit_cost(
        out_features,
        in_features,
        factor_rank,
        scale_bits=scale_bits,
        rank_alignment=rank_alignment,
    )
    return base.total + (scale_rank - 1) * (out_features + in_features) * scale_bits


def maximum_equal_bit_rank(
    out_features: int,
    in_features: int,
    bit_budget: int,
    scale_rank: int,
    *,
    scale_bits: int = 16,
    rank_alignment: int = 32,
) -> int:
    maximum = min(out_features, in_features)
    candidates = range(rank_alignment, maximum + 1, rank_alignment)
    accepted = 0
    for factor_rank in candidates:
        if cyclic_scale_bit_cost(
            out_features,
            in_features,
            factor_rank,
            scale_rank,
            scale_bits=scale_bits,
            rank_alignment=rank_alignment,
        ) <= bit_budget:
            accepted = factor_rank
    if accepted == 0:
        raise ValueError("bit budget cannot fund an aligned cyclic-scale factorization")
    return accepted


def cyclic_reconstruct(
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
) -> torch.Tensor:
    left = left_binary.float()
    right = right_binary.float()
    pre = scale_pre.float()
    mid = scale_mid.float().reshape(-1)
    post = scale_post.float()
    if (
        left.ndim != 2
        or right.ndim != 2
        or pre.ndim != 2
        or post.ndim != 2
        or left.shape[1] != right.shape[0]
        or mid.numel() != right.shape[0]
        or pre.shape[1] != right.shape[1]
        or post.shape[0] != left.shape[0]
        or pre.shape[0] != post.shape[1]
    ):
        raise ValueError("cyclic scale geometry is invalid")
    result = torch.zeros(
        (left.shape[0], right.shape[1]), device=left.device, dtype=torch.float32
    )
    components = torch.arange(right.shape[0], device=right.device)
    for group in range(pre.shape[0]):
        selected = components.remainder(pre.shape[0]) == group
        result += (left[:, selected] * post[:, group, None]) @ (
            right[selected]
            * mid[selected, None]
            * pre[group, None, :]
        )
    return result


def _weighted_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
) -> torch.Tensor:
    return (
        (target.float() - prediction.float()).square()
        * output_importance.float()[:, None]
        * input_importance.float()[None, :]
    ).sum()


def _fit_mid(
    target: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    groups = torch.arange(right.shape[0], device=right.device).remainder(pre.shape[0])
    component_pre = pre.index_select(0, groups)
    component_post = post.index_select(1, groups)
    scaled_left = left * component_post
    scaled_right = right * component_pre
    left_gram = scaled_left.mT @ (scaled_left * output_importance[:, None])
    weighted_right = scaled_right * input_importance.sqrt()[None, :]
    right_gram = weighted_right @ weighted_right.mT
    cross = scaled_left.mT @ (
        target * input_importance[None, :] * output_importance[:, None]
    )
    rhs = (cross * scaled_right).sum(dim=1)
    system = left_gram * right_gram
    system = 0.5 * (system + system.mT)
    ridge = torch.clamp(system.diagonal().mean().abs() * 1e-6, min=epsilon)
    system.diagonal().add_(ridge)
    cholesky, info = torch.linalg.cholesky_ex(system, upper=False)
    fitted = (
        torch.cholesky_solve(rhs[:, None], cholesky, upper=False).squeeze(1)
        if int(info.item()) == 0
        else torch.linalg.lstsq(system, rhs[:, None]).solution.squeeze(1)
    )
    return torch.nan_to_num(fitted)


def fit_cyclic_scales(
    target: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    scale_rank: int,
    *,
    alternating_passes: int = 2,
    epsilon: float = 1e-8,
    protected_columns: torch.Tensor | None = None,
) -> CyclicScaleFitResult:
    if scale_rank <= 0 or alternating_passes < 0 or epsilon <= 0:
        raise ValueError("cyclic scale-fit settings are invalid")
    left = torch.sign(left_binary.detach().float())
    right = torch.sign(right_binary.detach().float())
    target32 = target.detach().float()
    input_weight = input_importance.detach().float().reshape(-1).clamp_min(epsilon)
    output_weight = output_importance.detach().float().reshape(-1).clamp_min(epsilon)
    pre = scale_pre.detach().float().reshape(1, -1).repeat(scale_rank, 1)
    mid = scale_mid.detach().float().reshape(-1).clone()
    post = scale_post.detach().float().reshape(-1, 1).repeat(1, scale_rank)
    protected = None if protected_columns is None else protected_columns.detach().long().reshape(-1)
    if protected is not None:
        pre[:, protected] = 0
    prediction = cyclic_reconstruct(left, right, pre, mid, post)
    best_error = _weighted_error(target32, prediction, input_weight, output_weight)
    best = (pre.clone(), mid.clone(), post.clone(), prediction.clone())
    components = torch.arange(right.shape[0], device=right.device)
    for _ in range(alternating_passes):
        for group in range(scale_rank):
            selected = components.remainder(scale_rank) == group
            unscaled = left[:, selected] @ (
                right[selected] * mid[selected, None] * pre[group, None, :]
            )
            current = post[:, group, None] * unscaled
            residual_target = target32 - (prediction - current)
            numerator = (unscaled * residual_target * input_weight[None, :]).sum(dim=1)
            denominator = (unscaled.square() * input_weight[None, :]).sum(dim=1).clamp_min(epsilon)
            post[:, group] = torch.nan_to_num(numerator / denominator)
            prediction = prediction - current + post[:, group, None] * unscaled

        for group in range(scale_rank):
            selected = components.remainder(scale_rank) == group
            unscaled = (left[:, selected] * post[:, group, None]) @ (
                right[selected] * mid[selected, None]
            )
            current = unscaled * pre[group, None, :]
            residual_target = target32 - (prediction - current)
            numerator = (
                unscaled * residual_target * output_weight[:, None]
            ).sum(dim=0)
            denominator = (
                unscaled.square() * output_weight[:, None]
            ).sum(dim=0).clamp_min(epsilon)
            pre[group] = torch.nan_to_num(numerator / denominator)
            if protected is not None:
                pre[group, protected] = 0
            prediction = prediction - current + unscaled * pre[group, None, :]

        mid = _fit_mid(
            target32,
            left,
            right,
            pre,
            post,
            input_weight,
            output_weight,
            epsilon,
        )
        prediction = cyclic_reconstruct(left, right, pre, mid, post)
        error = _weighted_error(target32, prediction, input_weight, output_weight)
        if bool(torch.isfinite(error)) and float(error) < float(best_error):
            best_error = error
            best = (pre.clone(), mid.clone(), post.clone(), prediction.clone())
    best_pre, best_mid, best_post, best_prediction = best
    before = float(
        _weighted_error(
            target32,
            cyclic_reconstruct(
                left,
                right,
                scale_pre.reshape(1, -1),
                scale_mid,
                scale_post.reshape(-1, 1),
            ),
            input_weight,
            output_weight,
        )
    )
    after = float(best_error)
    return CyclicScaleFitResult(
        best_pre,
        best_mid,
        best_post,
        best_prediction,
        before,
        after,
        math.isfinite(after) and after <= before,
    )


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("integer lists must contain positive values")
    return result


def _parse_projections(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(result) - set(PROJECTION_PATHS))
    if not result or unknown:
        raise argparse.ArgumentTypeError(f"unsupported projections: {', '.join(unknown)}")
    return result


def _load_profiles(
    state_directory: Path,
    shrinkage: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    manifest = json.loads((state_directory / "manifest.json").read_text(encoding="utf-8"))
    sample_count = int(manifest["sample_count"])
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with safe_open(
        str(state_directory / "state.safetensors"), framework="pt", device="cpu"
    ) as handle:
        for index, layer in enumerate(manifest["layers"]):
            result[str(layer["path"])] = (
                shrink_importance(
                    handle.get_tensor(f"layer_{index}.inputs.total").float() / sample_count,
                    shrinkage,
                ),
                shrink_importance(
                    handle.get_tensor(f"layer_{index}.outputs.total").float() / sample_count,
                    shrinkage,
                ),
            )
    return result


def _logical_seed(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{key}".encode()).digest()[:8], "little") % (
        2**63 - 1
    )


def _normalized_error(error: float, target: torch.Tensor, weights: torch.Tensor) -> float:
    denominator = float((target.float().square() * weights).sum())
    return math.sqrt(error / max(denominator, 1e-30))


def _factorize_rank(
    weight: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    factor_rank: int,
    protocol: CyclicScaleProtocol,
    key: str,
) -> Any:
    generator = torch.Generator(device=protocol.device).manual_seed(
        _logical_seed(protocol.seed, key)
    )
    return factorize_admm_with_parameters(
        weight,
        input_importance,
        output_importance,
        factor_rank,
        generator,
        AdmmParameters(
            outer_iterations=protocol.outer_iterations,
            inner_iterations=protocol.inner_iterations,
            regularization=protocol.regularization,
            penalty_schedule=protocol.penalty_schedule,
            convergence_check_interval=protocol.convergence_check_interval,
            transpose_wide=True,
        ),
    )


def _evaluate_layer(
    handle: Any,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]],
    block: int,
    projection: str,
    protocol: CyclicScaleProtocol,
) -> dict[str, object]:
    path = PROJECTION_PATHS[projection]
    weight = handle.get_tensor(f"model.layers.{block}.{path}.weight").to(protocol.device)
    input_cpu, output_cpu = profiles[f"block.{block}.{path}"]
    input_importance = input_cpu.to(protocol.device).float()
    output_importance = output_cpu.to(protocol.device).float()
    out_features, in_features = weight.shape
    source_elements = weight.numel()
    target_bits = math.floor(protocol.target_bpw * source_elements)
    base_rank = maximum_equal_bit_rank(
        out_features,
        in_features,
        target_bits,
        1,
        scale_bits=protocol.scale_bits,
        rank_alignment=protocol.rank_alignment,
    )
    base_budget = cyclic_scale_bit_cost(
        out_features,
        in_features,
        base_rank,
        1,
        scale_bits=protocol.scale_bits,
        rank_alignment=protocol.rank_alignment,
    )
    element_weights = output_importance[:, None] * input_importance[None, :]
    ranks = {
        scale_rank: maximum_equal_bit_rank(
            out_features,
            in_features,
            base_budget,
            scale_rank,
            scale_bits=protocol.scale_bits,
            rank_alignment=protocol.rank_alignment,
        )
        for scale_rank in protocol.scale_ranks
    }
    factorized_by_rank: dict[int, Any] = {}
    controls: dict[int, dict[str, object]] = {}
    arms: dict[str, dict[str, object]] = {}
    for factor_rank in sorted(set(ranks.values())):
        started = time.perf_counter()
        factorized = _factorize_rank(
            weight,
            input_importance,
            output_importance,
            factor_rank,
            protocol,
            f"{block}|{projection}|rank={factor_rank}",
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
        error = float(_weighted_error(weight, fitted.reconstruction, input_importance, output_importance))
        factorized_by_rank[factor_rank] = (factorized, fitted)
        controls[factor_rank] = {
            "factor_rank": factor_rank,
            "weighted_error": error,
            "weighted_nrmse": _normalized_error(error, weight, element_weights),
            "wall_seconds": time.perf_counter() - started,
        }
    base_error = float(controls[base_rank]["weighted_error"])
    base_factorized, base_fitted = factorized_by_rank[base_rank]
    for scale_rank in protocol.scale_ranks:
        factor_rank = ranks[scale_rank]
        factorized, fitted = factorized_by_rank[factor_rank]
        started = time.perf_counter()
        cyclic = (
            CyclicScaleFitResult(
                fitted.scale_pre.reshape(1, -1),
                fitted.scale_mid,
                fitted.scale_post.reshape(-1, 1),
                fitted.reconstruction,
                float(controls[factor_rank]["weighted_error"]),
                float(controls[factor_rank]["weighted_error"]),
                True,
            )
            if scale_rank == 1
            else fit_cyclic_scales(
                weight,
                factorized.left_binary,
                factorized.right_binary,
                fitted.scale_pre,
                fitted.scale_mid,
                fitted.scale_post,
                input_importance,
                output_importance,
                scale_rank,
                alternating_passes=protocol.scale_fit_passes,
            )
        )
        cost = cyclic_scale_bit_cost(
            out_features,
            in_features,
            factor_rank,
            scale_rank,
            scale_bits=protocol.scale_bits,
            rank_alignment=protocol.rank_alignment,
        )
        control_error = float(controls[factor_rank]["weighted_error"])
        fixed_rank_cyclic = (
            cyclic
            if factor_rank == base_rank
            else fit_cyclic_scales(
                weight,
                base_factorized.left_binary,
                base_factorized.right_binary,
                base_fitted.scale_pre,
                base_fitted.scale_mid,
                base_fitted.scale_post,
                input_importance,
                output_importance,
                scale_rank,
                alternating_passes=protocol.scale_fit_passes,
            )
        )
        fixed_rank_cost = cyclic_scale_bit_cost(
            out_features,
            in_features,
            base_rank,
            scale_rank,
            scale_bits=protocol.scale_bits,
            rank_alignment=protocol.rank_alignment,
        )
        arms[str(scale_rank)] = {
            "scale_rank": scale_rank,
            "factor_rank": factor_rank,
            "total_bits": cost,
            "actual_bpw": cost / source_elements,
            "unspent_vs_baseline_bits": base_budget - cost,
            "weighted_error": cyclic.after_error,
            "weighted_nrmse": _normalized_error(cyclic.after_error, weight, element_weights),
            "same_factor_rank_control_error": control_error,
            "same_rank_error_change_fraction": cyclic.after_error / control_error - 1.0,
            "equal_bit_baseline_error_change_fraction": cyclic.after_error / base_error - 1.0,
            "fixed_factor_rank": base_rank,
            "fixed_rank_total_bits": fixed_rank_cost,
            "fixed_rank_bpw": fixed_rank_cost / source_elements,
            "fixed_rank_weighted_error": fixed_rank_cyclic.after_error,
            "fixed_rank_error_change_fraction": fixed_rank_cyclic.after_error / base_error - 1.0,
            "accepted": cyclic.accepted,
            "fit_wall_seconds": time.perf_counter() - started,
        }
    return {
        "block": block,
        "projection": projection,
        "shape": [out_features, in_features],
        "source_elements": source_elements,
        "base_rank": base_rank,
        "base_budget_bits": base_budget,
        "base_actual_bpw": base_budget / source_elements,
        "controls_by_factor_rank": {str(rank): value for rank, value in controls.items()},
        "arms": arms,
    }


def run(args: argparse.Namespace) -> int:
    protocol = CyclicScaleProtocol(
        2,
        args.model_revision,
        args.blocks,
        args.projections,
        args.target_bpw,
        args.scale_ranks,
        args.rank_alignment,
        args.scale_bits,
        args.outer_iterations,
        args.inner_iterations,
        args.regularization,
        args.penalty_schedule,
        args.convergence_check_interval,
        args.scale_fit_passes,
        args.seed,
        args.device,
        str(args.calibration_state.resolve()),
        args.calibration_shrinkage,
    )
    protocol_hash = semantic_hash(asdict(protocol))
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("protocol_hash") != protocol_hash:
            raise ValueError("existing cyclic-scale output uses another protocol")
    else:
        payload = {
            "schema_version": 1,
            "status": "running",
            "protocol_hash": protocol_hash,
            "protocol": asdict(protocol),
            "results": {},
        }
    profiles = _load_profiles(args.calibration_state, args.calibration_shrinkage)
    lease = acquire_device_lease(args.device) if args.device.startswith("cuda") else nullcontext()
    with lease, safe_open(str(args.model), framework="pt", device="cpu") as handle:
        for block in args.blocks:
            for projection in args.projections:
                key = f"{block}|{projection}"
                if key not in payload["results"]:
                    print(f"running {key}", flush=True)
                    payload["results"][key] = _evaluate_layer(
                        handle, profiles, block, projection, protocol
                    )
                    atomic_write_json(args.output, payload)
                print(json.dumps(payload["results"][key]["arms"], indent=2), flush=True)
    payload["status"] = "completed"
    atomic_write_json(args.output, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=_parse_ints, default=(0,))
    parser.add_argument("--projections", type=_parse_projections, default=("gate",))
    parser.add_argument("--scale-ranks", type=_parse_ints, default=(1, 2, 3, 5))
    parser.add_argument("--target-bpw", type=float, default=1.2)
    parser.add_argument("--rank-alignment", type=int, default=32)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--outer-iterations", type=int, default=800)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=3e-2)
    parser.add_argument("--penalty-schedule", default="cubic")
    parser.add_argument("--convergence-check-interval", type=int, default=100)
    parser.add_argument("--scale-fit-passes", type=int, default=2)
    parser.add_argument("--calibration-shrinkage", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
