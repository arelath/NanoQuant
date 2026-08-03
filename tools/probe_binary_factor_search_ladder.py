"""Measure bounded direct binary search on synthetic or real matrix crops."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from probe_tiny_factorization_optimality import exhaustive_sign_oracle
from safetensors import safe_open

from nanoquant.config.codec import semantic_hash
from nanoquant.domain.binary_factor_search import (
    BinaryFactorSearchResult,
    refine_binary_factors_separable,
)
from nanoquant.domain.calibration_math import shrink_importance
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.functional_binary_population import (
    canonical_binary_hash,
    canonical_binary_signs,
)
from nanoquant.domain.scale_fit import fit_scales, reconstruct
from nanoquant.infrastructure.io_utils import atomic_write_json


@dataclass(frozen=True, slots=True)
class LadderProtocol:
    schema_version: int
    rows: int
    columns: int
    rank: int
    seeds: int
    outer_iterations: int
    inner_iterations: int
    scale_passes: int
    search_outer_passes: int
    one_bit_passes: int
    one_bit_fraction: float
    max_one_bit_vectors: int
    codebook_passes: int
    codebook_size: int
    variable_depth_passes: int
    variable_depth_length: int
    pair_passes: int
    block_passes: int
    block_bits: int
    component_passes: int
    joint_passes: int
    joint_bits: int
    joint_candidate_refits: int
    joint_batch_size: int
    joint_screen_scale_passes: int
    exact_scale_starts: int
    exact_scale_passes: int
    exact_batch_size: int
    population_warm_starts: int
    include_oracle: bool
    synthetic_cases: int
    real_crops: int
    tensor_key: str | None
    profile_key: str | None
    calibration_shrinkage: float
    seed: int
    device: str


def _logical_seed(seed: int, key: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(f"{seed}|{key}".encode()).digest()[:8], "little") % (2**63 - 1)


def _weighted_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
) -> float:
    return float(
        (
            (prediction.float() - target.float()).square()
            * input_importance.float()[None, :]
            * output_importance.float()[:, None]
        ).sum()
    )


def _state_from_result(result: BinaryFactorSearchResult) -> tuple[torch.Tensor, ...]:
    return (
        result.left_binary,
        result.right_binary,
        result.scale_pre,
        result.scale_mid,
        result.scale_post,
    )


def _search_stage(
    name: str,
    target: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    state: tuple[torch.Tensor, ...],
    protocol: LadderProtocol,
    **settings: Any,
) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
    started = time.perf_counter()
    left, right, pre, mid, post = state
    search_settings: dict[str, Any] = {
        "outer_passes": protocol.search_outer_passes,
        "scale_passes": protocol.scale_passes,
        "hard_fraction": 1.0,
        "max_hard_vectors": max(target.shape),
    }
    search_settings.update(settings)
    result = refine_binary_factors_separable(
        target,
        left,
        right,
        pre,
        mid,
        post,
        input_importance,
        output_importance,
        **search_settings,
    )
    return (
        {
            "name": name,
            "weighted_error": result.after_error,
            "wall_seconds": time.perf_counter() - started,
            "accepted_outer_passes": result.accepted_outer_passes,
            "continuous_updates": result.continuous_updates,
            "one_bit_updates": result.one_bit_updates,
            "codebook_updates": result.codebook_updates,
            "variable_depth_updates": result.variable_depth_updates,
            "pair_updates": result.pair_updates,
            "block_updates": result.block_updates,
            "block_patterns_evaluated": result.block_patterns_evaluated,
            "component_updates": result.component_updates,
            "joint_updates": result.joint_updates,
            "joint_patterns_evaluated": result.joint_patterns_evaluated,
        },
        _state_from_result(result),
    )


def _factorization_candidates(
    target: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    protocol: LadderProtocol,
    key: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[torch.Tensor, ...]],
    list[tuple[dict[str, Any], tuple[torch.Tensor, ...]]],
]:
    records: list[dict[str, Any]] = []
    candidates: list[tuple[dict[str, Any], tuple[torch.Tensor, ...]]] = []
    best_states: dict[str, tuple[float, tuple[torch.Tensor, ...]]] = {}
    for method in ("power", "exact_svd"):
        for seed_index in range(protocol.seeds):
            started = time.perf_counter()
            result = factorize_admm_with_parameters(
                target,
                input_importance,
                output_importance,
                protocol.rank,
                torch.Generator(device=target.device).manual_seed(
                    _logical_seed(protocol.seed, f"{key}|{method}|{seed_index}")
                ),
                AdmmParameters(
                    outer_iterations=protocol.outer_iterations,
                    inner_iterations=protocol.inner_iterations,
                    regularization=3e-2,
                    penalty_schedule="cubic",
                    convergence_check_interval=100,
                    projection_method=method,
                ),
            )
            raw_error = _weighted_error(
                target, result.reconstruction, input_importance, output_importance
            )
            fitted = fit_scales(
                target,
                result.left_binary,
                result.right_binary,
                result.scale_pre,
                result.scale_mid,
                result.scale_post,
                input_importance,
                output_importance,
                alternating_passes=protocol.scale_passes,
            )
            state = (
                result.left_binary,
                result.right_binary,
                fitted.scale_pre,
                fitted.scale_mid,
                fitted.scale_post,
            )
            record = {
                "method": method,
                "seed": seed_index,
                "raw_error": raw_error,
                "scaled_error": fitted.after_error,
                "wall_seconds": time.perf_counter() - started,
                "canonical_hash": canonical_binary_hash(
                    result.left_binary, result.right_binary
                ),
            }
            records.append(record)
            candidates.append((record, state))
            current = best_states.get(method)
            if current is None or fitted.after_error < current[0]:
                best_states[method] = (fitted.after_error, state)
    return records, {key: value[1] for key, value in best_states.items()}, candidates


def _canonical_flat(state: tuple[torch.Tensor, ...]) -> torch.Tensor:
    left, right = canonical_binary_signs(state[0], state[1])
    return torch.cat((left.reshape(-1), right.reshape(-1))).gt(0)


def _diverse_warm_starts(
    candidates: list[tuple[dict[str, Any], tuple[torch.Tensor, ...]]],
    count: int,
) -> list[tuple[dict[str, Any], tuple[torch.Tensor, ...]]]:
    if count <= 0:
        return []
    unique: list[tuple[dict[str, Any], tuple[torch.Tensor, ...]]] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: float(item[0]["scaled_error"])):
        candidate_hash = str(candidate[0]["canonical_hash"])
        if candidate_hash not in seen:
            seen.add(candidate_hash)
            unique.append(candidate)
    if len(unique) <= count:
        return unique
    selected = [unique.pop(0)]
    while len(selected) < count:
        # Alternate a high-quality incumbent with the candidate farthest from
        # every selected canonical sign pair.
        if len(selected) % 2 == 1:
            selected.append(unique.pop(0))
            continue
        selected_bits = tuple(_canonical_flat(state) for _record, state in selected)
        distances = [
            min(
                int((_canonical_flat(state) != prior).sum())
                for prior in selected_bits
            )
            for _record, state in unique
        ]
        selected.append(unique.pop(max(range(len(unique)), key=distances.__getitem__)))
    return selected


def _population_polish(
    target: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    candidates: list[tuple[dict[str, Any], tuple[torch.Tensor, ...]]],
    protocol: LadderProtocol,
) -> list[dict[str, Any]]:
    records = []
    for source_record, state in _diverse_warm_starts(
        candidates, protocol.population_warm_starts
    ):
        left, right, pre, mid, post = state
        before_left, before_right = canonical_binary_signs(left, right)
        started = time.perf_counter()
        result = refine_binary_factors_separable(
            target,
            left,
            right,
            pre,
            mid,
            post,
            input_importance,
            output_importance,
            outer_passes=protocol.search_outer_passes,
            scale_passes=protocol.scale_passes,
            continuous_candidates=True,
            one_bit_passes=protocol.one_bit_passes,
            one_bit_fraction=protocol.one_bit_fraction,
            max_one_bit_vectors=protocol.max_one_bit_vectors,
            codebook_passes=protocol.codebook_passes,
            codebook_size=protocol.codebook_size,
            variable_depth_passes=protocol.variable_depth_passes,
            variable_depth_length=protocol.variable_depth_length,
            pair_passes=protocol.pair_passes,
            pair_pool_size=protocol.rank,
            block_bits=min(protocol.block_bits, protocol.rank),
            block_passes=protocol.block_passes,
            component_passes=protocol.component_passes,
            component_limit=protocol.rank,
            joint_passes=protocol.joint_passes,
            joint_bits=protocol.joint_bits,
            joint_candidate_refits=protocol.joint_candidate_refits,
            joint_batch_size=protocol.joint_batch_size,
            joint_screen_scale_passes=protocol.joint_screen_scale_passes,
            hard_fraction=1.0,
            max_hard_vectors=max(target.shape),
        )
        after_left, after_right = canonical_binary_signs(
            result.left_binary, result.right_binary
        )
        records.append(
            {
                "source_method": source_record["method"],
                "source_seed": source_record["seed"],
                "source_error": source_record["scaled_error"],
                "source_canonical_hash": source_record["canonical_hash"],
                "after_error": result.after_error,
                "gain_from_source_fraction": 1.0
                - result.after_error / max(float(source_record["scaled_error"]), 1e-30),
                "canonical_sign_distance": int(
                    (before_left != after_left).sum() + (before_right != after_right).sum()
                ),
                "result_canonical_hash": canonical_binary_hash(
                    result.left_binary, result.right_binary
                ),
                "wall_seconds": time.perf_counter() - started,
                "joint_patterns_evaluated": result.joint_patterns_evaluated,
            }
        )
    return records


def _score_case(
    name: str,
    target: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    protocol: LadderProtocol,
) -> dict[str, Any]:
    target = target.to(protocol.device).float()
    input_importance = input_importance.to(protocol.device).float()
    output_importance = output_importance.to(protocol.device).float()
    energy = float((target.square() * input_importance[None, :] * output_importance[:, None]).sum())
    factor_records, best_states, factor_candidates = _factorization_candidates(
        target, input_importance, output_importance, protocol, name
    )
    power_scaled = min(record["scaled_error"] for record in factor_records if record["method"] == "power")
    exact_scaled = min(record["scaled_error"] for record in factor_records if record["method"] == "exact_svd")
    state = best_states["power"] if power_scaled <= exact_scaled else best_states["exact_svd"]
    stages = [
        {
            "name": "selected_scaled_warm_start",
            "weighted_error": min(power_scaled, exact_scaled),
            "wall_seconds": 0.0,
        }
    ]
    stage, state = _search_stage(
        "continuous",
        target,
        input_importance,
        output_importance,
        state,
        protocol,
        continuous_candidates=True,
        one_bit_passes=0,
        pair_passes=0,
        block_bits=0,
        block_passes=0,
        component_passes=0,
        joint_passes=0,
    )
    stages.append(stage)
    population_records = _population_polish(
        target,
        input_importance,
        output_importance,
        factor_candidates,
        protocol,
    )
    population_error = (
        None
        if not population_records
        else min(float(record["after_error"]) for record in population_records)
    )
    stage, state = _search_stage(
        "one_bit",
        target,
        input_importance,
        output_importance,
        state,
        protocol,
        continuous_candidates=False,
        one_bit_passes=protocol.one_bit_passes,
        one_bit_fraction=protocol.one_bit_fraction,
        max_one_bit_vectors=protocol.max_one_bit_vectors,
        pair_passes=0,
        block_bits=0,
        block_passes=0,
        component_passes=0,
        joint_passes=0,
    )
    stages.append(stage)
    stage, state = _search_stage(
        "codebook",
        target,
        input_importance,
        output_importance,
        state,
        protocol,
        continuous_candidates=False,
        one_bit_passes=0,
        codebook_passes=protocol.codebook_passes,
        codebook_size=protocol.codebook_size,
        pair_passes=0,
        block_bits=0,
        block_passes=0,
        component_passes=0,
        joint_passes=0,
    )
    stages.append(stage)
    stage, state = _search_stage(
        "variable_depth",
        target,
        input_importance,
        output_importance,
        state,
        protocol,
        continuous_candidates=False,
        one_bit_passes=0,
        codebook_passes=0,
        variable_depth_passes=protocol.variable_depth_passes,
        variable_depth_length=protocol.variable_depth_length,
        pair_passes=0,
        block_bits=0,
        block_passes=0,
        component_passes=0,
        joint_passes=0,
    )
    stages.append(stage)
    stage, state = _search_stage(
        "pair",
        target,
        input_importance,
        output_importance,
        state,
        protocol,
        continuous_candidates=False,
        one_bit_passes=0,
        pair_passes=protocol.pair_passes,
        pair_pool_size=protocol.rank,
        block_bits=0,
        block_passes=0,
        component_passes=0,
        joint_passes=0,
    )
    stages.append(stage)
    stage, state = _search_stage(
        "block",
        target,
        input_importance,
        output_importance,
        state,
        protocol,
        continuous_candidates=False,
        one_bit_passes=0,
        pair_passes=0,
        block_bits=min(protocol.block_bits, protocol.rank),
        block_passes=protocol.block_passes,
        component_passes=0,
        joint_passes=0,
    )
    stages.append(stage)
    stage, state = _search_stage(
        "component",
        target,
        input_importance,
        output_importance,
        state,
        protocol,
        continuous_candidates=False,
        one_bit_passes=0,
        pair_passes=0,
        block_bits=0,
        block_passes=0,
        component_passes=protocol.component_passes,
        component_limit=protocol.rank,
        joint_passes=0,
    )
    stages.append(stage)
    stage, state = _search_stage(
        "joint",
        target,
        input_importance,
        output_importance,
        state,
        protocol,
        continuous_candidates=False,
        one_bit_passes=0,
        pair_passes=0,
        block_bits=0,
        block_passes=0,
        component_passes=0,
        joint_passes=protocol.joint_passes,
        joint_bits=protocol.joint_bits,
        joint_candidate_refits=protocol.joint_candidate_refits,
        joint_batch_size=protocol.joint_batch_size,
        joint_screen_scale_passes=protocol.joint_screen_scale_passes,
    )
    stages.append(stage)
    baseline = power_scaled
    oracle_error: float | None = None
    oracle_nrmse: float | None = None
    configurations: int | None = None
    oracle_wall_seconds: float | None = None
    if protocol.include_oracle:
        oracle_started = time.perf_counter()
        oracle, configurations = exhaustive_sign_oracle(
            target,
            input_importance,
            output_importance,
            starts=protocol.exact_scale_starts,
            passes=protocol.exact_scale_passes,
            seed=_logical_seed(protocol.seed, f"{name}|oracle"),
            device=protocol.device,
            batch_size=protocol.exact_batch_size,
            rank=protocol.rank,
        )
        oracle_error = float(oracle.errors[0])
        oracle_nrmse = math.sqrt(oracle_error / max(energy, 1e-30))
        oracle_wall_seconds = time.perf_counter() - oracle_started
    for stage in stages:
        stage["nrmse"] = math.sqrt(stage["weighted_error"] / max(energy, 1e-30))
        stage["gain_vs_power_fraction"] = 1.0 - stage["weighted_error"] / max(baseline, 1e-30)
        stage["gap_closed_fraction"] = (
            None
            if oracle_error is None
            else (baseline - stage["weighted_error"]) / max(baseline - oracle_error, 1e-30)
        )
    return {
        "name": name,
        "target_weighted_energy": energy,
        "factorization_candidates": factor_records,
        "best_power_scaled_error": power_scaled,
        "best_exact_svd_scaled_error": exact_scaled,
        "exact_svd_gain_vs_power_fraction": 1.0 - exact_scaled / max(power_scaled, 1e-30),
        "stages": stages,
        "population_polish": {
            "selected_warm_starts": len(population_records),
            "records": population_records,
            "best_error": population_error,
            "single_incumbent_combined_error": (
                None if not population_records else population_records[0]["after_error"]
            ),
            "gain_vs_single_incumbent_combined_fraction": (
                None
                if population_error is None or not population_records
                else 1.0
                - population_error
                / max(float(population_records[0]["after_error"]), 1e-30)
            ),
        },
        "oracle_error": oracle_error,
        "oracle_nrmse": oracle_nrmse,
        "oracle_sign_configurations": configurations,
        "oracle_wall_seconds": oracle_wall_seconds,
    }


def _synthetic_cases(
    rows: int, columns: int, rank: int, count: int, seed: int
) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    cases = []
    for index in range(count):
        geometry = f"{rows}x{columns}"
        generator = torch.Generator().manual_seed(
            _logical_seed(seed, f"synthetic|{geometry}|{index}")
        )
        cases.append(
            (
                f"gaussian-{geometry}-{index}",
                torch.randn((rows, columns), generator=generator),
                torch.ones(columns),
                torch.ones(rows),
            )
        )
        left = torch.randint(0, 2, (rows, rank), generator=generator).float().mul_(2).sub_(1)
        right = torch.randint(0, 2, (rank, columns), generator=generator).float().mul_(2).sub_(1)
        pre = torch.exp(0.35 * torch.randn(columns, generator=generator))
        mid = torch.exp(0.35 * torch.randn(rank, generator=generator))
        post = torch.exp(0.35 * torch.randn(rows, generator=generator))
        cases.append(
            (
                f"represented-{geometry}-{index}",
                reconstruct(left, right, pre, mid, post),
                torch.ones(columns),
                torch.ones(rows),
            )
        )
    return cases


def _load_profile(state: Path, key: str, shrinkage: float) -> tuple[torch.Tensor, torch.Tensor]:
    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    sample_count = int(manifest["sample_count"])
    matches = [index for index, layer in enumerate(manifest["layers"]) if layer["path"] == key]
    if len(matches) != 1:
        raise ValueError(f"calibration profile must resolve exactly once: {key}")
    index = matches[0]
    with safe_open(str(state / "state.safetensors"), framework="pt", device="cpu") as handle:
        return (
            shrink_importance(handle.get_tensor(f"layer_{index}.inputs.total").float() / sample_count, shrinkage),
            shrink_importance(handle.get_tensor(f"layer_{index}.outputs.total").float() / sample_count, shrinkage),
        )


def _real_cases(args: argparse.Namespace) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if args.real_crops == 0:
        return []
    if args.model is None or args.calibration_state is None or args.tensor_key is None or args.profile_key is None:
        raise ValueError("real crops require model, calibration state, tensor key, and profile key")
    input_importance, output_importance = _load_profile(
        args.calibration_state, args.profile_key, args.calibration_shrinkage
    )
    with safe_open(str(args.model), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(args.tensor_key).float()
    if weight.shape[0] < args.rows or weight.shape[1] < args.columns:
        raise ValueError("real tensor is smaller than the requested crop")
    cases = []
    for index in range(args.real_crops):
        geometry = f"{args.rows}x{args.columns}"
        generator = torch.Generator().manual_seed(
            _logical_seed(args.seed, f"real|{geometry}|{args.tensor_key}|{index}")
        )
        row = int(torch.randint(0, weight.shape[0] - args.rows + 1, (), generator=generator))
        column = int(
            torch.randint(0, weight.shape[1] - args.columns + 1, (), generator=generator)
        )
        cases.append(
            (
                f"real-{geometry}-{index}-r{row}-c{column}",
                weight[row : row + args.rows, column : column + args.columns],
                input_importance[column : column + args.columns],
                output_importance[row : row + args.rows],
            )
        )
    return cases


def run(args: argparse.Namespace) -> int:
    args.rows = args.size if args.rows is None else args.rows
    args.columns = args.size if args.columns is None else args.columns
    rank = min(args.rows, args.columns) if args.rank is None else args.rank
    if args.rows <= 0 or args.columns <= 0 or rank <= 0 or rank > min(args.rows, args.columns):
        raise ValueError("rank must be between one and the smaller matrix dimension")
    if (
        args.one_bit_passes < 0
        or not 0.0 <= args.one_bit_fraction <= 1.0
        or args.max_one_bit_vectors < 0
        or args.codebook_passes < 0
        or args.codebook_size < 0
        or args.variable_depth_passes < 0
        or args.variable_depth_length < 0
        or args.population_warm_starts < 0
    ):
        raise ValueError("binary search ladder settings are invalid")
    protocol = LadderProtocol(
        2,
        args.rows,
        args.columns,
        rank,
        args.seeds,
        args.outer_iterations,
        args.inner_iterations,
        args.scale_passes,
        args.search_outer_passes,
        args.one_bit_passes,
        args.one_bit_fraction,
        args.max_one_bit_vectors,
        args.codebook_passes,
        args.codebook_size,
        args.variable_depth_passes,
        args.variable_depth_length,
        args.pair_passes,
        args.block_passes,
        args.block_bits,
        args.component_passes,
        args.joint_passes,
        args.joint_bits,
        args.joint_candidate_refits,
        args.joint_batch_size,
        args.joint_screen_scale_passes,
        args.exact_scale_starts,
        args.exact_scale_passes,
        args.exact_batch_size,
        args.population_warm_starts,
        not args.skip_oracle,
        args.synthetic_cases,
        args.real_crops,
        args.tensor_key,
        args.profile_key,
        args.calibration_shrinkage,
        args.seed,
        args.device,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "protocol_hash": semantic_hash(asdict(protocol)),
        "protocol": asdict(protocol),
        "results": {},
    }
    atomic_write_json(args.output, payload)
    cases = _synthetic_cases(
        args.rows,
        args.columns,
        rank,
        args.synthetic_cases,
        args.seed,
    ) + _real_cases(args)
    for name, target, input_importance, output_importance in cases:
        print(f"running {name}", flush=True)
        result = _score_case(name, target, input_importance, output_importance, protocol)
        payload["results"][name] = result
        atomic_write_json(args.output, payload)
        last = result["stages"][-1]
        message = (
            f"power={math.sqrt(result['best_power_scaled_error'] / result['target_weighted_energy']):.8f} "
            f"direct={last['nrmse']:.8f} "
        )
        if result["oracle_nrmse"] is None:
            message += f"gain={100 * last['gain_vs_power_fraction']:.2f}%"
        else:
            message += (
                f"oracle={result['oracle_nrmse']:.8f} "
                f"gap_closed={100 * last['gap_closed_fraction']:.2f}%"
            )
        print(message, flush=True)
    payload["status"] = "completed"
    atomic_write_json(args.output, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--calibration-state", type=Path)
    parser.add_argument("--tensor-key")
    parser.add_argument("--profile-key")
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--outer-iterations", type=int, default=800)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--scale-passes", type=int, default=32)
    parser.add_argument("--search-outer-passes", type=int, default=8)
    parser.add_argument("--one-bit-passes", type=int, default=16)
    parser.add_argument("--one-bit-fraction", type=float, default=1.0)
    parser.add_argument("--max-one-bit-vectors", type=int, default=2**31 - 1)
    parser.add_argument("--codebook-passes", type=int, default=2)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--variable-depth-passes", type=int, default=2)
    parser.add_argument("--variable-depth-length", type=int, default=32)
    parser.add_argument("--pair-passes", type=int, default=4)
    parser.add_argument("--block-passes", type=int, default=4)
    parser.add_argument("--block-bits", type=int, default=10)
    parser.add_argument("--component-passes", type=int, default=4)
    parser.add_argument("--joint-passes", type=int, default=1)
    parser.add_argument("--joint-bits", type=int, default=10)
    parser.add_argument("--joint-candidate-refits", type=int, default=8)
    parser.add_argument("--joint-batch-size", type=int, default=64)
    parser.add_argument("--joint-screen-scale-passes", type=int, default=4)
    parser.add_argument("--exact-scale-starts", type=int, default=16)
    parser.add_argument("--exact-scale-passes", type=int, default=64)
    parser.add_argument("--exact-batch-size", type=int, default=65536)
    parser.add_argument(
        "--population-warm-starts",
        type=int,
        default=0,
        help="polish this many gauge-distinct ADMM/SVID starts, alternating quality and novelty",
    )
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--synthetic-cases", type=int, default=1)
    parser.add_argument("--real-crops", type=int, default=1)
    parser.add_argument("--calibration-shrinkage", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
