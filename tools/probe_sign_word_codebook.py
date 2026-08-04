"""Screen equal-bit compressed sign words on one pinned Gemma matrix.

This is an analysis-only probe.  It does not introduce a packed schema or
runtime.  Fitted-table, progressively-fixed, and relational arms are
constrained throughout ADMM and spend the saved sign bits on aligned rank.
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

from nanoquant.config.schema import BinaryFactorSearchConfig
from nanoquant.domain.binary_factor_search import (
    BinaryFactorSearchResult,
    refine_binary_factors_control_then_tabu,
)
from nanoquant.domain.calibration_math import shrink_importance
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.progressive_sign_fixing import (
    ProgressiveSignConstraint,
    factorize_progressive_sign_fixing_admm,
    maximum_progressive_rank_for_budget,
    progressive_sign_fixing_bit_cost,
)
from nanoquant.domain.relational_sign_code import (
    RelationalSignConstraint,
    factorize_relational_sign_admm,
    maximum_relational_rank_for_budget,
    relational_sign_bit_cost,
)
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.domain.sign_word_codebook import (
    CORRECTED_ASSIGNMENT_CANDIDATES,
    asymmetric_sign_word_codebook_bit_cost,
    codebook_index_metrics,
    corrected_asymmetric_codebook_bit_cost,
    factorize_sign_word_codebook_admm,
    maximum_asymmetric_codebook_rank_for_budget,
    maximum_codebook_rank_for_budget,
    maximum_corrected_asymmetric_rank_for_budget,
    mixed_right_corrected_codebook_bit_cost,
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
    candidate_rank: int | None
    right_free_rows: int
    right_codebook_banks: int
    right_codebook_bank_axis: str
    right_corrected_codebook_banks: int | None
    index_widths: tuple[int, ...]
    rank_multiple: int
    scale_bits: int
    outer_iterations: int
    inner_iterations: int
    regularization: float
    penalty_schedule: str
    convergence_check_interval: int
    codebook_update_interval: int
    codebook_warmup_fraction: float
    codebook_freeze_fraction: float
    progressive_warmup_fraction: float
    transpose_matrix: bool
    codebook_mode: str
    assignment_batch_words: int
    corrected_assignment_candidates: int
    scale_fit_passes: int
    binary_search: BinaryFactorSearchConfig
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


def _binary_search_result_metrics(result: BinaryFactorSearchResult) -> dict[str, int | float]:
    return {
        "before_error": result.before_error,
        "after_error": result.after_error,
        "accepted_outer_passes": result.accepted_outer_passes,
        "one_bit_updates": result.one_bit_updates,
        "variable_depth_updates": result.variable_depth_updates,
        "tabu_updates": result.tabu_updates,
    }


def _run_binary_search(
    weight: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    protocol: SignWordCodebookProtocol,
    *,
    right_free_rows: int | None,
) -> tuple[BinaryFactorSearchResult, dict[str, Any]]:
    search = protocol.binary_search
    if not search.enabled:
        raise ValueError("binary search was not enabled for this protocol")
    common_fit = fit_scales(
        weight,
        left_binary,
        right_binary,
        scale_pre,
        scale_mid,
        scale_post,
        input_importance,
        output_importance,
        alternating_passes=search.scale_passes,
    )
    right_mutable = None
    if right_free_rows is not None:
        right_mutable = torch.zeros(
            right_binary.shape[0],
            dtype=torch.bool,
            device=right_binary.device,
        )
        right_mutable[:right_free_rows] = True
    started = time.perf_counter()
    control, tabu = refine_binary_factors_control_then_tabu(
        weight,
        left_binary,
        right_binary,
        common_fit.scale_pre,
        common_fit.scale_mid,
        common_fit.scale_post,
        input_importance,
        output_importance,
        scale_passes=search.scale_passes,
        control_outer_passes=search.control_outer_passes,
        one_bit_passes=search.one_bit_passes,
        one_bit_fraction=search.one_bit_fraction,
        max_one_bit_vectors=search.max_one_bit_vectors,
        variable_depth_passes=search.variable_depth_passes,
        variable_depth_length=search.variable_depth_length,
        tabu_outer_passes=search.tabu_outer_passes,
        tabu_passes=search.tabu_passes,
        tabu_steps=search.tabu_steps,
        tabu_tenure=search.tabu_tenure,
        tabu_tenure_jitter=search.tabu_tenure_jitter,
        right_mutable_components=right_mutable,
    )
    if right_free_rows is not None and not torch.equal(
        tabu.right_binary[right_free_rows:],
        right_binary[right_free_rows:],
    ):
        raise RuntimeError("binary search changed codebook-constrained factor rows")
    return tabu, {
        "settings": asdict(search),
        "right_mutable_rows": (
            right_binary.shape[0] if right_free_rows is None else right_free_rows
        ),
        "right_immutable_rows": (
            0 if right_free_rows is None else right_binary.shape[0] - right_free_rows
        ),
        "common_refit_error": common_fit.after_error,
        "common_refit_metrics": _metrics(
            weight,
            common_fit.reconstruction,
            input_importance,
            output_importance,
        ),
        "control": _binary_search_result_metrics(control),
        "tabu": _binary_search_result_metrics(tabu),
        "left_sign_distance_from_admm": int(
            (tabu.left_binary != left_binary).sum()
        ),
        "right_sign_distance_from_admm": int(
            (tabu.right_binary != right_binary).sum()
        ),
        "tabu_sign_distance_from_control": int(
            (tabu.left_binary != control.left_binary).sum()
            + (tabu.right_binary != control.right_binary).sum()
        ),
        "wall_seconds": time.perf_counter() - started,
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
    pre_search_metrics = _metrics(
        weight,
        fitted.reconstruction,
        input_importance,
        output_importance,
    )
    binary_search = None
    reconstruction = fitted.reconstruction
    if protocol.binary_search.enabled:
        searched, binary_search = _run_binary_search(
            weight,
            factorized.left_binary,
            factorized.right_binary,
            fitted.scale_pre,
            fitted.scale_mid,
            fitted.scale_post,
            input_importance,
            output_importance,
            protocol,
            right_free_rows=None,
        )
        reconstruction = searched.reconstruction
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
            reconstruction,
            input_importance,
            output_importance,
        ),
        "pre_search_metrics": pre_search_metrics,
        "binary_search": binary_search,
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
    progressive = protocol.codebook_mode == "progressive"
    relational = protocol.codebook_mode == "relational"
    corrections_per_word = {
        "full-right-flip1": 1,
        "full-right-flip2": 2,
        "full-right-flip3": 3,
    }.get(protocol.codebook_mode, 0)
    correction_bits = {0: 0, 1: 5, 2: 9, 3: 13}[corrections_per_word]
    right_only = protocol.codebook_mode == "full-right" or corrections_per_word > 0
    corrected = corrections_per_word > 0
    if corrected:
        rank = maximum_corrected_asymmetric_rank_for_budget(
            weight.shape[0],
            weight.shape[1],
            target_bits,
            left_index_width=None,
            right_index_width=index_width,
            right_flip_bits=correction_bits,
            rank_multiple=protocol.rank_multiple,
            scale_width=protocol.scale_bits,
        )
    elif right_only:
        rank = maximum_asymmetric_codebook_rank_for_budget(
            weight.shape[0],
            weight.shape[1],
            target_bits,
            left_index_width=None,
            right_index_width=index_width,
            rank_multiple=protocol.rank_multiple,
            scale_width=protocol.scale_bits,
        )
    elif relational:
        rank = maximum_relational_rank_for_budget(
            weight.shape[0],
            weight.shape[1],
            target_bits,
            variable_bits_per_word=index_width,
            rank_multiple=protocol.rank_multiple,
            scale_width=protocol.scale_bits,
        )
    elif progressive:
        rank = maximum_progressive_rank_for_budget(
            weight.shape[0],
            weight.shape[1],
            target_bits,
            variable_bits_per_word=index_width,
            rank_multiple=protocol.rank_multiple,
            scale_width=protocol.scale_bits,
        )
    else:
        rank = maximum_codebook_rank_for_budget(
            weight.shape[0],
            weight.shape[1],
            target_bits,
            index_width=index_width,
            rank_multiple=protocol.rank_multiple,
            scale_width=protocol.scale_bits,
        )
    if protocol.candidate_rank is not None:
        rank = protocol.candidate_rank
    generator = torch.Generator(device=protocol.device).manual_seed(
        _logical_seed(
            protocol.seed,
            f"{protocol.codebook_mode}-{index_width}-rank-{rank}",
        )
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(protocol.device)
        torch.cuda.synchronize(protocol.device)
    started = time.perf_counter()
    bit_cost: Any
    if progressive:
        progressive_result = factorize_progressive_sign_fixing_admm(
            weight,
            input_importance,
            output_importance,
            rank,
            generator,
            variable_bits_per_word=index_width,
            outer_iterations=protocol.outer_iterations,
            inner_iterations=protocol.inner_iterations,
            regularization=protocol.regularization,
            penalty_schedule=protocol.penalty_schedule,
            convergence_check_interval=protocol.convergence_check_interval,
            fixing_warmup_fraction=protocol.progressive_warmup_fraction,
            fixing_fraction=protocol.codebook_freeze_fraction,
        )
        factors = progressive_result.factors
        representation_metrics: dict[str, Any] = {
            "constraint_metrics": {
                "left": _constraint_metrics(progressive_result.left_constraint),
                "right": _constraint_metrics(progressive_result.right_constraint),
            }
        }
        bit_cost = progressive_sign_fixing_bit_cost(
            weight.shape[0],
            weight.shape[1],
            rank,
            variable_bits_per_word=index_width,
            scale_width=protocol.scale_bits,
        )
        arm_name = f"progressive_k{index_width}"
    elif relational:
        relational_result = factorize_relational_sign_admm(
            weight,
            input_importance,
            output_importance,
            rank,
            generator,
            variable_bits_per_word=index_width,
            outer_iterations=protocol.outer_iterations,
            inner_iterations=protocol.inner_iterations,
            regularization=protocol.regularization,
            penalty_schedule=protocol.penalty_schedule,
            convergence_check_interval=protocol.convergence_check_interval,
            relation_warmup_fraction=protocol.progressive_warmup_fraction,
            relation_freeze_fraction=protocol.codebook_freeze_fraction,
        )
        factors = relational_result.factors
        representation_metrics = {
            "relation_metrics": {
                "left": _relation_metrics(relational_result.left_constraint),
                "right": _relation_metrics(relational_result.right_constraint),
            }
        }
        bit_cost = relational_sign_bit_cost(
            weight.shape[0],
            weight.shape[1],
            rank,
            variable_bits_per_word=index_width,
            scale_width=protocol.scale_bits,
        )
        arm_name = f"relational_k{index_width}"
    else:
        codebook_result = factorize_sign_word_codebook_admm(
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
            codebook_warmup_fraction=protocol.codebook_warmup_fraction,
            codebook_freeze_fraction=protocol.codebook_freeze_fraction,
            assignment_batch_words=protocol.assignment_batch_words,
            corrected_assignment_candidates=(
                protocol.corrected_assignment_candidates
            ),
            codebook_mode="full" if right_only else protocol.codebook_mode,
            constrain_left=not right_only,
            right_flips_per_word=corrections_per_word,
            right_free_rows=protocol.right_free_rows,
            right_codebook_banks=protocol.right_codebook_banks,
            right_codebook_bank_axis=protocol.right_codebook_bank_axis,
            right_corrected_codebook_banks=(
                protocol.right_corrected_codebook_banks
            ),
        )
        factors = codebook_result.factors
        representation_metrics = {
            "index_metrics": codebook_index_metrics(codebook_result)
        }
        if corrected and codebook_result.right_flip_positions is not None:
            counts = torch.bincount(
                codebook_result.right_flip_positions.reshape(-1).to(torch.int64),
                minlength=32,
            ).float()
            probabilities = counts / counts.sum().clamp_min(1)
            representation_metrics["correction_metrics"] = {
                "right_word_count": int(
                    codebook_result.right_flip_positions.shape[0]
                    * codebook_result.right_flip_positions.shape[1]
                ),
                "corrections_per_word": corrections_per_word,
                "assignment_candidate_count": (
                    protocol.corrected_assignment_candidates
                ),
                "position_entropy_bits": float(
                    -(probabilities[probabilities > 0]
                      * probabilities[probabilities > 0].log2()).sum()
                ),
                "maximum_position_frequency": float(probabilities.max()),
            }
        if corrected:
            corrected_rows = (
                None
                if protocol.right_corrected_codebook_banks is None
                else math.ceil(
                    (rank - protocol.right_free_rows)
                    * protocol.right_corrected_codebook_banks
                    / protocol.right_codebook_banks
                )
            )
            bit_cost = (
                mixed_right_corrected_codebook_bit_cost(
                    weight.shape[0],
                    weight.shape[1],
                    rank,
                    right_free_rows=protocol.right_free_rows,
                    right_index_width=index_width,
                    right_flip_bits=correction_bits,
                    scale_width=protocol.scale_bits,
                    right_codebook_count=protocol.right_codebook_banks,
                    right_corrected_rows=corrected_rows,
                )
                if protocol.right_free_rows
                else corrected_asymmetric_codebook_bit_cost(
                    weight.shape[0],
                    weight.shape[1],
                    rank,
                    left_index_width=None,
                    right_index_width=index_width,
                    right_flip_bits=correction_bits,
                    scale_width=protocol.scale_bits,
                    right_codebook_count=protocol.right_codebook_banks,
                )
            )
            arm_name = (
                f"right_codebook_flip{corrections_per_word}_k{index_width}"
            )
        elif right_only:
            bit_cost = asymmetric_sign_word_codebook_bit_cost(
                weight.shape[0],
                weight.shape[1],
                rank,
                left_index_width=None,
                right_index_width=index_width,
                scale_width=protocol.scale_bits,
            )
            arm_name = f"right_codebook_k{index_width}"
        else:
            bit_cost = sign_word_codebook_bit_cost(
                weight.shape[0],
                weight.shape[1],
                rank,
                index_width=index_width,
                scale_width=protocol.scale_bits,
            )
            arm_name = f"codebook_k{index_width}"
    fitted = fit_scales(
        weight,
        factors.left_binary,
        factors.right_binary,
        factors.scale_pre,
        factors.scale_mid,
        factors.scale_post,
        input_importance,
        output_importance,
        alternating_passes=protocol.scale_fit_passes,
    )
    pre_search_metrics = _metrics(
        weight,
        fitted.reconstruction,
        input_importance,
        output_importance,
    )
    binary_search = None
    reconstruction = fitted.reconstruction
    if protocol.binary_search.enabled:
        searched, binary_search = _run_binary_search(
            weight,
            factors.left_binary,
            factors.right_binary,
            fitted.scale_pre,
            fitted.scale_mid,
            fitted.scale_post,
            input_importance,
            output_importance,
            protocol,
            right_free_rows=protocol.right_free_rows,
        )
        reconstruction = searched.reconstruction
    if protocol.device.startswith("cuda"):
        torch.cuda.synchronize(protocol.device)
    return {
        "arm": arm_name,
        "rank": rank,
        "right_free_rows": protocol.right_free_rows,
        "rank_multiple_vs_baseline": rank / protocol.baseline_rank,
        "bit_cost": asdict(bit_cost),
        "total_bits": bit_cost.total,
        "unused_budget_bits": target_bits - bit_cost.total,
        "actual_bpw": bit_cost.total / weight.numel(),
        "signed_contributions_per_weight": rank,
        "factorized_work_over_dense": rank * sum(weight.shape) / weight.numel(),
        "metrics": _metrics(
            weight,
            reconstruction,
            input_importance,
            output_importance,
        ),
        "pre_search_metrics": pre_search_metrics,
        "binary_search": binary_search,
        **representation_metrics,
        "scale_fit_accepted": fitted.accepted,
        "scale_fit_rollback_reason": fitted.rollback_reason,
        "trace": _trace(factors.trace),
        "wall_seconds": time.perf_counter() - started,
        "peak_device_bytes": (
            int(torch.cuda.max_memory_allocated(protocol.device))
            if protocol.device.startswith("cuda")
            else 0
        ),
    }


def _constraint_metrics(constraint: ProgressiveSignConstraint) -> dict[str, Any]:
    majorities = [decision.majority_fraction for decision in constraint.decisions]
    return {
        "fixed_count": constraint.fixed_count,
        "fixed_positions": [
            decision.position for decision in constraint.decisions
        ],
        "fixed_values": [
            decision.value for decision in constraint.decisions
        ],
        "decisions": [asdict(decision) for decision in constraint.decisions],
        "minimum_majority_fraction": min(majorities) if majorities else 0.0,
        "mean_majority_fraction": (
            sum(majorities) / len(majorities) if majorities else 0.0
        ),
        "maximum_majority_fraction": max(majorities) if majorities else 0.0,
    }


def _relation_metrics(constraint: RelationalSignConstraint) -> dict[str, Any]:
    agreements = [
        decision.agreement_fraction for decision in constraint.decisions
    ]
    return {
        "root_count": constraint.root_count,
        "root_indices": constraint.root_indices.tolist(),
        "parities": constraint.parities.to(torch.int64).tolist(),
        "decisions": [asdict(decision) for decision in constraint.decisions],
        "minimum_agreement_fraction": min(agreements) if agreements else 0.0,
        "mean_agreement_fraction": (
            sum(agreements) / len(agreements) if agreements else 0.0
        ),
        "maximum_agreement_fraction": max(agreements) if agreements else 0.0,
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
    product_mode = not args.codebook_mode.startswith("full")
    if args.baseline_rank <= 0 or any(
        width <= 0 or (product_mode and width % 2)
        for width in args.index_widths
    ):
        raise ValueError(
            "baseline rank must be positive and index widths valid for the codebook"
        )
    if args.candidate_rank is not None and (
        args.candidate_rank <= 0
        or args.candidate_rank % args.rank_multiple
    ):
        raise ValueError("candidate rank must be positive and rank-aligned")
    if args.right_free_rows < 0 or args.right_free_rows % args.rank_multiple:
        raise ValueError("right free rows must be non-negative and rank-aligned")
    if args.right_free_rows and (
        args.candidate_rank is None
        or args.right_free_rows >= args.candidate_rank
        or not args.codebook_mode.startswith("full-right-flip")
    ):
        raise ValueError(
            "right free rows require a larger explicit corrected-code rank"
        )
    if args.binary_search and (
        args.right_free_rows <= 0
        or not args.codebook_mode.startswith("full-right-flip")
    ):
        raise ValueError(
            "representation-preserving binary search requires a corrected mixed-right codebook"
        )
    if (
        args.right_codebook_banks <= 0
        or args.right_codebook_banks & (args.right_codebook_banks - 1)
        or (
            args.right_codebook_banks != 1
            and not args.codebook_mode.startswith("full")
        )
    ):
        raise ValueError(
            "right codebook banks must be a positive power of two in full mode"
        )
    if args.right_corrected_codebook_banks is not None and (
        args.right_corrected_codebook_banks <= 0
        or args.right_corrected_codebook_banks > args.right_codebook_banks
        or args.right_free_rows == 0
        or (
            args.right_corrected_codebook_banks
            != args.right_codebook_banks
            and args.right_codebook_bank_axis != "row"
        )
    ):
        raise ValueError(
            "partial corrected banks require a valid row-banked prefix"
        )
    protocol = SignWordCodebookProtocol(
        20,
        args.model_revision,
        args.block,
        args.projection,
        args.baseline_rank,
        args.candidate_rank,
        args.right_free_rows,
        args.right_codebook_banks,
        args.right_codebook_bank_axis,
        args.right_corrected_codebook_banks,
        args.index_widths,
        args.rank_multiple,
        args.scale_bits,
        args.outer_iterations,
        args.inner_iterations,
        args.regularization,
        args.penalty_schedule,
        args.convergence_check_interval,
        args.codebook_update_interval,
        args.codebook_warmup_fraction,
        args.codebook_freeze_fraction,
        args.progressive_warmup_fraction,
        args.transpose_matrix,
        args.codebook_mode,
        args.assignment_batch_words,
        args.corrected_assignment_candidates,
        args.scale_fit_passes,
        BinaryFactorSearchConfig(enabled=args.binary_search),
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
        if args.transpose_matrix:
            weight = weight.mT.contiguous()
            input_importance, output_importance = (
                output_importance,
                input_importance,
            )
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
                "transposed_from_source": args.transpose_matrix,
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
        correction_count = {
            "full-right-flip1": 1,
            "full-right-flip2": 2,
            "full-right-flip3": 3,
        }.get(protocol.codebook_mode, 0)
        arm_prefix = {
            "progressive": "progressive",
            "relational": "relational",
            "full-right": "right_codebook",
        }.get(
            protocol.codebook_mode,
            (
                f"right_codebook_flip{correction_count}"
                if correction_count
                else "codebook"
            ),
        )
        for width in args.index_widths:
            key = f"{arm_prefix}_k{width}"
            result = output["results"].get(key)
            if result is None:
                if correction_count:
                    rank = maximum_corrected_asymmetric_rank_for_budget(
                        weight.shape[0],
                        weight.shape[1],
                        target_bits,
                        left_index_width=None,
                        right_index_width=width,
                        right_flip_bits={1: 5, 2: 9, 3: 13}[
                            correction_count
                        ],
                        rank_multiple=protocol.rank_multiple,
                        scale_width=protocol.scale_bits,
                    )
                elif protocol.codebook_mode == "full-right":
                    rank = maximum_asymmetric_codebook_rank_for_budget(
                        weight.shape[0],
                        weight.shape[1],
                        target_bits,
                        left_index_width=None,
                        right_index_width=width,
                        rank_multiple=protocol.rank_multiple,
                        scale_width=protocol.scale_bits,
                    )
                elif protocol.codebook_mode == "relational":
                    rank = maximum_relational_rank_for_budget(
                        weight.shape[0],
                        weight.shape[1],
                        target_bits,
                        variable_bits_per_word=width,
                        rank_multiple=protocol.rank_multiple,
                        scale_width=protocol.scale_bits,
                    )
                elif protocol.codebook_mode == "progressive":
                    rank = maximum_progressive_rank_for_budget(
                        weight.shape[0],
                        weight.shape[1],
                        target_bits,
                        variable_bits_per_word=width,
                        rank_multiple=protocol.rank_multiple,
                        scale_width=protocol.scale_bits,
                    )
                else:
                    rank = maximum_codebook_rank_for_budget(
                        weight.shape[0],
                        weight.shape[1],
                        target_bits,
                        index_width=width,
                        rank_multiple=protocol.rank_multiple,
                        scale_width=protocol.scale_bits,
                    )
                if protocol.candidate_rank is not None:
                    rank = protocol.candidate_rank
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
                for key in (
                    f"{arm_prefix}_k{width}" for width in args.index_widths
                )
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
    parser.add_argument("--candidate-rank", type=int)
    parser.add_argument("--right-free-rows", type=int, default=0)
    parser.add_argument("--right-codebook-banks", type=int, default=1)
    parser.add_argument(
        "--right-codebook-bank-axis",
        choices=("word", "row"),
        default="word",
    )
    parser.add_argument("--right-corrected-codebook-banks", type=int)
    parser.add_argument("--index-widths", type=_parse_ints, default=(8, 12))
    parser.add_argument("--rank-multiple", type=int, default=32)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--outer-iterations", type=int, default=400)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=3e-2)
    parser.add_argument("--penalty-schedule", default="cubic")
    parser.add_argument("--convergence-check-interval", type=int, default=100)
    parser.add_argument("--codebook-update-interval", type=int, default=10)
    parser.add_argument("--codebook-warmup-fraction", type=float, default=0.0)
    parser.add_argument("--codebook-freeze-fraction", type=float, default=0.5)
    parser.add_argument("--progressive-warmup-fraction", type=float, default=0.25)
    parser.add_argument("--transpose-matrix", action="store_true")
    parser.add_argument(
        "--codebook-mode",
        choices=(
            "product",
            "full",
            "full-right",
            "full-right-flip1",
            "full-right-flip2",
            "full-right-flip3",
            "progressive",
            "relational",
        ),
        default="product",
    )
    parser.add_argument("--assignment-batch-words", type=int, default=65_536)
    parser.add_argument(
        "--corrected-assignment-candidates",
        type=int,
        default=CORRECTED_ASSIGNMENT_CANDIDATES,
    )
    parser.add_argument("--scale-fit-passes", type=int, default=2)
    parser.add_argument(
        "--binary-search",
        action="store_true",
        help=(
            "apply the retained control-then-tabu search to the free-word baseline "
            "and only the representation-free signs of the mixed codebook arm"
        ),
    )
    parser.add_argument("--calibration-shrinkage", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
