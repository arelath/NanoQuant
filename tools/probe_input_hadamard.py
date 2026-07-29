"""Test input-only randomized Hadamard rotation after covariance headroom.

This analysis-only probe compares the existing diagonal binary factorization
with a runtime-compatible input transform.  A candidate applies a deterministic
sign/permutation/block-Hadamard transform to each projection input, factorizes
the correspondingly rotated weight with the diagonal of the rotated fit
covariance, and maps the dense reconstruction back only for evaluation.  The
shipped representation would instead retain the rotated binary factors and
apply the structured transform to activations.

Blocks 0, 12, and 24 use complete fused-QKV block inventories.  QKV, O, gate,
and up are transformed; down is held bit- and tensor-identical between arms.
Fit/held-out covariance rows and functional WikiText sequences are disjoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_covariance_headroom import (
    _capture_covariances,
    _group_importance,
    _input_capture_specs,
    _materialize_group_weight,
)
from probe_factor_grouping import (
    PINNED_MODEL_REVISION,
    PROJECTION_PATHS,
    GroupSpec,
    MemberSpec,
    ProbeProtocol,
    _logical_seed,
    _planned_group_rank,
    group_shape,
    load_calibration_profiles,
)
from probe_importance_shrinkage import (
    _capture_outputs,
    _dtype,
    _isolated_block_outputs,
    _paired_summary,
    _parse_ints,
)
from safetensors import safe_open

from nanoquant.config.codec import to_dict
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.metrics import dense_hessian_squared_error, weighted_squared_error
from nanoquant.domain.models import BitCost, BlockId, LayerId
from nanoquant.domain.objectives import regularize_covariance
from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.kl_splice import (
    DenseKlSpliceEvaluator,
    SpliceReconstruction,
    SpliceReconstructionSet,
)
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens

MODEL_SOURCE = "google/gemma-3-1b-it"
BASELINE_KEY = "plain-diagonal"
TRANSFORMED_GROUPS = frozenset({"qkv", "o", "gate", "up"})


def _parse_seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("transform seeds must be unique non-negative integers")
    return result


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _block_fwht(value: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply an orthonormal Walsh-Hadamard transform within column blocks."""

    if value.ndim < 1 or value.shape[-1] % block_size or not _is_power_of_two(block_size):
        raise ValueError("Hadamard block size must be a power of two dividing the last dimension")
    original_shape = value.shape
    result = value.float().reshape(-1, original_shape[-1] // block_size, block_size)
    width = 1
    while width < block_size:
        reshaped = result.reshape(*result.shape[:-1], -1, 2, width)
        first = reshaped[..., 0, :]
        second = reshaped[..., 1, :]
        result = torch.cat((first + second, first - second), dim=-1).reshape(
            *result.shape[:-1],
            block_size,
        )
        width *= 2
    return result.reshape(original_shape) / math.sqrt(block_size)


@dataclass(frozen=True, slots=True)
class StructuredHadamard:
    """A deterministic sign, permutation, and block-Hadamard input transform."""

    width: int
    block_size: int
    signs: torch.Tensor
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor

    def _metadata(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.signs.to(device=device),
            self.permutation.to(device=device),
            self.inverse_permutation.to(device=device),
        )

    def apply_right(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.width:
            raise ValueError("Hadamard transform width differs from the tensor")
        signs, permutation, _inverse = self._metadata(value.device)
        permuted = (value.float() * signs).index_select(-1, permutation)
        return _block_fwht(permuted, self.block_size)

    def inverse_right(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.width:
            raise ValueError("Hadamard transform width differs from the tensor")
        signs, _permutation, inverse = self._metadata(value.device)
        unpermuted = _block_fwht(value, self.block_size).index_select(-1, inverse)
        return unpermuted * signs

    def matrix(self, *, device: str | torch.device = "cpu") -> torch.Tensor:
        identity = torch.eye(self.width, dtype=torch.float32, device=device)
        return self.apply_right(identity)


def make_structured_hadamard(
    width: int,
    block_size: int,
    seed: int,
) -> StructuredHadamard:
    if width <= 0 or width % block_size or not _is_power_of_two(block_size):
        raise ValueError("Hadamard block size must be a power of two dividing the width")
    if seed < 0:
        raise ValueError("Hadamard seed must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int64)
    signs = signs.mul(2).sub(1).float()
    permutation = torch.randperm(width, generator=generator)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(width, dtype=permutation.dtype)
    return StructuredHadamard(width, block_size, signs, permutation, inverse)


def rotated_covariance_diagonal(
    covariance: torch.Tensor,
    transform: StructuredHadamard,
) -> torch.Tensor:
    if covariance.shape != (transform.width, transform.width):
        raise ValueError("covariance dimensions differ from the Hadamard transform")
    matrix = transform.matrix(device=covariance.device)
    return ((covariance.float() @ matrix) * matrix).sum(dim=0)


def _transform_seed(seed: int, block: int, role: str) -> int:
    digest = hashlib.sha256(f"{seed}|{block}|{role}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def block_groups(block: int) -> tuple[GroupSpec, ...]:
    members = {name: MemberSpec(block, name) for name in PROJECTION_PATHS}
    return (
        GroupSpec("qkv", tuple(members[name] for name in ("q", "k", "v"))),
        GroupSpec("o", (members["o"],)),
        GroupSpec("gate", (members["gate"],)),
        GroupSpec("up", (members["up"],)),
        GroupSpec("down", (members["down"],)),
    )


def _covariance_key(block: int, group: str) -> str:
    return f"{block}:{'mlp' if group in {'gate', 'up'} else group}"


def _transform_role(group: str) -> str:
    return "mlp" if group in {"gate", "up"} else group


def _member_reconstructions(
    group: GroupSpec,
    prediction: torch.Tensor,
    member_rows: tuple[int, ...],
) -> tuple[tuple[MemberSpec, torch.Tensor, float], ...]:
    if len(group.members) != len(member_rows) or sum(member_rows) != prediction.shape[0]:
        raise ValueError("member rows do not cover the reconstructed group")
    result = []
    offset = 0
    for member, rows in zip(group.members, member_rows, strict=True):
        value = prediction[offset : offset + rows]
        if member.transpose:
            value = value.mT
        energy = float(value.float().square().sum())
        result.append((member, value.detach().cpu().contiguous(), energy))
        offset += rows
    if offset != prediction.shape[0]:
        raise ValueError("member rows do not cover the reconstructed group")
    return tuple(result)


def _factorize_group(
    target: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    rank: int,
    *,
    logical_seed: int,
    protocol: ProbeProtocol,
) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device=protocol.device).manual_seed(logical_seed)
    parameters = AdmmParameters(
        outer_iterations=protocol.outer_iterations,
        inner_iterations=protocol.inner_iterations,
        regularization=protocol.regularization,
        penalty_schedule=protocol.penalty_schedule,
        convergence_check_interval=protocol.convergence_check_interval,
        transpose_wide=protocol.transpose_wide,
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(protocol.device)
        torch.cuda.synchronize(protocol.device)
    started = time.perf_counter()
    factorized = factorize_admm_with_parameters(
        target,
        input_importance,
        output_importance,
        rank,
        generator,
        parameters,
    )
    fitted = fit_scales(
        target,
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
    wall_seconds = time.perf_counter() - started
    factor_error = float(
        weighted_squared_error(
            target,
            fitted.reconstruction,
            input_importance,
            output_importance,
        )
    )
    factor_target = float(
        weighted_squared_error(
            target,
            torch.zeros_like(target),
            input_importance,
            output_importance,
        )
    )
    metadata = {
        "iterations_completed": factorized.iterations_completed,
        "scale_fit_accepted": fitted.accepted,
        "scale_fit_rollback_reason": fitted.rollback_reason,
        "factor_objective_error": factor_error,
        "factor_objective_target": factor_target,
        "factor_objective_normalized_rmse": math.sqrt(
            factor_error / max(factor_target, 1e-30)
        ),
        "wall_seconds": wall_seconds,
        "peak_device_bytes": (
            int(torch.cuda.max_memory_allocated(protocol.device))
            if protocol.device.startswith("cuda")
            else 0
        ),
    }
    prediction = fitted.reconstruction.detach()
    del factorized, fitted
    return prediction, metadata


def _evaluate_prediction(
    target: torch.Tensor,
    prediction: torch.Tensor,
    output_importance: torch.Tensor,
    fit_covariance: torch.Tensor | None,
    held_out_covariance: torch.Tensor | None,
    diagonal_input: torch.Tensor,
) -> dict[str, Any]:
    raw_error = float((target.float() - prediction.float()).square().sum())
    raw_target = float(target.float().square().sum())
    result: dict[str, Any] = {
        "original_error": raw_error,
        "original_target": raw_target,
        "original_normalized_rmse": math.sqrt(raw_error / max(raw_target, 1e-30)),
    }
    if fit_covariance is None or held_out_covariance is None:
        error = float(
            weighted_squared_error(
                target,
                prediction,
                diagonal_input,
                output_importance,
            )
        )
        norm = float(
            weighted_squared_error(
                target,
                torch.zeros_like(target),
                diagonal_input,
                output_importance,
            )
        )
        result["diagonal_weighted_error"] = error
        result["diagonal_weighted_target"] = norm
        result["diagonal_weighted_normalized_rmse"] = math.sqrt(
            error / max(norm, 1e-30)
        )
        return result
    fit_error = float(
        dense_hessian_squared_error(
            target,
            prediction,
            fit_covariance,
            output_importance,
        )
    )
    fit_target = float(
        dense_hessian_squared_error(
            target,
            torch.zeros_like(target),
            fit_covariance,
            output_importance,
        )
    )
    held_error = float(
        dense_hessian_squared_error(
            target,
            prediction,
            held_out_covariance,
            output_importance,
        )
    )
    held_target = float(
        dense_hessian_squared_error(
            target,
            torch.zeros_like(target),
            held_out_covariance,
            output_importance,
        )
    )
    result["fit_covariance_error"] = fit_error
    result["fit_covariance_target"] = fit_target
    result["fit_covariance_normalized_rmse"] = math.sqrt(
        fit_error / max(fit_target, 1e-30)
    )
    result["held_out_covariance_error"] = held_error
    result["held_out_covariance_target"] = held_target
    result["held_out_covariance_normalized_rmse"] = math.sqrt(
        held_error / max(held_target, 1e-30)
    )
    return result


def _group_result(
    handle: Any,
    group: GroupSpec,
    protocol: ProbeProtocol,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]],
    fit_covariance: torch.Tensor | None,
    held_out_covariance: torch.Tensor | None,
    *,
    transform: StructuredHadamard | None,
    damp_fraction: float,
) -> tuple[dict[str, Any], tuple[tuple[MemberSpec, torch.Tensor, float], ...]]:
    raw_target = _materialize_group_weight(handle, group).to(
        device=protocol.device,
        dtype=torch.bfloat16,
    )
    raw_input, output_importance = _group_importance(group, profiles)
    raw_input = raw_input.to(protocol.device).float()
    output_importance = output_importance.to(protocol.device).float()
    regularized = (
        None
        if fit_covariance is None
        else regularize_covariance(
            fit_covariance.to(protocol.device),
            damp_fraction=damp_fraction,
        )
    )
    diagonal = raw_input if regularized is None else regularized.diagonal().clone()
    if transform is None:
        factor_target = raw_target
        factor_input = diagonal
    else:
        factor_target = transform.apply_right(raw_target).to(raw_target.dtype)
        if regularized is None:
            raise ValueError("Hadamard candidates require a captured fit covariance")
        factor_input = rotated_covariance_diagonal(regularized, transform).clamp_min(1e-12)
    out_features, in_features, source_elements = group_shape(handle, group)
    member_rows = tuple(
        int(handle.get_slice(member.tensor_name).get_shape()[1 if member.transpose else 0])
        for member in group.members
    )
    rank, extra_scale_bits = _planned_group_rank(handle, group, protocol, profiles)
    cost = factor_bit_cost(
        out_features,
        in_features,
        rank,
        scale_bits=protocol.scale_bits,
        rank_alignment=protocol.rank_alignment,
    ) + BitCost(scale_bits=extra_scale_bits)
    prediction, factor_metadata = _factorize_group(
        factor_target,
        factor_input,
        output_importance,
        rank,
        logical_seed=_logical_seed(protocol.seed, f"{group.members[0].block}:{group.label}"),
        protocol=protocol,
    )
    effective = prediction.float() if transform is None else transform.inverse_right(prediction)
    evaluation = _evaluate_prediction(
        raw_target,
        effective,
        output_importance,
        None if fit_covariance is None else fit_covariance.to(protocol.device),
        None if held_out_covariance is None else held_out_covariance.to(protocol.device),
        diagonal,
    )
    result = {
        "block": group.members[0].block,
        "group": group.label,
        "members": [member.label for member in group.members],
        "shape": [out_features, in_features],
        "rank": rank,
        "source_elements": source_elements,
        "target_bits": math.floor(source_elements * protocol.target_bpw),
        "bit_cost": asdict(cost),
        "actual_bpw": cost.total / source_elements,
        "transformed": transform is not None,
        "factorization": factor_metadata,
        "evaluation": evaluation,
    }
    members = _member_reconstructions(group, effective, member_rows)
    del raw_target, factor_target, prediction, effective, output_importance, diagonal
    if protocol.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, members


def _build_reconstruction_set(
    group_results: dict[str, dict[str, Any]],
    member_results: dict[str, tuple[tuple[MemberSpec, torch.Tensor, float], ...]],
) -> SpliceReconstructionSet:
    reconstructions = []
    unit_members = []
    unit_errors = []
    for key in sorted(group_results):
        layers = []
        group_members = member_results[key]
        group_error = float(group_results[key]["evaluation"]["original_error"])
        group_target = float(group_results[key]["evaluation"]["original_target"])
        for member, weight, _energy in group_members:
            layer = LayerId(BlockId(member.block), PROJECTION_PATHS[member.projection])
            layers.append(layer)
            reconstructions.append(
                SpliceReconstruction(
                    layer,
                    weight,
                    None,
                    group_error / max(group_target, 1e-30),
                )
            )
        unit_members.append((key, tuple(layers)))
        unit_errors.append((key, group_error / max(group_target, 1e-30)))
    if len(reconstructions) != 21 or len({item.layer for item in reconstructions}) != 21:
        raise ValueError("Hadamard reconstruction inventory must contain three complete blocks")
    return SpliceReconstructionSet(
        tuple(reconstructions),
        tuple(unit_members),
        tuple(unit_errors),
    )


def _aggregate_groups(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = tuple(groups.values())
    supported = tuple(value for value in values if value["group"] in TRANSFORMED_GROUPS)
    source_elements = sum(int(value["source_elements"]) for value in values)
    actual_bits = sum(
        sum(int(part) for part in cast(dict[str, int], value["bit_cost"]).values())
        for value in values
    )
    original_error = math.fsum(
        float(value["evaluation"]["original_error"]) for value in values
    )
    original_target = math.fsum(
        float(value["evaluation"]["original_target"]) for value in values
    )
    held_error = math.fsum(
        float(value["evaluation"]["held_out_covariance_error"]) for value in supported
    )
    held_target = math.fsum(
        float(value["evaluation"]["held_out_covariance_target"]) for value in supported
    )
    fit_error = math.fsum(
        float(value["evaluation"]["fit_covariance_error"]) for value in supported
    )
    fit_target = math.fsum(
        float(value["evaluation"]["fit_covariance_target"]) for value in supported
    )
    return {
        "group_count": len(values),
        "covariance_group_count": len(supported),
        "source_elements": source_elements,
        "actual_bits": actual_bits,
        "actual_bpw": actual_bits / source_elements,
        "original_normalized_rmse": math.sqrt(
            original_error / max(original_target, 1e-30)
        ),
        "fit_covariance_error": fit_error,
        "fit_covariance_target": fit_target,
        "fit_covariance_normalized_rmse": math.sqrt(
            fit_error / max(fit_target, 1e-30)
        ),
        "held_out_covariance_error": held_error,
        "held_out_covariance_target": held_target,
        "held_out_covariance_normalized_rmse": math.sqrt(
            held_error / max(held_target, 1e-30)
        ),
        "factorization_wall_seconds": math.fsum(
            float(value["factorization"]["wall_seconds"]) for value in values
        ),
        "peak_device_bytes": max(
            int(value["factorization"]["peak_device_bytes"]) for value in values
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--blocks", type=_parse_ints, default=(0, 12, 24))
    parser.add_argument("--transform-seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--hadamard-block-size", type=int, default=128)
    parser.add_argument("--fit-tokens", type=int, default=2048)
    parser.add_argument("--held-out-tokens", type=int, default=2048)
    parser.add_argument("--wikitext-samples", type=int, default=12)
    parser.add_argument("--block-output-samples", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--target-bpw", type=float, default=1.0)
    parser.add_argument("--rank-alignment", type=int, default=1)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--outer-iterations", type=int, default=400)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=3e-2)
    parser.add_argument("--penalty-schedule", default="cubic")
    parser.add_argument("--convergence-check-interval", type=int, default=100)
    parser.add_argument("--scale-fit-passes", type=int, default=2)
    parser.add_argument("--damp-fraction", type=float, default=0.01)
    parser.add_argument("--covariance-promotion-threshold", type=float, default=0.10)
    parser.add_argument("--minimum-relative-kl-gain", type=float, default=0.05)
    parser.add_argument("--minimum-promoting-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.blocks != (0, 12, 24):
        raise ValueError("the first Hadamard screen requires blocks 0, 12, and 24")
    if (
        args.fit_tokens <= 0
        or args.held_out_tokens <= 0
        or args.wikitext_samples <= 0
        or args.block_output_samples <= 0
        or args.sequence_length < 2
    ):
        raise ValueError("Hadamard probe dataset dimensions must be positive")
    if not _is_power_of_two(args.hadamard_block_size):
        raise ValueError("Hadamard block size must be a power of two")
    if args.damp_fraction < 0 or not 0 <= args.covariance_promotion_threshold <= 1:
        raise ValueError("Hadamard covariance thresholds are invalid")
    if not 0 <= args.minimum_relative_kl_gain <= 1:
        raise ValueError("minimum relative KL gain must be in [0, 1]")
    if not 1 <= args.minimum_promoting_seeds <= len(args.transform_seeds):
        raise ValueError("minimum promoting seeds is outside the transform inventory")
    config_payload = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("model config must be a JSON object")
    config = cast(dict[str, object], config_payload)
    adapter = adapter_for_config(config)
    expected_blocks = adapter.decoder_block_count_from_config(config)
    if any(block >= expected_blocks for block in args.blocks):
        raise ValueError("requested block is outside the model")
    covariance_samples = math.ceil(
        (args.fit_tokens + args.held_out_tokens) / args.sequence_length
    )
    all_tokens, dataset_fingerprint, _bos = _wikitext_tokens(
        args.snapshot,
        samples=covariance_samples + args.wikitext_samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    covariance_tokens = all_tokens[:covariance_samples]
    functional_tokens = all_tokens[covariance_samples:]
    protocol = ProbeProtocol(
        1,
        args.model_revision,
        args.target_bpw,
        args.rank_alignment,
        args.scale_bits,
        args.outer_iterations,
        args.inner_iterations,
        args.regularization,
        args.penalty_schedule,
        args.convergence_check_interval,
        True,
        args.scale_fit_passes,
        args.seed,
        args.device,
        str(args.calibration_state.resolve()),
        0.0,
    )
    profiles = load_calibration_profiles(args.calibration_state, 0.0)
    with acquire_device_lease(args.device):
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        decoder_blocks = tuple(adapter.get_decoder_layers(teacher))
        covariances = _capture_covariances(
            teacher,
            _input_capture_specs(decoder_blocks, args.blocks),
            covariance_tokens,
            fit_tokens=args.fit_tokens,
            held_out_tokens=args.held_out_tokens,
            device=args.device,
        )
        teacher.to("cpu")
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        baseline_groups: dict[str, dict[str, Any]] = {}
        baseline_members: dict[
            str,
            tuple[tuple[MemberSpec, torch.Tensor, float], ...],
        ] = {}
        candidate_groups: dict[str, dict[str, dict[str, Any]]] = {}
        candidate_members: dict[
            str,
            dict[str, tuple[tuple[MemberSpec, torch.Tensor, float], ...]],
        ] = {}
        with safe_open(str(args.model), framework="pt", device="cpu") as handle:
            for block in args.blocks:
                for group in block_groups(block):
                    key = f"{block}:{group.label}"
                    covariance_key = (
                        _covariance_key(block, group.label)
                        if group.label in TRANSFORMED_GROUPS
                        else None
                    )
                    pair = None if covariance_key is None else covariances[covariance_key]
                    print(f"factorizing arm={BASELINE_KEY} group={key}", flush=True)
                    baseline_groups[key], baseline_members[key] = _group_result(
                        handle,
                        group,
                        protocol,
                        profiles,
                        None if pair is None else pair[0],
                        None if pair is None else pair[1],
                        transform=None,
                        damp_fraction=args.damp_fraction,
                    )
            for transform_seed in args.transform_seeds:
                arm = f"hadamard-seed-{transform_seed}"
                candidate_groups[arm] = {}
                candidate_members[arm] = {}
                transform_cache: dict[tuple[int, str, int], StructuredHadamard] = {}
                for block in args.blocks:
                    for group in block_groups(block):
                        key = f"{block}:{group.label}"
                        if group.label not in TRANSFORMED_GROUPS:
                            candidate_groups[arm][key] = baseline_groups[key]
                            candidate_members[arm][key] = baseline_members[key]
                            continue
                        covariance_key = _covariance_key(block, group.label)
                        pair = covariances[covariance_key]
                        width = int(pair[0].shape[0])
                        role = _transform_role(group.label)
                        transform_key = (block, role, width)
                        transform = transform_cache.get(transform_key)
                        if transform is None:
                            transform = make_structured_hadamard(
                                width,
                                args.hadamard_block_size,
                                _transform_seed(transform_seed, block, role),
                            )
                            transform_cache[transform_key] = transform
                        print(f"factorizing arm={arm} group={key}", flush=True)
                        candidate_groups[arm][key], candidate_members[arm][key] = _group_result(
                            handle,
                            group,
                            protocol,
                            profiles,
                            pair[0],
                            pair[1],
                            transform=transform,
                            damp_fraction=args.damp_fraction,
                        )
        reconstruction_sets = {
            BASELINE_KEY: _build_reconstruction_set(baseline_groups, baseline_members),
            **{
                arm: _build_reconstruction_set(candidate_groups[arm], candidate_members[arm])
                for arm in candidate_groups
            },
        }
        reconstruction_metrics = {
            BASELINE_KEY: {
                "aggregate": _aggregate_groups(baseline_groups),
                "groups": baseline_groups,
            },
            **{
                arm: {
                    "aggregate": _aggregate_groups(candidate_groups[arm]),
                    "groups": candidate_groups[arm],
                }
                for arm in candidate_groups
            },
        }
        baseline_bits = int(
            reconstruction_metrics[BASELINE_KEY]["aggregate"]["actual_bits"]
        )
        baseline_ranks = {
            key: int(value["rank"]) for key, value in baseline_groups.items()
        }
        for arm, groups in candidate_groups.items():
            if int(reconstruction_metrics[arm]["aggregate"]["actual_bits"]) != baseline_bits:
                raise ValueError("Hadamard arm changed the physical factor bit budget")
            if {key: int(value["rank"]) for key, value in groups.items()} != baseline_ranks:
                raise ValueError("Hadamard arm changed the factor rank inventory")
        teacher.to(args.device)
        teacher.eval()
        decoder_blocks = tuple(adapter.get_decoder_layers(teacher))
        output_tokens = functional_tokens[: args.block_output_samples]
        output_reference = _capture_outputs(
            teacher,
            {block: decoder_blocks[block] for block in args.blocks},
            output_tokens,
            device=args.device,
        )
        arms = ("full", *(f"block:{block}" for block in args.blocks))
        ordered_keys = (BASELINE_KEY, *candidate_groups)
        kl_results = {}
        block_outputs = {}
        teacher_batches: tuple[torch.Tensor, ...] | None = None
        baseline_nll = math.nan
        for key in ordered_keys:
            evaluator = DenseKlSpliceEvaluator(
                teacher,
                reconstruction_sets[key],
                functional_tokens,
                device=args.device,
                batch_size=1,
                token_chunk_size=128,
                teacher_cache_mode="cpu",
            )
            if teacher_batches is None:
                baseline_nll, teacher_batches = evaluator.teacher_cache_state()
            else:
                evaluator.install_teacher_cache(baseline_nll, teacher_batches)
            kl_results[key] = {arm: evaluator(arm) for arm in arms}
            block_outputs[key] = _isolated_block_outputs(
                evaluator,
                teacher,
                decoder_blocks,
                output_reference,
                output_tokens,
                device=args.device,
            )
            del evaluator
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        del teacher
    comparisons = {
        key: {
            arm: _paired_summary(kl_results[BASELINE_KEY][arm], results[arm])
            for arm in arms
        }
        for key, results in kl_results.items()
        if key != BASELINE_KEY
    }
    held_baseline = float(
        reconstruction_metrics[BASELINE_KEY]["aggregate"]["held_out_covariance_error"]
    )
    held_reductions = {
        key: (
            held_baseline
            - float(reconstruction_metrics[key]["aggregate"]["held_out_covariance_error"])
        )
        / max(held_baseline, 1e-30)
        for key in candidate_groups
    }
    functional_promoting = {
        key: (
            float(summary["full"]["relative_kl_delta"]) <= -args.minimum_relative_kl_gain
            and float(summary["full"]["upper_delta"]) < 0
        )
        for key, summary in comparisons.items()
    }
    promoting_seed_count = sum(functional_promoting.values())
    median_held_reduction = statistics.median(held_reductions.values())
    promotes_hadamard = (
        promoting_seed_count >= args.minimum_promoting_seeds
        and median_held_reduction >= args.covariance_promotion_threshold
    )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "role": "analysis-only input-Hadamard selection; not a compression artifact",
        "model_source": MODEL_SOURCE,
        "model_revision": args.model_revision,
        "blocks": list(args.blocks),
        "transformed_groups": sorted(TRANSFORMED_GROUPS),
        "held_identical_group": "down",
        "transform": {
            "kind": "random-sign-permutation-block-fwht",
            "block_size": args.hadamard_block_size,
            "seeds": list(args.transform_seeds),
            "runtime_metadata": "deterministically derived from global seed, block, and input role",
            "unique_sites_per_arm": len(args.blocks) * 3,
        },
        "protocol": {
            **to_dict(protocol),
            "fit_tokens": args.fit_tokens,
            "held_out_tokens": args.held_out_tokens,
            "wikitext_samples": args.wikitext_samples,
            "block_output_samples": args.block_output_samples,
            "sequence_length": args.sequence_length,
            "dataset_fingerprint": dataset_fingerprint,
            "covariance_slice_hash": _token_hash(covariance_tokens),
            "functional_slice_hash": _token_hash(functional_tokens),
            "damp_fraction": args.damp_fraction,
            "covariance_promotion_threshold": args.covariance_promotion_threshold,
            "minimum_relative_kl_gain": args.minimum_relative_kl_gain,
            "minimum_promoting_seeds": args.minimum_promoting_seeds,
        },
        "teacher_baseline_nll": baseline_nll,
        "reconstruction": reconstruction_metrics,
        "kl": {
            key: {arm: to_dict(result) for arm, result in results.items()}
            for key, results in kl_results.items()
        },
        "isolated_block_output_normalized_rmse": block_outputs,
        "paired_comparisons_vs_baseline": comparisons,
        "held_out_covariance_relative_error_reduction": held_reductions,
        "functional_seed_promotions": functional_promoting,
        "promotion": {
            "promoting_seed_count": promoting_seed_count,
            "median_held_out_covariance_relative_error_reduction": median_held_reduction,
            "promotes_hadamard": promotes_hadamard,
        },
    }
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "held_out_covariance_relative_error_reduction": held_reductions,
                "paired_comparisons": comparisons,
                "isolated_block_output_normalized_rmse": block_outputs,
                "promotion": payload["promotion"],
            },
            indent=2,
        )
    )
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
