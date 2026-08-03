"""Validate reactive tabu search on retained production-sized NanoQuant owners.

The probe starts from immutable factors committed by a complete resident run,
reconstructs the exact outlier-removed objective used by that owner, applies the
existing one-bit/variable-depth control, and then adds only the opt-in tabu tier.
It writes scalar evidence plus dense control/candidate weights for a later
functional splice gate. It never mutates the source run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nanoquant.config.codec import to_dict
from nanoquant.domain.binary_factor_search import (
    BinaryFactorSearchResult,
    refine_binary_factors_separable,
)
from nanoquant.domain.calibration_math import weighted_group_output_importance
from nanoquant.domain.models import (
    ArtifactRef,
    ArtifactTypes,
    BlockResult,
    FrozenNanoQuantState,
    FrozenOutlierState,
    FrozenSharedInputGroupState,
    LayerId,
    LayerResult,
    SharedInputGroupResult,
    TensorRef,
)
from nanoquant.domain.scale_fit import reconstruct
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import (
    CommitIdentity,
    latest_complete_identity,
    load_committed_block,
)
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.tensor_store import LocalTensorStore


@dataclass(frozen=True, slots=True)
class OwnerCase:
    block: int
    name: str
    members: tuple[LayerId, ...]
    member_rows: tuple[int, ...]
    target: torch.Tensor
    input_importance: torch.Tensor
    output_importance: torch.Tensor
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    outlier_indices: torch.Tensor | None
    outlier_values: torch.Tensor | None
    patch: torch.Tensor | None
    references: dict[str, Any]


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("blocks must be unique non-negative integers")
    return result


def _parse_names(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("owners must be unique non-empty names")
    return result


def _read_tensor(tensors: LocalTensorStore, reference: TensorRef) -> torch.Tensor:
    with tensors.read(reference, "cpu") as value:
        return value.clone()


def _state_tensors(
    state: FrozenNanoQuantState | FrozenSharedInputGroupState,
    tensors: LocalTensorStore,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if state.scales.mid is None:
        raise ValueError(f"retained owner {state.rank=} has no middle scale")
    return (
        _read_tensor(tensors, state.left_binary),
        _read_tensor(tensors, state.right_binary),
        _read_tensor(tensors, state.scales.pre),
        _read_tensor(tensors, state.scales.mid),
        _read_tensor(tensors, state.scales.post),
    )


def _outlier_tensors(
    state: FrozenOutlierState | None,
    tensors: LocalTensorStore,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if state is None:
        return None, None
    indices = _read_tensor(tensors, state.indices).long()
    values = _read_tensor(tensors, state.values).float()
    if state.scales is not None:
        values *= _read_tensor(tensors, state.scales).float()
    return indices, values


def _patch_tensor(
    state: FrozenNanoQuantState,
    tensors: LocalTensorStore,
) -> torch.Tensor | None:
    if state.patch_left is None or state.patch_right is None:
        return None
    return _read_tensor(tensors, state.patch_left).float() @ _read_tensor(
        tensors, state.patch_right
    ).float()


def _residual_target(
    source: torch.Tensor,
    input_importance: torch.Tensor,
    indices: torch.Tensor | None,
    *,
    removed_column_importance: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = source.float().clone()
    importance = input_importance.float().clone()
    if indices is not None and indices.numel():
        target[:, indices] = 0
        if removed_column_importance == "zero":
            floor = importance.median().clamp_min(1e-12) * 1e-4
            importance[indices] = floor
    return target, importance


def _layer_case(
    result: LayerResult,
    model_handle: Any,
    tensors: LocalTensorStore,
) -> OwnerCase:
    state = result.frozen_state
    source = model_handle.get_tensor(result.plan.source_weight.source_key).float()
    input_importance = _read_tensor(tensors, result.plan.objective.input_importance).float()
    output_importance = _read_tensor(tensors, result.plan.objective.output_importance).float()
    indices, values = _outlier_tensors(state.outliers, tensors)
    target, input_importance = _residual_target(
        source,
        input_importance,
        indices,
        removed_column_importance=result.plan.outliers.removed_column_importance,
    )
    left, right, pre, mid, post = _state_tensors(state, tensors)
    return OwnerCase(
        result.layer.block.index,
        result.layer.path,
        (result.layer,),
        (source.shape[0],),
        target,
        input_importance,
        output_importance,
        left,
        right,
        pre,
        mid,
        post,
        indices,
        values,
        _patch_tensor(state, tensors),
        {
            "factorization": to_dict(result.factorization),
            "rank": state.rank,
            "final_reconstruction": to_dict(result.final_reconstruction),
        },
    )


def _group_case(
    result: SharedInputGroupResult,
    model_handle: Any,
    tensors: LocalTensorStore,
) -> OwnerCase:
    state = result.frozen_state
    sources = tuple(
        model_handle.get_tensor(member.weight.source_key).float() for member in result.plan.members
    )
    inputs = tuple(
        _read_tensor(tensors, objective.input_importance).float()
        for objective in result.plan.objectives
    )
    canonical = inputs[0]
    for member, value in zip(result.plan.members[1:], inputs[1:], strict=True):
        if not torch.allclose(
            canonical / canonical.mean().clamp_min(1e-12),
            value / value.mean().clamp_min(1e-12),
            rtol=1e-4,
            atol=1e-6,
        ):
            raise ValueError(f"shared input objective differs for {member.layer}")
    outputs = tuple(
        _read_tensor(tensors, objective.output_importance).float()
        for objective in result.plan.objectives
    )
    multipliers = result.plan.objective_multipliers or (1.0,) * len(outputs)
    output_importance = weighted_group_output_importance(outputs, multipliers)
    source = torch.cat(sources, dim=0).contiguous()
    indices, values = _outlier_tensors(state.outliers, tensors)
    target, input_importance = _residual_target(
        source,
        canonical,
        indices,
        removed_column_importance=result.plan.outliers.removed_column_importance,
    )
    left, right, pre, mid, post = _state_tensors(state, tensors)
    return OwnerCase(
        result.block.index,
        result.name,
        tuple(member.layer for member in result.plan.members),
        tuple(source.shape[0] for source in sources),
        target,
        input_importance,
        output_importance,
        left,
        right,
        pre,
        mid,
        post,
        indices,
        values,
        None,
        {
            "factorization": to_dict(result.factorization),
            "rank": state.rank,
            "objective_multipliers": list(multipliers),
            "final_reconstruction": to_dict(result.final_reconstruction),
        },
    )


def _load_cases(
    run_output: Path,
    model: Path,
    blocks: tuple[int, ...],
    owners: tuple[str, ...],
    expected_blocks: int,
) -> tuple[CommitIdentity, tuple[OwnerCase, ...]]:
    artifacts = LocalArtifactStore(run_output / "artifacts")
    tensors = LocalTensorStore(artifacts)
    records = [
        json.loads(line)
        for line in (run_output / "state" / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identity, block_records = latest_complete_identity(records, expected_blocks)
    cases = []
    with safe_open(str(model), framework="pt", device="cpu") as model_handle:
        for block in blocks:
            if block not in block_records:
                raise ValueError(f"run has no complete block {block}")
            committed = load_committed_block(
                ArtifactRef(
                    ArtifactTypes.BLOCK_RESULT,
                    str(block_records[block]["artifact_id"]),
                    1,
                ),
                artifacts,
                identity,
            ).result
            cases.extend(_select_cases(committed, owners, model_handle, tensors))
    return identity, tuple(cases)


def _select_cases(
    block: BlockResult,
    owners: tuple[str, ...],
    model_handle: Any,
    tensors: LocalTensorStore,
) -> tuple[OwnerCase, ...]:
    layers = {result.layer.path: result for result in block.layers}
    groups = {result.name: result for result in block.shared_input_groups}
    selected = []
    for owner in owners:
        if owner in layers:
            selected.append(_layer_case(layers[owner], model_handle, tensors))
        elif owner in groups:
            selected.append(_group_case(groups[owner], model_handle, tensors))
        else:
            raise ValueError(f"block {block.block.index} has no retained owner {owner}")
    return tuple(selected)


def _state(result: BinaryFactorSearchResult) -> tuple[torch.Tensor, ...]:
    return (
        result.left_binary,
        result.right_binary,
        result.scale_pre,
        result.scale_mid,
        result.scale_post,
    )


def _run_search(
    case: OwnerCase,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    device = args.device
    target = case.target.to(device)
    input_importance = case.input_importance.to(device)
    output_importance = case.output_importance.to(device)
    initial = tuple(
        value.to(device)
        for value in (
            case.left_binary,
            case.right_binary,
            case.scale_pre,
            case.scale_mid,
            case.scale_post,
        )
    )
    energy = float(
        (
            target.float().square()
            * output_importance.float()[:, None]
            * input_importance.float()[None, :]
        ).sum()
    )
    started = time.perf_counter()
    refit = refine_binary_factors_separable(
        target,
        *initial,
        input_importance,
        output_importance,
        outer_passes=0,
        scale_passes=args.scale_passes,
        continuous_candidates=False,
        one_bit_passes=0,
        pair_passes=0,
        block_bits=0,
        block_passes=0,
        component_passes=0,
    )
    refit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    base_control = refine_binary_factors_separable(
        target,
        *_state(refit),
        input_importance,
        output_importance,
        outer_passes=args.search_outer_passes,
        scale_passes=args.scale_passes,
        continuous_candidates=False,
        one_bit_passes=args.one_bit_passes,
        one_bit_fraction=args.one_bit_fraction,
        max_one_bit_vectors=args.max_one_bit_vectors,
        variable_depth_passes=args.variable_depth_passes,
        variable_depth_length=args.variable_depth_length,
        pair_passes=0,
        block_bits=0,
        block_passes=0,
        component_passes=0,
    )
    base_control_seconds = time.perf_counter() - started
    started = time.perf_counter()
    common_refit = refine_binary_factors_separable(
        target,
        *_state(base_control),
        input_importance,
        output_importance,
        outer_passes=0,
        scale_passes=args.scale_passes,
        continuous_candidates=False,
        one_bit_passes=0,
        pair_passes=0,
        block_bits=0,
        block_passes=0,
        component_passes=0,
    )
    common_refit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    candidate = refine_binary_factors_separable(
        target,
        *_state(common_refit),
        input_importance,
        output_importance,
        outer_passes=args.search_outer_passes,
        scale_passes=args.scale_passes,
        continuous_candidates=False,
        one_bit_passes=0,
        variable_depth_passes=0,
        tabu_passes=args.tabu_passes,
        tabu_steps=args.tabu_steps,
        tabu_tenure=args.tabu_tenure,
        tabu_tenure_jitter=args.tabu_tenure_jitter,
        pair_passes=0,
        block_bits=0,
        block_passes=0,
        component_passes=0,
    )
    tabu_seconds = time.perf_counter() - started
    extended_control = common_refit
    extension_records = []
    extension_seconds = 0.0
    if args.compute_match_control:
        for round_index in range(args.max_control_extension_rounds):
            started = time.perf_counter()
            next_control = refine_binary_factors_separable(
                target,
                *_state(extended_control),
                input_importance,
                output_importance,
                outer_passes=args.search_outer_passes,
                scale_passes=args.scale_passes,
                continuous_candidates=False,
                one_bit_passes=args.one_bit_passes,
                one_bit_fraction=args.one_bit_fraction,
                max_one_bit_vectors=args.max_one_bit_vectors,
                variable_depth_passes=args.variable_depth_passes,
                variable_depth_length=args.variable_depth_length,
                pair_passes=0,
                block_bits=0,
                block_passes=0,
                component_passes=0,
            )
            round_seconds = time.perf_counter() - started
            extension_seconds += round_seconds
            extension_records.append(
                {
                    "round": round_index + 1,
                    "weighted_error": next_control.after_error,
                    "accepted_outer_passes": next_control.accepted_outer_passes,
                    "one_bit_updates": next_control.one_bit_updates,
                    "variable_depth_updates": next_control.variable_depth_updates,
                    "wall_seconds": round_seconds,
                }
            )
            extended_control = next_control
            if next_control.accepted_outer_passes == 0 or extension_seconds >= tabu_seconds:
                break
    control_dense = _dense_weight(case, extended_control).cpu()
    candidate_dense = _dense_weight(case, candidate).cpu()
    record = {
        "block": case.block,
        "owner": case.name,
        "members": [f"{member.block.index}:{member.path}" for member in case.members],
        "shape": list(case.target.shape),
        "rank": case.left_binary.shape[1],
        "target_weighted_energy": energy,
        "retained_weighted_error": _weighted_error(case, initial),
        "scale_refit_weighted_error": refit.after_error,
        "base_control_weighted_error": base_control.after_error,
        "common_refit_weighted_error": common_refit.after_error,
        "control_weighted_error": extended_control.after_error,
        "tabu_weighted_error": candidate.after_error,
        "control_nrmse": math.sqrt(extended_control.after_error / max(energy, 1e-30)),
        "tabu_nrmse": math.sqrt(candidate.after_error / max(energy, 1e-30)),
        "tabu_gain_fraction": 1.0
        - candidate.after_error / max(extended_control.after_error, 1e-30),
        "tabu_gain_vs_base_control_fraction": 1.0
        - candidate.after_error / max(base_control.after_error, 1e-30),
        "tabu_gain_vs_common_refit_fraction": 1.0
        - candidate.after_error / max(common_refit.after_error, 1e-30),
        "base_control_updates": {
            "one_bit": base_control.one_bit_updates,
            "variable_depth": base_control.variable_depth_updates,
            "accepted_outer_passes": base_control.accepted_outer_passes,
        },
        "control_extension": extension_records,
        "tabu_updates": candidate.tabu_updates,
        "tabu_accepted_outer_passes": candidate.accepted_outer_passes,
        "tabu_sign_distance": int(
            (candidate.left_binary != extended_control.left_binary).sum()
            + (candidate.right_binary != extended_control.right_binary).sum()
        ),
        "wall_seconds": {
            "scale_refit": refit_seconds,
            "base_control": base_control_seconds,
            "common_refit": common_refit_seconds,
            "control_extension": extension_seconds,
            "tabu": tabu_seconds,
        },
        "references": case.references,
    }
    del (
        target,
        input_importance,
        output_importance,
        refit,
        base_control,
        common_refit,
        extended_control,
        candidate,
    )
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    tensors = {}
    offset = 0
    for member, rows in zip(case.members, case.member_rows, strict=True):
        key = f"block_{case.block}.{member.path}"
        tensors[f"control.{key}"] = control_dense[offset : offset + rows].to(torch.bfloat16)
        tensors[f"tabu.{key}"] = candidate_dense[offset : offset + rows].to(torch.bfloat16)
        offset += rows
    return record, tensors


def _weighted_error(case: OwnerCase, state: tuple[torch.Tensor, ...]) -> float:
    prediction = reconstruct(*state).float().cpu()
    return float(
        (
            (prediction - case.target.float()).square()
            * case.output_importance.float()[:, None]
            * case.input_importance.float()[None, :]
        ).sum()
    )


def _dense_weight(case: OwnerCase, result: BinaryFactorSearchResult) -> torch.Tensor:
    dense = result.reconstruction.float().clone()
    if case.outlier_indices is not None and case.outlier_values is not None:
        dense[:, case.outlier_indices.to(dense.device)] += case.outlier_values.to(dense.device)
    if case.patch is not None:
        dense += case.patch.to(dense.device)
    return dense


def run(args: argparse.Namespace) -> int:
    if args.max_control_extension_rounds < 0:
        raise ValueError("maximum control extension rounds must be non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    identity, cases = _load_cases(
        args.run_output,
        args.model,
        args.blocks,
        args.owners,
        args.expected_blocks,
    )
    output = {
        "schema_version": 1,
        "status": "running",
        "role": "analysis-only retained-owner tabu validation",
        "run_output": str(args.run_output.resolve()),
        "identity": to_dict(identity),
        "protocol": {
            "blocks": list(args.blocks),
            "owners": list(args.owners),
            "scale_passes": args.scale_passes,
            "search_outer_passes": args.search_outer_passes,
            "one_bit_passes": args.one_bit_passes,
            "one_bit_fraction": args.one_bit_fraction,
            "max_one_bit_vectors": args.max_one_bit_vectors,
            "variable_depth_passes": args.variable_depth_passes,
            "variable_depth_length": args.variable_depth_length,
            "tabu_passes": args.tabu_passes,
            "tabu_steps": args.tabu_steps,
            "tabu_tenure": args.tabu_tenure,
            "tabu_tenure_jitter": args.tabu_tenure_jitter,
            "compute_match_control": args.compute_match_control,
            "max_control_extension_rounds": args.max_control_extension_rounds,
            "device": args.device,
        },
        "results": [],
    }
    atomic_write_json(args.output, output)
    dense_tensors: dict[str, torch.Tensor] = {}
    with acquire_device_lease(args.device):
        for case in cases:
            print(
                f"running block={case.block} owner={case.name} "
                f"shape={tuple(case.target.shape)} rank={case.left_binary.shape[1]}",
                flush=True,
            )
            record, tensors = _run_search(case, args)
            output["results"].append(record)
            dense_tensors.update(tensors)
            atomic_write_json(args.output, output)
            print(
                f"completed block={case.block} owner={case.name} "
                f"tabu_gain={100 * record['tabu_gain_fraction']:.4f}% "
                f"updates={record['tabu_updates']} wall={record['wall_seconds']['tabu']:.1f}s",
                flush=True,
            )
    temporary = args.weights.with_suffix(args.weights.suffix + ".tmp")
    save_file(dense_tensors, temporary)
    temporary.replace(args.weights)
    output["status"] = "completed"
    output["weights"] = str(args.weights.resolve())
    atomic_write_json(args.output, output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--blocks", type=_parse_ints, default=(12,))
    parser.add_argument(
        "--owners",
        type=_parse_names,
        default=("self_attn.attn_qkv", "mlp.gate_proj", "mlp.down_proj"),
    )
    parser.add_argument("--expected-blocks", type=int, default=26)
    parser.add_argument("--scale-passes", type=int, default=64)
    parser.add_argument("--search-outer-passes", type=int, default=8)
    parser.add_argument("--one-bit-passes", type=int, default=16)
    parser.add_argument("--one-bit-fraction", type=float, default=1.0)
    parser.add_argument("--max-one-bit-vectors", type=int, default=2**31 - 1)
    parser.add_argument("--variable-depth-passes", type=int, default=2)
    parser.add_argument("--variable-depth-length", type=int, default=64)
    parser.add_argument("--tabu-passes", type=int, default=2)
    parser.add_argument("--tabu-steps", type=int, default=256)
    parser.add_argument("--tabu-tenure", type=int, default=8)
    parser.add_argument("--tabu-tenure-jitter", type=int, default=4)
    parser.add_argument(
        "--compute-match-control",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-control-extension-rounds", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
