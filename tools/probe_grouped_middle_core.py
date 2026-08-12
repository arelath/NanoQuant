"""Fit equal-rate 2x2 middle cores on retained production factor owners.

This analysis-only probe loads immutable factors and objectives from a complete
resident run. It refits the diagonal-scale control, removes the few weakest
components required to fund a 2x2 block-diagonal middle core, and writes dense
BF16 control/candidate weights for a later functional splice gate. The source
run and its artifacts are never mutated.
"""

from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from probe_real_block_tabu import OwnerCase, _load_cases, _parse_ints, _parse_names
from safetensors.torch import save_file

from nanoquant.config.codec import to_dict
from nanoquant.domain.grouped_middle_core import fit_equal_rate_grouped_middle_core
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json


def _complete_dense(case: OwnerCase, reconstruction: torch.Tensor) -> torch.Tensor:
    dense = reconstruction.float().clone()
    if case.outlier_indices is not None and case.outlier_values is not None:
        dense[:, case.outlier_indices.to(dense.device)] += case.outlier_values.to(dense.device)
    if case.patch is not None:
        dense += case.patch.to(dense.device)
    return dense


def _run_case(
    case: OwnerCase,
    *,
    device: str,
    diagonal_scale_passes: int,
    grouped_alternating_passes: int,
    pairing: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    target = case.target.to(device)
    input_importance = case.input_importance.to(device)
    output_importance = case.output_importance.to(device)
    factors = (
        case.left_binary.to(device),
        case.right_binary.to(device),
        case.scale_pre.to(device),
        case.scale_mid.to(device),
        case.scale_post.to(device),
    )
    protected = case.outlier_indices.to(device) if case.outlier_indices is not None else None
    diagonal = fit_scales(
        target,
        *factors,
        input_importance,
        output_importance,
        alternating_passes=diagonal_scale_passes,
        protected_columns=protected,
    )
    grouped = fit_equal_rate_grouped_middle_core(
        target,
        factors[0],
        factors[1],
        diagonal.scale_pre,
        diagonal.scale_mid,
        diagonal.scale_post,
        input_importance,
        output_importance,
        alternating_passes=grouped_alternating_passes,
        protected_columns=protected,
        pairing=pairing,
    )
    target_energy = float(
        (
            target.float().square()
            * output_importance.float()[:, None]
            * input_importance.float()[None, :]
        ).sum()
    )
    diagonal_dense = _complete_dense(case, diagonal.reconstruction).cpu()
    grouped_dense = _complete_dense(case, grouped.reconstruction).cpu()
    record = {
        "block": case.block,
        "owner": case.name,
        "members": [f"{member.block.index}:{member.path}" for member in case.members],
        "shape": list(case.target.shape),
        "diagonal_rank": int(case.left_binary.shape[1]),
        "grouped_rank": int(grouped.component_indices.numel()),
        "dropped_components": int(case.left_binary.shape[1] - grouped.component_indices.numel()),
        "target_weighted_energy": target_energy,
        "diagonal_weighted_error": diagonal.after_error,
        "grouped_weighted_error": grouped.after_error,
        "diagonal_nrmse": math.sqrt(diagonal.after_error / max(target_energy, 1e-30)),
        "grouped_nrmse": math.sqrt(grouped.after_error / max(target_energy, 1e-30)),
        "grouped_error_change_fraction": grouped.after_error / max(diagonal.after_error, 1e-30) - 1.0,
        "rate": {
            "diagonal_factor_and_middle_bits": grouped.diagonal_bits,
            "grouped_factor_and_middle_bits": grouped.grouped_bits,
            "slack_bits": grouped.diagonal_bits - grouped.grouped_bits,
            "common_pre_post_outlier_patch_bits_identical": True,
        },
        "diagonal_scale_fit_accepted": diagonal.accepted,
        "grouped_fit_accepted": grouped.accepted,
        "references": case.references,
    }
    tensors: dict[str, torch.Tensor] = {}
    offset = 0
    for member, rows in zip(case.members, case.member_rows, strict=True):
        key = f"block_{case.block}.{member.path}"
        tensors[f"diagonal.{key}"] = diagonal_dense[offset : offset + rows].to(torch.bfloat16)
        tensors[f"grouped.{key}"] = grouped_dense[offset : offset + rows].to(torch.bfloat16)
        offset += rows
    return record, tensors


def run(args: argparse.Namespace) -> int:
    if args.diagonal_scale_passes < 0 or args.grouped_alternating_passes < 0:
        raise ValueError("grouped middle-core pass counts must be non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    identity, cases = _load_cases(
        args.run_output,
        args.model,
        args.blocks,
        args.owners,
        args.expected_blocks,
    )
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "role": "analysis-only equal-rate grouped-middle-core validation",
        "run_output": str(args.run_output.resolve()),
        "identity": to_dict(identity),
        "protocol": {
            "blocks": list(args.blocks),
            "owners": list(args.owners),
            "group_size": 2,
            "scale_bits": 16,
            "storage_dtype": "bfloat16",
            "diagonal_scale_passes": args.diagonal_scale_passes,
            "grouped_alternating_passes": args.grouped_alternating_passes,
            "pairing": args.pairing,
            "device": args.device,
        },
        "results": [],
    }
    atomic_write_json(args.output, output)
    dense_tensors: dict[str, torch.Tensor] = {}
    lease = acquire_device_lease(args.device) if args.device.startswith("cuda") else nullcontext()
    with lease:
        for case in cases:
            print(
                f"running block={case.block} owner={case.name} "
                f"shape={tuple(case.target.shape)} rank={case.left_binary.shape[1]}",
                flush=True,
            )
            record, tensors = _run_case(
                case,
                device=args.device,
                diagonal_scale_passes=args.diagonal_scale_passes,
                grouped_alternating_passes=args.grouped_alternating_passes,
                pairing=args.pairing,
            )
            output["results"].append(record)
            dense_tensors.update(tensors)
            atomic_write_json(args.output, output)
            print(
                f"completed block={case.block} owner={case.name} "
                f"delta={100 * record['grouped_error_change_fraction']:.4f}% "
                f"rank={record['grouped_rank']}/{record['diagonal_rank']}",
                flush=True,
            )
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
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
    parser.add_argument("--blocks", type=_parse_ints, default=(0, 12, 25))
    parser.add_argument(
        "--owners",
        type=_parse_names,
        default=("self_attn.attn_qkv", "mlp.gate_proj", "mlp.down_proj"),
    )
    parser.add_argument("--expected-blocks", type=int, default=26)
    parser.add_argument("--diagonal-scale-passes", type=int, default=2)
    parser.add_argument("--grouped-alternating-passes", type=int, default=2)
    parser.add_argument(
        "--pairing",
        choices=("fixed", "magnitude", "coupling", "residual"),
        default="magnitude",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
