"""Measure same-rank diagonal versus full-covariance reconstruction floors.

The probe captures disjoint fit and held-out input second moments from the
pinned teacher, then compares the best real-valued rank-r reconstruction under
the diagonal fit covariance with the best rank-r reconstruction under the
regularized dense fit covariance. Ranks use the production 1-BPW cost model.
This is a decision bound, not a runtime-compatible compression artifact.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_factor_grouping import (
    PINNED_MODEL_REVISION,
    PROJECTION_PATHS,
    GroupSpec,
    MemberSpec,
    ProbeProtocol,
    _planned_group_rank,
    load_calibration_profiles,
)
from probe_importance_shrinkage import _dtype, _parse_ints
from safetensors import safe_open

from nanoquant.application.covariance import SplitDenseCovarianceAccumulator
from nanoquant.domain.metrics import dense_hessian_squared_error
from nanoquant.domain.objectives import regularize_covariance, regularized_cholesky, unwhiten
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens

MODEL_SOURCE = "google/gemma-3-1b-it"
GROUP_LABELS = ("qkv", "o", "gate", "up")


def covariance_groups(block: int) -> tuple[GroupSpec, ...]:
    members = {name: MemberSpec(block, name) for name in PROJECTION_PATHS}
    return (
        GroupSpec("qkv", tuple(members[name] for name in ("q", "k", "v"))),
        GroupSpec("o", (members["o"],)),
        GroupSpec("gate", (members["gate"],)),
        GroupSpec("up", (members["up"],)),
    )


def _input_capture_specs(
    decoder_blocks: tuple[torch.nn.Module, ...],
    blocks: tuple[int, ...],
) -> dict[str, torch.nn.Module]:
    result = {}
    for block in blocks:
        module = decoder_blocks[block]
        result[f"{block}:qkv"] = module.get_submodule(PROJECTION_PATHS["q"])
        result[f"{block}:o"] = module.get_submodule(PROJECTION_PATHS["o"])
        # Gate and up consume the same post-attention-normalized tensor.
        result[f"{block}:mlp"] = module.get_submodule(PROJECTION_PATHS["gate"])
    return result


def _module_input_width(module: torch.nn.Module) -> int:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[1] <= 0:
        raise ValueError("projection module does not expose a two-dimensional weight")
    return int(weight.shape[1])


@torch.inference_mode()
def _capture_covariances(
    model: torch.nn.Module,
    modules: dict[str, torch.nn.Module],
    tokens: torch.Tensor,
    *,
    fit_tokens: int,
    held_out_tokens: int,
    device: str,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    accumulators = {
        key: SplitDenseCovarianceAccumulator(
            _module_input_width(module),
            fit_tokens,
            held_out_tokens,
            device=device,
        )
        for key, module in modules.items()
    }

    def capture(key: str, positional: tuple[object, ...]) -> None:
        if not positional or not isinstance(positional[0], torch.Tensor):
            raise TypeError(f"captured projection input is not a tensor: {key}")
        accumulators[key].update(positional[0])

    with ExitStack() as stack:
        for key, module in modules.items():
            handle = module.register_forward_pre_hook(
                lambda _module, positional, key=key: capture(key, positional)
            )
            stack.callback(handle.remove)
        for index in range(tokens.shape[0]):
            cast(Any, model)(input_ids=tokens[index : index + 1].to(device), use_cache=False)
    if any(not accumulator.complete for accumulator in accumulators.values()):
        raise ValueError("covariance capture did not collect every requested fit and held-out row")
    return {
        key: (
            accumulator.fit.materialize()[0],
            accumulator.held_out.materialize()[0],
        )
        for key, accumulator in accumulators.items()
    }


def _materialize_group_weight(handle: Any, group: GroupSpec) -> torch.Tensor:
    values = []
    for member in group.members:
        value = handle.get_tensor(member.tensor_name).float()
        values.append(value.mT if member.transpose else value)
    return values[0] if len(values) == 1 else torch.cat(values, dim=0)


def _group_importance(
    group: GroupSpec,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    input_values = []
    output_values = []
    for member in group.members:
        input_importance, output_importance = profiles[member.calibration_path]
        if member.transpose:
            input_importance, output_importance = output_importance, input_importance
        input_values.append(input_importance.float())
        output_values.append(output_importance.float())
    canonical_input = input_values[0]
    if any(
        not torch.allclose(canonical_input, candidate, rtol=1e-5, atol=1e-7)
        for candidate in input_values[1:]
    ):
        raise ValueError(f"group has inconsistent input importance: {group.label}")
    return canonical_input, torch.cat(output_values)


def _truncated_svd(value: torch.Tensor, rank: int) -> torch.Tensor:
    if value.ndim != 2 or not 0 < rank <= min(value.shape):
        raise ValueError("truncated SVD rank is outside the matrix dimensions")
    left, singular, right = torch.linalg.svd(value.float(), full_matrices=False)
    return cast(torch.Tensor, (left[:, :rank] * singular[:rank]) @ right[:rank])


def diagonal_rank_floor(
    target: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    if input_importance.numel() != target.shape[1] or output_importance.numel() != target.shape[0]:
        raise ValueError("diagonal importance dimensions do not match target")
    input_scale = input_importance.float().clamp_min(1e-12).sqrt()
    output_scale = output_importance.float().clamp_min(1e-12).sqrt()
    transformed = target.float() * output_scale[:, None] * input_scale[None, :]
    fitted = _truncated_svd(transformed, rank)
    return fitted / output_scale[:, None] / input_scale[None, :]


def covariance_rank_floor(
    target: torch.Tensor,
    covariance: torch.Tensor,
    output_importance: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    if covariance.shape != (target.shape[1], target.shape[1]) or output_importance.numel() != target.shape[0]:
        raise ValueError("covariance objective dimensions do not match target")
    cholesky = regularized_cholesky(covariance)
    output_scale = output_importance.float().clamp_min(1e-12).sqrt()
    transformed = (target.float() * output_scale[:, None]) @ cholesky
    fitted = _truncated_svd(transformed, rank)
    return unwhiten(fitted, cholesky) / output_scale[:, None]


def _error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    covariance: torch.Tensor,
    output_importance: torch.Tensor,
) -> float:
    return float(dense_hessian_squared_error(target, prediction, covariance, output_importance))


def compare_covariance_floors(
    target: torch.Tensor,
    fit_covariance: torch.Tensor,
    held_out_covariance: torch.Tensor,
    output_importance: torch.Tensor,
    rank: int,
    *,
    damp_fraction: float,
    promotion_threshold: float,
) -> dict[str, Any]:
    regularized = regularize_covariance(fit_covariance, damp_fraction=damp_fraction)
    diagonal = regularized.diagonal().clone()
    diagonal_fit = diagonal_rank_floor(target, diagonal, output_importance, rank)
    covariance_fit = covariance_rank_floor(target, regularized, output_importance, rank)
    zero = torch.zeros_like(target)
    fit_target = _error(target, zero, fit_covariance, output_importance)
    held_out_target = _error(target, zero, held_out_covariance, output_importance)
    fit_diagonal = _error(target, diagonal_fit, fit_covariance, output_importance)
    fit_covariance_error = _error(target, covariance_fit, fit_covariance, output_importance)
    held_out_diagonal = _error(target, diagonal_fit, held_out_covariance, output_importance)
    held_out_covariance_error = _error(target, covariance_fit, held_out_covariance, output_importance)
    relative_error_reduction = (held_out_diagonal - held_out_covariance_error) / max(
        held_out_diagonal,
        1e-30,
    )
    return {
        "rank": rank,
        "shape": list(target.shape),
        "damp_fraction": damp_fraction,
        "fit": {
            "target_error": fit_target,
            "diagonal_floor_error": fit_diagonal,
            "covariance_floor_error": fit_covariance_error,
            "diagonal_floor_normalized_rmse": math.sqrt(fit_diagonal / max(fit_target, 1e-30)),
            "covariance_floor_normalized_rmse": math.sqrt(
                fit_covariance_error / max(fit_target, 1e-30)
            ),
        },
        "held_out": {
            "target_error": held_out_target,
            "diagonal_floor_error": held_out_diagonal,
            "covariance_floor_error": held_out_covariance_error,
            "diagonal_floor_normalized_rmse": math.sqrt(
                held_out_diagonal / max(held_out_target, 1e-30)
            ),
            "covariance_floor_normalized_rmse": math.sqrt(
                held_out_covariance_error / max(held_out_target, 1e-30)
            ),
            "covariance_relative_error_reduction": relative_error_reduction,
        },
        "promotes_covariance": relative_error_reduction >= promotion_threshold,
    }


def _aggregate(values: tuple[dict[str, Any], ...], promotion_threshold: float) -> dict[str, Any]:
    fit_target = math.fsum(float(value["fit"]["target_error"]) for value in values)
    fit_diagonal = math.fsum(float(value["fit"]["diagonal_floor_error"]) for value in values)
    fit_covariance = math.fsum(float(value["fit"]["covariance_floor_error"]) for value in values)
    held_target = math.fsum(float(value["held_out"]["target_error"]) for value in values)
    held_diagonal = math.fsum(float(value["held_out"]["diagonal_floor_error"]) for value in values)
    held_covariance = math.fsum(float(value["held_out"]["covariance_floor_error"]) for value in values)
    reduction = (held_diagonal - held_covariance) / max(held_diagonal, 1e-30)
    return {
        "group_count": len(values),
        "fit_diagonal_floor_normalized_rmse": math.sqrt(fit_diagonal / max(fit_target, 1e-30)),
        "fit_covariance_floor_normalized_rmse": math.sqrt(fit_covariance / max(fit_target, 1e-30)),
        "held_out_diagonal_floor_normalized_rmse": math.sqrt(
            held_diagonal / max(held_target, 1e-30)
        ),
        "held_out_covariance_floor_normalized_rmse": math.sqrt(
            held_covariance / max(held_target, 1e-30)
        ),
        "held_out_covariance_relative_error_reduction": reduction,
        "promotes_covariance": reduction >= promotion_threshold,
        "individual_groups_promoting": sum(bool(value["promotes_covariance"]) for value in values),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--blocks", type=_parse_ints, default=(0, 12, 24))
    parser.add_argument("--fit-tokens", type=int, default=2048)
    parser.add_argument("--held-out-tokens", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--target-bpw", type=float, default=1.0)
    parser.add_argument("--rank-alignment", type=int, default=1)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--damp-fraction", type=float, default=0.01)
    parser.add_argument("--promotion-threshold", type=float, default=0.20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.fit_tokens <= 0 or args.held_out_tokens <= 0 or args.sequence_length < 2:
        raise ValueError("covariance token dimensions must be positive")
    if args.damp_fraction < 0:
        raise ValueError("covariance damp fraction must be non-negative")
    if not 0 <= args.promotion_threshold <= 1:
        raise ValueError("promotion threshold must be in [0, 1]")
    config_payload = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("model config must be a JSON object")
    config = cast(dict[str, object], config_payload)
    adapter = adapter_for_config(config)
    expected_blocks = adapter.decoder_block_count_from_config(config)
    if any(block >= expected_blocks for block in args.blocks):
        raise ValueError("requested block is outside the model")
    required_samples = math.ceil((args.fit_tokens + args.held_out_tokens) / args.sequence_length)
    tokens, dataset_fingerprint, _bos = _wikitext_tokens(
        args.snapshot,
        samples=required_samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    protocol = ProbeProtocol(
        1,
        args.model_revision,
        args.target_bpw,
        args.rank_alignment,
        args.scale_bits,
        0,
        0,
        0.0,
        "not-run",
        0,
        True,
        0,
        0,
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
            tokens,
            fit_tokens=args.fit_tokens,
            held_out_tokens=args.held_out_tokens,
            device=args.device,
        )
        del teacher, decoder_blocks
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        groups = {}
        with safe_open(str(args.model), framework="pt", device="cpu") as handle:
            for block in args.blocks:
                for group in covariance_groups(block):
                    key = f"{block}:{group.label}"
                    covariance_key = f"{block}:{'mlp' if group.label in {'gate', 'up'} else group.label}"
                    fit_covariance, held_out_covariance = covariances[covariance_key]
                    _input_importance, output_importance = _group_importance(group, profiles)
                    rank, extra_scale_bits = _planned_group_rank(handle, group, protocol, profiles)
                    target = _materialize_group_weight(handle, group).to(args.device)
                    groups[key] = compare_covariance_floors(
                        target,
                        fit_covariance.to(args.device),
                        held_out_covariance.to(args.device),
                        output_importance.to(args.device),
                        rank,
                        damp_fraction=args.damp_fraction,
                        promotion_threshold=args.promotion_threshold,
                    )
                    groups[key]["extra_input_scale_bits"] = extra_scale_bits
                    del target
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
    values = tuple(groups.values())
    aggregate = _aggregate(values, args.promotion_threshold)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "role": "analysis-only same-rank covariance decision bound; not a compression artifact",
        "model_source": MODEL_SOURCE,
        "model_revision": args.model_revision,
        "blocks": list(args.blocks),
        "groups": list(GROUP_LABELS),
        "excluded_group": {
            "label": "down",
            "reason": "6912-wide dense covariance is outside the cheap decision-probe workspace",
        },
        "protocol": {
            "fit_tokens": args.fit_tokens,
            "held_out_tokens": args.held_out_tokens,
            "sequence_length": args.sequence_length,
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_slice_hash": _token_hash(tokens),
            "target_bpw": args.target_bpw,
            "rank_alignment": args.rank_alignment,
            "scale_bits": args.scale_bits,
            "calibration_state": str(args.calibration_state.resolve()),
            "calibration_shrinkage": 0.0,
            "damp_fraction": args.damp_fraction,
            "promotion_threshold": args.promotion_threshold,
            "device": args.device,
        },
        "aggregate": aggregate,
        "results": groups,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"aggregate": aggregate}, indent=2))
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
