"""Probe covariance refinement after resident factorized tuning and block refit.

The candidate starts from a completed resident run's immutable, pre-global-KD
factors.  It changes only binary signs and the three existing scale vectors,
leaving ranks, outliers, patches, and the physical format unchanged.  This
isolates placement: the earlier probe refined before tuning, while this probe
refines the factors after every block-local tuning and refit step has finished.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from evaluate_wikitext import _evaluate as _evaluate_wikitext
from evaluate_wikitext import _protocol_tokens
from probe_covariance_binary import _reconstruction_set
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
    load_calibration_profiles,
)
from probe_importance_shrinkage import (
    _capture_outputs,
    _dtype,
    _isolated_block_outputs,
    _paired_summary,
    _parse_ints,
)
from probe_input_hadamard import (
    TRANSFORMED_GROUPS,
    _covariance_key,
    _evaluate_prediction,
    _member_reconstructions,
    block_groups,
)
from safetensors import safe_open
from torch import nn

from nanoquant.application.layers import FrozenReferenceLinear, SharedInputProjectionView
from nanoquant.config.codec import to_dict
from nanoquant.domain.covariance_refinement import refine_binary_factors_under_covariance
from nanoquant.domain.objectives import regularize_covariance
from nanoquant.domain.scale_fit import reconstruct
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.kl_splice import DenseKlSpliceEvaluator
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens

MODEL_SOURCE = "google/gemma-3-1b-it"
BASELINE_KEY = "post-refit-baseline"
CANDIDATE_KEY = "post-refit-covariance-refined"


@dataclass(frozen=True, slots=True)
class FrozenFactorSnapshot:
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    dense_weight: torch.Tensor
    bias: torch.Tensor | None
    protected_columns: torch.Tensor | None
    patch_present: bool
    rank: int


@dataclass(frozen=True, slots=True)
class RefinedFactorSnapshot:
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor


def _parse_groups(value: str) -> tuple[str, ...]:
    groups = tuple(item.strip() for item in value.split(",") if item.strip())
    if not groups or len(set(groups)) != len(groups) or any(
        group not in TRANSFORMED_GROUPS for group in groups
    ):
        raise argparse.ArgumentTypeError(
            f"refinement groups must be unique members of {sorted(TRANSFORMED_GROUPS)}"
        )
    return groups


def _module_at_path(block: nn.Module, path: str) -> nn.Module:
    current = block
    for part in path.split("."):
        child = current[part] if isinstance(current, nn.ModuleDict) else getattr(current, part, None)
        if not isinstance(child, nn.Module):
            raise KeyError(f"module path not found: {path}")
        current = child
    return current


def _owner_for_group(block: nn.Module, group: GroupSpec) -> FrozenReferenceLinear:
    modules = tuple(
        _module_at_path(block, PROJECTION_PATHS[member.projection])
        for member in group.members
    )
    owners = tuple(
        module.owner if isinstance(module, SharedInputProjectionView) else module
        for module in modules
    )
    if not all(isinstance(owner, FrozenReferenceLinear) for owner in owners):
        raise TypeError(f"post-refit group is not frozen and factorized: {group.label}")
    if any(owner is not owners[0] for owner in owners[1:]):
        raise ValueError(f"post-refit group members do not share one owner: {group.label}")
    return cast(FrozenReferenceLinear, owners[0])


def _snapshot(owner: FrozenReferenceLinear) -> FrozenFactorSnapshot:
    protected = (
        None
        if owner.outlier_indices is None
        else torch.unique(owner.outlier_indices.detach().long().reshape(-1)).cpu()
    )
    return FrozenFactorSnapshot(
        owner.left_binary.detach().cpu().clone(),
        owner.right_binary.detach().cpu().clone(),
        owner.scale_pre.detach().cpu().clone(),
        owner.scale_mid.detach().cpu().clone(),
        owner.scale_post.detach().cpu().clone(),
        owner.dense_weight().detach().cpu().clone(),
        None if owner.bias is None else owner.bias.detach().cpu().clone(),
        protected,
        owner.patch_left is not None,
        int(owner.left_binary.shape[1]),
    )


def _extract_post_refit_snapshots(
    run_output: Path,
    snapshot: Path,
    blocks: tuple[int, ...],
    *,
    revision: str,
) -> tuple[dict[str, FrozenFactorSnapshot], str]:
    loaded = load_frozen_run(
        run_output,
        snapshot,
        source_name=MODEL_SOURCE,
        revision=revision,
        device="cpu",
        verify_hashes=False,
        backend="factorized",
        use_global_tuning=False,
    )
    base = getattr(loaded.model, "model", None)
    decoder = getattr(base, "layers", None)
    if not isinstance(decoder, nn.ModuleList):
        raise TypeError("frozen model does not expose decoder blocks")
    result: dict[str, FrozenFactorSnapshot] = {}
    for block_index in blocks:
        for group in block_groups(block_index):
            result[f"{block_index}:{group.label}"] = _snapshot(
                _owner_for_group(decoder[block_index], group)
            )
    identity = (
        f"{loaded.identity.model_hash}|{loaded.identity.config_hash}|"
        f"{loaded.identity.plan_hash}"
    )
    del loaded
    gc.collect()
    return result, identity


def _residual_target_and_addition(
    original_weight: torch.Tensor,
    factors: FrozenFactorSnapshot,
) -> tuple[torch.Tensor, torch.Tensor]:
    factor_weight = reconstruct(
        factors.left_binary.float(),
        factors.right_binary.float(),
        factors.scale_pre.float(),
        factors.scale_mid.float(),
        factors.scale_post.float(),
    )
    addition = factors.dense_weight.float() - factor_weight
    return original_weight.float() - addition, addition


def _bias_members(
    group: GroupSpec,
    factors: FrozenFactorSnapshot,
    member_rows: tuple[int, ...],
) -> tuple[torch.Tensor | None, ...]:
    if factors.bias is None:
        return tuple(None for _ in group.members)
    values = []
    cursor = 0
    for rows in member_rows:
        values.append(factors.bias[cursor : cursor + rows].clone())
        cursor += rows
    if cursor != factors.bias.numel():
        raise ValueError("group bias rows do not cover the frozen owner")
    return tuple(values)


def _attach_biases(
    members: tuple[tuple[MemberSpec, torch.Tensor, float], ...],
    biases: tuple[torch.Tensor | None, ...],
) -> tuple[tuple[MemberSpec, torch.Tensor, float, torch.Tensor | None], ...]:
    return tuple(
        (member, weight, energy, bias)
        for (member, weight, energy), bias in zip(members, biases, strict=True)
    )


def _reconstruction_set_with_bias(
    group_results: dict[str, dict[str, Any]],
    member_results: dict[
        str,
        tuple[tuple[MemberSpec, torch.Tensor, float, torch.Tensor | None], ...],
    ],
):
    plain = {
        key: tuple((member, weight, energy) for member, weight, energy, _bias in values)
        for key, values in member_results.items()
    }
    result = _reconstruction_set(group_results, plain)
    biases = {
        (member.block, PROJECTION_PATHS[member.projection]): bias
        for values in member_results.values()
        for member, _weight, _energy, bias in values
    }
    layers = tuple(
        type(item)(
            item.layer,
            item.weight,
            biases[(item.layer.block.index, item.layer.path)],
            item.weighted_normalized_squared_error,
        )
        for item in result.layers
    )
    return type(result)(layers, result.unit_members, result.unit_weighted_normalized_squared_errors)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--blocks", type=_parse_ints, default=(0, 12, 24))
    parser.add_argument("--fit-tokens", type=int, default=8192)
    parser.add_argument("--held-out-tokens", type=int, default=2048)
    parser.add_argument("--covariance-reserved-samples", type=int)
    parser.add_argument("--wikitext-samples", type=int, default=12)
    parser.add_argument("--block-output-samples", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--scale-passes", type=int, default=2)
    parser.add_argument("--left-steps", type=int, default=32)
    parser.add_argument("--right-batches", type=int, default=16)
    parser.add_argument("--right-batch-size", type=int, default=128)
    parser.add_argument("--damp-fraction", type=float, default=0.01)
    parser.add_argument("--covariance-diagonal-blend", type=float, default=0.0)
    parser.add_argument("--minimum-relative-kl-gain", type=float, default=0.05)
    parser.add_argument(
        "--refine-groups",
        type=_parse_groups,
        default=tuple(sorted(TRANSFORMED_GROUPS)),
    )
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--unit-arms", action="store_true")
    parser.add_argument(
        "--direct-frozen-wikitext",
        action="store_true",
        help="Also evaluate the actual pre-KD frozen model before and after installing refined factors.",
    )
    parser.add_argument(
        "--direct-unit-arms",
        action="store_true",
        help="With direct frozen WikiText, evaluate each refined owner separately before the joint candidate.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if not args.blocks or len(set(args.blocks)) != len(args.blocks) or any(block < 0 for block in args.blocks):
        raise ValueError("placement blocks must be unique non-negative indices")
    if min(
        args.fit_tokens,
        args.held_out_tokens,
        args.wikitext_samples,
        args.block_output_samples,
        args.sequence_length,
    ) <= 0:
        raise ValueError("placement probe dataset dimensions must be positive")
    if (
        args.scale_passes < 0
        or args.left_steps < 0
        or args.right_batches < 0
        or args.right_batch_size <= 0
        or args.damp_fraction < 0
        or not 0 <= args.covariance_diagonal_blend <= 1
        or not 0 <= args.minimum_relative_kl_gain <= 1
    ):
        raise ValueError("placement refinement settings are invalid")
    config_payload = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("model config must be a JSON object")
    config = cast(dict[str, object], config_payload)
    adapter = adapter_for_config(config)
    expected_blocks = adapter.decoder_block_count_from_config(config)
    if any(block >= expected_blocks for block in args.blocks):
        raise ValueError("requested placement block is outside the model")
    covariance_samples = math.ceil(
        (args.fit_tokens + args.held_out_tokens) / args.sequence_length
    )
    reserved_samples = args.covariance_reserved_samples or covariance_samples
    if reserved_samples < covariance_samples:
        raise ValueError("reserved covariance samples do not cover fit and held-out rows")

    all_tokens, dataset_fingerprint, _bos = _wikitext_tokens(
        args.snapshot,
        samples=reserved_samples + args.wikitext_samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    covariance_tokens = all_tokens[:covariance_samples]
    functional_tokens = all_tokens[reserved_samples:]
    profiles = load_calibration_profiles(args.calibration_state, 0.0)
    snapshots, commit_identity = _extract_post_refit_snapshots(
        args.run_output,
        args.snapshot,
        args.blocks,
        revision=args.model_revision,
    )

    with acquire_device_lease(args.device):
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        decoder_blocks = tuple(adapter.get_decoder_layers(teacher))
        capture_roles = {
            "qkv": "qkv",
            "o": "o",
            "gate": "mlp",
            "up": "mlp",
        }
        required_capture_keys = {
            f"{block}:{capture_roles[group]}"
            for block in args.blocks
            for group in args.refine_groups
        }
        covariances = _capture_covariances(
            teacher,
            {
                key: module
                for key, module in _input_capture_specs(decoder_blocks, args.blocks).items()
                if key in required_capture_keys
            },
            covariance_tokens,
            fit_tokens=args.fit_tokens,
            held_out_tokens=args.held_out_tokens,
            device=args.device,
        )

        baseline_groups: dict[str, dict[str, Any]] = {}
        candidate_groups: dict[str, dict[str, Any]] = {}
        baseline_members: dict[
            str,
            tuple[tuple[MemberSpec, torch.Tensor, float, torch.Tensor | None], ...],
        ] = {}
        candidate_members: dict[
            str,
            tuple[tuple[MemberSpec, torch.Tensor, float, torch.Tensor | None], ...],
        ] = {}
        refined_snapshots: dict[str, RefinedFactorSnapshot] = {}
        with safe_open(str(args.model), framework="pt", device="cpu") as handle:
            for block_index in args.blocks:
                for group in block_groups(block_index):
                    key = f"{block_index}:{group.label}"
                    factors = snapshots[key]
                    original = _materialize_group_weight(handle, group).to(args.device)
                    member_rows = tuple(
                        int(handle.get_slice(member.tensor_name).get_shape()[0])
                        for member in group.members
                    )
                    raw_input, output_importance = _group_importance(group, profiles)
                    output_importance = output_importance.to(args.device).float()
                    pair = (
                        covariances[_covariance_key(block_index, group.label)]
                        if group.label in args.refine_groups
                        else None
                    )
                    baseline = factors.dense_weight.to(args.device).float()
                    candidate = baseline
                    metadata: dict[str, Any] = {
                        "rank": factors.rank,
                        "refined": False,
                        "left_flips": 0,
                        "right_flips": 0,
                        "patch_present": factors.patch_present,
                    }
                    if pair is not None:
                        fit_covariance, held_covariance = pair
                        metric = regularize_covariance(
                            fit_covariance.to(args.device),
                            damp_fraction=args.damp_fraction,
                            diagonal_blend=args.covariance_diagonal_blend,
                        )
                        residual, addition = _residual_target_and_addition(
                            original,
                            FrozenFactorSnapshot(
                                factors.left_binary.to(args.device),
                                factors.right_binary.to(args.device),
                                factors.scale_pre.to(args.device),
                                factors.scale_mid.to(args.device),
                                factors.scale_post.to(args.device),
                                factors.dense_weight.to(args.device),
                                None if factors.bias is None else factors.bias.to(args.device),
                                (
                                    None
                                    if factors.protected_columns is None
                                    else factors.protected_columns.to(args.device)
                                ),
                                factors.patch_present,
                                factors.rank,
                            ),
                        )
                        started = time.perf_counter()
                        refined = refine_binary_factors_under_covariance(
                            residual,
                            factors.left_binary.to(args.device),
                            factors.right_binary.to(args.device),
                            factors.scale_pre.to(args.device),
                            factors.scale_mid.to(args.device),
                            factors.scale_post.to(args.device),
                            metric,
                            output_importance,
                            protected_columns=(
                                None
                                if factors.protected_columns is None
                                else factors.protected_columns.to(args.device)
                            ),
                            scale_passes=args.scale_passes,
                            left_steps=args.left_steps,
                            right_batches=args.right_batches,
                            right_batch_size=args.right_batch_size,
                        )
                        candidate = refined.reconstruction.float() + addition
                        refined_snapshots[key] = RefinedFactorSnapshot(
                            refined.left_binary.detach().cpu().clone(),
                            refined.right_binary.detach().cpu().clone(),
                            refined.scale_pre.detach().cpu().clone(),
                            refined.scale_mid.detach().cpu().clone(),
                            refined.scale_post.detach().cpu().clone(),
                        )
                        metadata = {
                            "rank": factors.rank,
                            "refined": True,
                            "before_covariance_error": refined.before_error,
                            "after_covariance_error": refined.after_error,
                            "relative_covariance_error_reduction": (
                                (refined.before_error - refined.after_error)
                                / max(refined.before_error, 1e-30)
                            ),
                            "left_flips": refined.left_flips,
                            "right_flips": refined.right_flips,
                            "patch_present": factors.patch_present,
                            "wall_seconds": time.perf_counter() - started,
                        }
                    else:
                        fit_covariance = held_covariance = None
                    diagonal = (
                        raw_input.to(args.device).float()
                        if pair is None
                        else pair[0].diagonal().to(args.device).float()
                    )
                    common = {
                        "block": block_index,
                        "group": group.label,
                        "members": [member.label for member in group.members],
                        "shape": list(original.shape),
                        "rank": factors.rank,
                        "source_elements": original.numel(),
                    }
                    baseline_groups[key] = {
                        **common,
                        "evaluation": _evaluate_prediction(
                            original,
                            baseline,
                            output_importance,
                            None if pair is None else fit_covariance.to(args.device),
                            None if pair is None else held_covariance.to(args.device),
                            diagonal,
                        ),
                    }
                    candidate_groups[key] = {
                        **common,
                        "refinement": metadata,
                        "evaluation": _evaluate_prediction(
                            original,
                            candidate,
                            output_importance,
                            None if pair is None else fit_covariance.to(args.device),
                            None if pair is None else held_covariance.to(args.device),
                            diagonal,
                        ),
                    }
                    biases = _bias_members(group, factors, member_rows)
                    baseline_members[key] = _attach_biases(
                        _member_reconstructions(group, baseline, member_rows),
                        biases,
                    )
                    candidate_members[key] = _attach_biases(
                        _member_reconstructions(group, candidate, member_rows),
                        biases,
                    )
                    del original, baseline, candidate
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()

        reconstruction_sets = {
            BASELINE_KEY: _reconstruction_set_with_bias(baseline_groups, baseline_members),
            CANDIDATE_KEY: _reconstruction_set_with_bias(candidate_groups, candidate_members),
        }
        output_tokens = functional_tokens[: args.block_output_samples]
        output_reference = _capture_outputs(
            teacher,
            {block: decoder_blocks[block] for block in args.blocks},
            output_tokens,
            device=args.device,
        )
        arms = ("full",) if args.full_only else (
            "full",
            *(f"block:{block}" for block in args.blocks),
            *(
                f"unit:{block}:{group}"
                for block in args.blocks
                for group in args.refine_groups
                if args.unit_arms
            ),
        )
        kl_results = {}
        block_outputs = {}
        teacher_batches: tuple[torch.Tensor, ...] | None = None
        teacher_nll = math.nan
        for key in (BASELINE_KEY, CANDIDATE_KEY):
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
                teacher_nll, teacher_batches = evaluator.teacher_cache_state()
            else:
                evaluator.install_teacher_cache(teacher_nll, teacher_batches)
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
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        direct_wikitext = None
        if args.direct_frozen_wikitext:
            direct_tokens, direct_fingerprint, direct_bos = _protocol_tokens(
                args.snapshot,
                64,
                128,
            )
            loaded = load_frozen_run(
                args.run_output,
                args.snapshot,
                source_name=MODEL_SOURCE,
                revision=args.model_revision,
                device=args.device,
                verify_hashes=False,
                backend="factorized",
                use_global_tuning=False,
            )
            loaded.model.eval()
            direct_baseline = _evaluate_wikitext(
                loaded.model,
                direct_tokens,
                args.device,
                1,
            )
            loaded_base = getattr(loaded.model, "model", None)
            loaded_decoder = getattr(loaded_base, "layers", None)
            if not isinstance(loaded_decoder, nn.ModuleList):
                raise TypeError("direct frozen model does not expose decoder blocks")
            owners = {}
            for key in refined_snapshots:
                block_text, group_name = key.split(":", 1)
                group = next(
                    group
                    for group in block_groups(int(block_text))
                    if group.label == group_name
                )
                owners[key] = _owner_for_group(loaded_decoder[int(block_text)], group)

            def install(key: str, values: RefinedFactorSnapshot | FrozenFactorSnapshot) -> None:
                owner = owners[key]
                with torch.no_grad():
                    owner.left_binary.copy_(
                        values.left_binary.to(
                            device=owner.left_binary.device,
                            dtype=owner.left_binary.dtype,
                        )
                    )
                    owner.right_binary.copy_(
                        values.right_binary.to(
                            device=owner.right_binary.device,
                            dtype=owner.right_binary.dtype,
                        )
                    )
                    owner.scale_pre.copy_(
                        values.scale_pre.to(
                            device=owner.scale_pre.device,
                            dtype=owner.scale_pre.dtype,
                        )
                    )
                    owner.scale_mid.copy_(
                        values.scale_mid.to(
                            device=owner.scale_mid.device,
                            dtype=owner.scale_mid.dtype,
                        )
                    )
                    owner.scale_post.copy_(
                        values.scale_post.to(
                            device=owner.scale_post.device,
                            dtype=owner.scale_post.dtype,
                        )
                    )

            direct_units = {}
            if args.direct_unit_arms:
                for key, refined in refined_snapshots.items():
                    install(key, refined)
                    result = _evaluate_wikitext(
                        loaded.model,
                        direct_tokens,
                        args.device,
                        1,
                    )
                    direct_units[key] = {
                        **result,
                        "mean_negative_log_likelihood_delta": (
                            float(result["mean_negative_log_likelihood"])
                            - float(direct_baseline["mean_negative_log_likelihood"])
                        ),
                        "perplexity_delta": (
                            float(result["perplexity"])
                            - float(direct_baseline["perplexity"])
                        ),
                        "perplexity_relative_delta": (
                            float(result["perplexity"])
                            / float(direct_baseline["perplexity"])
                            - 1
                        ),
                    }
                    install(key, snapshots[key])
            for key, refined in refined_snapshots.items():
                install(key, refined)
            direct_candidate = _evaluate_wikitext(
                loaded.model,
                direct_tokens,
                args.device,
                1,
            )
            direct_wikitext = {
                "samples": 64,
                "sequence_length": 128,
                "dataset_fingerprint": direct_fingerprint,
                "bos_token_id": direct_bos,
                "token_hash": _token_hash(direct_tokens),
                "baseline": direct_baseline,
                "candidate": direct_candidate,
                "unit_candidates": direct_units,
                "mean_negative_log_likelihood_delta": (
                    float(direct_candidate["mean_negative_log_likelihood"])
                    - float(direct_baseline["mean_negative_log_likelihood"])
                ),
                "perplexity_delta": (
                    float(direct_candidate["perplexity"])
                    - float(direct_baseline["perplexity"])
                ),
                "perplexity_relative_delta": (
                    float(direct_candidate["perplexity"])
                    / float(direct_baseline["perplexity"])
                    - 1
                ),
            }
            del loaded
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    comparisons = {}
    for arm in arms:
        baseline_result = kl_results[BASELINE_KEY][arm]
        candidate_result = kl_results[CANDIDATE_KEY][arm]
        comparisons[arm] = {
            **_paired_summary(baseline_result, candidate_result),
            "negative_log_likelihood_delta": (
                candidate_result.negative_log_likelihood
                - baseline_result.negative_log_likelihood
            ),
        }
    held_baseline = math.fsum(
        float(value["evaluation"]["held_out_covariance_error"])
        for value in baseline_groups.values()
        if value["group"] in args.refine_groups
    )
    held_candidate = math.fsum(
        float(value["evaluation"]["held_out_covariance_error"])
        for value in candidate_groups.values()
        if value["group"] in args.refine_groups
    )
    held_reduction = (held_baseline - held_candidate) / max(held_baseline, 1e-30)
    full = comparisons["full"]
    functional_promotion = (
        float(full["relative_kl_delta"]) <= -args.minimum_relative_kl_gain
        and float(full["upper_delta"]) < 0
        and float(full["negative_log_likelihood_delta"]) < 0
    )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "role": "analysis-only post-refit placement probe; not a compression artifact",
        "model_source": MODEL_SOURCE,
        "model_revision": args.model_revision,
        "source_run": str(args.run_output.resolve()),
        "source_commit_identity": commit_identity,
        "placement": "after factorized tuning and post-block refit; before global KD",
        "blocks": list(args.blocks),
        "refined_groups": list(args.refine_groups),
        "held_identical_group": "down",
        "rank_inventory": {key: value.rank for key, value in snapshots.items()},
        "format_invariants": {
            "ranks_unchanged": True,
            "outliers_unchanged": True,
            "patches_unchanged": True,
            "physical_bits_unchanged": True,
        },
        "protocol": {
            "fit_tokens": args.fit_tokens,
            "held_out_tokens": args.held_out_tokens,
            "covariance_reserved_samples": reserved_samples,
            "wikitext_samples": args.wikitext_samples,
            "block_output_samples": args.block_output_samples,
            "sequence_length": args.sequence_length,
            "dataset_fingerprint": dataset_fingerprint,
            "covariance_slice_hash": _token_hash(covariance_tokens),
            "functional_slice_hash": _token_hash(functional_tokens),
            "scale_passes": args.scale_passes,
            "left_steps": args.left_steps,
            "right_batches": args.right_batches,
            "right_batch_size": args.right_batch_size,
            "damp_fraction": args.damp_fraction,
            "covariance_diagonal_blend": args.covariance_diagonal_blend,
            "minimum_relative_kl_gain": args.minimum_relative_kl_gain,
            "full_only": args.full_only,
            "unit_arms": args.unit_arms,
            "direct_frozen_wikitext": args.direct_frozen_wikitext,
            "direct_unit_arms": args.direct_unit_arms,
        },
        "teacher_baseline_nll": teacher_nll,
        "reconstruction": {
            BASELINE_KEY: {"groups": baseline_groups},
            CANDIDATE_KEY: {"groups": candidate_groups},
        },
        "kl": {
            key: {arm: to_dict(result) for arm, result in values.items()}
            for key, values in kl_results.items()
        },
        "isolated_block_output_normalized_rmse": block_outputs,
        "paired_comparisons_vs_post_refit_baseline": comparisons,
        "direct_frozen_wikitext": direct_wikitext,
        "promotion": {
            "held_out_covariance_relative_error_reduction": held_reduction,
            "functional_promotion": functional_promotion,
            "promotes_post_refit_covariance_refinement": functional_promotion,
        },
    }
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "promotion": payload["promotion"],
                "paired_comparisons": comparisons,
                "isolated_block_output_normalized_rmse": block_outputs,
            },
            indent=2,
        )
    )
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
