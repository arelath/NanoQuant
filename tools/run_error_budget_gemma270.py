"""Run a resumable Gemma-3-270M error-budget candidate in the torch runtime."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import _paths  # noqa: F401
from recipes import (
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
)

from nanoquant.config.codec import to_dict
from nanoquant.config.schema import (
    AllocationStrategy,
    BiasCorrectionConfig,
    KlAllocationObjective,
    KlSensitivityGranularity,
    LowRankPatchConfig,
    RankResponseSource,
    ReconstructionImportanceConfig,
    ReconstructionRankPlanningConfig,
    SharedInputMemberMultiplierConfig,
)
from nanoquant.infrastructure.hf_calibration_dataset import materialize_pinned_calibration
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.resident_quantization import run_resident_quantization
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    resident_request_from_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-source", type=Path, required=True)
    parser.add_argument("--kl-profile", type=Path)
    parser.add_argument("--kl-profile-key")
    parser.add_argument(
        "--uniform-control",
        action="store_true",
        help="build a fresh same-campaign uniform control without KL or inherited response values",
    )
    parser.add_argument(
        "--kl-granularity",
        type=KlSensitivityGranularity,
        choices=tuple(KlSensitivityGranularity),
        default=KlSensitivityGranularity.EXACT,
    )
    parser.add_argument(
        "--legacy-d2-proxy",
        action="store_true",
        help="explicitly reproduce the historical transferred-response KL/error proxy",
    )
    parser.add_argument("--v-multiplier", type=float, default=2.0)
    parser.add_argument("--patch-rank", type=int, default=0)
    parser.add_argument("--rank-trust-reference-run", type=Path)
    parser.add_argument("--rank-trust-fraction", type=float, default=1.0)
    parser.add_argument(
        "--bias-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interrupt-after-block-commits", type=int)
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    if args.uniform_control:
        if args.kl_profile is not None or args.kl_profile_key is not None:
            raise ValueError("uniform control forbids KL inputs")
        if args.legacy_d2_proxy:
            raise ValueError("uniform control has no D2 proxy mode")
    elif args.kl_profile is None or args.kl_profile_key is None:
        raise ValueError("a KL candidate requires both profile and profile key")
    if not args.uniform_control and not args.legacy_d2_proxy:
        if args.kl_granularity is not KlSensitivityGranularity.EXACT:
            raise ValueError("self-measured D2 requires exact physical-unit KL arms")
        if args.rank_trust_reference_run is not None or args.rank_trust_fraction != 1:
            raise ValueError("self-measured D2 forbids imported rank references")
    if args.v_multiplier <= 0:
        raise ValueError("v multiplier must be positive")
    if args.patch_rank < 0:
        raise ValueError("patch rank must not be negative")
    if not 0 <= args.rank_trust_fraction <= 1:
        raise ValueError("rank trust fraction must be in [0, 1]")
    if (args.rank_trust_fraction == 1) != (args.rank_trust_reference_run is None):
        raise ValueError(
            "rank trust reference must be set exactly when rank trust fraction is below one"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    base = replace(
        ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
        model=GEMMA_3_270M_COMPRESSION_TEMPLATE.model,
    )
    shared = base.factorization.shared_input
    weighted_groups = (
        shared.groups
        if args.uniform_control
        else tuple(
            replace(
                group,
                member_multipliers=(
                    SharedInputMemberMultiplierConfig("self_attn.v_proj", args.v_multiplier),
                ),
            )
            for group in shared.groups
        )
    )
    patch = (
        LowRankPatchConfig()
        if args.uniform_control
        else LowRankPatchConfig(
            enabled=args.patch_rank > 0,
            rank=max(1, args.patch_rank),
        )
    )
    if args.uniform_control:
        allocation = replace(
            base.allocation,
            strategy=AllocationStrategy.UNIFORM,
            kl_profile_artifact=None,
            kl_profile_key=None,
            reconstruction=ReconstructionRankPlanningConfig(),
        )
    else:
        if args.kl_profile is None or args.kl_profile_key is None:
            raise AssertionError("validated KL inputs disappeared")
        reconstruction = (
            base.allocation.reconstruction
            if args.legacy_d2_proxy
            else replace(
                base.allocation.reconstruction,
                objective_mode="calibration_weighted",
                response_source=RankResponseSource.MEASURED,
                response_curves=(),
                response_profile_provenance="",
                kl_objective=KlAllocationObjective.MEASURED_UNIT_KL,
                importance=ReconstructionImportanceConfig(),
                sensitivity_strength=1,
                protect_sensitive_units=False,
                target_protected_error_reduction_fraction=0,
            )
        )
        allocation = replace(
            base.allocation,
            strategy=AllocationStrategy.KL_CALIBRATED,
            kl_profile_artifact=str(args.kl_profile.resolve()),
            kl_profile_key=args.kl_profile_key,
            kl_sensitivity_granularity=args.kl_granularity,
            reconstruction=replace(
                reconstruction,
                rank_trust_reference_run=(
                    None
                    if args.rank_trust_reference_run is None
                    else str(args.rank_trust_reference_run.resolve())
                ),
                rank_trust_fraction=args.rank_trust_fraction,
            ),
        )
    config = replace(
        base,
        intent=replace(
            base.intent,
            experiment_number=None,
            name=args.output.name,
            purpose=(
                "Build a fresh uniform D2 control without inherited experimental values."
                if args.uniform_control
                else "Measure KL-calibrated allocation, closed-form bias correction, and member-weighted "
                "stacked QKV at the configured bit budget."
            ),
            hypothesis=(
                "Selected KL sensitivity granularity plus unbiased output bias and alpha_v weighting "
                "reduce held-out NLL and KL without increasing effective BPW."
            ),
            baseline_run=None if args.uniform_control else "d2-uniform-control",
            tags=(*base.intent.tags, "error-budget", "kl-calibrated", "bias-correction"),
        ),
        allocation=allocation,
        factorization=replace(
            base.factorization,
            bias_correction=BiasCorrectionConfig(enabled=args.bias_correction and not args.uniform_control),
            low_rank_patch=patch,
            shared_input=replace(shared, groups=weighted_groups),
        ),
        distillation=replace(base.distillation, enabled=False),
        runtime=replace(base.runtime, compute_device=args.device),
    )
    calibration = materialize_pinned_calibration(
        args.calibration_source,
        args.output,
        sample_count=config.calibration.sample_count,
        sequence_length=config.model.sequence_length,
        seed=config.reproducibility.seed,
        preparation_id=None,
        tokenizer_identity=f"{config.model.source}@{config.model.revision}",
    )
    inputs = ResolvedResidentInputs(
        snapshot=args.snapshot,
        output=args.output,
        registry_root=args.output.parent,
        token_ids=calibration.input_ids,
        quality_token_ids=calibration.input_ids[:1, :8],
        launcher_path=Path(__file__),
    )
    request = resident_request_from_config(
        config,
        inputs,
        ResidentExecutionOptions(
            interrupt_after_block_commits=args.interrupt_after_block_commits,
            maximum_wddm_shared_bytes=int(0.75 * 2**30),
        ),
    )
    try:
        result = run_resident_quantization(request)
    except InterruptedError as error:
        atomic_write_json(
            args.output / "candidate-summary.json",
            {"status": "interrupted", "reason": str(error), "config": to_dict(config)},
        )
        return 2
    payload = {
        "status": "completed",
        "config": to_dict(config),
        "blocks": len(result.blocks),
        "factor_owners": sum(len(block.layers) + len(block.shared_input_groups) for block in result.blocks),
        "effective_bpw": result.frozen_model.effective_bpw,
        "actual_total_bits": result.frozen_model.actual_total_bits,
        "reference_nll": result.reference_nll,
        "compressed_nll": result.compressed_nll,
        "logit_mse": result.logit_mse,
        "argmax_agreement": result.argmax_agreement,
        "peak_device_bytes": result.peak_device_bytes,
        "peak_host_bytes": result.peak_host_bytes,
        "artifact_bytes": result.artifact_bytes,
        "elapsed_seconds": result.elapsed_seconds,
        "bias_owner_count": sum(
            layer.frozen_state.bias is not None
            for block in result.blocks
            for layer in block.layers
        )
        + sum(
            group.frozen_state.bias is not None
            for block in result.blocks
            for group in block.shared_input_groups
        ),
        "patch_owner_count": sum(
            layer.frozen_state.patch_left is not None
            for block in result.blocks
            for layer in block.layers
        ),
        "actual_bias_bits": sum(
            layer.actual_bit_cost.bias_bits
            for block in result.blocks
            for layer in block.layers
        )
        + sum(
            group.actual_bit_cost.bias_bits
            for block in result.blocks
            for group in block.shared_input_groups
        ),
        "actual_patch_bits": sum(
            layer.actual_bit_cost.patch_bits
            for block in result.blocks
            for layer in block.layers
        ),
        "rank_inventory": [
            *(
                {
                    "unit_id": f"{block.block.index}:{layer.layer.path}",
                    "block": block.block.index,
                    "name": layer.layer.path,
                    "rank": layer.frozen_state.rank,
                    "factor_bits": layer.actual_bit_cost.binary_factor_bits,
                }
                for block in result.blocks
                for layer in block.layers
            ),
            *(
                {
                    "unit_id": f"{block.block.index}:{group.name}",
                    "block": block.block.index,
                    "name": group.name,
                    "rank": group.frozen_state.rank,
                    "factor_bits": group.actual_bit_cost.binary_factor_bits,
                }
                for block in result.blocks
                for group in block.shared_input_groups
            ),
        ],
    }
    atomic_write_json(args.output / "candidate-summary.json", payload)
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
