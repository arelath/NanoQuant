"""Run the pinned Gemma 3 1B Experiment 018/019 parity protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nanoquant.config.schema import (
    ADMMConfig,
    AllocationStrategy,
    DType,
    OutlierConfig,
    OutlierSelector,
    ProfilingConfig,
    ProfilingLevel,
    ResidualProbeConfig,
    ScaleFitConfig,
)
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    run_resident_factorization_slice,
    run_resident_quantization,
)

MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
CALIBRATION_ARTIFACT = "sha256-ad1f609729f86db7598eed5c703c55aacbb9cb024cab816ca7b300d574b7a4c8"
LAYER_ORDER = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path("evidence/m3/experiment018-calibration"))
    parser.add_argument("--factorized-tuning-epochs", type=int, default=0)
    parser.add_argument("--factorized-tuning-batch-size", type=int, default=8)
    parser.add_argument("--nonfactorized-tuning-epochs", type=int, default=0)
    parser.add_argument("--nonfactorized-tuning-schedule", default="")
    parser.add_argument("--nonfactorized-tuning-batch-size", type=int, default=8)
    parser.add_argument("--post-block-refit-epochs", type=int, default=0)
    parser.add_argument("--post-block-refit-batch-size", type=int, default=8)
    parser.add_argument(
        "--tuning-microbatch-size",
        type=int,
        default=1,
        help="activation microbatch used to accumulate each tuning optimizer batch (default: 1)",
    )
    parser.add_argument("--activation-retention", choices=("rolling", "all"), default="rolling")
    parser.add_argument("--interrupt-after-layer-commits", type=int)
    parser.add_argument("--interrupt-after-block-commits", type=int)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--admm-outer-iterations", type=int, default=800)
    parser.add_argument("--admm-inner-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-forward-batch-size", type=int, default=8)
    parser.add_argument("--calibration-artifact")
    parser.add_argument("--objectives-artifact")
    parser.add_argument("--plan-artifact")
    parser.add_argument("--defer-model-restore", action="store_true")
    parser.add_argument("--defer-layer-loss-snapshots", action="store_true")
    parser.add_argument("--factor-only", action="store_true")
    parser.add_argument("--factor-only-count", type=int, default=1)
    parser.add_argument("--skip-source-hash-verification", action="store_true")
    parser.add_argument(
        "--profile",
        choices=(ProfilingLevel.OFF.value, ProfilingLevel.MACRO.value, ProfilingLevel.MICRO.value),
        default="macro",
    )
    parser.add_argument("--profile-span-events", action="store_true")
    parser.add_argument("--profile-cuda-timing", action="store_true")
    parser.add_argument("--profile-cuda-sample-every", type=int, default=16)
    parser.add_argument("--profile-memory-counters", action="store_true")
    args = parser.parse_args()
    nonfactorized_schedule = tuple(
        int(value.strip()) for value in args.nonfactorized_tuning_schedule.split(",") if value.strip()
    )

    calibration = load_pinned_calibration(
        args.calibration,
        ArtifactRef("calibration-dataset-manifest", CALIBRATION_ARTIFACT, 1),
    )
    request = ResidentQuantizationRequest(
        snapshot=args.snapshot,
        output=args.output,
        source="google/gemma-3-1b-it",
        revision=MODEL_REVISION,
        token_ids=calibration.input_ids[: args.samples],
        quality_token_ids=calibration.input_ids[:1, :8],
        device=args.device,
        verify_hashes=not args.skip_source_hash_verification,
        target_bpw=1.0,
        rank_multiple=32,
        allocation_strategy=AllocationStrategy.SENSITIVITY,
        rank_floor_fraction=0.9,
        rank_ceiling_fraction=1.1,
        rank_sensitivity_alpha=0.5,
        rank_edge_boost=0.15,
        layer_order=LAYER_ORDER,
        admm=ADMMConfig(
            outer_iterations=args.admm_outer_iterations,
            inner_iterations=args.admm_inner_iterations,
        ),
        outliers=OutlierConfig(
            selector=OutlierSelector.RESIDUAL,
            fraction=0.001,
            storage_dtype=DType.BFLOAT16,
            charge_to_bit_budget=False,
            count_multiple=1,
            removed_column_importance="zero",
            residual_probe=ResidualProbeConfig(iterations=80, chunk_rows=512),
        ),
        scale_fit=ScaleFitConfig(
            enabled=True,
            alternating_passes=2,
            epsilon=1e-8,
            chunk_rows=512,
            rollback_on_regression=True,
        ),
        factorized_tuning_epochs=args.factorized_tuning_epochs,
        factorized_tuning_batch_size=args.factorized_tuning_batch_size,
        factorized_tuning_learning_rate=1e-5,
        nonfactorized_tuning_epochs=args.nonfactorized_tuning_epochs,
        nonfactorized_tuning_epochs_by_layer=nonfactorized_schedule,
        nonfactorized_tuning_batch_size=args.nonfactorized_tuning_batch_size,
        nonfactorized_tuning_learning_rate=1e-4,
        post_block_refit_epochs=args.post_block_refit_epochs,
        post_block_refit_batch_size=args.post_block_refit_batch_size,
        post_block_refit_learning_rate=1e-5,
        tuning_microbatch_size=args.tuning_microbatch_size,
        legacy_tuning_seed_reset=True,
        seed=args.seed,
        activation_retention=args.activation_retention,
        calibration_method="online_fisher",
        calibration_shrinkage=0.6,
        calibration_batch_size=1,
        block_forward_batch_size=args.block_forward_batch_size,
        interrupt_after_layer_commits=args.interrupt_after_layer_commits,
        interrupt_after_block_commits=args.interrupt_after_block_commits,
        precomputed_calibration=(
            None
            if args.calibration_artifact is None
            else ArtifactRef("calibration-stats", args.calibration_artifact, 1)
        ),
        precomputed_objectives=(
            None if args.objectives_artifact is None else ArtifactRef("objective-specs", args.objectives_artifact, 1)
        ),
        precomputed_plan=(
            None if args.plan_artifact is None else ArtifactRef("quantization-plan", args.plan_artifact, 1)
        ),
        restore_completed_blocks=not args.defer_model_restore,
        evaluate_inline_quality=not args.defer_model_restore,
        defer_layer_loss_snapshots=args.defer_layer_loss_snapshots,
        profiling=ProfilingConfig(
            level=ProfilingLevel(args.profile),
            cuda_timing=args.profile_cuda_timing,
            cuda_sample_every=args.profile_cuda_sample_every,
            memory_counters=args.profile_memory_counters,
            emit_span_events=args.profile_span_events,
        ),
    )
    if args.factor_only:
        if args.factor_only_count <= 0:
            raise ValueError("factor-only count must be positive")
        slices = []
        for _index in range(args.factor_only_count):
            sliced = run_resident_factorization_slice(request)
            slices.append(sliced)
            if sliced.layer is None:
                break
        print(
            json.dumps(
                {
                    "status": "complete" if slices[-1].layer is None else "committed",
                    "slices": [
                        {
                            "layer": None
                            if item.layer is None
                            else f"{item.layer.layer.block.index}:{item.layer.layer.path}",
                            "rank": None if item.layer is None else item.layer.plan.rank,
                            "peak_device_bytes": item.peak_device_bytes,
                            "elapsed_seconds": item.elapsed_seconds,
                            "remaining_layers": item.remaining_layers,
                        }
                        for item in slices
                    ],
                },
                indent=2,
            )
        )
        return
    try:
        result = run_resident_quantization(request)
    except InterruptedError as exc:
        peak = torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else 0
        print(json.dumps({"status": "interrupted", "reason": str(exc), "peak_device_bytes": peak}, indent=2))
        return
    print(
        json.dumps(
            {
                "blocks": len(result.blocks),
                "layers": sum(len(block.layers) for block in result.blocks),
                "reused": result.reused_commit_count,
                "bpw": result.frozen_model.effective_bpw,
                "reference_nll": result.reference_nll,
                "compressed_nll": result.compressed_nll,
                "logit_mse": result.logit_mse,
                "peak_device_bytes": result.peak_device_bytes,
                "peak_host_bytes": result.peak_host_bytes,
                "artifact_bytes": result.artifact_bytes,
                "elapsed_seconds": result.elapsed_seconds,
                "report": result.report.artifact_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
