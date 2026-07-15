"""Run the pinned Gemma 3 1B Experiment 018/019 parity protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from nanoquant.config.schema import (
    ActivationRetention,
    ObservabilityConfig,
    ProfilingConfig,
    ProfilingLevel,
)
from nanoquant.domain.models import ArtifactRef, ArtifactTypes
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.infrastructure.resource_usage import peak_device_memory_bytes
from nanoquant.recipes import EXPERIMENT_018_CONFIG
from nanoquant.resident_quantization import (
    run_resident_factorization_slice,
    run_resident_quantization,
)
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    resident_request_from_config,
)

MODEL_REVISION = str(EXPERIMENT_018_CONFIG.model.revision)
CALIBRATION_ARTIFACT = str(EXPERIMENT_018_CONFIG.dataset.prepared_artifact)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path("evidence/m3/experiment018-calibration"))
    parser.add_argument("--factorized-tuning-epochs", type=int, default=0)
    parser.add_argument("--factorized-tuning-batch-size", type=int, default=8)
    parser.add_argument("--factorized-tuning-epoch-cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--initial-cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--nonfactorized-tuning-epochs", type=int, default=0)
    parser.add_argument("--nonfactorized-tuning-schedule", default="")
    parser.add_argument("--nonfactorized-tuning-batch-size", type=int, default=8)
    parser.add_argument("--nonfactorized-tuning-epoch-cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--post-block-refit-epochs", type=int, default=0)
    parser.add_argument("--post-block-refit-batch-size", type=int, default=8)
    parser.add_argument("--post-block-refit-epoch-cooldown-seconds", type=float, default=0.0)
    parser.add_argument(
        "--tuning-microbatch-size",
        type=int,
        default=None,
        help=(
            "activation microbatch used to accumulate each tuning optimizer batch "
            "(default: inherit the logical batch size; use 1 explicitly as a memory fallback)"
        ),
    )
    parser.add_argument("--activation-retention", choices=("rolling", "all"), default="rolling")
    parser.add_argument("--interrupt-after-layer-commits", type=int)
    parser.add_argument("--interrupt-after-block-commits", type=int)
    parser.add_argument("--interrupt-after-factorized-tuning-epoch-commits", type=int)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--admm-outer-iterations", type=int, default=800)
    parser.add_argument("--admm-inner-iterations", type=int, default=5)
    parser.add_argument(
        "--transpose-wide",
        action="store_true",
        help="Use legacy source's transposed wide-matrix ADMM orientation instead of the validated native policy.",
    )
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
    parser.add_argument("--event-level", choices=("debug", "info", "warning", "error"), default="info")
    parser.add_argument("--console-level", choices=("debug", "info", "warning", "error"), default="info")
    parser.add_argument("--record-admm-steps", action="store_true")
    parser.add_argument("--resource-interval-seconds", type=float, default=5.0)
    parser.add_argument("--capture-cuda-trace", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    nonfactorized_schedule = tuple(
        int(value.strip()) for value in args.nonfactorized_tuning_schedule.split(",") if value.strip()
    )

    calibration = load_pinned_calibration(
        args.calibration,
        ArtifactRef("calibration-dataset-manifest", CALIBRATION_ARTIFACT, 1),
    )
    base = EXPERIMENT_018_CONFIG
    factorized_loop = replace(
        base.block_tuning.factorized.loop,
        enabled=args.factorized_tuning_epochs > 0,
        epochs=args.factorized_tuning_epochs,
        batch_size=args.factorized_tuning_batch_size,
    )
    schedule_epochs = max(nonfactorized_schedule, default=0)
    nonfactorized_loop = replace(
        base.block_tuning.non_factorized.loop,
        enabled=args.nonfactorized_tuning_epochs > 0 or bool(nonfactorized_schedule),
        epochs=max(args.nonfactorized_tuning_epochs, schedule_epochs),
        batch_size=args.nonfactorized_tuning_batch_size,
    )
    refit = replace(
        base.block_tuning.post_block_refit,
        enabled=args.post_block_refit_epochs > 0,
        epochs=args.post_block_refit_epochs,
        batch_size=args.post_block_refit_batch_size,
    )
    config = replace(
        base,
        intent=replace(base.intent, experiment_number=None, name=args.output.name),
        calibration=replace(base.calibration, sample_count=args.samples),
        reproducibility=replace(base.reproducibility, seed=args.seed),
        factorization=replace(
            base.factorization,
            admm=replace(
                base.factorization.admm,
                outer_iterations=args.admm_outer_iterations,
                inner_iterations=args.admm_inner_iterations,
                transpose_wide=args.transpose_wide,
            ),
        ),
        block_tuning=replace(
            base.block_tuning,
            factorized=replace(base.block_tuning.factorized, loop=factorized_loop),
            non_factorized=replace(
                base.block_tuning.non_factorized,
                loop=nonfactorized_loop,
                epochs_by_layer_position=nonfactorized_schedule,
            ),
            post_block_refit=refit,
            microbatch_size=args.tuning_microbatch_size,
        ),
        distillation=replace(base.distillation, enabled=False),
        runtime=replace(
            base.runtime,
            compute_device=args.device,
            block_forward_batch_size=args.block_forward_batch_size,
            source_streaming=replace(
                base.runtime.source_streaming,
                verify_tensor_hashes=not args.skip_source_hash_verification,
            ),
            checkpoints=replace(
                base.runtime.checkpoints,
                activation_retention=ActivationRetention(args.activation_retention),
            ),
        ),
        profiling=ProfilingConfig(
            level=ProfilingLevel(args.profile),
            cuda_timing=args.profile_cuda_timing,
            cuda_sample_every=args.profile_cuda_sample_every,
            memory_counters=args.profile_memory_counters,
            emit_span_events=args.profile_span_events,
        ),
        observability=ObservabilityConfig(
            event_level=args.event_level,
            console_level=args.console_level,
            record_admm_steps=args.record_admm_steps,
            record_resource_interval_seconds=args.resource_interval_seconds,
            capture_cuda_trace=args.capture_cuda_trace,
        ),
    )
    inputs = ResolvedResidentInputs(
        snapshot=args.snapshot,
        output=args.output,
        registry_root=args.run_root,
        token_ids=calibration.input_ids[: args.samples],
        quality_token_ids=calibration.input_ids[:1, :8],
        launcher_path=Path(__file__),
        precomputed_calibration=(
            None
            if args.calibration_artifact is None
            else ArtifactRef("calibration-stats", args.calibration_artifact, 1)
        ),
        precomputed_objectives=(
            None if args.objectives_artifact is None else ArtifactRef("objective-specs", args.objectives_artifact, 1)
        ),
        precomputed_plan=(
            None if args.plan_artifact is None else ArtifactRef(ArtifactTypes.QUANTIZATION_PLAN, args.plan_artifact, 1)
        ),
    )
    options = ResidentExecutionOptions(
        factorized_tuning_epoch_cooldown_seconds=args.factorized_tuning_epoch_cooldown_seconds,
        initial_cooldown_seconds=args.initial_cooldown_seconds,
        nonfactorized_tuning_epoch_cooldown_seconds=args.nonfactorized_tuning_epoch_cooldown_seconds,
        post_block_refit_epoch_cooldown_seconds=args.post_block_refit_epoch_cooldown_seconds,
        interrupt_after_layer_commits=args.interrupt_after_layer_commits,
        interrupt_after_block_commits=args.interrupt_after_block_commits,
        interrupt_after_factorized_tuning_epoch_commits=args.interrupt_after_factorized_tuning_epoch_commits,
        restore_completed_blocks=not args.defer_model_restore,
        defer_layer_loss_snapshots=args.defer_layer_loss_snapshots,
    )
    request = resident_request_from_config(config, inputs, options)
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
        peak = peak_device_memory_bytes(args.device)
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
