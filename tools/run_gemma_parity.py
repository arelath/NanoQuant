"""Run the pinned Gemma 3 1B Experiment 018/019 parity protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanoquant.config.schema import (
    ADMMConfig,
    AllocationStrategy,
    DType,
    OutlierConfig,
    OutlierSelector,
    ResidualProbeConfig,
    ScaleFitConfig,
)
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.resident_quantization import ResidentQuantizationRequest, run_resident_quantization

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
    parser.add_argument("--interrupt-after-layer-commits", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    calibration = load_pinned_calibration(
        args.calibration,
        ArtifactRef("calibration-dataset-manifest", CALIBRATION_ARTIFACT, 1),
    )
    result = run_resident_quantization(
        ResidentQuantizationRequest(
            snapshot=args.snapshot,
            output=args.output,
            source="google/gemma-3-1b-it",
            revision=MODEL_REVISION,
            token_ids=calibration.input_ids,
            quality_token_ids=calibration.input_ids[:1, :8],
            device=args.device,
            target_bpw=1.0,
            rank_multiple=32,
            allocation_strategy=AllocationStrategy.SENSITIVITY,
            rank_floor_fraction=0.9,
            rank_ceiling_fraction=1.1,
            rank_sensitivity_alpha=0.5,
            rank_edge_boost=0.15,
            layer_order=LAYER_ORDER,
            admm=ADMMConfig(outer_iterations=800, inner_iterations=5),
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
            factorized_tuning_batch_size=1,
            factorized_tuning_learning_rate=1e-5,
            calibration_method="online_fisher",
            calibration_shrinkage=0.6,
            calibration_batch_size=1,
            block_forward_batch_size=4,
            interrupt_after_layer_commits=args.interrupt_after_layer_commits,
        )
    )
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
