"""Materialize a completed Fisher checkpoint into calibration/objective/plan artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from nanoquant.application.calibration import materialize_causal_online_state
from nanoquant.application.calibration_artifacts import build_objectives, persist_calibration
from nanoquant.application.planning import PlanningRequest, build_quantization_plan, persist_plan
from nanoquant.config.codec import to_dict
from nanoquant.config.schema import (
    AllocationStrategy,
    DType,
    ObjectiveConfig,
    OutlierConfig,
    OutlierSelector,
    RankAllocationConfig,
    RankBoundsConfig,
    RankRetryConfig,
    ResidualProbeConfig,
    RetryThresholdConfig,
)
from nanoquant.domain.models import ArtifactRef, BlockId, DatasetIdentity, LayerId
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.calibration_checkpoint import load_causal_calibration_state
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.resident_quantization import _legacy_sensitivity_profile

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
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path("evidence/m3/experiment018-calibration"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    state = load_causal_calibration_state(args.state)
    dataset_values = load_pinned_calibration(
        args.calibration,
        ArtifactRef("calibration-dataset-manifest", CALIBRATION_ARTIFACT, 1),
    )
    if state.sample_count != dataset_values.input_ids.shape[0]:
        raise ValueError("Fisher checkpoint is not complete")
    source = SafetensorsModelSource(
        args.snapshot,
        source="google/gemma-3-1b-it",
        revision=MODEL_REVISION,
        verify_hashes=True,
    )
    checkpoint = source.inventory()
    adapter = adapter_for_config(checkpoint.config)
    inventory = adapter.model_inventory(source)
    inventory = replace(
        inventory,
        blocks=tuple(
            replace(
                block,
                quantizable_layers=tuple(
                    {layer.layer.path: layer for layer in block.quantizable_layers}[path] for path in LAYER_ORDER
                ),
            )
            for block in inventory.blocks
        ),
    )
    token_bytes = dataset_values.input_ids.contiguous().view(torch.uint8).numpy().tobytes()
    dataset = DatasetIdentity(
        "sha256:" + hashlib.sha256(token_bytes).hexdigest(),
        ("deterministic-token-fixture",),
        ("1",),
        checkpoint.tokenizer_hash,
        "raw-token-ids-v1",
    )
    materialized = materialize_causal_online_state(state, shrinkage=0.6)
    calibration_values = tuple(
        (LayerId(BlockId(int(item.path.split(".", 2)[1])), item.path.split(".", 2)[2]), item)
        for item in materialized
    )
    artifacts = LocalArtifactStore(args.output / "artifacts")
    tensors = LocalTensorStore(artifacts)
    calibration = persist_calibration(
        calibration_values,
        inventory.model,
        dataset,
        "online_fisher",
        "float32",
        artifacts,
        tensors,
        total_tokens=dataset_values.input_ids.numel(),
    )
    objectives = build_objectives(calibration, ObjectiveConfig(), artifacts)
    outliers = OutlierConfig(
        selector=OutlierSelector.RESIDUAL,
        fraction=0.001,
        storage_dtype=DType.BFLOAT16,
        charge_to_bit_budget=False,
        count_multiple=1,
        removed_column_importance="zero",
        residual_probe=ResidualProbeConfig(iterations=80, chunk_rows=512),
    )
    with acquire_device_lease(args.device):
        sensitivity = _legacy_sensitivity_profile(
            inventory,
            calibration,
            source,
            tensors,
            alpha=0.5,
            edge_boost=0.15,
            device=args.device,
        )
    allocation = RankAllocationConfig(
        target_bpw=1.0,
        strategy=AllocationStrategy.SENSITIVITY,
        sensitivity_alpha=0.5,
        bounds=RankBoundsConfig(
            multiple=32,
            floor_fraction_of_uniform=0.9,
            ceiling_fraction_of_uniform=1.1,
            edge_block_boost=0.15,
        ),
        retry=RankRetryConfig(
            enabled=True,
            thresholds=RetryThresholdConfig(
                weighted_normalized_error=0.5,
                raw_normalized_error=0.5,
            ),
            rank_increase_fraction=0.25,
            maximum_attempts=3,
            extra_bit_budget_fraction=0.02,
            allow_above_allocator_cap=True,
        ),
    )
    plan = persist_plan(
        build_quantization_plan(
            PlanningRequest(
                inventory,
                calibration.stats,
                calibration.reference,
                objectives.objectives,
                allocation,
                outliers,
                sensitivity,
            )
        ),
        artifacts,
    )
    payload = {
        "schema_version": 1,
        "calibration": to_dict(calibration.reference),
        "objectives": to_dict(objectives.reference),
        "plan": to_dict(plan.reference),
    }
    (args.output / "preprocessing.json").write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
