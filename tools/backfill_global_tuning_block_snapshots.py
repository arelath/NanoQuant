"""Add bounded pre/post-KD block snapshots to an existing global-tuning result."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanoquant.application.block_snapshots import compare_block_snapshots, select_block_snapshot_tokens
from nanoquant.domain.models import ArtifactRef
from nanoquant.global_distillation import _checkpoint_dtype, _decoder_layers, _hidden_states
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.block_snapshot_probe import capture_block_output_reference, measure_block_output_mse
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.global_tuning import (
    activate_global_tuning,
    active_global_tuning,
    commit_global_tuning,
    load_global_tuning,
)
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration

MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
CALIBRATION_ARTIFACT = "sha256-ad1f609729f86db7598eed5c703c55aacbb9cb024cab816ca7b300d574b7a4c8"


def _offload(model: nn.Module, device: str) -> None:
    model.cpu()
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _load_source(snapshot: Path, attention_implementation: str, device: str) -> nn.Module:
    model = cast(
        nn.Module,
        AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            torch_dtype=_checkpoint_dtype(snapshot),
            attn_implementation=attention_implementation,
        ),
    ).to(device)
    cast(Any, model).config.use_cache = False
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path("evidence/m3/experiment018-calibration"))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    calibration = load_pinned_calibration(
        args.calibration,
        ArtifactRef("calibration-dataset-manifest", CALIBRATION_ARTIFACT, 1),
    )
    selection = select_block_snapshot_tokens(
        calibration.input_ids,
        maximum_samples=args.samples,
        maximum_tokens=args.tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    artifacts = LocalArtifactStore(args.run_output / "artifacts")
    active = active_global_tuning(args.run_output)
    if active is None:
        raise ValueError("run has no active global tuning result")
    tuning = load_global_tuning(active, artifacts).result
    if tuning.block_metrics and not args.replace:
        raise ValueError("active global tuning already has block snapshots; pass --replace to remeasure")

    def execute() -> dict[str, object]:
        pre = load_frozen_run(
            args.run_output,
            args.snapshot,
            source_name="google/gemma-3-1b-it",
            revision=MODEL_REVISION,
            device="cpu",
            backend="factorized",
            use_global_tuning=False,
        )
        attention = cast(Any, pre.model).config._attn_implementation
        source = _load_source(args.snapshot, attention, args.device)
        reference = capture_block_output_reference(
            source,
            _decoder_layers(source),
            selection.token_ids,
            _hidden_states,
            device=args.device,
        )
        _offload(source, args.device)
        del source

        pre.model.to(args.device)
        pre_losses = measure_block_output_mse(
            pre.model,
            _decoder_layers(pre.model),
            selection.token_ids,
            reference,
            _hidden_states,
            device=args.device,
            pad_token_id=tokenizer.pad_token_id,
        )
        _offload(pre.model, args.device)
        del pre

        post = load_frozen_run(
            args.run_output,
            args.snapshot,
            source_name="google/gemma-3-1b-it",
            revision=MODEL_REVISION,
            device=args.device,
            backend="factorized",
            use_global_tuning=True,
        )
        if post.global_tuning != active:
            raise ValueError("active global tuning changed while snapshots were being measured")
        post_losses = measure_block_output_mse(
            post.model,
            _decoder_layers(post.model),
            selection.token_ids,
            reference,
            _hidden_states,
            device=args.device,
            pad_token_id=tokenizer.pad_token_id,
        )
        blocks = tuple(block.block for block in post.blocks)
        _offload(post.model, args.device)
        del post

        metrics = compare_block_snapshots(blocks, pre_losses, post_losses, selection.protocol)
        upgraded = replace(
            tuning,
            schema_version=2,
            block_snapshot_protocol_hash=selection.protocol.semantic_key,
            block_metrics=metrics,
        )
        committed = commit_global_tuning(upgraded, artifacts)
        if active_global_tuning(args.run_output) != active:
            raise ValueError("active global tuning changed before snapshot activation")
        activate_global_tuning(args.run_output, committed.reference)
        return {
            "previous_artifact": active.artifact_id,
            "artifact": committed.reference.artifact_id,
            "block_snapshot_protocol_hash": selection.protocol.semantic_key,
            "samples": selection.protocol.sample_count,
            "tokens": selection.protocol.sequence_length,
            "blocks": [
                {
                    "block": item.block.index,
                    "final_frozen_pre_kd": item.final_frozen_pre_kd,
                    "final_post_kd": item.final_post_kd,
                    "absolute_delta": item.post_kd_vs_pre_kd.absolute_delta,
                    "relative_delta": item.post_kd_vs_pre_kd.relative_delta,
                }
                for item in metrics
            ],
        }

    if args.device.startswith("cuda"):
        with acquire_device_lease(args.device):
            payload = execute()
    else:
        payload = execute()
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
