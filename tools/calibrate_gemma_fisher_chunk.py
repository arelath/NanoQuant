"""Advance pinned Gemma online-Fisher calibration by one resumable sample slice."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from nanoquant.application.calibration import CausalOnlineCalibrationState, calibrate_causal_model
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.calibration_checkpoint import (
    load_causal_calibration_state,
    save_causal_calibration_state,
)
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.infrastructure.resource_usage import peak_process_memory_bytes

CALIBRATION_ARTIFACT = "sha256-ad1f609729f86db7598eed5c703c55aacbb9cb024cab816ca7b300d574b7a4c8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path("evidence/m3/experiment018-calibration"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("calibration chunk count must be positive")

    calibration = load_pinned_calibration(
        args.calibration,
        ArtifactRef("calibration-dataset-manifest", CALIBRATION_ARTIFACT, 1),
    )
    state = load_causal_calibration_state(args.state) if (args.state / "manifest.json").exists() else None
    start = 0 if state is None else state.sample_count
    end = min(start + args.count, calibration.input_ids.shape[0])
    if start >= end:
        print(json.dumps({"status": "complete", "sample_count": start}, indent=2))
        return

    started = time.perf_counter()
    with acquire_device_lease(args.device):
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        model = AutoModelForCausalLM.from_pretrained(
            args.snapshot,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            # Experiment 018 forces eager Gemma attention. SDPA changes both
            # forward rounding and the Fisher gradients accumulated here.
            attn_implementation="eager",
        ).to(args.device)
        model.eval()
        layers = tuple(
            (f"block.{index}.{name}", module)
            for index, block in enumerate(model.model.layers)
            for name, module in block.named_modules()
            if isinstance(module, nn.Linear)
        )
        updated: list[CausalOnlineCalibrationState] = []
        calibrate_causal_model(
            model,
            tuple(calibration.input_ids[index : index + 1].to(args.device) for index in range(start, end)),
            layers,
            initial_state=state,
            state_sink=updated.append,
        )
        save_causal_calibration_state(args.state, updated[-1])
        peak_device = torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else 0
    print(
        json.dumps(
            {
                "status": "complete" if end == calibration.input_ids.shape[0] else "partial",
                "start": start,
                "end": end,
                "sample_count": updated[-1].sample_count,
                "peak_device_bytes": peak_device,
                "peak_host_bytes": peak_process_memory_bytes(),
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
