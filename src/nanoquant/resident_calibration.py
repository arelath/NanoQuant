"""Resident calibration composition for pinned local Transformers snapshots."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from nanoquant.application.calibration import MaterializedLayerCalibration, calibrate_block
from nanoquant.application.calibration_artifacts import build_objectives, persist_calibration
from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.config.codec import to_dict
from nanoquant.config.schema import ObjectiveConfig, ProfilingConfig, ProfilingLevel
from nanoquant.domain.models import (
    ArtifactRef,
    DatasetIdentity,
    LayerId,
    ModelInventory,
)
from nanoquant.domain.profiling import NULL_RECORDER, PhaseRecorder
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.profiling import profiled_run
from nanoquant.infrastructure.resource_usage import peak_device_memory_bytes
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.tensor_store import LocalTensorStore


@dataclass(frozen=True, slots=True)
class ResidentCalibrationRequest:
    snapshot: Path
    output: Path
    source: str
    revision: str
    token_ids: tuple[tuple[int, ...], ...]
    device: str = "cuda"
    verify_hashes: bool = True
    shrinkage: float = 0.0
    profiling: ProfilingConfig = ProfilingConfig()


@dataclass(frozen=True, slots=True)
class ResidentCalibrationResult:
    inventory: ModelInventory
    calibration: ArtifactRef
    objectives: ArtifactRef
    report: ArtifactRef
    layer_count: int
    total_tokens: int
    maximum_logit_difference: float
    logit_mse: float
    peak_device_bytes: int
    elapsed_seconds: float


def _token_fingerprint(tokens: torch.Tensor) -> str:
    value = tokens.detach().cpu().contiguous()
    return "sha256:" + hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _layers(model: nn.Module) -> tuple[nn.Module, ...]:
    base = getattr(model, "model", None)
    values = getattr(base, "layers", None)
    if not isinstance(values, nn.ModuleList):
        raise TypeError("model does not expose a supported decoder layer stack")
    return tuple(values)


def _checkpoint_dtype(config: dict[str, object]) -> torch.dtype:
    value = config.get("torch_dtype")
    if not isinstance(value, str):
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(value, torch.float32)


def _run_resident_calibration(
    request: ResidentCalibrationRequest,
    recorder: PhaseRecorder,
) -> ResidentCalibrationResult:
    """Calibrate every quantizable layer and validate adapter replay against the full model."""
    started = time.perf_counter()
    if not request.token_ids or any(not row for row in request.token_ids):
        raise ValueError("resident calibration requires non-empty token rows")
    if len({len(row) for row in request.token_ids}) != 1:
        raise ValueError("resident calibration token rows must have equal lengths")
    if request.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA resident calibration requested without CUDA")

    micro_recorder = recorder if request.profiling.level is ProfilingLevel.MICRO else NULL_RECORDER
    artifacts = LocalArtifactStore(request.output / "artifacts", recorder=micro_recorder)
    tensors = LocalTensorStore(artifacts)
    with recorder.phase("source"):
        source = SafetensorsModelSource(
            request.snapshot,
            source=request.source,
            revision=request.revision,
            verify_hashes=request.verify_hashes,
        )
        checkpoint = source.inventory()
        adapter = adapter_for_config(checkpoint.config)
        inventory = adapter.model_inventory(source)
    tokens = torch.tensor(request.token_ids, dtype=torch.long, device=request.device)
    if request.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(request.device)

    with recorder.phase("model_load"):
        model = cast(
            nn.Module,
            AutoModelForCausalLM.from_pretrained(
                request.snapshot,
                local_files_only=True,
                torch_dtype=_checkpoint_dtype(checkpoint.config),
                attn_implementation=adapter.attention_implementation,
            ),
        ).to(request.device)
        model.eval()
        decoder_layers = _layers(model)
        if len(decoder_layers) != len(inventory.blocks):
            raise ValueError("adapter inventory and loaded model disagree on decoder block count")

    with recorder.phase("reference"):
        with torch.no_grad():
            reference_output = cast(Any, model)(input_ids=tokens, use_cache=False)
            reference_logits = cast(torch.Tensor, reference_output.logits).detach()
    with recorder.phase("prefix_capture"):
        capture = capture_prefix_invocations(
            decoder_layers[0],
            (lambda: cast(Any, model)(input_ids=tokens, use_cache=False),),
        )[0]
    hidden = capture.positional[0]
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("captured first-block hidden state is not a tensor")
    metadata = capture.keyword
    materialized: list[tuple[LayerId, MaterializedLayerCalibration]] = []

    for block_inventory in inventory.blocks:
        block_id = block_inventory.block
        with recorder.phase("block", block=block_id.index):
            with recorder.phase("load"):
                block = adapter.load_block(source, block_id, request.device)
                block.eval()
                layer_paths = tuple(layer.path for layer in adapter.quantizable_layers(block, block_id))

            def runner(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
                return adapter.run_block(module, value, **metadata)

            with recorder.phase("calibrate"):
                stats = calibrate_block(
                    block,
                    (hidden,),
                    layer_paths,
                    runner,
                    method="forward_only",
                    shrinkage=request.shrinkage,
                    recorder=micro_recorder,
                )
            materialized.extend((LayerId(block_id, item.path), item) for item in stats)
            recorder.add("calibration.layers", len(stats))
            with recorder.phase("propagate"):
                with torch.no_grad():
                    hidden = runner(block, hidden).detach()
            del block
            recorder.add("calibration.blocks", 1)

    with recorder.phase("suffix"):
        with torch.no_grad():
            replay_logits = adapter.run_suffix(model, hidden).detach()
        difference = (reference_logits.float() - replay_logits.float()).abs()
        maximum_difference = float(difference.max())
        logit_mse = float(difference.square().mean())
    total_tokens = tokens.numel()
    dataset = DatasetIdentity(
        _token_fingerprint(tokens),
        ("deterministic-token-fixture",),
        ("1",),
        checkpoint.tokenizer_hash,
        "raw-token-ids-v1",
    )
    recorder.add("calibration.tokens", total_tokens)
    with recorder.phase("persist_calibration"):
        calibration = persist_calibration(
            tuple(materialized),
            inventory.model,
            dataset,
            "forward_only",
            "float32",
            artifacts,
            tensors,
            total_tokens=total_tokens,
        )
    with recorder.phase("build_objectives"):
        objectives = build_objectives(calibration, ObjectiveConfig(), artifacts)
    peak_device_bytes = peak_device_memory_bytes(request.device)
    elapsed = time.perf_counter() - started
    report_payload = {
        "schema_version": 1,
        "source": request.source,
        "revision": request.revision,
        "model": to_dict(inventory.model),
        "calibration_artifact": calibration.reference.artifact_id,
        "objectives_artifact": objectives.reference.artifact_id,
        "block_count": len(inventory.blocks),
        "layer_count": len(materialized),
        "total_tokens": total_tokens,
        "maximum_logit_difference": maximum_difference,
        "logit_mse": logit_mse,
        "peak_device_bytes": peak_device_bytes,
        "elapsed_seconds": elapsed,
    }
    with recorder.phase("report"):
        with artifacts.begin_write("resident-calibration-report") as writer:
            (writer.path / "report.json").write_text(
                json.dumps(report_payload, sort_keys=True, indent=2), encoding="utf-8"
            )
            descriptor = writer.commit()
    report = ArtifactRef("resident-calibration-report", descriptor.artifact_id, descriptor.schema_version)
    return ResidentCalibrationResult(
        inventory,
        calibration.reference,
        objectives.reference,
        report,
        len(materialized),
        total_tokens,
        maximum_difference,
        logit_mse,
        peak_device_bytes,
        elapsed,
    )


def run_resident_calibration(request: ResidentCalibrationRequest) -> ResidentCalibrationResult:
    def execute() -> ResidentCalibrationResult:
        with profiled_run(
            request.profiling,
            request.output,
            None,
            run_id="resident-calibration",
        ) as recorder:
            with recorder.phase("run"):
                return _run_resident_calibration(request, recorder)

    if request.device.startswith("cuda"):
        with acquire_device_lease(request.device):
            return execute()
    return execute()
