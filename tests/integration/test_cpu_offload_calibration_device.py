from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from nanoquant.config.schema import ADMMConfig, ExecutorKind, ProfilingConfig, ProfilingLevel
from nanoquant.resident_quantization import ResidentQuantizationRequest, run_resident_quantization


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA offload placement requires a GPU")
def test_cpu_offload_forward_calibration_streams_each_source_block_to_cuda(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    Gemma3ForCausalLM(
        Gemma3TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
    ).save_pretrained(snapshot, safe_serialization=True)
    output = tmp_path / "run"

    run_resident_quantization(
        ResidentQuantizationRequest(
            snapshot,
            output,
            "fixture/gemma3",
            "pinned-test-revision",
            ((1, 2, 3, 4),),
            device="cuda",
            executor=ExecutorKind.CPU_OFFLOAD,
            target_bpw=8.0,
            rank_multiple=1,
            admm=ADMMConfig(outer_iterations=1, inner_iterations=1),
            restore_completed_blocks=False,
            evaluate_inline_quality=False,
            profiling=ProfilingConfig(level=ProfilingLevel.OFF),
        )
    )

    events = tuple(json.loads(line) for line in (output / "events.jsonl").read_text().splitlines())
    preparation = next(event for event in events if event["name"] == "calibration_block_prepare.started")
    calibration = next(event for event in events if event["name"] == "calibration_block.started")
    assert preparation["fields"]["device"] == "cuda"
    assert calibration["fields"]["device"] == "cuda:0"
    assert calibration["fields"]["streamed_source"] is True
