from pathlib import Path
from typing import Any, cast
from weakref import ReferenceType, ref

import torch
from torch import nn

from nanoquant.application.calibration import calibrate_block, calibrate_block_streamed
from nanoquant.application.streaming_blocks import StreamingBlockExecutor, StreamingBlockRequest
from nanoquant.domain.models import BlockId
from nanoquant.infrastructure.activation_store import MmapActivationStore
from nanoquant.infrastructure.resident_executor import ResidentExecutor
from nanoquant.ports.model_adapter import ModelAdapter
from nanoquant.ports.model_source import ModelSource


class LinearBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(4) * 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


class TrackingAdapter:
    def __init__(self) -> None:
        self.loads = 0
        self.source_reference: ReferenceType[nn.Module] | None = None

    def load_block(self, source: object, block: BlockId, device: str) -> nn.Module:
        del source, block
        if self.loads == 1:
            assert self.source_reference is not None and self.source_reference() is None
        value = LinearBlock().to(device)
        self.loads += 1
        if self.loads == 1:
            self.source_reference = ref(value)
        return value

    def run_block(self, block: nn.Module, inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return block(inputs)


def test_streaming_block_releases_teacher_before_working_and_commits_generations(tmp_path: Path) -> None:
    inputs = MmapActivationStore(tmp_path / "inputs")
    outputs = MmapActivationStore(tmp_path / "outputs")
    values = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    inputs.put("teacher-input", values)
    inputs.put("compressed-input", values + 1)
    adapter = TrackingAdapter()

    def prepare(block: nn.Module) -> nn.Module:
        with torch.no_grad():
            cast(LinearBlock, block).linear.weight.mul_(0.5)
        return block

    result = StreamingBlockExecutor(ResidentExecutor(), "cpu", batch_size=2).execute(
        StreamingBlockRequest(
            BlockId(0),
            "teacher-input",
            "compressed-input",
            "teacher-output",
            "compressed-output",
            {},
        ),
        cast(ModelAdapter, adapter),
        cast(ModelSource, object()),
        inputs,
        outputs,
        prepare,
    )

    assert adapter.loads == 2
    assert result.teacher.batch_count == result.compressed.batch_count == 3
    with outputs.read("teacher-output") as teacher:
        assert torch.equal(teacher, values * 2)
    with outputs.read("compressed-output") as compressed:
        assert torch.equal(compressed, values + 1)


def test_forward_only_streamed_calibration_matches_in_memory_batches(tmp_path: Path) -> None:
    store = MmapActivationStore(tmp_path / "activations")
    values = torch.arange(40, dtype=torch.float32).reshape(5, 2, 4) / 10
    store.put("block-inputs", values)
    block = LinearBlock()

    def runner(module: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        return cast(Any, module)(batch)

    streamed = calibrate_block_streamed(
        block,
        store,
        "block-inputs",
        ("linear",),
        runner,
        batch_size=2,
    )
    direct = calibrate_block(
        block,
        (values[:2], values[2:4], values[4:]),
        ("linear",),
        runner,
        method="forward_only",
    )

    assert streamed[0].sample_count == 5
    assert torch.equal(streamed[0].input_importance, direct[0].input_importance)
    assert torch.equal(streamed[0].output_importance, direct[0].output_importance)
