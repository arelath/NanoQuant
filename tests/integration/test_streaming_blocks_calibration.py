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


class IndexedLinearBlock(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        matrices = (
            torch.tensor(
                (
                    (1.0, 1.0, 0.0, 0.0),
                    (0.0, 1.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0, 1.0),
                    (1.0, 0.0, 0.0, 1.0),
                )
            ),
            torch.tensor(
                (
                    (1.0, 0.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0, 1.0),
                    (1.0, 0.0, 0.0, 1.0),
                    (0.0, 1.0, 1.0, 0.0),
                )
            ),
        )
        self.index = index
        self.linear = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(matrices[index])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


class IndexedAdapter:
    def __init__(self) -> None:
        self.loads: list[int] = []

    def load_block(self, source: object, block: BlockId, device: str) -> nn.Module:
        del source
        self.loads.append(block.index)
        return IndexedLinearBlock(block.index).to(device)

    def run_block(self, block: nn.Module, inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
        scale = float(kwargs.get("output_scale", 1.0))
        return block(inputs) * scale


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


def test_tiny_multiblock_streaming_matches_resident_execution_with_mmap_boundaries(tmp_path: Path) -> None:
    values = torch.arange(56, dtype=torch.float32).reshape(7, 2, 4) / 4
    metadata = ({"output_scale": 0.5}, {"output_scale": 2.0})

    def prepare_working(block: nn.Module) -> nn.Module:
        with torch.no_grad():
            cast(IndexedLinearBlock, block).linear.weight.mul_(0.5)
        return block

    resident_adapter = IndexedAdapter()
    resident_teacher = values.clone()
    resident_compressed = values.clone()
    for index in range(2):
        source_block = resident_adapter.load_block(object(), BlockId(index), "cpu")
        resident_teacher = resident_adapter.run_block(source_block, resident_teacher, **metadata[index])
        working_block = prepare_working(
            resident_adapter.load_block(object(), BlockId(index), "cpu")
        )
        resident_compressed = resident_adapter.run_block(
            working_block,
            resident_compressed,
            **metadata[index],
        )

    activations = MmapActivationStore(tmp_path / "multiblock-activations")
    activations.put("teacher-0", values)
    activations.put("compressed-0", values)
    streaming_adapter = IndexedAdapter()
    executor = StreamingBlockExecutor(ResidentExecutor(), "cpu", batch_size=3)
    results = []
    for index in range(2):
        results.append(
            executor.execute(
                StreamingBlockRequest(
                    BlockId(index),
                    f"teacher-{index}",
                    f"compressed-{index}",
                    f"teacher-{index + 1}",
                    f"compressed-{index + 1}",
                    metadata[index],
                ),
                cast(ModelAdapter, streaming_adapter),
                cast(ModelSource, object()),
                activations,
                activations,
                prepare_working,
            )
        )

    with (
        activations.read("teacher-2") as streamed_teacher,
        activations.read("compressed-2") as streamed_compressed,
    ):
        assert torch.equal(streamed_teacher, resident_teacher)
        assert torch.equal(streamed_compressed, resident_compressed)
        assert not torch.equal(streamed_teacher, streamed_compressed)
    assert resident_adapter.loads == streaming_adapter.loads == [0, 0, 1, 1]
    assert all(result.teacher.batch_count == result.compressed.batch_count == 3 for result in results)
    assert all(
        result.teacher.maximum_staged_rows == result.compressed.maximum_staged_rows == 3
        for result in results
    )
    expected_bytes = values.numel() * values.element_size()
    assert all(result.teacher.bytes_read == result.teacher.bytes_written == expected_bytes for result in results)
    assert all(
        result.compressed.bytes_read == result.compressed.bytes_written == expected_bytes
        for result in results
    )
