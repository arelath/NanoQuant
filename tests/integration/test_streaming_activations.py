from pathlib import Path

import torch

from nanoquant.application.streaming_activations import DoubleBufferedActivationPropagator
from nanoquant.infrastructure.activation_store import MmapActivationStore
from nanoquant.infrastructure.resident_executor import ResidentExecutor


def test_double_buffered_mmap_propagation_is_batch_bounded_and_equivalent(tmp_path: Path) -> None:
    source = MmapActivationStore(tmp_path / "source")
    destination = MmapActivationStore(tmp_path / "destination")
    inputs = torch.arange(30, dtype=torch.float32).reshape(5, 3, 2)
    source.put("inputs", inputs)
    seen_pointers: list[int] = []
    seen_rows: list[int] = []

    def forward(batch: torch.Tensor) -> torch.Tensor:
        seen_pointers.append(batch.untyped_storage().data_ptr())
        seen_rows.append(batch.shape[0])
        return batch * 2 + 1

    with destination.begin_generation("outputs", tuple(inputs.shape), inputs.dtype) as writer:
        metrics = DoubleBufferedActivationPropagator(ResidentExecutor(), "cpu", batch_size=2).propagate(
            source, "inputs", writer, forward
        )
        writer.commit()

    assert metrics.batch_count == 3
    assert metrics.maximum_staged_rows == 2
    assert seen_rows == [2, 2, 1]
    assert seen_pointers[0] == seen_pointers[2]
    assert seen_pointers[0] != seen_pointers[1]
    assert metrics.bytes_read == inputs.numel() * inputs.element_size()
    with destination.read("outputs") as outputs:
        assert torch.equal(outputs, inputs * 2 + 1)
