"""Bounded double-buffered propagation between activation generations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from nanoquant.ports.activation_store import ActivationStore, BatchGenerationWriter
from nanoquant.ports.executor import Executor


@dataclass(frozen=True, slots=True)
class PropagationMetrics:
    batch_count: int
    maximum_staged_rows: int
    bytes_read: int
    bytes_written: int


class DoubleBufferedActivationPropagator:
    def __init__(self, executor: Executor, device: str, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("activation propagation batch size must be positive")
        self.executor = executor
        self.device = device
        self.batch_size = batch_size

    def propagate(
        self,
        source: ActivationStore,
        source_key: str,
        destination: BatchGenerationWriter,
        forward: Callable[[torch.Tensor], torch.Tensor],
    ) -> PropagationMetrics:
        batches = 0
        maximum_rows = 0
        bytes_read = 0
        bytes_written = 0
        with source.read(source_key, self.device) as values:
            if values.ndim == 0 or values.shape[0] == 0:
                raise ValueError("activation generation must have a non-empty batch dimension")
            row_shape = tuple(values.shape[1:])
            for start in range(0, values.shape[0], self.batch_size):
                stop = min(start + self.batch_size, values.shape[0])
                rows = stop - start
                slot = batches % 2
                staging = self.executor.buffer(
                    f"activation-stage-{slot}",
                    (self.batch_size, *row_shape),
                    values.dtype,
                    self.device,
                )
                staged = staging[:rows]
                staged.copy_(values[start:stop], non_blocking=values.is_pinned())
                with torch.no_grad():
                    output = forward(staged)
                if output.shape[0] != rows:
                    raise ValueError("activation forward changed the batch dimension")
                destination.write(slice(start, stop), output)
                batches += 1
                maximum_rows = max(maximum_rows, rows)
                bytes_read += staged.numel() * staged.element_size()
                bytes_written += output.numel() * output.element_size()
        return PropagationMetrics(batches, maximum_rows, bytes_read, bytes_written)
