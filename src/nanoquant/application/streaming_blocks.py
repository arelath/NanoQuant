"""One-block-at-a-time teacher and working execution over activation stores."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from nanoquant.application.streaming_activations import (
    DoubleBufferedActivationPropagator,
    PropagationMetrics,
)
from nanoquant.domain.models import BlockId
from nanoquant.infrastructure.activation_store import MmapActivationStore
from nanoquant.ports.activation_store import ActivationStore
from nanoquant.ports.executor import Executor
from nanoquant.ports.model_adapter import ModelAdapter
from nanoquant.ports.model_source import ModelSource


@dataclass(frozen=True, slots=True)
class StreamingBlockRequest:
    block: BlockId
    teacher_input_key: str
    compressed_input_key: str
    teacher_output_key: str
    compressed_output_key: str
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class StreamingBlockResult:
    teacher: PropagationMetrics
    compressed: PropagationMetrics
    source_block_bytes: int
    working_block_bytes: int


def _module_bytes(module: nn.Module) -> int:
    return sum(value.numel() * value.element_size() for value in (*module.parameters(), *module.buffers()))


def _block_forward(
    adapter: ModelAdapter, block: nn.Module, metadata: dict[str, object]
) -> Callable[[torch.Tensor], torch.Tensor]:
    def forward(batch: torch.Tensor) -> torch.Tensor:
        return adapter.run_block(block, batch, **metadata)

    return forward


class StreamingBlockExecutor:
    def __init__(self, executor: Executor, device: str, batch_size: int) -> None:
        self.propagator = DoubleBufferedActivationPropagator(executor, device, batch_size)
        self.device = device

    def execute(
        self,
        request: StreamingBlockRequest,
        adapter: ModelAdapter,
        source: ModelSource,
        inputs: ActivationStore,
        outputs: MmapActivationStore,
        prepare_working: Callable[[nn.Module], nn.Module],
    ) -> StreamingBlockResult:
        with inputs.read(request.teacher_input_key, self.device) as teacher_inputs:
            output_shape = tuple(teacher_inputs.shape)
            output_dtype = teacher_inputs.dtype
        source_block = adapter.load_block(source, request.block, self.device)
        source_block.eval()
        source_bytes = _module_bytes(source_block)
        teacher_forward = _block_forward(adapter, source_block, request.metadata)
        with outputs.begin_generation(request.teacher_output_key, output_shape, output_dtype) as writer:
            teacher_metrics = self.propagator.propagate(
                inputs,
                request.teacher_input_key,
                writer,
                teacher_forward,
            )
            writer.commit()
        del teacher_forward
        del source_block

        working_block = prepare_working(adapter.load_block(source, request.block, self.device))
        working_block.eval()
        working_bytes = _module_bytes(working_block)
        compressed_forward = _block_forward(adapter, working_block, request.metadata)
        with outputs.begin_generation(request.compressed_output_key, output_shape, output_dtype) as writer:
            compressed_metrics = self.propagator.propagate(
                inputs,
                request.compressed_input_key,
                writer,
                compressed_forward,
            )
            writer.commit()
        del compressed_forward
        del working_block
        return StreamingBlockResult(teacher_metrics, compressed_metrics, source_bytes, working_bytes)
