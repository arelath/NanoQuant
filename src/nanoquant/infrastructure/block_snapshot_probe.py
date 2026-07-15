"""Memory-bounded hidden-state probes for model-level block comparisons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

BlockForward = Callable[[nn.Module, torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class BlockOutputReference:
    outputs: tuple[tuple[torch.Tensor, ...], ...]
    block_count: int


def _output_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError("decoder block did not return a tensor or tensor-first tuple")


def _capture_one(
    model: nn.Module,
    blocks: tuple[nn.Module, ...],
    token_ids: torch.Tensor,
    forward: BlockForward,
) -> tuple[torch.Tensor, ...]:
    captured: list[torch.Tensor] = []
    handles = [
        block.register_forward_hook(
            lambda _module, _inputs, output: captured.append(
                _output_tensor(output).detach().to(device="cpu", dtype=torch.bfloat16)
            )
        )
        for block in blocks
    ]
    try:
        with torch.no_grad():
            forward(model, token_ids)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(blocks):
        raise ValueError(f"block snapshot captured {len(captured)} outputs for {len(blocks)} blocks")
    return tuple(captured)


def capture_block_output_reference(
    model: nn.Module,
    blocks: tuple[nn.Module, ...],
    token_ids: torch.Tensor,
    forward: BlockForward,
    *,
    device: str | torch.device,
) -> BlockOutputReference:
    if not blocks or token_ids.ndim != 2 or token_ids.shape[0] <= 0:
        raise ValueError("block output reference requires blocks and non-empty rank-two tokens")
    prior_training = model.training
    model.eval()
    try:
        outputs = tuple(
            _capture_one(model, blocks, token_ids[index : index + 1].to(device), forward)
            for index in range(token_ids.shape[0])
        )
    finally:
        model.train(prior_training)
    return BlockOutputReference(outputs, len(blocks))


def measure_block_output_mse(
    model: nn.Module,
    blocks: tuple[nn.Module, ...],
    token_ids: torch.Tensor,
    reference: BlockOutputReference,
    forward: BlockForward,
    *,
    device: str | torch.device,
    pad_token_id: int | None = None,
) -> tuple[float, ...]:
    if len(blocks) != reference.block_count or len(reference.outputs) != token_ids.shape[0]:
        raise ValueError("block snapshot model/reference geometry differs")
    sums = [0.0] * len(blocks)
    counts = [0] * len(blocks)
    prior_training = model.training
    model.eval()
    try:
        for sample_index in range(token_ids.shape[0]):
            sample = token_ids[sample_index : sample_index + 1]
            observed = _capture_one(model, blocks, sample.to(device), forward)
            mask = torch.ones_like(sample, dtype=torch.bool) if pad_token_id is None else sample.ne(pad_token_id)
            if not bool(mask.any()):
                continue
            for block_index, (candidate, baseline) in enumerate(
                zip(observed, reference.outputs[sample_index], strict=True)
            ):
                if candidate.shape != baseline.shape or candidate.shape[:2] != mask.shape:
                    raise ValueError("block snapshot output/reference shape differs")
                difference = candidate[mask].float() - baseline[mask].float()
                sums[block_index] += float(difference.square().sum())
                counts[block_index] += difference.numel()
    finally:
        model.train(prior_training)
    if any(count == 0 for count in counts):
        raise ValueError("block snapshot probe contains no non-padding output elements")
    return tuple(total / count for total, count in zip(sums, counts, strict=True))
