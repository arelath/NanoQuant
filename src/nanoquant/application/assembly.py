"""Assemble a frozen logical model solely from committed block artifacts."""

from __future__ import annotations

from nanoquant.domain.models import ArtifactRef, BlockResult, FrozenModelResult, ModelIdentity, TensorRef


def assemble_frozen_model(
    model: ModelIdentity,
    plan: ArtifactRef,
    blocks: tuple[tuple[ArtifactRef, BlockResult], ...],
    shared_tensors: tuple[TensorRef, ...],
    original_quantized_elements: int,
    global_tuning: ArtifactRef | None = None,
) -> FrozenModelResult:
    if original_quantized_elements <= 0:
        raise ValueError("original quantized element count must be positive")
    indexes = [result.block.index for _, result in blocks]
    if indexes != list(range(len(blocks))):
        raise ValueError(f"block results are not complete and contiguous: {indexes}")
    total_bits = sum(layer.actual_bit_cost.total for _, block in blocks for layer in block.layers)
    return FrozenModelResult(
        1,
        model,
        plan,
        tuple(reference for reference, _ in blocks),
        shared_tensors,
        global_tuning,
        total_bits,
        total_bits / original_quantized_elements,
    )
