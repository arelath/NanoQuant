"""Bounded GPU residency and pinned-host prefetch for immutable sensitivity targets."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import torch

from nanoquant.infrastructure.kl_teacher_dataset import KlTeacherDataset


def cache_teacher_targets(dataset: KlTeacherDataset, device: str, gpu_bytes: int) -> KlTeacherDataset:
    """Cache a prefix in VRAM and pin remaining batches for asynchronous copies."""
    if gpu_bytes <= 0 or not device.startswith("cuda"):
        return dataset
    batches = []
    remaining = gpu_bytes
    for batch in dataset.cache.batches:
        cost = batch.numel() * batch.element_size()
        if cost <= remaining:
            batches.append(batch.to(device))
            remaining -= cost
        else:
            batches.append(batch.pin_memory())
    cache = replace(dataset.cache, batches=tuple(batches))
    return replace(dataset, cache=cache)


def teacher_batches(dataset: KlTeacherDataset, device: str) -> Iterator[torch.Tensor]:
    """Copy the next pinned batch while the current window executes."""
    batches = dataset.cache.batches
    if not device.startswith("cuda") or not any(batch.is_pinned() for batch in batches):
        yield from batches
        return
    stream = torch.cuda.Stream(device=device)

    def transfer(index: int) -> torch.Tensor:
        with torch.cuda.stream(stream):
            return batches[index].to(device, non_blocking=True)

    current = transfer(0)
    for index in range(len(batches)):
        main = torch.cuda.current_stream(device)
        main.wait_stream(stream)
        current.record_stream(main)
        following = transfer(index + 1) if index + 1 < len(batches) else None
        yield current
        if following is not None:
            current = following
