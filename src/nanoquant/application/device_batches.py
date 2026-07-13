"""Order-preserving pinned-host batch prefetch for CUDA consumers."""

from __future__ import annotations

from collections.abc import Iterator

import torch


def iter_device_batches(
    values: tuple[torch.Tensor, ...],
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, ...]]:
    if not values:
        raise ValueError("device batch iterator requires at least one tensor")
    if values[0].shape[0] == 0:
        return
    if (
        device.type != "cuda"
        or any(value.device.type != "cpu" or not value.is_pinned() for value in values)
    ):
        for start in range(0, values[0].shape[0], batch_size):
            end = min(start + batch_size, values[0].shape[0])
            yield tuple(value[start:end].to(device, non_blocking=True) for value in values)
        return

    copy_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
    compute_stream = torch.cuda.current_stream(device)
    ready_events = tuple(torch.cuda.Event() for _ in range(2))  # type: ignore[no-untyped-call]

    def schedule(start: int, slot: int) -> tuple[tuple[torch.Tensor, ...], torch.cuda.Event]:
        end = min(start + batch_size, values[0].shape[0])
        with torch.cuda.stream(copy_stream):
            batches = tuple(value[start:end].to(device, non_blocking=True) for value in values)
            ready = ready_events[slot]
            ready.record(copy_stream)
        return batches, ready

    current = schedule(0, 0)
    next_slot = 1
    try:
        for next_start in range(batch_size, values[0].shape[0] + batch_size, batch_size):
            batches, ready = current
            compute_stream.wait_event(ready)
            for batch in batches:
                batch.record_stream(compute_stream)
            following = schedule(next_start, next_slot) if next_start < values[0].shape[0] else None
            next_slot = (next_slot + 1) % len(ready_events)
            yield batches
            if following is None:
                break
            current = following
    finally:
        compute_stream.wait_stream(copy_stream)
