import pytest
import torch

from nanoquant.infrastructure.resource_usage import (
    peak_device_memory_bytes,
    peak_process_memory_bytes,
    process_memory_snapshot,
)


def test_process_memory_snapshot_reports_current_and_peak_working_set() -> None:
    snapshot = process_memory_snapshot()

    assert snapshot.working_set_bytes > 0
    assert snapshot.peak_working_set_bytes >= snapshot.working_set_bytes
    assert snapshot.private_bytes >= 0
    assert snapshot.peak_private_bytes >= snapshot.private_bytes
    assert peak_process_memory_bytes() >= snapshot.peak_working_set_bytes


def test_peak_device_memory_uses_reserved_allocator_high_water(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 6_000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 9_000)

    assert peak_device_memory_bytes("cuda:0") == 9_000
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 4_000)
    assert peak_device_memory_bytes("cuda:0") == 6_000
    assert peak_device_memory_bytes("cpu") == 0
