import os
from pathlib import Path

import pytest
import torch

from nanoquant.domain.resources import ResourceComponents, ResourceMargins
from nanoquant.domain.stages import HostInventory
from nanoquant.infrastructure.activation_store import MmapActivationStore, activation_store_for_plan
from nanoquant.infrastructure.resource_planning import ResourcePlanningRequest, build_resource_plan


def test_constrained_plan_forces_preallocated_batched_mmap_generation(tmp_path: Path) -> None:
    mib = 1024**2
    plan = build_resource_plan(
        ResourcePlanningRequest(
            ResourceComponents(10 * mib, 2 * mib, 1 * mib, 1 * mib, 0, 8 * mib, 0, 2 * mib),
            margins=ResourceMargins(0, 0, 0),
        ),
        HostInventory(4 * mib, 2 * mib, 40 * mib),
    )
    store = activation_store_for_plan(plan, tmp_path / "activations")
    assert isinstance(store, MmapActivationStore)
    expected = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
    with store.begin_generation("compressed-block-1", tuple(expected.shape), expected.dtype) as writer:
        assert writer.temporary.stat().st_size == expected.numel() * expected.element_size()
        writer.write(slice(0, 2), expected[:2])
        writer.write(slice(2, 4), expected[2:])
        content_hash = writer.commit()
    assert content_hash.startswith("sha256:")
    with store.read("compressed-block-1", selection=slice(1, 3)) as batch:
        assert torch.equal(batch, expected[1:3])

    data = next((tmp_path / "activations").glob("*.bin"))
    with data.open("r+b") as output:
        output.seek(0)
        output.write(b"bad!")
    with pytest.raises(OSError, match="ACT001.*corrupt"):
        with store.read("compressed-block-1"):
            pass


def test_incomplete_generation_is_invisible_and_cleanup_removes_orphans(tmp_path: Path) -> None:
    store = MmapActivationStore(tmp_path / "activations")
    writer = store.begin_generation("teacher", (2, 2), torch.float32)
    writer.write(slice(0, 1), torch.ones(1, 2))
    with pytest.raises(ValueError, match="unwritten"):
        writer.commit()
    writer.close()
    with pytest.raises(KeyError, match="not stored"):
        with store.read("teacher"):
            pass
    assert store.cleanup_uncommitted() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sequential_generation_hashes_during_write_without_rereading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    store = MmapActivationStore(tmp_path / "activations")
    expected = torch.arange(12, dtype=torch.float32).reshape(4, 3).to(dtype)

    def reject_reread(_path: Path) -> str:
        raise AssertionError("sequential commit reread its completed data file")

    with monkeypatch.context() as commit_patch:
        commit_patch.setattr("nanoquant.infrastructure.activation_store._hash_file", reject_reread)
        with store.begin_generation("sequential", tuple(expected.shape), dtype) as writer:
            writer.write(slice(0, 2), expected[:2])
            writer.write(slice(2, 4), expected[2:])
            writer.commit()

    with store.read("sequential") as actual:
        assert torch.equal(actual, expected)


def test_out_of_order_generation_falls_back_to_commit_time_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MmapActivationStore(tmp_path / "activations")
    expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    calls = 0

    from nanoquant.infrastructure.activation_store import _hash_file

    def count_reread(path: Path) -> str:
        nonlocal calls
        calls += 1
        return _hash_file(path)

    monkeypatch.setattr("nanoquant.infrastructure.activation_store._hash_file", count_reread)
    with store.begin_generation("out-of-order", tuple(expected.shape)) as writer:
        writer.write(slice(2, 4), expected[2:])
        writer.write(slice(0, 2), expected[:2])
        writer.commit()
    assert calls == 1


def test_disk_full_during_descriptor_commit_leaves_invisible_recoverable_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MmapActivationStore(tmp_path / "activations")
    real_replace = os.replace
    calls = 0

    def fail_descriptor(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        real_replace(source, destination)

    monkeypatch.setattr("nanoquant.infrastructure.activation_store.os.replace", fail_descriptor)
    with store.begin_generation("teacher", (2, 2), torch.float32) as writer:
        writer.write(slice(0, 2), torch.ones(2, 2))
        with pytest.raises(OSError, match="disk full"):
            writer.commit()
    with pytest.raises(KeyError, match="not stored"):
        with store.read("teacher"):
            pass
    assert store.cleanup_uncommitted() >= 1
