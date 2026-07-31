from __future__ import annotations

import pytest
import torch

from nanoquant.infrastructure.probe_reconstruction_cache import (
    ProbeReconstructionCache,
    ProbeReconstructionCacheEntry,
)


def _entry() -> ProbeReconstructionCacheEntry:
    return ProbeReconstructionCacheEntry(
        baseline=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        candidate=torch.tensor([[0.5, 1.5], [2.5, 3.5]]),
        baseline_metrics={"weighted_normalized_rmse": 0.5},
        candidate_metrics={"weighted_normalized_rmse": 0.4},
        candidate_index_metrics={"right": {"used_entries": 2}},
        rank=2,
        matrix_shape=(2, 2),
    )


def test_probe_reconstruction_cache_round_trips(tmp_path) -> None:
    cache = ProbeReconstructionCache(tmp_path)
    identity = {"block": 2, "projection": "gate", "version": 1}

    cache_key = cache.store(identity, _entry())
    loaded = cache.load(identity)

    assert cache_key == cache.key(identity)
    assert loaded is not None
    assert torch.equal(loaded.baseline, _entry().baseline.to(torch.bfloat16))
    assert torch.equal(loaded.candidate, _entry().candidate.to(torch.bfloat16))
    assert loaded.baseline_metrics == {
        "weighted_normalized_rmse": 0.5
    }
    assert cache.load({**identity, "block": 3}) is None


def test_probe_reconstruction_cache_rejects_corruption(tmp_path) -> None:
    cache = ProbeReconstructionCache(tmp_path)
    identity = {"block": 2, "projection": "gate", "version": 1}
    cache.store(identity, _entry())
    tensor_path = (
        tmp_path
        / cache.key(identity)
        / "reconstructions.safetensors"
    )
    content = tensor_path.read_bytes()
    tensor_path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))

    with pytest.raises(ValueError, match="identity or hash"):
        cache.load(identity)


def test_probe_reconstruction_cache_rejects_invalid_entry() -> None:
    with pytest.raises(ValueError, match="entry is invalid"):
        ProbeReconstructionCacheEntry(
            baseline=torch.ones((2, 2)),
            candidate=torch.ones((2, 3)),
            baseline_metrics={},
            candidate_metrics={},
            candidate_index_metrics={},
            rank=2,
            matrix_shape=(2, 2),
        )
