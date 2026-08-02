from __future__ import annotations

import os

import pytest
import torch

from nanoquant.infrastructure.reproducibility import deterministic_torch_execution


def test_deterministic_torch_execution_replays_rng_and_restores_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    original = torch.are_deterministic_algorithms_enabled()
    observed = []
    for _attempt in range(2):
        with deterministic_torch_execution(17, "cpu"):
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.backends.cudnn.deterministic
            assert not torch.backends.cudnn.benchmark
            observed.append(torch.rand(4))

    assert torch.equal(observed[0], observed[1])
    assert torch.are_deterministic_algorithms_enabled() is original
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_deterministic_torch_execution_rejects_incompatible_cublas_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        with deterministic_torch_execution(0, "cpu"):
            raise AssertionError("unreachable")
