from pathlib import Path

import torch
from safetensors.torch import save_file

import tools.replay_gemma_gate_tuning as replay
from nanoquant.infrastructure.device_lease import DeviceLeaseError
from tools.replay_gemma_gate_tuning import (
    _acquire_or_wait,
    _comparison,
    _legacy_initial,
    _rewrite_initial,
    _rewrite_pre_scale_fit,
)


def test_initial_state_loaders_map_legacy_and_rewrite_names(tmp_path: Path) -> None:
    left = torch.tensor([[1.0], [-1.0]])
    right = torch.tensor([[1.0, -1.0]])
    scales = {
        "scale_pre": torch.ones(2),
        "scale_mid": torch.ones(1),
        "scale_post": torch.ones(2),
    }
    outliers = {
        "outlier_indices": torch.tensor([1]),
        "outlier_values": torch.tensor([[2.0], [3.0]]),
    }
    legacy_path = tmp_path / "legacy.safetensors"
    factor_path = tmp_path / "factor.safetensors"
    scale_path = tmp_path / "scale.safetensors"
    frozen_path = tmp_path / "frozen.safetensors"
    save_file(
        {
            "U_latent": left,
            "V_latent": right,
            **scales,
            "salient_idx": outliers["outlier_indices"].int(),
            "salient_weight": outliers["outlier_values"],
        },
        legacy_path,
    )
    save_file({"left_latent": left, "right_latent": right, **scales}, factor_path)
    save_file(scales, scale_path)
    save_file(outliers, frozen_path)

    legacy = _legacy_initial(legacy_path)
    rewrite = _rewrite_initial(factor_path, scale_path, frozen_path)
    rewrite_pre_fit = _rewrite_pre_scale_fit(factor_path, frozen_path)
    comparison = _comparison(legacy, rewrite)

    assert comparison["outlier_indices_exact"] is True
    assert comparison["left"]["agreement"] == 1.0  # type: ignore[index]
    assert torch.equal(legacy["outlier_indices"], rewrite["outlier_indices"])
    assert torch.equal(rewrite_pre_fit["scale_mid"], scales["scale_mid"])


def test_acquire_or_wait_retries_without_bypassing_lease(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    attempts = 0
    sleeps: list[float] = []

    class Lease:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    def acquire(_device: str) -> Lease:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DeviceLeaseError("busy")
        return Lease()

    monotonic = iter((0.0, 1.0))
    monkeypatch.setattr(replay, "acquire_device_lease", acquire)
    monkeypatch.setattr(replay.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(replay.time, "sleep", sleeps.append)

    with _acquire_or_wait("cuda", 10.0, poll_seconds=2.0):
        pass

    assert attempts == 2
    assert sleeps == [2.0]
