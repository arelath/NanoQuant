from __future__ import annotations

from pathlib import Path

import torch

from nanoquant.application.foldable_mlp_multipliers import InstalledMultipliers
from nanoquant.foldable_mlp_tuning import (
    _checkpoint_state,
    _learning_rate,
    _restore_checkpoint,
)


def _installed() -> InstalledMultipliers:
    first = torch.nn.Parameter(torch.tensor([0.1, -0.2]))
    second = torch.nn.Parameter(torch.tensor([0.3]))
    return InstalledMultipliers({}, {"first": (first,), "second": (second,)})


def test_foldable_mlp_checkpoint_restores_parameters_optimizer_and_progress(
    tmp_path: Path,
) -> None:
    installed = _installed()
    optimizer = torch.optim.AdamW(installed.parameters, lr=1e-4)
    loss = sum(parameter.square().sum() for parameter in installed.parameters)
    loss.backward()
    optimizer.step()
    expected_values = tuple(parameter.detach().clone() for parameter in installed.parameters)
    expected_states = tuple(
        (
            optimizer.state[parameter]["step"].detach().clone(),
            optimizer.state[parameter]["exp_avg"].detach().clone(),
            optimizer.state[parameter]["exp_avg_sq"].detach().clone(),
        )
        for parameter in installed.parameters
    )
    _checkpoint_state(
        installed,
        optimizer,
        protocol_hash="sha256:protocol",
        completed_steps=1,
        losses=[1.25],
        destination=tmp_path / "checkpoint",
    )

    restored = _installed()
    restored_optimizer = torch.optim.AdamW(restored.parameters, lr=1e-4)
    completed, losses = _restore_checkpoint(
        restored,
        restored_optimizer,
        protocol_hash="sha256:protocol",
        source=tmp_path / "checkpoint",
    )

    assert completed == 1
    assert losses == [1.25]
    for parameter, expected_value, expected_state in zip(
        restored.parameters,
        expected_values,
        expected_states,
        strict=True,
    ):
        torch.testing.assert_close(parameter, expected_value)
        actual = restored_optimizer.state[parameter]
        torch.testing.assert_close(actual["step"], expected_state[0])
        torch.testing.assert_close(actual["exp_avg"], expected_state[1])
        torch.testing.assert_close(actual["exp_avg_sq"], expected_state[2])


def test_foldable_mlp_cosine_schedule_reaches_zero() -> None:
    assert _learning_rate(1e-4, 0, 64) == 1e-4
    assert _learning_rate(1e-4, 64, 64) == 0.0
