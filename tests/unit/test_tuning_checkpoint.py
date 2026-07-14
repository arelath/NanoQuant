from pathlib import Path

import torch

from nanoquant.application.tuning import TuningOptimizerState, TuningResumeState
from nanoquant.infrastructure.tuning_checkpoint import (
    TuningCheckpointIdentity,
    active_tuning_checkpoint,
    clear_tuning_checkpoint,
    save_tuning_checkpoint,
)


def _identity(*, layer: str = "mlp.up_proj") -> TuningCheckpointIdentity:
    return TuningCheckpointIdentity("config", "model", "plan", 3, layer, "factorized")


def _state(epoch: int) -> TuningResumeState:
    value = torch.tensor([float(epoch), 2.0])
    best = torch.tensor([1.0, 2.0])
    step = torch.tensor(epoch * 2, dtype=torch.int64)
    optimizer = TuningOptimizerState(
        "quant.weight",
        step,
        torch.tensor([0.1, 0.2]),
        torch.tensor([0.01, 0.02]),
        torch.tensor([0.001, 0.002]),
    )
    return TuningResumeState(
        epoch,
        tuple([4.0] + [3.0 - index * 0.1 for index in range(epoch)]),
        epoch * 2,
        (("quant.weight", value),),
        (("quant.weight", best),),
        (optimizer,),
        epoch - 1 if epoch else -1,
        False,
    )


def test_tuning_checkpoint_roundtrips_and_bounds_generations(tmp_path: Path) -> None:
    first = save_tuning_checkpoint(tmp_path, _state(1), _identity())
    second = save_tuning_checkpoint(tmp_path, _state(2), _identity())

    loaded = active_tuning_checkpoint(tmp_path, _identity())

    assert loaded is not None
    assert loaded.generation == second.generation
    assert loaded.generation != first.generation
    assert loaded.state.completed_epochs == 2
    assert torch.equal(loaded.state.parameter_values[0][1], torch.tensor([2.0, 2.0]))
    assert torch.equal(
        loaded.state.optimizer_states[0].kahan_compensation,
        torch.tensor([0.001, 0.002]),
    )
    generations = list((tmp_path / "state" / "tuning-checkpoint").glob("generation-*"))
    assert [path.name for path in generations] == [second.generation]


def test_tuning_checkpoint_ignores_other_identity_and_clears(tmp_path: Path) -> None:
    save_tuning_checkpoint(tmp_path, _state(1), _identity())

    assert active_tuning_checkpoint(tmp_path, _identity(layer="mlp.down_proj")) is None

    clear_tuning_checkpoint(tmp_path)
    assert not (tmp_path / "state" / "tuning-checkpoint").exists()
