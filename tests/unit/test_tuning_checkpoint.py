import json
from dataclasses import replace
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


def test_tuning_checkpoint_roundtrips_legacy_mode_without_best_state(tmp_path: Path) -> None:
    state = replace(
        _state(2),
        epoch_losses=(None, 3.0, 2.9),
        best_parameter_values=(),
        best_epoch=1,
    )

    stored = save_tuning_checkpoint(tmp_path, state, _identity())
    loaded = active_tuning_checkpoint(tmp_path, _identity())

    assert loaded is not None
    assert loaded.state.epoch_losses == (None, 3.0, 2.9)
    assert loaded.state.best_parameter_values == ()
    manifest = json.loads(
        (
            tmp_path
            / "state"
            / "tuning-checkpoint"
            / stored.generation
            / "checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 2
    assert manifest["has_best_state"] is False


def test_tuning_checkpoint_reads_v1_generation(tmp_path: Path) -> None:
    stored = save_tuning_checkpoint(tmp_path, _state(1), _identity())
    manifest_path = (
        tmp_path
        / "state"
        / "tuning-checkpoint"
        / stored.generation
        / "checkpoint.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest.pop("has_best_state")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = active_tuning_checkpoint(tmp_path, _identity())

    assert loaded is not None
    assert loaded.state.completed_epochs == stored.state.completed_epochs
    assert loaded.state.epoch_losses == stored.state.epoch_losses
    assert loaded.state.steps_completed == stored.state.steps_completed
    assert loaded.state.best_epoch == stored.state.best_epoch
    assert loaded.state.stopped_early == stored.state.stopped_early
    assert [name for name, _value in loaded.state.parameter_values] == [
        name for name, _value in stored.state.parameter_values
    ]
    assert torch.equal(loaded.state.parameter_values[0][1], stored.state.parameter_values[0][1])
    assert torch.equal(
        loaded.state.best_parameter_values[0][1],
        stored.state.best_parameter_values[0][1],
    )
