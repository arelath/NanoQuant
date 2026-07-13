from pathlib import Path

import pytest
import torch

from nanoquant.application.distillation import DistillationOptimizerState, DistillationResumeState
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.distillation_checkpoint import (
    DistillationCheckpointIdentity,
    activate_distillation_checkpoint,
    active_distillation_checkpoint,
    commit_distillation_checkpoint,
    load_distillation_checkpoint,
)


def _state() -> DistillationResumeState:
    value = torch.tensor((1.0, 2.0))
    return DistillationResumeState(
        2,
        (1.5, 1.25),
        6,
        (("scale", value),),
        (
            DistillationOptimizerState(
                "scale",
                torch.tensor(6.0),
                torch.tensor((0.1, 0.2)),
                torch.tensor((0.01, 0.04)),
                torch.tensor((0.001, 0.002)),
            ),
        ),
    )


def test_distillation_checkpoint_roundtrips_and_activates_atomically(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    identity = DistillationCheckpointIdentity(
        (ArtifactRef("block-result", "sha256-" + "a" * 64, 1),),
        "sha256:protocol",
        "sha256:tokens",
    )
    committed = commit_distillation_checkpoint(_state(), identity, artifacts)
    activate_distillation_checkpoint(tmp_path, committed.reference)

    loaded = active_distillation_checkpoint(tmp_path, identity, artifacts)
    assert loaded is not None
    assert loaded.reference == committed.reference
    assert loaded.state.completed_epochs == 2
    assert loaded.state.steps_completed == 6
    assert torch.equal(dict(loaded.state.parameter_values)["scale"], torch.tensor((1.0, 2.0)))
    assert torch.equal(loaded.state.optimizer_states[0].exponential_average, torch.tensor((0.1, 0.2)))
    assert torch.equal(
        loaded.state.optimizer_states[0].kahan_compensation,
        torch.tensor((0.001, 0.002)),
    )
    assert (
        active_distillation_checkpoint(
            tmp_path,
            DistillationCheckpointIdentity(identity.source_blocks, "different", identity.token_hash),
            artifacts,
        )
        is None
    )
    with pytest.raises(ValueError, match="does not match"):
        load_distillation_checkpoint(
            committed.reference,
            DistillationCheckpointIdentity(identity.source_blocks, "different", identity.token_hash),
            artifacts,
        )
