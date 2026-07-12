from pathlib import Path
from time import perf_counter

import torch

from nanoquant.domain.factorization import factorize_admm
from nanoquant.domain.models import BlockId, LayerId
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.fixtures import capture_block, capture_layer, replay_block, replay_layer
from nanoquant.infrastructure.tiny_model import TinyBlock, TinyModelConfig


def test_layer_capture_replays_deterministically_from_canonical_artifact(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    weight = torch.tensor([[1.0, -2.0], [0.5, 1.5]])
    importance = torch.ones(2)
    accepted = factorize_admm(
        weight, importance, importance, 1, torch.Generator().manual_seed(4), outer_iterations=5, inner_iterations=2
    ).reconstruction
    reference = capture_layer(
        LayerId(BlockId(0), "linear"),
        weight,
        weight,
        importance,
        importance,
        1,
        4,
        artifacts,
        accepted_reconstruction=accepted,
        outer_iterations=5,
        inner_iterations=2,
    )
    started = perf_counter()
    replay = replay_layer(reference, artifacts)
    elapsed = perf_counter() - started
    assert replay.expected_close is True
    assert replay.maximum_absolute_difference == 0
    assert elapsed < 60


def test_block_capture_replays_state_inputs_targets_and_accepted_loss(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    config = TinyModelConfig(hidden_size=8, intermediate_size=12, block_count=1)
    with torch.random.fork_rng():
        torch.manual_seed(3)
        block = TinyBlock(config)
    inputs = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(7))

    def runner(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
        return module(value)

    with torch.no_grad():
        targets = runner(block, inputs) + 0.1
    reference = capture_block(0, block, inputs, targets, artifacts, runner)
    replay = replay_block(reference, artifacts, lambda: TinyBlock(config), runner)
    assert replay.loss == replay.expected_loss
    assert replay.absolute_difference == 0
