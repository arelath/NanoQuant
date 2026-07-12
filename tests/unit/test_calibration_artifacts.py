from pathlib import Path

import pytest
import torch

from nanoquant.application.calibration import MaterializedLayerCalibration
from nanoquant.application.calibration_artifacts import build_objectives, persist_calibration
from nanoquant.config.schema import ObjectiveConfig, ObjectiveKind
from nanoquant.domain.models import BlockId, ComponentRef, DatasetIdentity, LayerId, ModelIdentity
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.tensor_store import LocalTensorStore


def _identities() -> tuple[ModelIdentity, DatasetIdentity]:
    model = ModelIdentity(
        "fixture/model", "model-rev", "sha256:config", "fixture/tokenizer", "tok-rev", ComponentRef("tiny", "1")
    )
    dataset = DatasetIdentity("sha256:data", ("fixture",), ("data-rev",), "sha256:tokenizer", "format-v1")
    return model, dataset


def test_calibration_stats_and_objectives_are_portable_artifacts(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    tensors = LocalTensorStore(artifacts)
    model, dataset = _identities()
    layer = LayerId(BlockId(0), "mlp.up_proj")
    materialized = MaterializedLayerCalibration(
        "mlp.up_proj", torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]), 2, "online_fisher"
    )
    persisted = persist_calibration(
        ((layer, materialized),), model, dataset, "online_fisher", "float32", artifacts, tensors
    )
    artifacts.validate(persisted.reference.artifact_id)
    stats = persisted.stats.layers[0]
    with tensors.read(stats.input_importance) as value:
        assert torch.equal(value, materialized.input_importance)
    objectives = build_objectives(persisted, ObjectiveConfig(kind=ObjectiveKind.DIAGONAL), artifacts)
    artifacts.validate(objectives.reference.artifact_id)
    assert objectives.objectives[0].layer == layer
    assert objectives.objectives[0].source_calibration == persisted.reference
    assert objectives.objectives[0].kind == "diagonal"


def test_non_finite_calibration_is_rejected_before_commit(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    tensors = LocalTensorStore(artifacts)
    model, dataset = _identities()
    layer = LayerId(BlockId(0), "bad")
    bad = MaterializedLayerCalibration("bad", torch.tensor([float("nan")]), torch.ones(1), 1, "online_fisher")
    with pytest.raises(ValueError, match="non-finite"):
        persist_calibration(((layer, bad),), model, dataset, "online_fisher", "float32", artifacts, tensors)
