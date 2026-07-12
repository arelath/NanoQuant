from __future__ import annotations

import math
from pathlib import Path

import torch

from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity
from nanoquant.infrastructure.progress import ProgressJournal
from nanoquant.tiny_pipeline import run_tiny_pipeline


def test_tiny_pipeline_runs_entirely_on_rewrite_components(tmp_path: Path) -> None:
    result = run_tiny_pipeline(tmp_path)

    assert len(result.blocks) == 2
    assert sum(len(block.layers) for block in result.blocks) == 12
    assert len(result.frozen_model.blocks) == 2
    assert result.teacher_logits.shape == result.compressed_logits.shape == (2, 4, 32)
    assert torch.isfinite(result.teacher_logits).all()
    assert torch.isfinite(result.compressed_logits).all()
    assert math.isfinite(float((result.teacher_logits - result.compressed_logits).square().mean()))
    assert "Per-layer objective-weighted reconstruction" in result.report
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == result.report
    assert result.elapsed_seconds < 600

    identity = CommitIdentity(
        "tiny-config-v1", result.frozen_model.model.config_hash, result.frozen_model.plan.artifact_id
    )
    discovery = ProgressJournal(tmp_path / "state", "tiny-run", LocalArtifactStore(tmp_path / "artifacts")).discover(
        result.plan, identity
    )
    assert discovery.first_incomplete is None
    assert len(discovery.valid_records) == 14
    assert not discovery.orphan_records
