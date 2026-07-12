from dataclasses import replace
from pathlib import Path

import torch

from nanoquant.application.assembly import assemble_frozen_model
from nanoquant.application.reconstruction_report import render_reconstruction_tables
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity, commit_block
from tests.integration.test_commits_resume import _objects


def test_frozen_model_assembly_needs_no_mutable_model(tmp_path: Path) -> None:
    layer, plan, frozen_block, losses = _objects()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    committed = commit_block(
        frozen_block.block,
        (layer,),
        frozen_block,
        losses,
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2),
        0,
        artifacts,
        CommitIdentity("c", "m", "p"),
    )
    plan_ref = plan.calibration
    result = assemble_frozen_model(plan.model, plan_ref, ((committed.reference, committed.result),), (), 4)
    assert result.blocks == (committed.reference,)
    assert result.actual_total_bits == layer.actual_bit_cost.total
    assert result.effective_bpw == layer.actual_bit_cost.total / 4


def test_reconstruction_report_preserves_named_baselines_and_na(tmp_path: Path) -> None:
    layer, _plan, frozen_block, losses = _objects()
    losses = replace(
        losses,
        source_reference=0.0,
        final_vs_source_reference=replace(losses.final_vs_source_reference, baseline_loss=0.0, relative_delta=None),
    )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    committed = commit_block(
        frozen_block.block,
        (layer,),
        frozen_block,
        losses,
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2),
        0,
        artifacts,
        CommitIdentity("c", "m", "p"),
    )
    report = render_reconstruction_tables((committed.result,))
    assert "Per-layer objective-weighted reconstruction" in report
    assert "Block entry pre-quantization" in report
    assert "n/a" in report
