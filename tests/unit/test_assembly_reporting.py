from dataclasses import replace
from pathlib import Path

import torch

from nanoquant.application.assembly import assemble_frozen_model
from nanoquant.application.block_snapshots import compare_block_snapshots, select_block_snapshot_tokens
from nanoquant.application.reconstruction_report import render_reconstruction_tables
from nanoquant.domain.models import GlobalTuningResult
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


def test_reconstruction_report_preserves_local_pre_kd_and_probe_post_kd_snapshots(
    tmp_path: Path,
) -> None:
    layer, _plan, frozen_block, losses = _objects()
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
    protocol = select_block_snapshot_tokens(torch.tensor(((1, 2),)), maximum_samples=1, maximum_tokens=2).protocol
    metrics = compare_block_snapshots((frozen_block.block,), (2.0,), (1.0,), protocol)
    tuning = GlobalTuningResult(
        schema_version=2,
        source_blocks=(committed.result.teacher_outputs.artifact,),
        tuned_blocks=(frozen_block,),
        auxiliary_parameters=(),
        protocol_hash="sha256:training",
        token_hash="sha256:training-tokens",
        epoch_losses=(1.0,),
        steps_completed=1,
        selected_parameter_count=1,
        teacher_cache_bytes=1,
        wall_seconds=1.0,
        peak_gpu_bytes=1,
        peak_host_bytes=1,
        block_snapshot_protocol_hash=protocol.semantic_key,
        block_metrics=metrics,
    )

    report = render_reconstruction_tables((committed.result,), tuning)

    assert "Final frozen block error before model-level KD" in report
    assert "Final block error after model-level KD" in report
    assert protocol.semantic_key in report
    assert "Local final pre-KD" in report
    assert "Probe final pre-KD" in report
    assert "Probe final post-KD" in report
    assert "| 0 | 1.150000 | 2.000000 | 1.000000 | -1.000000 | -0.5000 |" in report
