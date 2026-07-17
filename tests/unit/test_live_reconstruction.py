import json
import os
from pathlib import Path

import torch

from nanoquant.application.planning import persist_plan
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity, commit_block, commit_layer
from nanoquant.infrastructure.live_reconstruction import (
    initialize_live_weight_error_report,
    live_weight_error_path,
    rebuild_live_weight_error_report,
    update_live_weight_error_report,
)
from nanoquant.infrastructure.progress import ProgressJournal
from tests.integration.test_commits_resume import _objects


def test_live_report_is_published_early_and_updates_through_the_same_link(tmp_path: Path) -> None:
    run = tmp_path / "evidence" / "run"

    published = initialize_live_weight_error_report(
        tmp_path,
        6,
        run,
        expected_blocks=2,
        layer_order=("mlp.gate_proj",),
    )
    source = live_weight_error_path(run)
    destination = tmp_path / "Results" / "006" / "weight-errors.md"
    update_live_weight_error_report(
        run,
        (),
        (),
        expected_blocks=2,
        layer_order=("mlp.gate_proj",),
        status="running",
    )

    assert source.is_file()
    assert destination.is_file()
    assert os.path.samefile(source, destination)
    assert "Status: **running**" in destination.read_text(encoding="utf-8")
    assert published.results_directory == tmp_path / "Results" / "006"
    manifest = json.loads(published.manifest.read_text(encoding="utf-8"))
    assert manifest["artifacts"][0]["published"] == "Results/006/weight-errors.md"


def test_live_report_rebuilds_partial_and_completed_rows_from_the_journal(tmp_path: Path) -> None:
    run = tmp_path / "run"
    artifacts = LocalArtifactStore(run / "artifacts")
    layer, plan, frozen_block, losses = _objects()
    persisted = persist_plan(plan, artifacts)
    identity = CommitIdentity("config", "model", persisted.reference.artifact_id)
    journal = ProgressJournal(run / "state", "run", artifacts)
    committed_layer = commit_layer(layer, artifacts, identity)
    journal.append("layer", 0, "linear", committed_layer.reference.artifact_id, identity)

    report = rebuild_live_weight_error_report(tmp_path, 9, run)

    assert "Durable progress: **1/1 layers**, **0/1 blocks**" in report.read_text(encoding="utf-8")
    committed_block = commit_block(
        frozen_block.block,
        (layer,),
        frozen_block,
        losses,
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2),
        0,
        artifacts,
        identity,
    )
    journal.append("block", 0, None, committed_block.reference.artifact_id, identity)

    rebuild_live_weight_error_report(tmp_path, 9, run, status="compression complete")

    text = report.read_text(encoding="utf-8")
    assert "Durable progress: **1/1 layers**, **1/1 blocks**" in text
    assert "Status: **compression complete**" in text
    assert "block final" in text
    assert "Entry normalized" in text
    assert "Final normalized" in text
