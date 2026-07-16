"""Incremental reconstruction reporting and early numbered-results publication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from nanoquant.application.reconstruction_report import render_live_weight_error_report
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import ArtifactRef, ArtifactTypes, BlockResult, LayerResult, QuantizationPlan

from .artifacts import LocalArtifactStore
from .commits import CommitIdentity, load_committed_block, load_committed_layer
from .io_utils import atomic_write_text, rewrite_linked_text
from .publication import (
    PublicationResult,
    PublishableArtifact,
    PublishableArtifactKind,
    publish_experiment_artifacts,
)

LIVE_WEIGHT_ERROR_REPORT = "weight-errors.md"


def live_weight_error_path(run_output: str | Path) -> Path:
    return Path(run_output) / LIVE_WEIGHT_ERROR_REPORT


def initialize_live_weight_error_report(
    repository_root: str | Path,
    experiment_number: int,
    run_output: str | Path,
    *,
    expected_blocks: int,
    layer_order: tuple[str, ...],
) -> PublicationResult:
    """Create the discoverable empty report once and publish it before compression starts."""

    report = live_weight_error_path(run_output)
    if not report.exists():
        atomic_write_text(
            report,
            render_live_weight_error_report(
                (),
                (),
                expected_blocks=expected_blocks,
                layer_order=layer_order,
                status="initializing",
            ),
        )
    return publish_experiment_artifacts(
        repository_root,
        experiment_number,
        (PublishableArtifact(report, PublishableArtifactKind.REPORT),),
    )


def update_live_weight_error_report(
    run_output: str | Path,
    layers: tuple[LayerResult, ...],
    blocks: tuple[BlockResult, ...],
    *,
    expected_blocks: int,
    layer_order: tuple[str, ...],
    status: str = "running",
) -> Path:
    """Rewrite the live report after a durable journal commit while retaining publication links."""

    report = live_weight_error_path(run_output)
    rewrite_linked_text(
        report,
        render_live_weight_error_report(
            layers,
            blocks,
            expected_blocks=expected_blocks,
            layer_order=layer_order,
            status=status,
        ),
    )
    return report


def rebuild_live_weight_error_report(
    repository_root: str | Path,
    experiment_number: int,
    run_output: str | Path,
    *,
    status: str = "running",
) -> Path:
    """Rebuild and publish live state from the latest durable journal identity."""

    run = Path(run_output)
    journal_path = run / "state" / "journal.jsonl"
    payloads = [
        cast(dict[str, Any], json.loads(line))
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    durable = [item for item in payloads if item.get("kind") in {"layer", "block"}]
    if not durable:
        raise ValueError("run journal contains no durable layer or block records")
    identity = from_dict(CommitIdentity, cast(dict[str, Any], durable[-1]["identity"]), path="identity")
    identity_payload = to_dict(identity)
    active = [item for item in durable if item.get("identity") == identity_payload]
    artifacts = LocalArtifactStore(run / "artifacts")
    artifacts.validate(identity.plan_hash)
    plan_payload = json.loads(
        (artifacts.path_for(identity.plan_hash) / "plan.json").read_text(encoding="utf-8")
    )
    plan = from_dict(QuantizationPlan, cast(dict[str, Any], plan_payload), path="plan")
    layer_records: dict[tuple[int, str], dict[str, Any]] = {}
    block_records: dict[int, dict[str, Any]] = {}
    for item in active:
        block = int(item["block"])
        if item["kind"] == "block":
            block_records[block] = item
            continue
        layer = item.get("layer")
        if not isinstance(layer, str):
            raise ValueError("layer journal record is missing its layer path")
        layer_records[(block, layer)] = item
    blocks = tuple(
        load_committed_block(
            ArtifactRef(ArtifactTypes.BLOCK_RESULT, str(item["artifact_id"]), 1),
            artifacts,
            identity,
        ).result
        for _block, item in sorted(block_records.items())
    )
    completed = {block.block.index for block in blocks}
    layers = [layer for block in blocks for layer in block.layers]
    layers.extend(
        load_committed_layer(
            ArtifactRef(ArtifactTypes.LAYER_RESULT, str(item["artifact_id"]), 1),
            artifacts,
            identity,
        ).result
        for (block, _path), item in sorted(layer_records.items())
        if block not in completed
    )
    layer_order = tuple(layer.layer.path for layer in plan.blocks[0].layers) if plan.blocks else ()
    initialize_live_weight_error_report(
        repository_root,
        experiment_number,
        run,
        expected_blocks=len(plan.blocks),
        layer_order=layer_order,
    )
    return update_live_weight_error_report(
        run,
        tuple(layers),
        blocks,
        expected_blocks=len(plan.blocks),
        layer_order=layer_order,
        status=status,
    )


__all__ = [
    "LIVE_WEIGHT_ERROR_REPORT",
    "initialize_live_weight_error_report",
    "live_weight_error_path",
    "rebuild_live_weight_error_report",
    "update_live_weight_error_report",
]
