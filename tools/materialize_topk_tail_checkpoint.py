"""Materialize an analysis KD checkpoint as an isolated, loadable derived run."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_distillation_checkpoint_tail_mass import discover_checkpoints
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION

from nanoquant.config.codec import from_dict, semantic_hash, to_dict
from nanoquant.domain.models import ArtifactRef, GlobalTuningResult
from nanoquant.domain.runs import RunManifest, RunStatus
from nanoquant.global_distillation import (
    _freeze_tuned_blocks,
    _selected_parameters,
    _thaw_frozen_layers,
)
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.distillation_checkpoint import (
    CommittedDistillationCheckpoint,
    DistillationCheckpointIdentity,
    load_distillation_checkpoint,
)
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.global_tuning import (
    activate_global_tuning,
    commit_global_tuning,
    load_global_tuning,
)
from nanoquant.infrastructure.io_utils import atomic_workspace, atomic_write_json
from nanoquant.infrastructure.resource_usage import peak_process_memory_bytes
from nanoquant.infrastructure.runs import (
    initial_manifest_from_resolved,
    launcher_provenance,
    transition,
)
from nanoquant.infrastructure.tensor_store import LocalTensorStore, _tensor_hash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument(
        "--epoch",
        type=int,
        help="Materialize a specific durable epoch instead of the active checkpoint.",
    )
    parser.add_argument("--derived-run-output", type=Path, required=True)
    parser.add_argument("--model-source", default=MODEL_SOURCE)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    return parser


def _hardlink_tree(source: Path, destination: Path) -> int:
    linked = 0
    for root, directories, filenames in os.walk(source):
        root_path = Path(root)
        relative = root_path.relative_to(source)
        target_root = destination / relative
        target_root.mkdir(parents=True, exist_ok=True)
        directories.sort()
        for filename in sorted(filenames):
            source_path = root_path / filename
            if source_path.is_symlink():
                raise ValueError(f"derived run source contains a symbolic link: {source_path}")
            os.link(source_path, target_root / filename)
            linked += 1
    return linked


def _load_checkpoint(
    checkpoint_output: Path,
    epoch: int | None = None,
) -> CommittedDistillationCheckpoint:
    if epoch is not None:
        if epoch <= 0:
            raise ValueError("materialized checkpoint epoch must be positive")
        candidate = discover_checkpoints(checkpoint_output, {epoch})[0]
        return load_distillation_checkpoint(
            candidate.reference,
            candidate.identity,
            LocalArtifactStore(checkpoint_output / "artifacts"),
        )
    pointer = from_dict(
        ArtifactRef,
        json.loads(
            (checkpoint_output / "global-distillation-training.json").read_text(
                encoding="utf-8"
            )
        ),
        path="topk_tail_checkpoint.reference",
    )
    artifacts = LocalArtifactStore(checkpoint_output / "artifacts")
    manifest = json.loads(
        (artifacts.path_for(pointer.artifact_id) / "checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    identity = from_dict(
        DistillationCheckpointIdentity,
        manifest["identity"],
        path="topk_tail_checkpoint.identity",
    )
    return load_distillation_checkpoint(pointer, identity, artifacts)


def _tensor_inventory(values: dict[str, torch.Tensor]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "content_hash": _tensor_hash(value),
        }
        for name, value in sorted(values.items())
    )


def _exact_reload_audit(
    checkpoint_values: dict[str, torch.Tensor],
    reloaded_values: dict[str, torch.Tensor],
) -> dict[str, object]:
    expected = _tensor_inventory(checkpoint_values)
    actual = _tensor_inventory(reloaded_values)
    expected_by_name = {str(item["name"]): item for item in expected}
    actual_by_name = {str(item["name"]): item for item in actual}
    missing = sorted(set(expected_by_name) - set(actual_by_name))
    unexpected = sorted(set(actual_by_name) - set(expected_by_name))
    mismatched = sorted(
        name
        for name in set(expected_by_name) & set(actual_by_name)
        if expected_by_name[name] != actual_by_name[name]
        or not torch.equal(checkpoint_values[name].cpu(), reloaded_values[name].cpu())
    )
    passed = not missing and not unexpected and not mismatched
    audit = {
        "passed": passed,
        "comparison": "exact-name-shape-dtype-and-value-equality",
        "parameter_count": len(expected),
        "element_count": sum(value.numel() for value in checkpoint_values.values()),
        "checkpoint_inventory_hash": semantic_hash(expected),
        "reloaded_inventory_hash": semantic_hash(actual),
        "parameters": expected,
        "missing_parameters": missing,
        "unexpected_parameters": unexpected,
        "mismatched_parameters": mismatched,
    }
    if not passed:
        raise ValueError(
            "materialized global tuning differs from its checkpoint: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    return audit


def _derived_manifest(
    source: RunManifest,
    *,
    global_tuning_artifact_id: str,
    arguments: tuple[str, ...],
) -> RunManifest:
    created = initial_manifest_from_resolved(
        source.config_hash,
        source.resolved_config,
        launcher_provenance(Path(__file__), None, arguments),
        source.environment,
        parent_run_id=source.run_id,
        forked_from_stage="distillation-checkpoint-materialization",
    )
    running = transition(created, RunStatus.RUNNING)
    artifacts = tuple(
        dict.fromkeys((*source.artifacts, global_tuning_artifact_id))
    )
    return transition(running, RunStatus.COMPLETED, artifacts=artifacts)


def run(args: argparse.Namespace) -> int:
    source = args.run_output.resolve()
    destination = args.derived_run_output.resolve()
    if destination.exists():
        raise ValueError(f"derived run output already exists: {destination}")
    checkpoint_report = json.loads(
        (args.checkpoint_output / "report.json").read_text(encoding="utf-8")
    )
    if checkpoint_report.get("status") != "completed":
        raise ValueError("top-k tail checkpoint experiment is not complete")
    checkpoint = _load_checkpoint(args.checkpoint_output, args.epoch)
    source_manifest = from_dict(
        RunManifest,
        json.loads((source / "manifest.json").read_text(encoding="utf-8")),
        path="source_manifest",
    )
    started = time.perf_counter()
    source_artifacts = LocalArtifactStore(source / "artifacts")
    loaded = load_frozen_run(
        source,
        args.snapshot,
        source_name=args.model_source,
        revision=args.model_revision,
        device="cpu",
        verify_hashes=False,
        backend="factorized",
        use_global_tuning=False,
    )
    trainable = _thaw_frozen_layers(loaded, LocalTensorStore(source_artifacts))
    selected_ids, auxiliary_names = _selected_parameters(loaded.model, trainable)
    selected_parameters = {
        name: parameter
        for name, parameter in loaded.model.named_parameters()
        if id(parameter) in selected_ids
    }
    checkpoint_values = dict(checkpoint.state.parameter_values)
    if set(selected_parameters) != set(checkpoint_values):
        raise ValueError("checkpoint parameters differ from the retained pre-KD selector")
    with torch.no_grad():
        for name, parameter in selected_parameters.items():
            parameter.copy_(checkpoint_values[name].to(dtype=parameter.dtype))

    with atomic_workspace(destination) as temporary:
        linked_files = _hardlink_tree(source, temporary)
        destination_artifacts = LocalArtifactStore(temporary / "artifacts")
        destination_tensors = LocalTensorStore(destination_artifacts)
        tuned_blocks = _freeze_tuned_blocks(loaded, trainable, destination_tensors)
        parameter_map = dict(loaded.model.named_parameters())
        auxiliary_refs = (
            destination_tensors.put(
                "global-tuning-parameters",
                {
                    name: parameter_map[name].detach().cpu()
                    for name in auxiliary_names
                },
            )
            if auxiliary_names
            else {}
        )
        result = GlobalTuningResult(
            2,
            tuple(block.teacher_outputs.artifact for block in loaded.blocks),
            tuned_blocks,
            tuple((name, auxiliary_refs[name]) for name in auxiliary_names),
            checkpoint.identity.protocol_hash,
            checkpoint.identity.token_hash,
            checkpoint.state.epoch_losses,
            checkpoint.state.steps_completed,
            len(selected_parameters),
            int(checkpoint_report.get("teacher_cache_bytes", 0)),
            float(checkpoint_report.get("wall_seconds", 0.0)),
            0,
            peak_process_memory_bytes(),
        )
        committed = commit_global_tuning(result, destination_artifacts)
        activate_global_tuning(temporary, committed.reference)
        reloaded_tuning = load_global_tuning(
            committed.reference,
            destination_artifacts,
        ).result
        reloaded = load_frozen_run(
            temporary,
            args.snapshot,
            source_name=args.model_source,
            revision=args.model_revision,
            device="cpu",
            verify_hashes=True,
            backend="factorized",
            use_global_tuning=True,
        )
        reloaded_trainable = _thaw_frozen_layers(
            reloaded,
            destination_tensors,
            frozen_states=reloaded_tuning.tuned_blocks,
        )
        reloaded_selected_ids, _reloaded_auxiliary = _selected_parameters(
            reloaded.model,
            reloaded_trainable,
        )
        reloaded_values = {
            name: parameter.detach().cpu().clone()
            for name, parameter in reloaded.model.named_parameters()
            if id(parameter) in reloaded_selected_ids
        }
        reload_audit = _exact_reload_audit(checkpoint_values, reloaded_values)
        derived_manifest = _derived_manifest(
            source_manifest,
            global_tuning_artifact_id=committed.reference.artifact_id,
            arguments=(
                "--run-output",
                str(source),
                "--checkpoint-output",
                str(args.checkpoint_output.resolve()),
                "--epoch",
                str(args.epoch) if args.epoch is not None else "active",
                "--derived-run-output",
                str(destination),
            ),
        )
        atomic_write_json(temporary / "manifest.json", to_dict(derived_manifest))
        atomic_write_json(
            temporary / "topk-tail-materialization.json",
            {
                "schema_version": 2,
                "source_run_output": str(source),
                "checkpoint_output": str(args.checkpoint_output.resolve()),
                "checkpoint": to_dict(checkpoint.reference),
                "checkpoint_identity": to_dict(checkpoint.identity),
                "global_tuning": to_dict(committed.reference),
                "derived_run_id": derived_manifest.run_id,
                "parent_run_id": derived_manifest.parent_run_id,
                "steps_completed": checkpoint.state.steps_completed,
                "linked_source_files": linked_files,
                "exact_reload_audit": reload_audit,
                "materialization_wall_seconds": time.perf_counter() - started,
            },
        )
        del reloaded, reloaded_trainable, reloaded_values
    del loaded, trainable
    gc.collect()
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
