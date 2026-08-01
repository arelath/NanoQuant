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
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION

from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import ArtifactRef, GlobalTuningResult
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
from nanoquant.infrastructure.global_tuning import activate_global_tuning, commit_global_tuning
from nanoquant.infrastructure.io_utils import atomic_workspace, atomic_write_json
from nanoquant.infrastructure.resource_usage import peak_process_memory_bytes
from nanoquant.infrastructure.tensor_store import LocalTensorStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
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


def _load_checkpoint(checkpoint_output: Path) -> CommittedDistillationCheckpoint:
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
    checkpoint = _load_checkpoint(args.checkpoint_output)
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
        atomic_write_json(
            temporary / "topk-tail-materialization.json",
            {
                "schema_version": 1,
                "source_run_output": str(source),
                "checkpoint_output": str(args.checkpoint_output.resolve()),
                "checkpoint": to_dict(checkpoint.reference),
                "global_tuning": to_dict(committed.reference),
                "steps_completed": checkpoint.state.steps_completed,
                "linked_source_files": linked_files,
                "materialization_wall_seconds": time.perf_counter() - started,
            },
        )
    del loaded, trainable
    gc.collect()
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
