"""Capture and replay a committed resident layer from a pinned source snapshot."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from nanoquant.config.codec import from_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.domain.scale_fit import reconstruct
from nanoquant.domain.seeds import logical_seed
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity, load_committed_layer
from nanoquant.infrastructure.fixtures import LayerReplayResult, capture_layer, replay_layer
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.tensor_store import LocalTensorStore


@dataclass(frozen=True, slots=True)
class ResidentLayerReplay:
    fixture: ArtifactRef
    replay: LayerReplayResult
    elapsed_seconds: float


def capture_and_replay_resident_layer(
    run_output: str | Path,
    snapshot: str | Path,
    *,
    source_name: str,
    revision: str,
    block: int,
    path: str,
    outer_iterations: int,
    inner_iterations: int,
    seed: int = 0,
    device: str = "cuda",
) -> ResidentLayerReplay:
    started = time.perf_counter()
    run_output = Path(run_output)
    artifacts = LocalArtifactStore(run_output / "artifacts")
    tensors = LocalTensorStore(artifacts)
    records = [json.loads(line) for line in (run_output / "state" / "journal.jsonl").read_text().splitlines()]
    matching = [
        record
        for record in records
        if record.get("kind") == "layer" and record.get("block") == block and record.get("layer") == path
    ]
    if not matching:
        raise ValueError(f"run has no committed layer {block}:{path}")
    record = matching[-1]
    identity = from_dict(CommitIdentity, record["identity"], path="identity")
    reference = ArtifactRef("layer-result", str(record["artifact_id"]), 1)
    layer = load_committed_layer(reference, artifacts, identity).result
    if layer.frozen_state.outliers is not None:
        raise NotImplementedError("resident replay capture does not yet reconstruct outlier residuals")
    mid_ref = layer.frozen_state.scales.mid
    if mid_ref is None:
        raise ValueError("committed layer is missing a mid scale")
    model_source = SafetensorsModelSource(snapshot, source=source_name, revision=revision, verify_hashes=True)
    with (
        model_source.read_tensor(layer.plan.source_weight, device="cpu") as weight,
        tensors.read(layer.plan.objective.input_importance, device) as input_importance,
        tensors.read(layer.plan.objective.output_importance, device) as output_importance,
        tensors.read(layer.frozen_state.left_binary, device) as left,
        tensors.read(layer.frozen_state.right_binary, device) as right,
        tensors.read(layer.frozen_state.scales.pre, device) as scale_pre,
        tensors.read(mid_ref, device) as scale_mid,
        tensors.read(layer.frozen_state.scales.post, device) as scale_post,
    ):
        accepted = reconstruct(left, right, scale_pre, scale_mid, scale_post)
        fixture = capture_layer(
            layer.layer,
            weight,
            weight,
            input_importance,
            output_importance,
            layer.plan.rank,
            logical_seed(seed, "factorize", block, path, 0),
            artifacts,
            accepted_reconstruction=accepted,
            outer_iterations=outer_iterations,
            inner_iterations=inner_iterations,
            device=device,
        )
    replay = replay_layer(fixture, artifacts)
    return ResidentLayerReplay(fixture, replay, time.perf_counter() - started)
