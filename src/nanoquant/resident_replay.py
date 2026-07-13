"""Capture and replay a committed resident layer from a pinned source snapshot."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from nanoquant.config.codec import from_dict
from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.domain.models import ArtifactRef
from nanoquant.domain.profiling import NULL_RECORDER, PhaseRecorder
from nanoquant.domain.scale_fit import reconstruct
from nanoquant.domain.seeds import logical_seed
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity, load_committed_layer
from nanoquant.infrastructure.fixtures import LayerReplayResult, capture_layer, replay_layer
from nanoquant.infrastructure.profiling import profiled_run
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.tensor_store import LocalTensorStore

_DEFAULT_PROFILING = ProfilingConfig()


@dataclass(frozen=True, slots=True)
class ResidentLayerReplay:
    fixture: ArtifactRef
    replay: LayerReplayResult
    elapsed_seconds: float


def _capture_and_replay_resident_layer(
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
    legacy_seed_reset: bool = False,
    device: str = "cuda",
    recorder: PhaseRecorder,
    profiling: ProfilingConfig,
) -> ResidentLayerReplay:
    started = time.perf_counter()
    run_output = Path(run_output)
    micro_recorder = recorder if profiling.level is ProfilingLevel.MICRO else NULL_RECORDER
    artifacts = LocalArtifactStore(run_output / "artifacts", recorder=micro_recorder)
    tensors = LocalTensorStore(artifacts)
    with recorder.phase("journal"):
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
    with recorder.phase("load_commit"):
        layer = load_committed_layer(reference, artifacts, identity).result
    accepted_attempt = layer.attempts[layer.accepted_attempt]
    if layer.frozen_state.outliers is not None:
        raise NotImplementedError("resident replay capture does not yet reconstruct outlier residuals")
    mid_ref = layer.frozen_state.scales.mid
    if mid_ref is None:
        raise ValueError("committed layer is missing a mid scale")
    with recorder.phase("source"):
        model_source = SafetensorsModelSource(snapshot, source=source_name, revision=revision, verify_hashes=True)
    with (
        recorder.phase("load_tensors"),
        model_source.read_tensor(layer.plan.source_weight, device="cpu") as weight,
        tensors.read(layer.plan.objective.input_importance, device) as input_importance,
        tensors.read(layer.plan.objective.output_importance, device) as output_importance,
        tensors.read(layer.frozen_state.left_binary, device) as left,
        tensors.read(layer.frozen_state.right_binary, device) as right,
        tensors.read(layer.frozen_state.scales.pre, device) as scale_pre,
        tensors.read(mid_ref, device) as scale_mid,
        tensors.read(layer.frozen_state.scales.post, device) as scale_post,
    ):
        with recorder.phase("reconstruct"):
            accepted = reconstruct(left, right, scale_pre, scale_mid, scale_post)
        with recorder.phase("capture"):
            fixture = capture_layer(
                layer.layer,
                weight,
                weight,
                input_importance,
                output_importance,
                accepted_attempt.rank,
                (
                    seed
                    if legacy_seed_reset
                    else logical_seed(seed, "factorize-attempt", block, path, accepted_attempt.attempt)
                ),
                artifacts,
                accepted_reconstruction=accepted,
                outer_iterations=outer_iterations,
                inner_iterations=inner_iterations,
                device=device,
            )
    with recorder.phase("replay"):
        replay = replay_layer(fixture, artifacts)
    return ResidentLayerReplay(fixture, replay, time.perf_counter() - started)


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
    legacy_seed_reset: bool = False,
    device: str = "cuda",
    profiling: ProfilingConfig = _DEFAULT_PROFILING,
) -> ResidentLayerReplay:
    output = Path(run_output)
    with profiled_run(profiling, output, None, run_id="resident-layer-replay") as recorder:
        with recorder.phase("run"):
            return _capture_and_replay_resident_layer(
                output,
                snapshot,
                source_name=source_name,
                revision=revision,
                block=block,
                path=path,
                outer_iterations=outer_iterations,
                inner_iterations=inner_iterations,
                seed=seed,
                legacy_seed_reset=legacy_seed_reset,
                device=device,
                recorder=recorder,
                profiling=profiling,
            )
