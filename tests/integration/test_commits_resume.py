import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from nanoquant.application.loss_snapshots import BlockLossRecorder
from nanoquant.config.codec import to_dict
from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.domain.models import (
    ArtifactRef,
    AttemptSummary,
    BinaryFactorSearchMetrics,
    BitCost,
    BlockId,
    BlockPlan,
    ComponentRef,
    FrozenBlockState,
    FrozenNanoQuantState,
    LayerId,
    LayerPlan,
    LayerResult,
    ModelIdentity,
    ObjectiveSpec,
    OutlierPlan,
    QuantizationPlan,
    ReconstructionMetrics,
    RetryPolicy,
    ScaleState,
    SourceTensor,
    TensorId,
    TensorRef,
    TensorSpec,
)
from nanoquant.domain.runs import BudgetState
from nanoquant.infrastructure import commits, optimizer_continuation
from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore
from nanoquant.infrastructure.commits import (
    CommitIdentity,
    commit_block,
    commit_layer,
    load_block_activations,
    load_committed_block,
    load_committed_layer,
    retire_block_activations,
)
from nanoquant.infrastructure.profiling import Profiler
from nanoquant.infrastructure.progress import ProgressJournal


def _objects() -> tuple[LayerResult, QuantizationPlan, FrozenBlockState, object]:
    block = BlockId(0)
    layer = LayerId(block, "linear")
    artifact = ArtifactRef("tensors", "sha256-" + "0" * 64, 1)
    tensor = TensorRef(artifact, "value", TensorSpec((2, 2), "float32"), "tensor-hash")
    vector = TensorRef(artifact, "vector", TensorSpec((2,), "float32"), "vector-hash")
    source = SourceTensor(
        TensorId(layer, "weight"), "linear.weight", "shard", TensorSpec((2, 2), "float32"), "source-hash"
    )
    objective = ObjectiveSpec(
        1, layer, "diagonal", vector, vector, None, 0.01, "target_weighted_norm_squared", None, artifact
    )
    cost = BitCost(binary_factor_bits=8, scale_bits=16)
    layer_plan = LayerPlan(
        1,
        layer,
        source,
        1,
        1,
        2,
        objective,
        OutlierPlan("none", 0, "float16", True),
        RetryPolicy(1, 0.25, 1.0, None, 2, 0),
        cost,
    )
    scales = ScaleState(vector, vector, vector)
    metrics = ReconstructionMetrics("diagonal", 1, None, None, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    attempt = AttemptSummary(0, 1, artifact, 0.1, 0.1, cost, 0.1, True, "accepted")
    frozen = FrozenNanoQuantState(layer, 1, tensor, tensor, scales, None, None, "nanoquant-v1")
    binary_search = BinaryFactorSearchMetrics(
        1.0,
        0.9,
        0.8,
        2,
        1,
        3,
        4,
        5,
        6,
        False,
        True,
        7.0,
    )
    result = LayerResult(
        1,
        layer,
        layer_plan,
        (attempt,),
        0,
        artifact,
        None,
        None,
        frozen,
        metrics,
        cost,
        0,
        (),
        binary_factor_search=binary_search,
    )
    model = ModelIdentity("fixture", "rev", "config", "fixture", "tok", ComponentRef("tiny", "1"))
    plan = QuantizationPlan(
        1, ComponentRef("planner", "1"), model, artifact, (BlockPlan(block, (layer,), (layer_plan,), 16),), 1.0, cost
    )
    frozen_block = FrozenBlockState(block, (frozen,), ())
    recorder = BlockLossRecorder()
    recorder.record_source_reference(1.0)
    recorder.record_block_entry(1.1)
    recorder.record_after_layer(layer, 1.2)
    recorder.record_post_block_refit(1.15)
    recorder.record_final_frozen_pre_kd(1.15)
    return result, plan, frozen_block, recorder.finalize()


def _fail_at(target: str):
    def inject(point: str) -> None:
        if point == target:
            raise RuntimeError(target)

    return inject


def test_named_loss_snapshots_include_near_zero_na_semantics() -> None:
    layer = LayerId(BlockId(0), "linear")
    recorder = BlockLossRecorder(denominator_floor=1e-6)
    recorder.record_source_reference(0.0)
    recorder.record_target_weighted_mean_square(4.0)
    recorder.record_block_entry(2.0)
    recorder.record_after_layer(layer, 1.0)
    recorder.record_final_frozen_pre_kd(1.0)
    result = recorder.finalize()
    assert result.final_vs_block_entry.relative_delta == -0.5
    assert result.final_vs_source_reference.relative_delta is None
    assert result.final_vs_source_reference.baseline_name == "source_reference"
    assert result.target_weighted_mean_square == 4.0
    assert result.block_entry_normalized_error == 0.5
    assert result.final_frozen_normalized_error == 0.25

    near_zero = BlockLossRecorder(denominator_floor=1e-6)
    near_zero.record_source_reference(0.0)
    near_zero.record_target_weighted_mean_square(0.0000005)
    near_zero.record_block_entry(2.0)
    near_zero.record_final_frozen_pre_kd(1.0)
    assert near_zero.finalize().final_frozen_normalized_error is None


def test_layer_block_commits_are_atomic_and_post_commit_failures_are_discoverable(tmp_path: Path) -> None:
    layer_result, plan, frozen_block, losses = _objects()
    identity = CommitIdentity("config", "model", "plan")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    journal = ProgressJournal(tmp_path / "state", "run", artifacts)
    with pytest.raises(RuntimeError, match="after_layer"):
        commit_layer(layer_result, artifacts, identity, _fail_at("after_layer_commit"))
    discovery = journal.discover(plan, identity)
    assert len(discovery.orphan_records) == 1
    assert discovery.first_incomplete is not None and discovery.first_incomplete.layer is None

    with pytest.raises(RuntimeError, match="after_block"):
        commit_block(
            BlockId(0),
            (layer_result,),
            frozen_block,
            losses,
            torch.ones(2, 3, 2),
            torch.zeros(2, 3, 2),
            0,
            artifacts,
            identity,
            inject=_fail_at("after_block_commit"),
        )
    discovery = journal.discover(plan, identity)
    assert any(record.kind == "block" for record in discovery.orphan_records)
    assert discovery.first_incomplete is None


def test_before_commit_failure_leaves_no_discoverable_output(tmp_path: Path) -> None:
    layer_result, plan, _frozen_block, _losses = _objects()
    identity = CommitIdentity("config", "model", "plan")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    with pytest.raises(RuntimeError, match="before_layer"):
        commit_layer(layer_result, artifacts, identity, _fail_at("before_layer_commit"))
    assert ProgressJournal(tmp_path / "state", "run", artifacts).discover(plan, identity).orphan_records == ()


def test_journal_validates_identity_hashes_and_builds_run_state(tmp_path: Path) -> None:
    layer_result, plan, _frozen_block, _losses = _objects()
    identity = CommitIdentity("config", "model", "plan")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    committed = commit_layer(layer_result, artifacts, identity)
    journal = ProgressJournal(tmp_path / "state", "run", artifacts)
    journal.append("layer", 0, "linear", committed.reference.artifact_id, identity)
    discovery = journal.discover(plan, identity)
    assert len(discovery.valid_records) == 1 and not discovery.orphan_records
    state = journal.state_from_discovery(discovery, BudgetState(100, 24, 0))
    journal.write_state(state)
    assert (tmp_path / "state" / "run-state.json").is_file()
    incompatible = journal.discover(plan, CommitIdentity("changed", "model", "plan"))
    assert incompatible.valid_records == ()


def test_post_commit_failure_artifacts_equal_uninterrupted_controls(tmp_path: Path) -> None:
    layer_result, _plan, frozen_block, losses = _objects()
    identity = CommitIdentity("config", "model", "plan")
    control_store = LocalArtifactStore(tmp_path / "control")
    failure_store = LocalArtifactStore(tmp_path / "failure")
    control_layer = commit_layer(layer_result, control_store, identity)
    with pytest.raises(RuntimeError):
        commit_layer(layer_result, failure_store, identity, _fail_at("after_layer_commit"))
    failure_layer_id = next(
        path.parent.name
        for path in failure_store.root.glob("??/sha256-*/descriptor.json")
        if '"artifact_type": "layer-result"' in path.read_text(encoding="utf-8")
    )
    assert failure_layer_id == control_layer.reference.artifact_id

    teacher = torch.ones(2, 3, 2)
    compressed = torch.zeros(2, 3, 2)
    control_block = commit_block(
        BlockId(0), (layer_result,), frozen_block, losses, teacher, compressed, 0, control_store, identity
    )
    with pytest.raises(RuntimeError):
        commit_block(
            BlockId(0),
            (layer_result,),
            frozen_block,
            losses,
            teacher,
            compressed,
            0,
            failure_store,
            identity,
            inject=_fail_at("after_block_commit"),
        )
    failure_block_id = next(
        path.parent.name
        for path in failure_store.root.glob("??/sha256-*/descriptor.json")
        if '"artifact_type": "block-result"' in path.read_text(encoding="utf-8")
    )
    assert failure_block_id == control_block.reference.artifact_id


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("background", [False, True])
def test_commit_io_profile_preserves_layer_block_and_activation_artifact_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    background: bool,
) -> None:
    layer_result, _plan, frozen_block, losses = _objects()
    identity = CommitIdentity("config", "model", "plan")
    control_store = LocalArtifactStore(tmp_path / "control")
    profiler = Profiler(
        ProfilingConfig(level=ProfilingLevel.MICRO, emit_span_events=False),
        run_id="commit-io-micro",
    )
    profiled_store = LocalArtifactStore(tmp_path / "profiled", recorder=profiler)
    teacher = torch.arange(12, dtype=dtype).reshape(2, 2, 3).transpose(1, 2)
    compressed = teacher + 1

    control_layer = commit_layer(layer_result, control_store, identity)
    profiled_layer = commit_layer(layer_result, profiled_store, identity)
    with monkeypatch.context() as patch:
        patch.setattr(commits, "activation_tensor_chunks", lambda *_args: None)
        control_block = commit_block(
            BlockId(0), (layer_result,), frozen_block, losses, teacher, compressed, 0, control_store, identity
        )
    if background:
        with profiled_store.begin_write("activation-generation") as writer, ThreadPoolExecutor(max_workers=1) as pool:
            for name, tensor in (("teacher_outputs", teacher), ("compressed_outputs", compressed)):
                pool.submit(commits.write_activation_member, writer, name, tensor).result()
            profiled_block = commit_block(
                BlockId(0), (layer_result,), frozen_block, losses, teacher, compressed, 0, profiled_store, identity,
                staged_activations=writer,
            )
    else:
        profiled_block = commit_block(
            BlockId(0), (layer_result,), frozen_block, losses, teacher, compressed, 0, profiled_store, identity
        )

    assert profiled_layer.reference.artifact_id == control_layer.reference.artifact_id
    assert profiled_block.reference.artifact_id == control_block.reference.artifact_id
    assert (
        profiled_block.result.teacher_outputs.artifact.artifact_id
        == control_block.result.teacher_outputs.artifact.artifact_id
    )
    fresh_store = LocalArtifactStore(profiled_store.root, use_persistent_validation_cache=False)
    loaded_teacher, loaded_compressed = load_block_activations(profiled_block.reference, fresh_store)
    assert torch.equal(loaded_teacher, teacher)
    assert torch.equal(loaded_compressed, compressed)
    payload = profiler.snapshot()
    phase_paths = {str(phase["path"]) for phase in payload["phases"]}  # type: ignore[index]
    assert {"serialize", "hash", "write"} <= phase_paths
    counters = {str(counter["name"]): counter for counter in payload["counters"]}  # type: ignore[index]
    assert counters["io.artifacts"]["total"] == 3
    assert counters["io.commit_bytes"]["total"] > 0
    assert counters["io.activation_bytes_written"]["total"] > 0


def test_committed_layer_block_and_activations_round_trip_as_typed_results(tmp_path: Path) -> None:
    layer_result, _plan, frozen_block, losses = _objects()
    identity = CommitIdentity("config", "model", "plan")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    layer = commit_layer(layer_result, artifacts, identity)
    teacher = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    compressed = teacher + 1
    block = commit_block(
        BlockId(0),
        (layer_result,),
        frozen_block,
        losses,
        teacher,
        compressed,
        0,
        artifacts,
        identity,
        wall_seconds=1.25,
        peak_gpu_bytes=10,
        peak_host_bytes=20,
    )

    assert load_committed_layer(layer.reference, artifacts, identity) == layer
    loaded_block = load_committed_block(block.reference, artifacts, identity)
    assert loaded_block == block
    loaded_teacher, loaded_compressed = load_block_activations(block.reference, artifacts)
    assert torch.equal(loaded_teacher, teacher)
    assert torch.equal(loaded_compressed, compressed)
    block_descriptor = artifacts.validate(block.reference.artifact_id)
    assert [item.path for item in block_descriptor.files] == ["block-result.json"]
    assert block.result.teacher_outputs.artifact.artifact_type == "activation-generation"


def test_external_activation_generation_can_retire_without_invalidating_block_evidence(tmp_path: Path) -> None:
    layer_result, _plan, frozen_block, losses = _objects()
    identity = CommitIdentity("config", "model", "plan")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    committed = commit_block(
        BlockId(0),
        (layer_result,),
        frozen_block,
        losses,
        torch.ones(2, 3, 2),
        torch.zeros(2, 3, 2),
        0,
        artifacts,
        identity,
    )
    generation = committed.result.teacher_outputs.artifact

    assert retire_block_activations(committed.result, artifacts) > 0
    assert not artifacts.path_for(generation.artifact_id).exists()
    assert load_committed_block(committed.reference, artifacts, identity) == committed
    with pytest.raises(ArtifactCorruptionError, match="descriptor unavailable"):
        load_block_activations(committed.reference, artifacts)


def test_failure_after_activation_generation_leaves_no_discoverable_block(tmp_path: Path) -> None:
    layer_result, plan, frozen_block, losses = _objects()
    identity = CommitIdentity("config", "model", "plan")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")

    with pytest.raises(RuntimeError, match="after_activation"):
        commit_block(
            BlockId(0),
            (layer_result,),
            frozen_block,
            losses,
            torch.ones(2, 3, 2),
            torch.zeros(2, 3, 2),
            0,
            artifacts,
            identity,
            inject=_fail_at("after_activation_commit"),
        )

    discovery = ProgressJournal(tmp_path / "state", "run", artifacts).discover(plan, identity)
    assert discovery.orphan_records == ()
    assert any(
        '"artifact_type": "activation-generation"' in path.read_text(encoding="utf-8")
        for path in artifacts.root.glob("??/sha256-*/descriptor.json")
    )


@pytest.mark.parametrize("interrupt_import", [False, True])
def test_optimizer_continuation_preserves_payloads_and_recovers_transaction(tmp_path, monkeypatch, interrupt_import):
    layer, plan, frozen, losses = _objects()
    store = LocalArtifactStore(tmp_path / "artifacts")
    with store.begin_write("quantization-plan") as writer:
        (writer.path / "plan.json").write_text(json.dumps(to_dict(plan)))
        plan_id = writer.commit().artifact_id
    identity = CommitIdentity("old-config", "model", plan_id)
    journal = ProgressJournal(tmp_path / "state", "run", store)
    committed_layer = commit_layer(layer, store, identity)
    committed_block = commit_block(BlockId(0), (layer,), frozen, losses, torch.ones(2, 3, 2),
                                  torch.zeros(2, 3, 2), 0, store, identity)
    journal.append("layer", 0, "linear", committed_layer.reference.artifact_id, identity)
    journal.append("block", 0, None, committed_block.reference.artifact_id, identity)
    source_journal = journal.path.read_bytes()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"config_hash": "old-manifest"}))
    preprocessing = tmp_path / "state/preprocessing.json"
    preprocessing.write_text(json.dumps({"resident_config_hash": "old-config", "plan": {"artifact_id": plan_id}}))
    arguments = dict(source_journal_sha256=hashlib.sha256(source_journal).hexdigest(),
                     source_manifest_config_hash="old-manifest", source_config_hash="old-config",
                     target_config_hash="new-config", target_manifest={"config_hash": "new-manifest"}, start_block=1)
    with pytest.raises(ValueError, match="source journal changed"):
        optimizer_continuation.import_optimizer_prefix(
            tmp_path, **{**arguments, "source_journal_sha256": "0" * 64},
        )
    assert journal.path.read_bytes() == source_journal
    assert json.loads(manifest.read_text())["config_hash"] == "old-manifest"
    write_bytes = optimizer_continuation._write_bytes
    if interrupt_import:
        def interrupted_write(path, data):
            if path == manifest:
                raise InterruptedError("during transaction")
            write_bytes(path, data)
        monkeypatch.setattr(optimizer_continuation, "_write_bytes", interrupted_write)
        with pytest.raises(InterruptedError, match="during transaction"):
            optimizer_continuation.import_optimizer_prefix(tmp_path, **arguments)
        monkeypatch.setattr(optimizer_continuation, "_write_bytes", write_bytes)
    receipt = optimizer_continuation.import_optimizer_prefix(tmp_path, **arguments)
    assert receipt["complete_blocks_preserved"] == [0]
    assert receipt["records_preserved"] == 2
    assert receipt["status"] == "complete"
    assert (tmp_path / "state/optimizer-continuation/source-journal.jsonl").read_bytes() == source_journal
    new_identity = CommitIdentity("new-config", "model", plan_id)
    discovered = journal.discover(plan, new_identity)
    assert len(discovered.valid_records) == 2 and not discovered.orphan_records
    restored = load_committed_block(
        ArtifactRef("block-result", discovered.valid_records[-1].artifact_id, 1), store, new_identity,
    )
    assert restored.result == committed_block.result
    teacher, compressed = load_block_activations(restored.reference, store)
    assert torch.equal(teacher, torch.ones(2, 3, 2))
    assert torch.equal(compressed, torch.zeros(2, 3, 2))
    assert json.loads(preprocessing.read_text())["resident_config_hash"] == "new-config"
    assert optimizer_continuation.import_optimizer_prefix(tmp_path, **arguments) == receipt
    for record in json.loads("[" + ",".join(source_journal.decode().splitlines()) + "]"):
        store.validate(record["artifact_id"])
    changed = {**arguments, "source_journal_sha256": "0" * 64}
    with pytest.raises(ValueError, match="receipt"):
        optimizer_continuation.import_optimizer_prefix(tmp_path, **changed)
