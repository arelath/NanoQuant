from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nanoquant.application.loss_snapshots import BlockLossRecorder
from nanoquant.domain.models import (
    BlockId,
    FrozenBlockState,
    FrozenNanoQuantState,
    FrozenOutlierState,
    GlobalTuningResult,
    LayerId,
    ScaleState,
)
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity, commit_block
from nanoquant.infrastructure.global_tuning import activate_global_tuning, commit_global_tuning
from nanoquant.infrastructure.progress import ProgressJournal
from nanoquant.infrastructure.runtime_export import (
    export_frozen_run_logical,
    validate_frozen_run_logical,
)
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.runtime import FactorizedReferenceBackend, RuntimeModelMetadata


def _metadata() -> RuntimeModelMetadata:
    return RuntimeModelMetadata("fixture/model", "revision", "fixture", "model-config", "tokenizer-hash")


def _frozen_block(
    block_index: int,
    scale: float,
    tensors: LocalTensorStore,
) -> FrozenBlockState:
    references = tensors.put(
        "frozen-layer",
        {
            "left": torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]),
            "right": torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]]),
            "pre": torch.full((4,), scale),
            "mid": torch.full((2,), scale + 0.25),
            "post": torch.full((3,), scale + 0.5),
            "bias": torch.tensor([0.1, -0.2, 0.3]),
            "indices": torch.tensor([1], dtype=torch.int32),
            "values": torch.tensor([[2], [-1], [3]], dtype=torch.int8),
            "outlier_scales": torch.tensor([0.125]),
        },
    )
    block = BlockId(block_index)
    state = FrozenNanoQuantState(
        LayerId(block, "linear"),
        2,
        references["left"],
        references["right"],
        ScaleState(references["pre"], references["mid"], references["post"]),
        FrozenOutlierState(
            references["indices"],
            references["values"],
            references["outlier_scales"],
        ),
        references["bias"],
        "nanoquant-v1",
    )
    return FrozenBlockState(block, (state,), ())


def _losses():  # type: ignore[no-untyped-def]
    recorder = BlockLossRecorder()
    recorder.record_source_reference(1.0)
    recorder.record_block_entry(1.0)
    recorder.record_final_frozen_pre_kd(1.0)
    return recorder.finalize()


def _run(tmp_path: Path) -> tuple[Path, tuple[FrozenBlockState, ...], tuple[FrozenBlockState, ...]]:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"resolved_config": {"source": "fixture/model", "revision": "revision"}}),
        encoding="utf-8",
    )
    artifacts = LocalArtifactStore(run / "artifacts")
    tensors = LocalTensorStore(artifacts)
    identity = CommitIdentity("run-config", "model-config", "plan")
    journal = ProgressJournal(run / "state", "run", artifacts)
    committed_states = tuple(_frozen_block(index, 1.0 + index, tensors) for index in range(2))
    committed = tuple(
        commit_block(
            state.block,
            (),
            state,
            _losses(),
            torch.ones(1, 2, 3),
            torch.ones(1, 2, 3),
            0,
            artifacts,
            identity,
        )
        for state in committed_states
    )
    for value in committed:
        journal.append("block", value.result.block.index, None, value.reference.artifact_id, identity)
    tuned_states = tuple(_frozen_block(index, 4.0 + index, tensors) for index in range(2))
    tuning = GlobalTuningResult(
        1,
        tuple(value.result.teacher_outputs.artifact for value in committed),
        tuned_states,
        (),
        "protocol",
        "tokens",
        (1.0,),
        1,
        2,
        0,
        1.0,
        0,
        0,
    )
    activate_global_tuning(run, commit_global_tuning(tuning, artifacts).reference)
    return run, committed_states, tuned_states


def test_export_frozen_run_streams_active_global_tuning_into_runtime_artifact(tmp_path: Path) -> None:
    run, _committed, tuned = _run(tmp_path)

    result = export_frozen_run_logical(run, tmp_path / "logical", _metadata(), 2)
    layer = result.output.joinpath("nanoquant-model.json")

    assert layer.is_file()
    assert result.block_count == 2
    assert result.layer_count == 2
    assert result.global_tuning is not None
    from nanoquant.runtime import open_logical_artifact

    opened = open_logical_artifact(result.output)
    loaded = opened.load_layer("blocks.0.linear")
    assert torch.equal(loaded.scale_pre, torch.full((4,), 4.0))
    assert loaded.spec.outlier_count == 1
    assert loaded.spec.outlier_value_dtype == "int8"
    backend = FactorizedReferenceBackend()
    prepared = backend.prepare(loaded, "cpu")
    actual = backend.linear(torch.ones(1, 4), prepared)
    expected_state = tuned[0].quantized_layers[0]
    source_tensors = LocalTensorStore(LocalArtifactStore(run / "artifacts"))
    with (
        source_tensors.read(expected_state.left_binary) as left,
        source_tensors.read(expected_state.right_binary) as right,
        source_tensors.read(expected_state.scales.pre) as scale_pre,
        source_tensors.read(expected_state.scales.mid) as scale_mid,
        source_tensors.read(expected_state.scales.post) as scale_post,
        source_tensors.read(expected_state.bias) as bias,
        source_tensors.read(expected_state.outliers.indices) as indices,
        source_tensors.read(expected_state.outliers.values) as values,
        source_tensors.read(expected_state.outliers.scales) as outlier_scales,
    ):
        assert torch.equal(loaded.left_binary, left)
        assert torch.equal(loaded.right_binary, right)
        assert torch.equal(loaded.scale_pre, scale_pre)
        assert torch.equal(loaded.scale_mid, scale_mid)
        assert torch.equal(loaded.scale_post, scale_post)
        assert torch.equal(loaded.bias, bias)
        assert torch.equal(loaded.outlier_indices, indices)
        assert torch.equal(loaded.outlier_values, values)
        assert torch.equal(loaded.outlier_scales, outlier_scales)
    assert loaded.spec.rank == expected_state.rank
    assert actual.shape == (1, 3)
    assert bool(torch.all(torch.isfinite(actual)))
    validation = validate_frozen_run_logical(run, result.output, 2)
    assert validation.block_count == 2
    assert validation.layer_count == 2
    assert validation.tensor_count == 18
    assert validation.tensor_bytes > 0
    assert validation.global_tuning == result.global_tuning
    assert validation.exact


def test_export_rejects_model_identity_mismatch_before_writing(tmp_path: Path) -> None:
    run, _committed, _tuned = _run(tmp_path)
    metadata = replace(_metadata(), config_hash="different-model")

    with pytest.raises(ValueError, match="config hash does not match"):
        export_frozen_run_logical(run, tmp_path / "logical", metadata, 2)
    assert not (tmp_path / "logical").exists()


def test_export_rejects_declared_source_mismatch(tmp_path: Path) -> None:
    run, _committed, _tuned = _run(tmp_path)
    metadata = replace(_metadata(), source="different/model")

    with pytest.raises(ValueError, match="source does not match"):
        export_frozen_run_logical(run, tmp_path / "logical", metadata, 2)
    assert not (tmp_path / "logical").exists()


def test_export_frozen_run_can_select_pre_tuning_state(tmp_path: Path) -> None:
    run, _committed, _tuned = _run(tmp_path)

    result = export_frozen_run_logical(
        run,
        tmp_path / "logical",
        _metadata(),
        2,
        use_global_tuning=False,
    )

    from nanoquant.runtime import open_logical_artifact

    assert result.global_tuning is None
    assert torch.equal(
        open_logical_artifact(result.output).load_layer("blocks.0.linear").scale_pre,
        torch.ones(4),
    )


def test_export_uses_latest_complete_identity_instead_of_newer_partial_run(tmp_path: Path) -> None:
    run, committed, _tuned = _run(tmp_path)
    artifacts = LocalArtifactStore(run / "artifacts")
    newer = CommitIdentity("newer-partial-config", "model-config", "newer-plan")
    partial = commit_block(
        committed[0].block,
        (),
        committed[0],
        _losses(),
        torch.ones(1, 2, 3),
        torch.ones(1, 2, 3),
        0,
        artifacts,
        newer,
    )
    ProgressJournal(run / "state", "run", artifacts).append(
        "block",
        0,
        None,
        partial.reference.artifact_id,
        newer,
    )

    result = export_frozen_run_logical(run, tmp_path / "logical", _metadata(), 2)

    assert result.identity == CommitIdentity("run-config", "model-config", "plan")


def test_export_rejects_global_tuning_from_different_committed_blocks(tmp_path: Path) -> None:
    run, _committed, _tuned = _run(tmp_path)
    artifacts = LocalArtifactStore(run / "artifacts")
    active = run / "global-tuning.json"
    from nanoquant.infrastructure.global_tuning import active_global_tuning, load_global_tuning

    reference = active_global_tuning(run)
    assert reference is not None
    result = load_global_tuning(reference, artifacts).result
    mismatched = replace(result, source_blocks=tuple(reversed(result.source_blocks)))
    activate_global_tuning(run, commit_global_tuning(mismatched, artifacts).reference)

    with pytest.raises(ValueError, match="does not match"):
        export_frozen_run_logical(run, tmp_path / "logical", _metadata(), 2)
    assert not (tmp_path / "logical").exists()
    assert active.is_file()
