import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

import nanoquant.resident_quantization as resident
from nanoquant.config.schema import ADMMConfig, ExecutorKind, ProfilingConfig, ProfilingLevel
from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore
from nanoquant.infrastructure.commits import load_block_activations
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.progress import ProgressJournal
from nanoquant.infrastructure.run_registry import select_run
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    _clone_forward_metadata,
    _epoch_cooldown_observer,
    _resident_config_hash,
    run_resident_quantization,
)
from nanoquant.resident_replay import capture_and_replay_resident_layer


def test_resident_quantization_commits_complete_transformers_model(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    Gemma3ForCausalLM(config).save_pretrained(snapshot, safe_serialization=True)
    output = tmp_path / "run"

    result = run_resident_quantization(
        ResidentQuantizationRequest(
            snapshot,
            output,
            "fixture/gemma3",
            "pinned-test-revision",
            ((1, 2, 3, 4),),
            device="cpu",
            target_bpw=8.0,
            rank_multiple=1,
            admm=ADMMConfig(outer_iterations=2, inner_iterations=1),
            profiling=ProfilingConfig(level=ProfilingLevel.OFF),
            registry_root=tmp_path / "runs",
        )
    )

    assert len(result.blocks) == 1
    assert len(result.blocks[0].layers) == 7
    retried_layers = [layer for layer in result.blocks[0].layers if len(layer.attempts) > 1]
    assert retried_layers
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    event_names = [event["name"] for event in events]
    assert "preprocessing.selected" in event_names
    assert "resume.discovery_completed" in event_names
    assert event_names.count("inventory.started") == event_names.count("inventory.completed") == 1
    assert event_names.count("model_load.started") == event_names.count("model_load.completed") == 1
    assert event_names.count("calibration_block.started") == event_names.count("calibration_block.completed") == 1
    assert event_names.count("calibration_persist.started") == 1
    assert event_names.count("calibration_persist.completed") == 1
    assert event_names.count("rank_planning.started") == event_names.count("rank_planning.completed") == 1
    assert event_names.count("block.started") == 1
    assert event_names.count("block_teacher_forward.started") == 1
    assert event_names.count("block_teacher_forward.completed") == 1
    assert event_names.count("layer.started") == 7
    assert event_names.count("layer.committed") == 7
    assert event_names.count("layer.completed") == 7
    assert event_names.count("block.completed") == 1
    assert event_names.count("quality_evaluation.completed") == 1
    assert event_names.count("report_write.completed") == 1
    run_started = next(event for event in events if event["name"] == "run.started")
    assert run_started["fields"]["component"] == "resident-quantization"
    assert run_started["fields"]["device"] == "cpu"
    assert run_started["fields"]["calibration_samples"] == 1
    completed_event = next(event for event in events if event["name"] == "block.completed")
    assert completed_event["fields"]["journal_sequence"] == 8
    assert completed_event["fields"]["host_peak_bytes"] > 0
    assert completed_event["fields"]["target_weighted_mean_square"] > 0
    assert completed_event["fields"]["entry_normalized_error"] == pytest.approx(
        completed_event["fields"]["entry_loss"]
        / completed_event["fields"]["target_weighted_mean_square"]
    )
    assert completed_event["fields"]["final_normalized_error"] == pytest.approx(
        completed_event["fields"]["final_loss"]
        / completed_event["fields"]["target_weighted_mean_square"]
    )
    outlier_attempts = sum(
        event["name"] == "stage.completed" and event["stage"] == "select-outliers"
        for event in events
    )
    assert outlier_attempts == sum(len(layer.attempts) for layer in result.blocks[0].layers)
    for layer in retried_layers:
        accepted = layer.attempts[layer.accepted_attempt]
        assert accepted.accepted is True
        assert layer.frozen_state.rank == accepted.rank
        assert layer.actual_bit_cost.total == layer.plan.estimated_cost.total + layer.extra_retry_bits
    assert result.frozen_model.effective_bpw <= 8.0
    assert result.frozen_model.actual_total_bits > 0
    assert math.isfinite(result.reference_nll)
    assert math.isfinite(result.compressed_nll)
    assert math.isfinite(result.logit_mse)
    assert 0 <= result.argmax_agreement <= 1
    assert result.peak_host_bytes > 0
    assert all(block.peak_host_bytes > 0 for block in result.blocks)
    assert all(block.peak_gpu_bytes == 0 for block in result.blocks)
    assert result.artifact_bytes > 0
    offload = run_resident_quantization(
        ResidentQuantizationRequest(
            snapshot,
            tmp_path / "cpu-offload",
            "fixture/gemma3",
            "pinned-test-revision",
            ((1, 2, 3, 4),),
            device="cpu",
            executor=ExecutorKind.CPU_OFFLOAD,
            target_bpw=8.0,
            rank_multiple=1,
            admm=ADMMConfig(outer_iterations=2, inner_iterations=1),
            restore_completed_blocks=False,
            evaluate_inline_quality=False,
            profiling=ProfilingConfig(level=ProfilingLevel.OFF),
        )
    )
    assert offload.plan == result.plan
    assert offload.identity == result.identity
    for offload_layer, resident_layer in zip(offload.blocks[0].layers, result.blocks[0].layers, strict=True):
        assert offload_layer.final_reconstruction == resident_layer.final_reconstruction
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"].startswith("run_")
    assert manifest["status"] == "completed"
    assert manifest["resolved_config"]["component"] == "resident-quantization"
    assert "run.completed" in (output / "run.log").read_text(encoding="utf-8")
    assert select_run(tmp_path / "runs", "latest").path == output.resolve()
    artifacts = LocalArtifactStore(output / "artifacts")
    artifacts.validate(result.report.artifact_id)
    journal_path = output / "state" / "journal.jsonl"
    journal_lines = journal_path.read_text().splitlines()
    stale = json.loads(next(line for line in journal_lines if json.loads(line)["kind"] == "block"))
    stale["identity"] = {**stale["identity"], "config_hash": "stale-config"}
    journal_path.write_text(json.dumps(stale) + "\n" + "\n".join(journal_lines) + "\n")
    loaded = load_frozen_run(
        output,
        snapshot,
        source_name="fixture/gemma3",
        revision="pinned-test-revision",
        device="cpu",
    )
    evaluation_tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        logits = cast(Any, loaded.model)(input_ids=evaluation_tokens, use_cache=False).logits
    loaded_nll = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        evaluation_tokens[:, 1:].reshape(-1),
    )
    assert float(loaded_nll) == pytest.approx(result.compressed_nll)
    discovery = ProgressJournal(output / "state", "resident-quantization", artifacts).discover(
        result.plan, result.identity
    )
    assert discovery.first_incomplete is None

    resumed_output = tmp_path / "resumed"
    interrupted_request = ResidentQuantizationRequest(
        snapshot,
        resumed_output,
        "fixture/gemma3",
        "pinned-test-revision",
        ((1, 2, 3, 4),),
        device="cpu",
        target_bpw=8.0,
        rank_multiple=1,
        admm=ADMMConfig(outer_iterations=2, inner_iterations=1),
        interrupt_after_layer_commits=3,
    )
    with pytest.raises(InterruptedError, match="after 3"):
        run_resident_quantization(interrupted_request)
    live_report = (resumed_output / "weight-errors.md").read_text(encoding="utf-8")
    assert "Durable progress: **3/7 layers**, **0/1 blocks**" in live_report
    assert live_report.count("| layer commit |") == 3
    interrupted_manifest = json.loads((resumed_output / "manifest.json").read_text(encoding="utf-8"))
    assert interrupted_manifest["status"] == "interrupted"
    interrupted_run_id = interrupted_manifest["run_id"]
    resumed = run_resident_quantization(replace(interrupted_request, interrupt_after_layer_commits=None))
    completed_manifest = json.loads((resumed_output / "manifest.json").read_text(encoding="utf-8"))
    assert completed_manifest["status"] == "completed"
    assert completed_manifest["run_id"] == interrupted_run_id
    completed_live_report = (resumed_output / "weight-errors.md").read_text(encoding="utf-8")
    assert "Status: **compression complete**" in completed_live_report
    assert "Durable progress: **7/7 layers**, **1/1 blocks**" in completed_live_report

    assert (resumed_output / "profile.json").is_file()
    assert (resumed_output / "profile.2.json").is_file()
    resumed_profile = json.loads((resumed_output / "profile.2.json").read_text(encoding="utf-8"))
    assert resumed_profile["coverage"]["fraction"] >= 0.90
    assert any(
        phase["path"].endswith("/factorize/attempt")
        for phase in resumed_profile["phases"]
    )

    assert resumed.reused_commit_count == 3
    assert resumed.plan == result.plan
    assert resumed.frozen_model.actual_total_bits == result.frozen_model.actual_total_bits
    for resumed_layer, control_layer in zip(resumed.blocks[0].layers, result.blocks[0].layers, strict=True):
        assert resumed_layer.frozen_state.rank == control_layer.frozen_state.rank
        assert resumed_layer.actual_bit_cost == control_layer.actual_bit_cost
        assert resumed_layer.final_reconstruction.export_weighted_normalized_error == pytest.approx(
            control_layer.final_reconstruction.export_weighted_normalized_error,
            rel=1e-6,
            abs=1e-7,
        )
        assert resumed_layer.final_reconstruction.raw_normalized_error == pytest.approx(
            control_layer.final_reconstruction.raw_normalized_error,
            rel=1e-6,
            abs=1e-7,
        )
    assert resumed.compressed_nll == pytest.approx(result.compressed_nll)
    replay = capture_and_replay_resident_layer(
        resumed_output,
        snapshot,
        source_name="fixture/gemma3",
        revision="pinned-test-revision",
        block=0,
        path="self_attn.k_proj",
        outer_iterations=2,
        inner_iterations=1,
        device="cpu",
        profiling=ProfilingConfig(level=ProfilingLevel.MICRO, emit_span_events=False),
    )
    assert replay.replay.expected_close is True
    assert replay.elapsed_seconds < 60
    replay_profiles = [
        json.loads(profile.read_text(encoding="utf-8"))
        for profile in resumed_output.glob("profile*.json")
        if json.loads(profile.read_text(encoding="utf-8"))["run_id"] == "resident-layer-replay"
    ]
    assert len(replay_profiles) == 1
    replay_paths = {str(phase["path"]) for phase in replay_profiles[0]["phases"]}
    assert {
        "run/journal",
        "run/load_commit",
        "run/source",
        "run/load_tensors/reconstruct",
        "run/load_tensors/capture",
        "run/replay",
    } <= replay_paths


def test_resident_tuning_recipe_refits_blocks_and_resumes_exactly(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    config = Gemma3TextConfig(
        vocab_size=24,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    Gemma3ForCausalLM(config).save_pretrained(snapshot, safe_serialization=True)
    base = ResidentQuantizationRequest(
        snapshot,
        tmp_path / "control",
        "fixture/gemma3",
        "pinned-test-revision",
        ((1, 2, 3, 4), (4, 3, 2, 1)),
        device="cpu",
        target_bpw=8.0,
        rank_multiple=1,
        admm=ADMMConfig(outer_iterations=1, inner_iterations=1),
        nonfactorized_tuning_epochs_by_layer=(1, 0),
        nonfactorized_tuning_batch_size=1,
        factorized_tuning_epochs=1,
        factorized_tuning_batch_size=1,
        post_block_refit_epochs=1,
        post_block_refit_batch_size=1,
    )

    control = run_resident_quantization(base)
    assert control.blocks[0].losses.after_post_block_refit is not None
    assert all(layer.tuning is not None for layer in control.blocks[0].layers)
    assert control.blocks[0].frozen_state.auxiliary_parameters
    tuning_events = [
        json.loads(line) for line in (base.output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    epoch_summaries = [event for event in tuning_events if event["name"] == "tuning.epoch_completed"]
    assert {event["fields"]["tuning_kind"] for event in epoch_summaries} == {
        "nonfactorized",
        "post_block_refit",
    }
    assert all(event["fields"]["target_weighted_mean_square"] > 0 for event in epoch_summaries)
    assert all(
        event["fields"]["normalized_loss"]
        == pytest.approx(
            event["fields"]["loss"] / event["fields"]["target_weighted_mean_square"]
        )
        for event in epoch_summaries
    )
    factorized_summaries = [
        event
        for event in tuning_events
        if event["name"] == "factorized_tuning.epoch_checkpoint_committed"
    ]
    assert factorized_summaries
    assert all(event["fields"]["normalized_loss"] is not None for event in factorized_summaries)
    loaded = load_frozen_run(
        base.output,
        snapshot,
        source_name="fixture/gemma3",
        revision="pinned-test-revision",
        device="cpu",
    )
    evaluation_tokens = torch.tensor(base.token_ids)
    with torch.no_grad():
        loaded_logits = cast(Any, loaded.model)(input_ids=evaluation_tokens, use_cache=False).logits
    loaded_nll = torch.nn.functional.cross_entropy(
        loaded_logits[:, :-1].float().reshape(-1, loaded_logits.shape[-1]),
        evaluation_tokens[:, 1:].reshape(-1),
    )
    assert float(loaded_nll) == pytest.approx(control.compressed_nll, rel=1e-6, abs=1e-7)

    resumed_request = replace(base, output=tmp_path / "resumed", interrupt_after_layer_commits=3)
    with pytest.raises(InterruptedError, match="after 3"):
        run_resident_quantization(resumed_request)
    resumed = run_resident_quantization(replace(resumed_request, interrupt_after_layer_commits=None))

    assert resumed.reused_commit_count == 3
    assert resumed.compressed_nll == pytest.approx(control.compressed_nll, rel=1e-6, abs=1e-7)
    assert resumed.blocks[0].losses.after_post_block_refit == pytest.approx(
        control.blocks[0].losses.after_post_block_refit,
        rel=1e-6,
        abs=1e-7,
    )

    epoch_output = tmp_path / "epoch-resumed"
    with pytest.raises(InterruptedError, match="after 1"):
        run_resident_quantization(
            replace(base, output=epoch_output, interrupt_after_layer_commits=1)
        )
    epoch_request = replace(
        base,
        output=epoch_output,
        interrupt_after_factorized_tuning_epoch_commits=1,
    )
    with pytest.raises(InterruptedError, match="factorized tuning epoch checkpoint"):
        run_resident_quantization(epoch_request)
    checkpoint_pointer = epoch_request.output / "state" / "tuning-checkpoint" / "active.json"
    first_checkpoint_layer = json.loads(checkpoint_pointer.read_text(encoding="utf-8"))["identity"]["layer"]

    with pytest.raises(InterruptedError, match="factorized tuning epoch checkpoint"):
        run_resident_quantization(epoch_request)
    second_checkpoint_layer = json.loads(checkpoint_pointer.read_text(encoding="utf-8"))["identity"]["layer"]
    assert second_checkpoint_layer != first_checkpoint_layer

    epoch_resumed = run_resident_quantization(
        replace(epoch_request, interrupt_after_factorized_tuning_epoch_commits=None)
    )

    assert epoch_resumed.compressed_nll == pytest.approx(control.compressed_nll, rel=1e-6, abs=1e-7)
    assert (
        epoch_resumed.blocks[0].frozen_state.quantized_layers
        == control.blocks[0].frozen_state.quantized_layers
    )
    assert not (epoch_request.output / "state" / "tuning-checkpoint").exists()


def test_numerical_batch_shapes_invalidate_resume_identity(tmp_path: Path) -> None:
    request = ResidentQuantizationRequest(
        tmp_path / "snapshot",
        tmp_path / "output",
        "fixture/model",
        "revision",
        ((1, 2),),
        device="cpu",
    )

    assert _resident_config_hash(replace(request, tuning_microbatch_size=2)) != _resident_config_hash(request)
    assert _resident_config_hash(replace(request, block_forward_batch_size=2)) != _resident_config_hash(request)
    assert _resident_config_hash(
        replace(request, restore_best_tuning_state=False)
    ) != _resident_config_hash(request)
    assert _resident_config_hash(
        replace(request, factorized_tuning_epoch_cooldown_seconds=5.0)
    ) == _resident_config_hash(request)
    assert _resident_config_hash(
        replace(request, nonfactorized_tuning_epoch_cooldown_seconds=5.0)
    ) == _resident_config_hash(request)
    assert _resident_config_hash(
        replace(request, post_block_refit_epoch_cooldown_seconds=5.0)
    ) == _resident_config_hash(request)
    assert _resident_config_hash(replace(request, initial_cooldown_seconds=30.0)) == _resident_config_hash(request)


def test_epoch_cooldown_skips_initial_loss_and_sleeps_after_training_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("nanoquant.resident_quantization.time.sleep", sleeps.append)

    observer = _epoch_cooldown_observer(2.5)
    assert observer is not None
    observer(0, 10.0)
    observer(1, 9.0)
    observer(2, 8.0)

    assert sleeps == [2.5, 2.5]
    assert _epoch_cooldown_observer(0.0) is None


def test_rolling_retention_keeps_only_latest_resume_generation(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    Gemma3ForCausalLM(config).save_pretrained(snapshot, safe_serialization=True)
    output = tmp_path / "rolling"
    request = ResidentQuantizationRequest(
        snapshot,
        output,
        "fixture/gemma3",
        "pinned-test-revision",
        ((1, 2, 3, 4),),
        device="cpu",
        target_bpw=8.0,
        rank_multiple=1,
        admm=ADMMConfig(outer_iterations=1, inner_iterations=1),
        interrupt_after_block_commits=1,
    )
    with pytest.raises(InterruptedError, match="after 1"):
        run_resident_quantization(request)

    result = run_resident_quantization(replace(request, interrupt_after_block_commits=None))
    artifacts = LocalArtifactStore(output / "artifacts")
    generations = list(artifacts.root.glob("??/sha256-*/activation-generation.json"))

    assert len(result.blocks) == 2
    assert len(generations) == 1
    assert not artifacts.path_for(result.blocks[0].teacher_outputs.artifact.artifact_id).exists()
    with pytest.raises(ArtifactCorruptionError, match="descriptor unavailable"):
        load_block_activations(result.frozen_model.blocks[0], artifacts)
    teacher, compressed = load_block_activations(result.frozen_model.blocks[1], artifacts)
    assert teacher.shape == compressed.shape == (1, 4, 16)


def test_continuous_multiblock_run_reloads_committed_activation_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    Gemma3ForCausalLM(config).save_pretrained(snapshot, safe_serialization=True)
    loaded_boundaries: list[str] = []
    cloned_metadata: list[dict[str, object]] = []

    def recording_load(reference: Any, artifacts: Any, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        loaded_boundaries.append(reference.artifact_id)
        return load_block_activations(reference, artifacts, device)

    def recording_clone(metadata: dict[str, object]) -> dict[str, object]:
        cloned = _clone_forward_metadata(metadata)
        cloned_metadata.append(cloned)
        return cloned

    monkeypatch.setattr("nanoquant.resident_quantization.load_block_activations", recording_load)
    monkeypatch.setattr("nanoquant.resident_quantization._clone_forward_metadata", recording_clone)
    result = run_resident_quantization(
        ResidentQuantizationRequest(
            snapshot,
            tmp_path / "continuous",
            "fixture/gemma3",
            "pinned-test-revision",
            ((1, 2, 3, 4),),
            device="cpu",
            target_bpw=8.0,
            rank_multiple=1,
            admm=ADMMConfig(outer_iterations=1, inner_iterations=1),
        )
    )

    assert loaded_boundaries == [result.frozen_model.blocks[0].artifact_id]
    assert len(cloned_metadata) == 4
    assert len({id(metadata) for metadata in cloned_metadata}) == 4
    assert len(result.blocks) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA resident transfer requires a GPU")
def test_cuda_multiblock_run_keeps_complete_activation_streams_pageable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    Gemma3ForCausalLM(config).save_pretrained(snapshot, safe_serialization=True)
    observed_source_pinning: list[bool] = []
    original = resident.iter_device_batches

    def recording_batches(
        values: tuple[torch.Tensor, ...], batch_size: int, device: torch.device
    ):  # type: ignore[no-untyped-def]
        if device.type == "cuda" and all(value.device.type == "cpu" for value in values):
            observed_source_pinning.extend(value.is_pinned() for value in values)
        yield from original(values, batch_size, device)

    monkeypatch.setattr(resident, "iter_device_batches", recording_batches)
    output = tmp_path / "cuda-pageable"
    result = run_resident_quantization(
        ResidentQuantizationRequest(
            snapshot,
            output,
            "fixture/gemma3",
            "pinned-test-revision",
            ((1, 2, 3, 4),),
            device="cuda",
            target_bpw=8.0,
            rank_multiple=1,
            admm=ADMMConfig(outer_iterations=1, inner_iterations=1),
            block_forward_batch_size=1,
            evaluate_inline_quality=False,
            profiling=ProfilingConfig(level=ProfilingLevel.OFF),
            registry_root=tmp_path / "runs",
        )
    )
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]

    assert len(result.blocks) == 2
    assert observed_source_pinning
    assert not any(observed_source_pinning)
    cache_events = [event for event in events if event["name"] == "host_pinned_cache.released"]
    assert len(cache_events) == 2
    if os.name == "nt":
        assert all("wddm.shared_bytes" in event["fields"] for event in cache_events)


def test_forward_metadata_clone_isolates_nested_tensor_mutation() -> None:
    source = {
        "attention_mask": torch.tensor([[1.0, 2.0]]),
        "position_embeddings": (torch.tensor([3.0]), {"sin": torch.tensor([4.0])}),
        "flag": True,
    }

    cloned = _clone_forward_metadata(source)
    cast(torch.Tensor, cloned["attention_mask"]).zero_()
    position_embeddings = cast(tuple[torch.Tensor, dict[str, torch.Tensor]], cloned["position_embeddings"])
    position_embeddings[0].add_(10)
    position_embeddings[1]["sin"].mul_(0)

    assert torch.equal(cast(torch.Tensor, source["attention_mask"]), torch.tensor([[1.0, 2.0]]))
    source_positions = cast(tuple[torch.Tensor, dict[str, torch.Tensor]], source["position_embeddings"])
    assert torch.equal(source_positions[0], torch.tensor([3.0]))
    assert torch.equal(source_positions[1]["sin"], torch.tensor([4.0]))
