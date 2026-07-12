import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from nanoquant.config.schema import ADMMConfig
from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore
from nanoquant.infrastructure.commits import load_block_activations
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.progress import ProgressJournal
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
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
        )
    )

    assert len(result.blocks) == 1
    assert len(result.blocks[0].layers) == 7
    retried_layers = [layer for layer in result.blocks[0].layers if len(layer.attempts) > 1]
    assert retried_layers
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
    assert result.artifact_bytes > 0
    artifacts = LocalArtifactStore(output / "artifacts")
    artifacts.validate(result.report.artifact_id)
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
    resumed = run_resident_quantization(replace(interrupted_request, interrupt_after_layer_commits=None))

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
    )
    assert replay.replay.expected_close is True
    assert replay.elapsed_seconds < 60


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


def test_tuning_microbatch_is_execution_only_for_resume_identity(tmp_path: Path) -> None:
    request = ResidentQuantizationRequest(
        tmp_path / "snapshot",
        tmp_path / "output",
        "fixture/model",
        "revision",
        ((1, 2),),
        device="cpu",
    )

    assert _resident_config_hash(replace(request, tuning_microbatch_size=2)) == _resident_config_hash(request)


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
