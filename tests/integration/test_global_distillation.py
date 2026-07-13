import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from nanoquant.application.distillation import TopKDistillationConfig
from nanoquant.config.schema import ADMMConfig
from nanoquant.global_distillation import GlobalDistillationRequest, run_global_topk_distillation
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.distillation_checkpoint import (
    DistillationCheckpointIdentity,
    active_distillation_checkpoint,
)
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.global_tuning import active_global_tuning, load_global_tuning
from nanoquant.resident_quantization import ResidentQuantizationRequest, run_resident_quantization


def test_complete_frozen_run_can_be_distilled_committed_and_reloaded(tmp_path: Path) -> None:
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
    output = tmp_path / "run"
    tokens = torch.tensor(
        (
            (1, 2, 3, 4, 5),
            (5, 4, 3, 2, 1),
            (1, 3, 5, 7, 9),
            (2, 4, 6, 8, 10),
        )
    )
    run_resident_quantization(
        ResidentQuantizationRequest(
            snapshot,
            output,
            "fixture/gemma3",
            "pinned-test-revision",
            tokens,
            device="cpu",
            target_bpw=8.0,
            rank_multiple=1,
            admm=ADMMConfig(outer_iterations=1, inner_iterations=1),
        )
    )
    before = load_frozen_run(
        output,
        snapshot,
        source_name="fixture/gemma3",
        revision="pinned-test-revision",
        device="cpu",
    )
    with torch.no_grad():
        before_logits = cast(Any, before.model)(input_ids=tokens, use_cache=False).logits.detach()

    request = GlobalDistillationRequest(
        output,
        snapshot,
        "fixture/gemma3",
        "pinned-test-revision",
        tokens,
        TopKDistillationConfig(
            epochs=3,
            batch_size=2,
            learning_rate=0.02,
            top_k=8,
            vocabulary_chunk_size=7,
            token_chunk_size=4,
            maximum_tokens_per_batch=8,
            gradient_checkpointing=False,
            weight_decay=0.0,
        ),
        device="cpu",
    )
    with pytest.raises(InterruptedError, match="after 1 distillation epoch checkpoint"):
        run_global_topk_distillation(replace(request, interrupt_after_epoch_commits=1))
    distilled = run_global_topk_distillation(request)

    active = active_global_tuning(output)
    assert active == distilled.reference
    cache_journal = json.loads((output / "global-distillation-cache.json").read_text(encoding="utf-8"))
    assert len(cache_journal["epochs"]) == 3
    assert all(reference is not None for reference in cache_journal["epochs"])
    persisted = load_global_tuning(distilled.reference, LocalArtifactStore(output / "artifacts"))
    assert persisted.result == distilled.result
    assert distilled.metrics.steps_completed == 6
    assert distilled.metrics.epoch_losses[-1] <= distilled.metrics.epoch_losses[0]
    assert distilled.result.source_blocks == tuple(block.teacher_outputs.artifact for block in before.blocks)
    assert len(distilled.result.tuned_blocks) == 1
    assert distilled.result.auxiliary_parameters
    training_checkpoint = active_distillation_checkpoint(
        output,
        DistillationCheckpointIdentity(
            distilled.result.source_blocks,
            distilled.result.protocol_hash,
            distilled.result.token_hash,
        ),
        LocalArtifactStore(output / "artifacts"),
    )
    assert training_checkpoint is not None
    assert training_checkpoint.state.completed_epochs == 3
    assert training_checkpoint.state.steps_completed == 6
    parameter_values = dict(training_checkpoint.state.parameter_values)
    optimizer_states = {
        state.parameter_name: state for state in training_checkpoint.state.optimizer_states
    }
    scale_names = tuple(name for name in parameter_values if ".scale_" in name)
    assert scale_names
    assert all(parameter_values[name].dtype is torch.bfloat16 for name in scale_names)
    assert all(optimizer_states[name].kahan_compensation is not None for name in scale_names)

    loaded = load_frozen_run(
        output,
        snapshot,
        source_name="fixture/gemma3",
        revision="pinned-test-revision",
        device="cpu",
    )
    assert loaded.global_tuning == distilled.reference
    with torch.no_grad():
        after_logits = cast(Any, loaded.model)(input_ids=tokens, use_cache=False).logits.detach()
    assert not torch.equal(after_logits, before_logits)

    pre_distillation = load_frozen_run(
        output,
        snapshot,
        source_name="fixture/gemma3",
        revision="pinned-test-revision",
        device="cpu",
        use_global_tuning=False,
    )
    assert pre_distillation.global_tuning is None
    with torch.no_grad():
        pre_distillation_logits = cast(Any, pre_distillation.model)(input_ids=tokens, use_cache=False).logits
    assert torch.equal(pre_distillation_logits, before_logits)
