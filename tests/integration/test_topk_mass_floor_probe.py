import argparse
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from nanoquant.application.distillation import (
    TopKDistillationConfig,
    cache_topk_teacher_targets,
)
from nanoquant.config.schema import ADMMConfig, SharedInputGroupConfig
from nanoquant.global_distillation import GlobalDistillationRequest, run_global_topk_distillation
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.resident_quantization import ResidentQuantizationRequest, run_resident_quantization
from tools import probe_topk_tail_distillation as probe


def test_mass_floor_correction_resume_matches_uninterrupted_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    with torch.random.fork_rng():
        torch.manual_seed(71)
        model_config = Gemma3TextConfig(
            vocab_size=24,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
        Gemma3ForCausalLM(model_config).save_pretrained(snapshot, safe_serialization=True)
    run_output = tmp_path / "run"
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
            run_output,
            "fixture/gemma3",
            "pinned-test-revision",
            tokens,
            device="cpu",
            target_bpw=8.0,
            rank_multiple=1,
            admm=ADMMConfig(outer_iterations=1, inner_iterations=1),
            shared_input_groups=(
                SharedInputGroupConfig(
                    "self_attn.attn_qkv",
                    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                ),
            ),
        )
    )
    config = TopKDistillationConfig(
        epochs=2,
        batch_size=2,
        learning_rate=0.01,
        top_k=8,
        vocabulary_chunk_size=7,
        token_chunk_size=4,
        maximum_tokens_per_batch=8,
        gradient_checkpointing=False,
    )
    run_global_topk_distillation(
        GlobalDistillationRequest(
            run_output,
            snapshot,
            "fixture/gemma3",
            "pinned-test-revision",
            tokens,
            config,
            device="cpu",
        )
    )

    teacher = Gemma3ForCausalLM.from_pretrained(snapshot)
    teacher_cache = cache_topk_teacher_targets(
        teacher,
        tokens,
        teacher.lm_head,
        lambda model, token_ids: model.model(token_ids)[0],
        replace(config, objective="top_k_tail"),
        device="cpu",
        pad_token_id=None,
    )
    normalizers = tuple(
        tuple(batch.teacher_log_normalizers for batch in epoch)
        for epoch in teacher_cache.epochs
    )
    assert all(value is not None for epoch in normalizers for value in epoch)
    del teacher

    monkeypatch.setattr(
        probe,
        "_distillation_config",
        lambda _run, epochs: replace(config, epochs=epochs),
    )
    monkeypatch.setattr(probe, "_load_calibration", lambda _run: tokens)
    monkeypatch.setattr(probe, "_load_training_cache", lambda _run, *, epochs: teacher_cache)
    monkeypatch.setattr(
        probe,
        "_load_or_create_teacher_normalizers",
        lambda *_args, **_kwargs: normalizers,
    )
    monkeypatch.setattr(
        probe,
        "_split_tokens",
        lambda *_args, **_kwargs: (tokens, "fixture-monitor", 1),
    )
    monkeypatch.setattr(
        probe,
        "_monitor",
        lambda *_args, **_kwargs: {"distribution": {"student_teacher_topk_mass": 0.8}},
    )
    monkeypatch.setattr(probe, "MODEL_SOURCE", "fixture/gemma3")

    def arguments(output: Path, *, interrupt: bool = False) -> argparse.Namespace:
        values = [
            "--run-output",
            str(run_output),
            "--snapshot",
            str(snapshot),
            "--output-directory",
            str(output),
            "--model-revision",
            "pinned-test-revision",
            "--objective",
            "mass_floor_correction",
            "--minimum-teacher-mass-ratio",
            "0.8",
            "--mass-loss-weight",
            "0.5",
            "--epochs",
            "2",
            "--monitor-offset",
            "0",
            "--monitor-samples",
            "1",
            "--monitor-sequence-length",
            "5",
            "--device",
            "cpu",
        ]
        if interrupt:
            values.extend(("--interrupt-after-epoch", "1"))
        return probe._parser().parse_args(values)

    resumed_output = tmp_path / "resumed"
    with pytest.raises(InterruptedError, match="epoch 1"):
        probe.run(arguments(resumed_output, interrupt=True))
    probe.run(arguments(resumed_output))
    control_output = tmp_path / "control"
    probe.run(arguments(control_output))

    resumed = probe._load_report(
        resumed_output / "report.json",
        probe._report_protocol(arguments(resumed_output), config, tokens, tokens[:1]),
    )
    control = probe._load_report(
        control_output / "report.json",
        probe._report_protocol(arguments(control_output), config, tokens, tokens[:1]),
    )
    assert resumed["status"] == control["status"] == "completed"
    assert resumed["epoch_losses"] == control["epoch_losses"]
    assert resumed["checkpoints"][-1]["checkpoint"] == control["checkpoints"][-1]["checkpoint"]

    loaded = load_frozen_run(
        run_output,
        snapshot,
        source_name="fixture/gemma3",
        revision="pinned-test-revision",
        device="cpu",
    )
    assert loaded.global_tuning is not None
