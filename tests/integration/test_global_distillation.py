import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

import nanoquant.global_distillation as global_distillation_module
from nanoquant.application.distillation import TopKDistillationConfig
from nanoquant.application.layers import TrainableFactorizedLinear
from nanoquant.config.codec import semantic_hash, to_dict
from nanoquant.config.schema import ADMMConfig, SharedInputGroupConfig
from nanoquant.final_norm_calibration import calibrate_global_tuning_final_norm
from nanoquant.global_distillation import GlobalDistillationRequest, run_global_topk_distillation
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.distillation_checkpoint import (
    DistillationCheckpointIdentity,
    active_distillation_checkpoint,
)
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.global_tuning import active_global_tuning, load_global_tuning
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.resident_quantization import ResidentQuantizationRequest, run_resident_quantization


def test_global_distillation_selects_budgeted_side_tensors_but_not_binary_latents() -> None:
    factorized = TrainableFactorizedLinear(
        torch.ones(3, 2),
        torch.ones(2, 4),
        torch.ones(4),
        torch.ones(2),
        torch.ones(3),
        bias=torch.zeros(3),
        outlier_indices=torch.tensor([1]),
        outlier_values=torch.ones(3, 1),
        patch_left=torch.ones(3, 1),
        patch_right=torch.ones(1, 4),
        immutable_binary_factors=True,
    )
    model = torch.nn.ModuleDict(
        {
            "proj": factorized,
            "norm": torch.nn.LayerNorm(3),
        }
    )

    selected, auxiliary = global_distillation_module._selected_parameters(
        model,
        {(0, "proj"): factorized},
    )
    selected_names = {
        name for name, parameter in model.named_parameters() if id(parameter) in selected
    }

    assert selected_names == {
        "proj.scale_pre",
        "proj.scale_mid",
        "proj.scale_post",
        "proj.bias",
        "proj.outlier_values",
        "proj.patch_left",
        "proj.patch_right",
        "norm.weight",
        "norm.bias",
    }
    assert set(auxiliary) == {"norm.weight", "norm.bias"}


@pytest.mark.parametrize("objective", ["top_k", "top_k_tail", "top_k_mass_floor"])
def test_complete_frozen_run_can_be_distilled_committed_and_reloaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    objective: str,
) -> None:
    snapshot = tmp_path / "snapshot"
    with torch.random.fork_rng():
        torch.manual_seed(0)
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
            shared_input_groups=(
                SharedInputGroupConfig(
                    "self_attn.attn_qkv",
                    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                ),
            ),
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

    cooldowns: list[float] = []
    offloads: list[str] = []
    monkeypatch.setattr("nanoquant.global_distillation.time.sleep", cooldowns.append)
    original_offload = global_distillation_module._offload_student

    def observe_offload(student: torch.nn.Module, device: str) -> None:
        offloads.append(device)
        original_offload(student, device)

    monkeypatch.setattr(global_distillation_module, "_offload_student", observe_offload)
    request = GlobalDistillationRequest(
        output,
        snapshot,
        "fixture/gemma3",
        "pinned-test-revision",
        tokens,
        TopKDistillationConfig(
            objective=objective,
            epochs=3,
            batch_size=2,
            learning_rate=0.02,
            top_k=8,
            vocabulary_chunk_size=7,
            token_chunk_size=4,
            maximum_tokens_per_batch=8,
            gradient_checkpointing=False,
            weight_decay=0.0,
            tail_mass_weight=0.5,
            minimum_teacher_mass_ratio=0.8,
            mass_floor_weight=2.0,
        ),
        device="cpu",
        initial_cooldown_seconds=1.5,
        epoch_cooldown_seconds=3.25,
    )
    with pytest.raises(InterruptedError, match="after 1 distillation epoch checkpoint"):
        run_global_topk_distillation(replace(request, interrupt_after_epoch_commits=1))
    assert offloads == ["cpu"]
    distilled = run_global_topk_distillation(request)
    assert cooldowns == [1.5, 1.5, 3.25]
    assert offloads == ["cpu", "cpu"]
    protocol = to_dict(request.config)
    protocol.pop("scheduler_total_steps")
    if objective == "top_k":
        protocol.pop("objective")
        protocol.pop("maximum_batches_per_epoch")
        protocol.pop("tail_mass_weight")
        protocol.pop("minimum_teacher_mass_ratio")
        protocol.pop("mass_floor_weight")
    else:
        protocol["teacher_normalizer_version"] = (
            global_distillation_module.TAIL_NORMALIZER_VERSION
        )
        if objective == "top_k_tail":
            protocol.pop("minimum_teacher_mass_ratio")
            protocol.pop("mass_floor_weight")
        else:
            protocol.pop("tail_mass_weight")
    assert distilled.result.protocol_hash == semantic_hash(protocol)
    profiles = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(output.glob("profile*.json"))]
    distillation_profiles = [profile for profile in profiles if profile["run_id"] == "global-distillation"]
    assert len(distillation_profiles) == 2
    phase_paths = {str(phase["path"]) for profile in distillation_profiles for phase in profile["phases"]}
    assert {
        "run/load_frozen",
        "run/load_frozen/inventory",
        "run/load_frozen/commits",
        "run/load_frozen/model_load",
        "run/load_frozen/install_block",
        "run/load_frozen/install_block/install_layer",
        "run/thaw",
        "run/teacher_cache_epoch",
        "run/student_setup",
        "run/train",
        "run/train/checkpoint_commit",
        "run/offload",
        "run/freeze",
        "run/commit",
    } <= phase_paths

    active = active_global_tuning(output)
    assert active == distilled.reference
    cache_journal = json.loads((output / "global-distillation-cache.json").read_text(encoding="utf-8"))
    assert len(cache_journal["epochs"]) == 3
    assert all(reference is not None for reference in cache_journal["epochs"])
    first_epoch_reference = cache_journal["epochs"][0]
    assert isinstance(first_epoch_reference, dict)
    first_epoch_manifest = json.loads(
        (
            LocalArtifactStore(output / "artifacts").path_for(
                str(first_epoch_reference["artifact_id"])
            )
            / "epoch.json"
        ).read_text(encoding="utf-8")
    )
    assert first_epoch_manifest["schema_version"] == (
        3 if objective in {"top_k_tail", "top_k_mass_floor"} else 1
    )
    persisted = load_global_tuning(distilled.reference, LocalArtifactStore(output / "artifacts"))
    assert persisted.result == distilled.result
    assert distilled.metrics.steps_completed == 6
    assert distilled.metrics.epoch_losses[-1] <= distilled.metrics.epoch_losses[0]
    assert distilled.result.source_blocks == tuple(block.teacher_outputs.artifact for block in before.blocks)
    assert len(distilled.result.tuned_blocks) == 1
    assert len(distilled.result.tuned_blocks[0].shared_input_groups) == 1
    assert distilled.result.auxiliary_parameters
    assert distilled.result.schema_version == 2
    assert distilled.result.block_snapshot_protocol_hash is not None
    assert len(distilled.result.block_metrics) == 1
    block_metrics = distilled.result.block_metrics[0]
    assert block_metrics.block.index == 0
    assert block_metrics.final_frozen_pre_kd >= 0
    assert block_metrics.final_post_kd >= 0
    assert block_metrics.post_kd_vs_pre_kd.baseline_name == "final_frozen_pre_kd"
    assert block_metrics.post_kd_vs_pre_kd.candidate_name == "final_post_kd"
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
    optimizer_states = {state.parameter_name: state for state in training_checkpoint.state.optimizer_states}
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

    tuned = load_global_tuning(distilled.reference, LocalArtifactStore(output / "artifacts"))
    global_distillation_module._thaw_frozen_layers(
        loaded,
        LocalTensorStore(LocalArtifactStore(output / "artifacts")),
        frozen_states=tuned.result.tuned_blocks,
    )
    after_thaw = dict(loaded.model.named_parameters())
    assert parameter_values.keys() <= after_thaw.keys()
    assert all(
        torch.equal(after_thaw[name].detach().cpu(), value)
        for name, value in parameter_values.items()
    )

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

    if objective == "top_k":
        control_output = tmp_path / "control-run"
        shutil.copytree(output, control_output)
        primary_cache_pointer = (output / "global-distillation-cache.json").read_bytes()
        primary_training_pointer = (output / "global-distillation-training.json").read_bytes()
        correction_request = GlobalDistillationRequest(
            output,
            snapshot,
            "fixture/gemma3",
            "pinned-test-revision",
            tokens,
            TopKDistillationConfig(
                objective="top_k_mass_floor",
                epochs=2,
                batch_size=2,
                learning_rate=0.005,
                top_k=8,
                vocabulary_chunk_size=7,
                token_chunk_size=4,
                maximum_tokens_per_batch=8,
                maximum_batches_per_epoch=1,
                gradient_checkpointing=False,
                weight_decay=0.0,
                minimum_teacher_mass_ratio=0.8,
                mass_floor_weight=2.0,
            ),
            device="cpu",
            initializer_global_tuning=distilled.reference,
            state_namespace="global-distillation-mass-floor",
        )
        control = run_global_topk_distillation(
            replace(correction_request, run_output=control_output)
        )
        with pytest.raises(InterruptedError, match="after 1 distillation epoch checkpoint"):
            run_global_topk_distillation(
                replace(correction_request, interrupt_after_epoch_commits=1)
            )
        assert active_global_tuning(output) == distilled.reference
        assert (output / "global-distillation-cache.json").read_bytes() == primary_cache_pointer
        assert (output / "global-distillation-training.json").read_bytes() == primary_training_pointer
        assert (output / "global-distillation-mass-floor-cache.json").exists()
        assert (output / "global-distillation-mass-floor-training.json").exists()

        resumed = run_global_topk_distillation(correction_request)
        assert active_global_tuning(output) == resumed.reference
        assert resumed.result.protocol_hash == control.result.protocol_hash
        assert resumed.result.tuned_blocks == control.result.tuned_blocks
        assert resumed.result.auxiliary_parameters == control.result.auxiliary_parameters
        assert resumed.result.epoch_losses == control.result.epoch_losses
        assert resumed.result.steps_completed == control.result.steps_completed
        assert resumed.result.block_metrics == control.result.block_metrics
        reused = run_global_topk_distillation(correction_request)
        assert reused.reference == resumed.reference
        assert reused.result == resumed.result
        assert active_global_tuning(output) == resumed.reference
        assert (output / "global-distillation-result.json").exists()
        assert (output / "global-distillation-mass-floor-result.json").exists()

        source_auxiliary = dict(resumed.result.auxiliary_parameters)
        calibrated = calibrate_global_tuning_final_norm(output, resumed.reference, 1.015)
        calibrated_auxiliary = dict(calibrated.result.auxiliary_parameters)
        tensor_store = LocalTensorStore(LocalArtifactStore(output / "artifacts"))
        with tensor_store.read(source_auxiliary["model.norm.weight"]) as source_norm:
            with tensor_store.read(calibrated_auxiliary["model.norm.weight"]) as calibrated_norm:
                assert torch.allclose(
                    1 + calibrated_norm.float(),
                    (1 + source_norm.float()) * 1.015,
                    atol=2e-3,
                    rtol=0,
                )
        assert calibrate_global_tuning_final_norm(
            output,
            resumed.reference,
            1.015,
        ).reference == calibrated.reference
        assert active_global_tuning(output) == calibrated.reference

        primary_override = load_frozen_run(
            output,
            snapshot,
            source_name="fixture/gemma3",
            revision="pinned-test-revision",
            device="cpu",
            global_tuning_override=distilled.reference,
        )
        assert primary_override.global_tuning == distilled.reference
        assert primary_override.global_tuning != active_global_tuning(output)
        with pytest.raises(ValueError, match="requires global tuning"):
            load_frozen_run(
                output,
                snapshot,
                source_name="fixture/gemma3",
                revision="pinned-test-revision",
                device="cpu",
                use_global_tuning=False,
                global_tuning_override=distilled.reference,
            )
