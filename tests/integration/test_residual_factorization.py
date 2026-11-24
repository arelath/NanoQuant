import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nanoquant.application.layers import (
    LayerFreezer,
    SharedInputGroupFreezer,
    TrainableFactorizedLinear,
    TrainableSharedInputFactorGroup,
    materialize_residual_modules,
    thaw_residual_stages,
)
from nanoquant.application.quantization_stages import FactorizationAttemptStage, ScaleFitStage
from nanoquant.application.residual_factorization import fit_residual_stages, reconstruct_factors
from nanoquant.application.stages import StageContext
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.config.schema import ADMMConfig
from nanoquant.domain.models import (
    ArtifactRef,
    BlockId,
    FactorizationRequest,
    FrozenNanoQuantState,
    LayerId,
    ObjectiveSpec,
)
from nanoquant.domain.scale_fit import reconstruct
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.events import JsonlEventSink
from nanoquant.infrastructure.resident_executor import Cancellation, ResidentExecutor
from nanoquant.infrastructure.tensor_store import LocalTensorStore


def _term(seed: int) -> TrainableFactorizedLinear:
    generator = torch.Generator().manual_seed(seed)
    return TrainableFactorizedLinear(
        torch.randn(4, 2, generator=generator),
        torch.randn(2, 6, generator=generator),
        torch.rand(6, generator=generator),
        torch.rand(2, generator=generator),
        torch.rand(4, generator=generator),
    )


@pytest.mark.parametrize("backend", ["dense", "factorized", "compact"])
def test_independent_outer_scales_survive_freeze_reload_and_thaw(tmp_path: Path, backend: str):
    first, second, third = (_term(seed) for seed in range(3))
    module = TrainableFactorizedLinear(
        first.left_latent,
        first.right_latent,
        first.scale_pre,
        first.scale_mid,
        first.scale_post,
        bias=torch.ones(4),
        residual_stages=(second, third),
    )
    inputs = torch.randn(5, 6)
    expected = first(inputs) + second(inputs) + third(inputs) + 1
    torch.testing.assert_close(module(inputs), expected)
    torch.testing.assert_close(inputs @ module.dense_weight().T + 1, expected)
    module(inputs).square().sum().backward()
    assert all(parameter.grad is not None for stage in module.residual_terms for parameter in stage.parameters())
    tensors = LocalTensorStore(LocalArtifactStore(tmp_path / "artifacts"))
    frozen = LayerFreezer().freeze(LayerId(BlockId(0), "proj"), module, tensors)
    state = from_dict(FrozenNanoQuantState, to_dict(frozen.state))
    assert state.total_rank == 6 and state.rank == 2
    assert state.logical_format == "nanoquant-residual-v2"
    loaded = (
        LayerFreezer()
        .load(
            state,
            tensors,
            backend="factorized" if backend == "factorized" else "dense",
            compact_dense=backend == "compact",
        )
        .module
    )
    torch.testing.assert_close(loaded(inputs), expected)
    if backend != "compact":
        thawed = thaw_residual_stages(loaded)
        assert len(thawed) == 2
        torch.testing.assert_close(thawed[0](inputs), second(inputs))


def test_shared_group_slices_include_every_residual_term(tmp_path: Path):
    first, extra = _term(10), _term(11)
    group = TrainableSharedInputFactorGroup(
        first.left_latent,
        first.right_latent,
        first.scale_pre,
        first.scale_mid,
        first.scale_post,
        residual_stages=(extra,),
    )
    assert extra.coupled_shared_input_scales
    tensors = LocalTensorStore(LocalArtifactStore(tmp_path / "artifacts"))
    frozen = SharedInputGroupFreezer().freeze(
        (LayerId(BlockId(0), "q"), LayerId(BlockId(0), "k")),
        "qk",
        (3, 1),
        group,
        tensors,
    )
    inputs = torch.randn(3, 6)
    expected = group(inputs)
    assert frozen.state.total_rank == 4
    for member, view in frozen.views:
        torch.testing.assert_close(view(inputs), expected[:, :3] if member.path == "q" else expected[:, 3:])


def test_each_admm_stage_fits_the_remaining_residual_and_is_deterministic(tmp_path: Path):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    tensors = LocalTensorStore(artifacts)
    context = StageContext(
        "test",
        ResidentExecutor(),
        artifacts,
        tensors,
        JsonlEventSink(tmp_path / "events.jsonl", "test"),
        Cancellation(),
    )
    weight = torch.randn(4, 6, generator=torch.Generator().manual_seed(17))
    refs = tensors.put("source", {"weight": weight, "input": torch.ones(6), "output": torch.ones(4)})
    layer = LayerId(BlockId(0), "proj")
    objective = ObjectiveSpec(
        1,
        layer,
        "diagonal",
        refs["input"],
        refs["output"],
        None,
        0.01,
        "target_weighted_norm_squared",
        None,
        ArtifactRef("calibration", "fixture", 1),
    )
    request = FactorizationRequest(1, layer, refs["weight"], refs["weight"], objective, 10, 19, "config")
    factor_stage = FactorizationAttemptStage(ADMMConfig(outer_iterations=20, inner_iterations=2))
    kwargs = dict(factor_stage=factor_stage, scale_stage=ScaleFitStage(), fit_scales=True, multiple=2, device="cpu")
    result = fit_residual_stages(request, context, **kwargs)
    repeat = fit_residual_stages(request, context, **kwargs)
    assert result.rank == 10
    assert tuple(stage.rank for stage in result.stage_factorizations) == (4, 4, 2)
    assert len(result.stage_scale_fits) == 3
    prediction = reconstruct_factors(result.factors, context, "cpu")
    torch.testing.assert_close(prediction, reconstruct_factors(repeat.factors, context, "cpu"), rtol=0, atol=0)
    assert result.metrics.raw_error == pytest.approx(float((weight - prediction).square().sum()), rel=1e-5)
    assert result.metrics.raw_error < result.stage_factorizations[0].metrics.raw_error
    factors = result.factors
    with (
        tensors.read(factors.left_binary) as left,
        tensors.read(factors.right_binary) as right,
        tensors.read(factors.scales.pre) as pre,
        tensors.read(factors.scales.mid) as mid,
        tensors.read(factors.scales.post) as post,
    ):
        first = reconstruct(left, right, pre, mid, post)
        module = TrainableFactorizedLinear(
            left,
            right,
            pre,
            mid,
            post,
            residual_stages=materialize_residual_modules(
                factors,
                tensors,
                device="cpu",
                tune_binary=False,
            ),
        )
    torch.testing.assert_close(module.dense_weight(), prediction)
    assert not torch.allclose(first, prediction)


@pytest.mark.parametrize(("full_rank_stages", "adaptive"), [(None, False), (3, False), (3, True)],
                         ids=["budget-allocated", "three-full-rank", "adaptive-block-retry"])
def test_resident_staged_rank_tuning_resume_and_export_accounting(
    tmp_path: Path, full_rank_stages: int | None, adaptive: bool,
):
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

    from nanoquant.config.schema import AllocationStrategy, BlockADMMRetryConfig, SharedInputGroupConfig
    from nanoquant.domain.planning import factor_bit_cost
    from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
    from nanoquant.infrastructure.packed_model_loader import load_packed_model
    from nanoquant.infrastructure.runtime_export import export_frozen_run_logical, validate_frozen_run_logical
    from nanoquant.resident_quantization import (
        ResidentQuantizationRequest,
        load_completed_resident_quantization,
        run_resident_quantization,
    )
    from nanoquant.runtime.artifact import RuntimeModelMetadata
    from nanoquant.runtime.llamacpp import export_llamacpp_checkpoint
    from nanoquant.runtime.packed_artifact import convert_logical_to_packed

    snapshot = tmp_path / "snapshot"
    Gemma3ForCausalLM(
        Gemma3TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
    ).save_pretrained(snapshot, safe_serialization=True)
    request = ResidentQuantizationRequest(
        snapshot,
        tmp_path / "run",
        "fixture/gemma3",
        "pinned",
        ((1, 2, 3, 4), (4, 3, 2, 1)),
        device="cpu",
        target_bpw=6.0,
        rank_multiple=1,
        overcomplete_rank_ceiling_fraction=None if full_rank_stages is None else float(full_rank_stages),
        maximum_rank_layer_patterns=() if full_rank_stages is None else ("*",),
        residual_stages=True,
        admm=ADMMConfig(outer_iterations=3, inner_iterations=1,
                        block_retry=BlockADMMRetryConfig(1e-16, 5) if adaptive else None),
        allocation_strategy=AllocationStrategy.UNIFORM,
        factorized_tuning_epochs=1,
        factorized_tuning_batch_size=1,
        post_block_refit_epochs=1,
        shared_input_groups=(
            SharedInputGroupConfig(
                "self_attn.attn_qkv",
                (
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                ),
            ),
        ),
        registry_root=tmp_path / "registry",
    )
    if adaptive:
        with pytest.raises(InterruptedError, match="new .*commits"):
            run_resident_quantization(replace(request, interrupt_after_layer_commits=1))
    with pytest.raises(InterruptedError, match="new block commits"):
        run_resident_quantization(replace(request, interrupt_after_block_commits=1))
    result = run_resident_quantization(request)
    block = result.blocks[0]
    events = [json.loads(line) for line in (request.output / "events.jsonl").read_text().splitlines()]
    if adaptive:
        decisions = [e["fields"] for e in events if e["name"] == "block.admm_attempt_measured"]
        assert [(d["outer_iterations"], d["retry"]) for d in decisions] == [(3, True), (6, False)]
        assert decisions[-1]["weighted_error"] == block.losses.final_frozen_normalized_error
    for item in block.layers:
        committed = [event["fields"] for event in events if event["name"] == "layer.completed"
                     and event["fields"].get("layer") == item.layer.path
                     and event["fields"].get("status") == "committed"]
        assert committed[-1]["rank"] == item.frozen_state.total_rank
    assert any(layer.frozen_state.residual_stages for layer in block.layers)
    assert block.shared_input_groups[0].frozen_state.residual_stages
    for item in (*block.layers, *block.shared_input_groups):
        state = item.frozen_state
        if full_rank_stages is not None:
            physical_rank = min(state.left_binary.spec.shape[0], state.right_binary.spec.shape[1])
            assert (state.rank, *(stage.rank for stage in state.residual_stages)) == (physical_rank,) * full_rank_stages
        assert item.actual_bit_cost == factor_bit_cost(
            state.left_binary.spec.shape[0],
            state.right_binary.spec.shape[1],
            state.total_rank,
            residual_stages=True,
        )
    repeated = load_completed_resident_quantization(request)
    assert repeated.identity == result.identity
    assert repeated.blocks[0].frozen_state == block.frozen_state
    export_frozen_run_logical(
        request.output,
        tmp_path / "logical",
        RuntimeModelMetadata("fixture/gemma3", "pinned", "gemma3", result.identity.model_hash, "t"),
        1,
    )
    assert validate_frozen_run_logical(request.output, tmp_path / "logical", 1).exact
    convert_logical_to_packed(tmp_path / "logical", tmp_path / "packed")
    if full_rank_stages is not None:
        from safetensors import safe_open

        checkpoint_root = tmp_path / "checkpoint"
        checkpoint = export_llamacpp_checkpoint(tmp_path / "packed", checkpoint_root)
        assert checkpoint.schema_version == 2
        with safe_open(checkpoint_root / checkpoint.shards[0].path, framework="pt", device="cpu") as shard:
            count_keys = [key for key in shard.keys() if key.endswith(".residual_stage_count")]
            assert len(count_keys) == checkpoint.layer_count == 5
            assert all(int(shard.get_tensor(key).item()) == full_rank_stages - 1 for key in count_keys)
    tokens = torch.tensor([[1, 2, 3, 4]])
    for backend in ("dense", "factorized"):
        packed_model = load_packed_model(tmp_path / "packed", request.output, snapshot,
                                         source_name=request.source, revision=request.revision,
                                         device="cpu", backend=backend).model
        frozen_model = load_frozen_run(request.output, snapshot, source_name=request.source,
                                       revision=request.revision, device="cpu", backend=backend).model
        with torch.no_grad():
            torch.testing.assert_close(packed_model(input_ids=tokens).logits, frozen_model(input_ids=tokens).logits)
