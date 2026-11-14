from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from nanoquant.application.layers import (
    BlockEditor,
    SharedInputGroupFreezer,
    TrainableSharedInputFactorGroup,
)
from nanoquant.application.planning import PlanningRequest, build_quantization_plan
from nanoquant.config.schema import OutlierConfig, RankAllocationConfig, RankBoundsConfig
from nanoquant.domain.calibration_math import weighted_group_output_importance
from nanoquant.domain.models import (
    ArtifactRef,
    BlockId,
    BlockInventory,
    CalibrationStats,
    ComponentRef,
    DatasetIdentity,
    LayerCalibrationStats,
    LayerId,
    LayerInventory,
    ModelIdentity,
    ModelInventory,
    ObjectiveSpec,
    SharedInputGroupCandidate,
    SourceTensor,
    StatisticSummary,
    TensorId,
    TensorRef,
    TensorSpec,
)
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.runtime import (
    LogicalLayerState,
    ProjectionMemberSpec,
    QuantizedLinearSpec,
    RuntimeModelMetadata,
    open_logical_artifact,
    pack_logical_layer,
    write_logical_artifact,
)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 6, bias=False)
        self.k_proj = nn.Linear(8, 2, bias=False)
        self.v_proj = nn.Linear(8, 2, bias=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


def _planning_request() -> PlanningRequest:
    block = BlockId(0)
    component = ComponentRef("tiny", "1")
    model = ModelIdentity("fixture", "rev", "hash", "fixture", "tok", component)
    artifact = ArtifactRef("fixture", "sha256-" + "0" * 64, 1)
    layers = []
    stats = []
    objectives = []
    for path, outputs in (("self_attn.q_proj", 6), ("self_attn.k_proj", 2), ("self_attn.v_proj", 2)):
        layer = LayerId(block, path)
        source = SourceTensor(
            TensorId(layer, "weight"),
            f"blocks.0.{path}.weight",
            "model.safetensors",
            TensorSpec((outputs, 8), "float32"),
            f"hash-{path}",
        )
        layers.append(LayerInventory(layer, source, None, 8, outputs))
        input_ref = TensorRef(artifact, f"{path}.input", TensorSpec((8,), "float32"), f"input-{path}")
        output_ref = TensorRef(
            artifact,
            f"{path}.output",
            TensorSpec((outputs,), "float32"),
            f"output-{path}",
        )
        summary = StatisticSummary(1.0, 1.0, 1.0, 0.0, 0)
        stats.append(LayerCalibrationStats(layer, input_ref, output_ref, None, summary, summary))
        objectives.append(
            ObjectiveSpec(
                1,
                layer,
                "diagonal",
                input_ref,
                output_ref,
                None,
                0.0,
                "target_weighted_norm_squared",
                None,
                artifact,
            )
        )
    inventory = ModelInventory(
        1,
        model,
        (BlockInventory(block, tuple(item.weight for item in layers), tuple(layers)),),
        (),
        320,
    )
    calibration = CalibrationStats(
        1,
        component,
        model,
        DatasetIdentity("data", ("fixture",), ("rev",), "tok", "v1"),
        "forward_only",
        "float32",
        tuple(stats),
        1,
        8,
    )
    group = SharedInputGroupCandidate(block, "self_attn.attn_qkv", tuple(item.layer for item in layers))
    return PlanningRequest(
        inventory,
        calibration,
        artifact,
        tuple(objectives),
        RankAllocationConfig(target_bpw=8.0, bounds=RankBoundsConfig(multiple=1)),
        OutlierConfig(),
        shared_input_groups=(group,),
    )


def test_group_planning_owns_one_factor_cost_and_partitions_members() -> None:
    request = _planning_request()
    plan = build_quantization_plan(request)
    block = plan.blocks[0]

    assert plan.schema_version == 2
    assert block.layers == ()
    assert block.unit_order == ("self_attn.attn_qkv",)
    assert len(block.shared_input_groups) == 1
    group = block.shared_input_groups[0]
    assert tuple(member.layer.path for member in group.members) == (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    )
    assert group.out_features == 10
    assert group.estimated_cost == plan.planned_cost
    assert group.estimated_cost.total <= 8 * sum(member.in_features * member.out_features for member in group.members)


def test_group_member_multiplier_is_persisted_and_squared_then_mean_normalized() -> None:
    request = _planning_request()
    candidate = request.shared_input_groups[0]
    weighted_request = replace(
        request,
        shared_input_groups=(replace(candidate, objective_multipliers=(1.0, 1.0, 2.0)),),
    )
    group = build_quantization_plan(weighted_request).blocks[0].shared_input_groups[0]
    values = (torch.tensor([2.0, 4.0]), torch.tensor([6.0]), torch.tensor([8.0]))

    weighted = weighted_group_output_importance(values, group.objective_multipliers)

    assert group.objective_multipliers == (1.0, 1.0, 2.0)
    assert weighted.mean().item() == torch.cat(values).mean().item()
    assert weighted[-1].item() / values[-1].item() == pytest.approx(
        4 * weighted[0].item() / values[0][0].item()
    )


def test_group_owner_views_share_parameters_and_round_trip(tmp_path: Path) -> None:
    block = _Block()
    owner = TrainableSharedInputFactorGroup(
        torch.sign(torch.randn(10, 4, generator=torch.Generator().manual_seed(1))),
        torch.sign(torch.randn(4, 8, generator=torch.Generator().manual_seed(2))),
        torch.ones(8),
        torch.ones(4),
        torch.ones(10),
    )
    members = tuple(LayerId(BlockId(0), path) for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"))
    BlockEditor().install_trainable_group(block, "self_attn.attn_qkv", members, (6, 2, 2), owner)
    value = torch.randn(3, 8, generator=torch.Generator().manual_seed(3))

    stacked = owner(value)
    assert torch.equal(block.self_attn.q_proj(value), stacked[:, :6])
    assert torch.equal(block.self_attn.k_proj(value), stacked[:, 6:8])
    assert torch.equal(block.self_attn.v_proj(value), stacked[:, 8:10])
    parameter_names = tuple(name for name, _parameter in block.named_parameters())
    assert parameter_names == (
        "_nanoquant_shared_input_groups.self_attn__attn_qkv.left_latent",
        "_nanoquant_shared_input_groups.self_attn__attn_qkv.right_latent",
        "_nanoquant_shared_input_groups.self_attn__attn_qkv.scale_pre",
        "_nanoquant_shared_input_groups.self_attn__attn_qkv.scale_mid",
        "_nanoquant_shared_input_groups.self_attn__attn_qkv.scale_post",
    )

    tensors = LocalTensorStore(LocalArtifactStore(tmp_path / "artifacts"))
    frozen = SharedInputGroupFreezer().freeze(
        members,
        "self_attn.attn_qkv",
        (6, 2, 2),
        owner,
        tensors,
    )
    restored = SharedInputGroupFreezer().load(frozen.state, tensors)
    assert torch.equal(restored.owner.dense_weight(), frozen.owner.dense_weight())
    assert tuple((member.row_start, member.row_end) for member in frozen.state.members) == (
        (0, 6),
        (6, 8),
        (8, 10),
    )


def _installed_group(*, share: bool) -> tuple[_Block, TrainableSharedInputFactorGroup, list[int]]:
    block = _Block()
    owner = TrainableSharedInputFactorGroup(
        torch.sign(torch.randn(10, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1))),
        torch.sign(torch.randn(4, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(2))),
        torch.rand(8, dtype=torch.float64, generator=torch.Generator().manual_seed(3)) + 0.5,
        torch.rand(4, dtype=torch.float64, generator=torch.Generator().manual_seed(4)) + 0.5,
        torch.rand(10, dtype=torch.float64, generator=torch.Generator().manual_seed(5)) + 0.5,
        immutable_binary_factors=True,
    )
    owner.left_latent.requires_grad_(False)
    owner.right_latent.requires_grad_(False)
    members = tuple(LayerId(BlockId(0), path) for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"))
    BlockEditor().install_trainable_group(
        block, "self_attn.attn_qkv", members, (6, 2, 2), owner, share_member_evaluations=share,
    )
    evaluations: list[int] = []
    owner.register_forward_hook(lambda *_: evaluations.append(1))
    return block, owner, evaluations


def _member_outputs(block: _Block, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention = block.self_attn
    return attention.q_proj(value), attention.k_proj(value), attention.v_proj(value)


def test_shared_group_evaluation_runs_the_owner_once_with_unchanged_values_and_gradients() -> None:
    separate, separate_owner, separate_evaluations = _installed_group(share=False)
    shared, shared_owner, shared_evaluations = _installed_group(share=True)
    value = torch.randn(3, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    weights = torch.randn(3, 10, dtype=torch.float64, generator=torch.Generator().manual_seed(7))

    expected_input = value.clone().requires_grad_()
    actual_input = value.clone().requires_grad_()
    expected = torch.cat(_member_outputs(separate, expected_input), dim=-1)
    actual = torch.cat(_member_outputs(shared, actual_input), dim=-1)

    assert (len(separate_evaluations), len(shared_evaluations)) == (3, 1)
    assert torch.equal(actual, expected)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    assert torch.allclose(actual_input.grad, expected_input.grad, rtol=1e-12, atol=0)
    for name, parameter in separate_owner.named_parameters():
        if parameter.grad is not None:
            assert torch.allclose(dict(shared_owner.named_parameters())[name].grad, parameter.grad, rtol=1e-12, atol=0)
    # Nothing may outlive the last member: a retained output would pin the block's autograd graph.
    evaluation = shared.self_attn.q_proj._evaluation
    assert evaluation is shared.self_attn.v_proj._evaluation
    assert evaluation._input is None and evaluation._output is None
    assert separate.self_attn.q_proj._evaluation is None


def test_shared_group_evaluation_never_reuses_a_stale_owner_output() -> None:
    shared, owner, evaluations = _installed_group(share=True)
    attention = shared.self_attn
    first = torch.randn(3, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    second = first + 1

    assert torch.equal(attention.q_proj(first), owner(first)[:, :6])
    evaluations.clear()
    # A different tensor, even mid-group, is a different evaluation.
    assert torch.equal(attention.k_proj(second), owner(second)[:, 6:8])
    assert torch.equal(attention.v_proj(second), owner(second)[:, 8:10])
    assert len(evaluations) == 3  # one shared evaluation plus the two reference calls above

    evaluations.clear()
    mutable = first.clone()
    before = attention.q_proj(mutable).clone()
    mutable.add_(1)
    assert torch.equal(attention.k_proj(mutable), owner(second)[:, 6:8])
    assert not torch.equal(attention.q_proj(mutable), before)
    assert len(evaluations) == 3  # the in-place edit forced a second shared evaluation

    evaluations.clear()
    tracked = first.clone().requires_grad_()
    with torch.no_grad():
        assert not attention.q_proj(tracked).requires_grad
    assert attention.k_proj(tracked).requires_grad
    assert attention.v_proj(tracked).requires_grad
    assert len(evaluations) == 2  # the no-grad result is not handed to a gradient pass


def test_group_logical_and_packed_state_preserve_member_slices(tmp_path: Path) -> None:
    spec = QuantizedLinearSpec(
        "blocks.0.self_attn.attn_qkv",
        "nanoquant-v1",
        8,
        10,
        4,
        "float32",
        "float32",
        members=(
            ProjectionMemberSpec("blocks.0.self_attn.q_proj", 0, 6),
            ProjectionMemberSpec("blocks.0.self_attn.k_proj", 6, 8),
            ProjectionMemberSpec("blocks.0.self_attn.v_proj", 8, 10),
        ),
    )
    state = LogicalLayerState(
        spec,
        torch.ones(10, 4),
        torch.ones(4, 8),
        torch.ones(8),
        torch.ones(4),
        torch.ones(10),
    )
    metadata = RuntimeModelMetadata("fixture", "rev", "tiny", "config", "tokenizer")
    written = write_logical_artifact(tmp_path / "logical", metadata, {0: (state,)})
    opened = open_logical_artifact(written.root)
    loaded = opened.load_layer(spec.name)
    packed = pack_logical_layer(loaded)

    assert loaded.spec.members == spec.members
    assert packed.spec.members == spec.members
    assert torch.equal(packed.to_logical().right_binary, state.right_binary)
