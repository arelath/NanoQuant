from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from nanoquant.application.layers import (
    BlockEditor,
    SharedInputGroupFreezer,
    TrainableSharedInputFactorGroup,
)
from nanoquant.application.planning import PlanningRequest, build_quantization_plan
from nanoquant.config.schema import OutlierConfig, RankAllocationConfig, RankBoundsConfig
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
