from dataclasses import replace
from pathlib import Path

from nanoquant.application.planning import PlanningRequest, build_quantization_plan, persist_plan
from nanoquant.config.schema import (
    AllocationStrategy,
    OutlierConfig,
    RankAllocationConfig,
    RankBoundsConfig,
    RankRetryConfig,
)
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
    SourceTensor,
    StatisticSummary,
    TensorId,
    TensorRef,
    TensorSpec,
)
from nanoquant.infrastructure.artifacts import LocalArtifactStore


def _request(strategy: AllocationStrategy = AllocationStrategy.UNIFORM) -> PlanningRequest:
    component = ComponentRef("tiny", "1")
    model = ModelIdentity("fixture", "rev", "hash", "fixture", "tok", component)
    artifact = ArtifactRef("fixture", "sha256-" + "0" * 64, 1)
    blocks = []
    stats = []
    objectives = []
    for index, importance in enumerate((1.0, 10.0)):
        layer_id = LayerId(BlockId(index), "mlp.up_proj")
        source = SourceTensor(
            TensorId(layer_id, "weight"),
            f"blocks.{index}.weight",
            "model.safetensors",
            TensorSpec((64, 64), "float32"),
            f"hash-{index}",
        )
        layer = LayerInventory(layer_id, source, None, 64, 64)
        blocks.append(BlockInventory(BlockId(index), (source,), (layer,)))
        tensor = TensorRef(artifact, f"importance-{index}", TensorSpec((64,), "float32"), f"tensor-{index}")
        summary = StatisticSummary(importance, importance, importance, 0, 0)
        stats.append(LayerCalibrationStats(layer_id, tensor, tensor, None, summary, summary))
        objectives.append(
            ObjectiveSpec(
                1, layer_id, "diagonal", tensor, tensor, None, 0.01, "target_weighted_norm_squared", None, artifact
            )
        )
    inventory = ModelInventory(1, model, tuple(blocks), (), 32768)
    dataset = DatasetIdentity("data", ("fixture",), ("rev",), "tok", "v1")
    calibration = CalibrationStats(1, component, model, dataset, "online_fisher", "float32", tuple(stats), 2, 128)
    allocation = RankAllocationConfig(
        target_bpw=1.0,
        strategy=strategy,
        bounds=RankBoundsConfig(
            multiple=1, floor_fraction_of_uniform=0.5, ceiling_fraction_of_uniform=2.0, edge_block_boost=0
        ),
    )
    return PlanningRequest(inventory, calibration, artifact, tuple(objectives), allocation, OutlierConfig())


def test_uniform_plan_is_complete_budgeted_and_persisted_before_mutation(tmp_path: Path) -> None:
    request = _request()
    plan = build_quantization_plan(request)
    assert len(plan.blocks) == 2 and all(len(block.layers) == 1 for block in plan.blocks)
    ranks = [layer.rank for block in plan.blocks for layer in block.layers]
    assert ranks[0] == ranks[1]
    assert plan.planned_cost.total <= 64 * 64 * 2
    stored = persist_plan(plan, LocalArtifactStore(tmp_path / "artifacts"))
    assert stored.plan == plan


def test_sensitivity_plan_allocates_no_less_rank_to_more_sensitive_layer() -> None:
    plan = build_quantization_plan(_request(AllocationStrategy.SENSITIVITY))
    ranks = [layer.rank for block in plan.blocks for layer in block.layers]
    assert ranks[1] >= ranks[0]


def test_edge_boost_changes_utility_without_relaxing_rank_ceiling() -> None:
    request = _request(AllocationStrategy.SENSITIVITY)
    unboosted = build_quantization_plan(request)
    boosted_allocation = replace(
        request.allocation,
        bounds=replace(request.allocation.bounds, edge_block_boost=0.75),
    )
    boosted = build_quantization_plan(replace(request, allocation=boosted_allocation))

    assert [layer.allocator_cap for block in boosted.blocks for layer in block.layers] == [
        layer.allocator_cap for block in unboosted.blocks for layer in block.layers
    ]


def test_uniform_bpw_uses_shape_specific_ranks_for_heterogeneous_layers() -> None:
    request = _request()
    dimensions = ((1152, 6912), (1152, 256))
    blocks = []
    for block, (inputs, outputs) in zip(request.inventory.blocks, dimensions, strict=True):
        layer = block.quantizable_layers[0]
        source = replace(layer.weight, spec=TensorSpec((outputs, inputs), "float32"))
        updated = replace(layer, weight=source, in_features=inputs, out_features=outputs)
        blocks.append(replace(block, source_tensors=(source,), quantizable_layers=(updated,)))
    inventory = replace(request.inventory, blocks=tuple(blocks))
    allocation = replace(request.allocation, bounds=replace(request.allocation.bounds, multiple=32))

    plan = build_quantization_plan(replace(request, inventory=inventory, allocation=allocation))
    ranks = [layer.rank for block in plan.blocks for layer in block.layers]
    elements = sum(inputs * outputs for inputs, outputs in dimensions)

    assert ranks[0] > ranks[1]
    assert ranks == [960, 160]
    assert 0.98 <= plan.planned_cost.total / elements <= 1.0


def test_utility_profile_must_cover_every_layer() -> None:
    request = _request(AllocationStrategy.UTILITY_PROFILE)
    try:
        build_quantization_plan(request)
    except ValueError as error:
        assert "missing layer" in str(error)
    else:
        raise AssertionError("incomplete utility profile was accepted")


def test_retry_plan_honors_enabled_and_above_allocator_cap_policy() -> None:
    request = _request()
    bounded = build_quantization_plan(request).blocks[0].layers[0]
    assert bounded.retry.maximum_attempts == 2
    assert bounded.retry.hard_rank_cap == bounded.allocator_cap

    disabled = replace(request.allocation, retry=RankRetryConfig(enabled=False, maximum_attempts=3))
    disabled_layer = build_quantization_plan(replace(request, allocation=disabled)).blocks[0].layers[0]
    assert disabled_layer.retry.maximum_attempts == 1

    above_cap = replace(
        request.allocation,
        retry=RankRetryConfig(enabled=True, maximum_attempts=3, allow_above_allocator_cap=True),
    )
    above_cap_layer = build_quantization_plan(replace(request, allocation=above_cap)).blocks[0].layers[0]
    assert above_cap_layer.retry.maximum_attempts == 3
    assert above_cap_layer.retry.hard_rank_cap == 64
    assert above_cap_layer.retry.hard_rank_cap >= above_cap_layer.allocator_cap
