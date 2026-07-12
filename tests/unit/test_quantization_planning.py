from pathlib import Path

from nanoquant.application.planning import PlanningRequest, build_quantization_plan, persist_plan
from nanoquant.config.schema import AllocationStrategy, OutlierConfig, RankAllocationConfig, RankBoundsConfig
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


def test_utility_profile_must_cover_every_layer() -> None:
    request = _request(AllocationStrategy.UTILITY_PROFILE)
    try:
        build_quantization_plan(request)
    except ValueError as error:
        assert "missing layer" in str(error)
    else:
        raise AssertionError("incomplete utility profile was accepted")
