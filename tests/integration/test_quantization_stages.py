from pathlib import Path

import torch

from nanoquant.application.quantization_stages import (
    FactorizationAttemptStage,
    MaterializedScaleFitStageRequest,
    OutlierSelectionStage,
    ScaleFitStage,
)
from nanoquant.application.stages import StageContext, execute_stage
from nanoquant.domain.models import (
    ArtifactRef,
    BlockId,
    FactorizationRequest,
    LayerId,
    ObjectiveSpec,
    OutlierPlan,
    OutlierSelectionRequest,
    ScaleFitRequest,
)
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.events import JsonlEventSink
from nanoquant.infrastructure.resident_executor import Cancellation, ResidentExecutor
from nanoquant.infrastructure.tensor_store import LocalTensorStore


def _context(tmp_path: Path) -> tuple[StageContext, LocalTensorStore]:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    tensors = LocalTensorStore(artifacts)
    context = StageContext(
        "run", ResidentExecutor(), artifacts, tensors, JsonlEventSink(tmp_path / "events.jsonl", "run"), Cancellation()
    )
    return context, tensors


def test_outlier_factorization_and_scale_fit_stages_commit_typed_results(tmp_path: Path) -> None:
    context, tensors = _context(tmp_path)
    weight = torch.tensor([[3.0, 0.5, -1.0], [2.0, -0.5, 1.0], [1.0, 0.25, -2.0]])
    refs = tensors.put(
        "source-fixture",
        {
            "weight": weight,
            "input_importance": torch.tensor([2.0, 1.0, 1.0]),
            "output_importance": torch.ones(3),
        },
    )
    layer = LayerId(BlockId(0), "mlp.up_proj")
    source_calibration = ArtifactRef("calibration", "sha256-" + "0" * 64, 1)
    objective = ObjectiveSpec(
        1,
        layer,
        "diagonal",
        refs["input_importance"],
        refs["output_importance"],
        None,
        0.01,
        "target_weighted_norm_squared",
        None,
        source_calibration,
    )
    outlier_request = OutlierSelectionRequest(
        layer, refs["weight"], objective, OutlierPlan("fisher", 1, "float16", True), 1, 7
    )
    outliers = execute_stage(OutlierSelectionStage(), outlier_request, context)
    with tensors.read(outliers.indices) as indices:
        assert indices.numel() == 1
    factor_request = FactorizationRequest(
        1, layer, refs["weight"], outliers.residual_weight, objective, 1, 19, "factor-config"
    )
    factorized = execute_stage(FactorizationAttemptStage(), factor_request, context)
    assert factorized.rank == 1
    assert factorized.metrics.export_weighted_normalized_error >= 0
    scale_request = MaterializedScaleFitStageRequest(
        ScaleFitRequest(layer, outliers.residual_weight, factorized.factors, objective, outliers.indices),
        refs["input_importance"],
        refs["output_importance"],
    )
    fitted = execute_stage(ScaleFitStage(), scale_request, context)
    assert fitted.after.export_weighted_error <= fitted.before.export_weighted_error + 1e-5
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8").count("stage.completed") == 3
