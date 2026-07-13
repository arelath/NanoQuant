from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nanoquant.application.quantization_stages import (
    FactorizationAttemptStage,
    MaterializedScaleFitStageRequest,
    OutlierSelectionStage,
    ScaleFitStage,
)
from nanoquant.application.stages import StageContext, execute_stage
from nanoquant.config.schema import ADMMConfig
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
    with (
        tensors.read(outliers.indices) as indices,
        tensors.read(outliers.factor_input_importance) as factor_input_importance,
    ):
        assert indices.numel() == 1
        assert factor_input_importance[indices.long()].item() == pytest.approx(1e-4)
    factor_objective = replace(objective, input_importance=outliers.factor_input_importance)
    factor_request = FactorizationRequest(
        1,
        layer,
        refs["weight"],
        outliers.residual_weight,
        factor_objective,
        1,
        19,
        "factor-config",
        outliers.factor_generator_state,
    )
    factorized = execute_stage(FactorizationAttemptStage(), factor_request, context)
    assert factorized.rank == 1
    assert factorized.metrics.export_weighted_normalized_error >= 0
    scale_request = MaterializedScaleFitStageRequest(
        ScaleFitRequest(layer, outliers.residual_weight, factorized.factors, factor_objective, outliers.indices),
        outliers.factor_input_importance,
        refs["output_importance"],
    )
    fitted = execute_stage(ScaleFitStage(), scale_request, context)
    assert fitted.after.export_weighted_error <= fitted.before.export_weighted_error + 1e-5
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8").count("stage.completed") == 3


def test_residual_probe_uses_configured_inner_iterations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed = []

    def factorize(weight: torch.Tensor, *_args: object, **kwargs: object) -> SimpleNamespace:
        observed.append(kwargs["inner_iterations"])
        return SimpleNamespace(reconstruction=torch.zeros_like(weight))

    monkeypatch.setattr("nanoquant.application.quantization_stages.factorize_admm", factorize)
    context, tensors = _context(tmp_path)
    refs = tensors.put(
        "residual-probe-fixture",
        {
            "weight": torch.tensor([[1.0, 2.0], [3.0, 1.0]]),
            "input_importance": torch.ones(2),
            "output_importance": torch.ones(2),
        },
    )
    layer = LayerId(BlockId(0), "linear")
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
        ArtifactRef("calibration", "sha256-" + "0" * 64, 1),
    )

    execute_stage(
        OutlierSelectionStage(residual_probe_iterations=11, residual_probe_inner_iterations=7),
        OutlierSelectionRequest(
            layer,
            refs["weight"],
            objective,
            OutlierPlan("residual", 1, "bfloat16", True),
            1,
            3,
        ),
        context,
    )

    assert observed == [7]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA resident stage requires a GPU")
def test_factorization_stage_honors_cuda_device_and_admm_settings(tmp_path: Path) -> None:
    context, tensors = _context(tmp_path)
    refs = tensors.put(
        "gpu-factor-fixture",
        {
            "weight": torch.randn(8, 8, generator=torch.Generator().manual_seed(2)),
            "input_importance": torch.ones(8),
            "output_importance": torch.ones(8),
        },
    )
    layer = LayerId(BlockId(0), "linear")
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
        ArtifactRef("calibration", "sha256-" + "0" * 64, 1),
    )
    result = execute_stage(
        FactorizationAttemptStage(ADMMConfig(outer_iterations=2, inner_iterations=1), device="cuda"),
        FactorizationRequest(1, layer, refs["weight"], refs["weight"], objective, 2, 3, "gpu-test"),
        context,
    )

    assert result.convergence.iterations_completed == 2
    assert result.wall_seconds > 0
    assert result.peak_workspace_bytes > 0
