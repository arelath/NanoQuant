from contextlib import ExitStack

import pytest
import torch
from profile_admm_budgets import fit_owner, score, summarize

from nanoquant.application.quantization_stages import FactorizationAttemptStage, ScaleFitStage
from nanoquant.application.residual_factorization import fit_residual_stages
from nanoquant.application.stages import StageContext
from nanoquant.config.schema import ADMMConfig
from nanoquant.domain.factorization import AdmmParameters
from nanoquant.domain.models import ArtifactRef, BlockId, FactorizationRequest, LayerId, ObjectiveSpec
from nanoquant.domain.scale_fit import reconstruct
from nanoquant.domain.seeds import logical_seed
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.events import JsonlEventSink
from nanoquant.infrastructure.resident_executor import Cancellation, ResidentExecutor
from nanoquant.infrastructure.tensor_store import LocalTensorStore


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_budget_probe_matches_production_residual_stages_at_frozen_precision(tmp_path, dtype):
    target = (torch.randn(4, 6, generator=torch.Generator().manual_seed(17)) * 0.1).to(dtype)
    v, u = torch.arange(1, 7).float(), torch.arange(1, 5).float()
    parameters = AdmmParameters(outer_iterations=4, inner_iterations=2)
    result, stages = fit_owner(target, v, u, (4, 2), 11, 0, "proj", parameters, {})
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    tensors = LocalTensorStore(artifacts)
    context = StageContext("test", ResidentExecutor(), artifacts, tensors,
                           JsonlEventSink(tmp_path / "events.jsonl", "test"), Cancellation())
    refs = tensors.put("source", dict(weight=target, input=v, output=u))
    layer = LayerId(BlockId(0), "proj")
    objective = ObjectiveSpec(1, layer, "diagonal", refs["input"], refs["output"], None, .01,
                              "target_weighted_norm_squared", None, ArtifactRef("calibration", "fixture", 1))
    request = FactorizationRequest(1, layer, refs["weight"], refs["weight"], objective, 6, 11, "config")
    production = fit_residual_stages(
        request, context, factor_stage=FactorizationAttemptStage(ADMMConfig(outer_iterations=4, inner_iterations=2)),
        scale_stage=ScaleFitStage(), fit_scales=True, multiple=2, device="cpu",
    )
    frozen = torch.zeros_like(target, dtype=torch.float32)
    for factors in (production.factors, *production.factors.residual_stages):
        with ExitStack() as stack:
            left, right, pre, mid, post = [stack.enter_context(tensors.read(ref)) for ref in (
                factors.left_binary, factors.right_binary, factors.scales.pre, factors.scales.mid, factors.scales.post,
            )]
            frozen += reconstruct(left, right, pre.bfloat16(), mid.bfloat16(), post.bfloat16())
    torch.testing.assert_close(result, frozen, rtol=0, atol=0)
    assert [s["seed"] for s in stages] == [11, logical_seed(11, "residual-factor-stage", 0, "proj", 1)]
    assert [s["iterations"] for s in stages] == [4, 4]
    metrics = score(result, target, v, u, torch.eye(6)[None])
    assert metrics["activation_nmse"] == pytest.approx(metrics["relative_frobenius_error"] ** 2)


def test_summary_requires_matched_pairs_and_reports_relative_error_increases():
    rows = [dict(block=0, owner="proj", seed=seed, iterations=budget, inner_iterations=5, seconds=budget / 100,
                 metrics=dict(weighted_nmse=error, activation_nmse=error))
            for seed, budget, error in ((0, 800, .1), (1, 800, .2), (0, 400, .11), (1, 400, .18))]
    result = summarize(rows)
    assert result[0]["time_reduction_percent"] == 50
    assert result[0]["errors"]["weighted_nmse"]["mean_percent"] == pytest.approx(0, abs=1e-12)
    assert result[0]["errors"]["weighted_nmse"]["max_percent"] == pytest.approx(10)
    inner_rows = [dict(row, inner_iterations=2) for row in rows[:2]]
    assert [(r["iterations"], r["inner_iterations"]) for r in summarize(rows + inner_rows)] == [
        (400, 5), (800, 2), (800, 5),
    ]
    with pytest.raises(ValueError, match="unmatched"):
        summarize(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows + rows[:1])
