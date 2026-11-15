from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

REPOSITORY = Path(__file__).resolve().parents[2]


def _tool() -> ModuleType:
    tools = str(REPOSITORY / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("measure_joint_kl_targets")


def _fit(status: str = "within_measured") -> dict:
    curve = {"coefficient": 1.0, "exponent": 2.0}
    return {
        "kl_targets": [0.01],
        "curves": {
            "0:a": {**curve, "kl_targets": [{"kl_above_control": 0.01, "target_error": 0.1,
                                              "range_status": "within_measured"}]},
            "1:b": {**curve, "kl_targets": [{"kl_above_control": 0.01, "target_error": 0.2,
                                              "range_status": status}]},
        },
    }


def test_joint_perturbation_uses_each_targets_error_and_sweep_noise_then_restores() -> None:
    tool = _tool()
    first, second = torch.ones((32, 32)), torch.full((32, 32), 3.0)
    weights = {"0:a": first, "1:b": second}

    with tool.perturb_jointly(weights, {"0:a": 0.10, "1:b": 0.20}, seed=4) as metrics:
        assert metrics["0:a"]["relative_frobenius_error"] == pytest.approx(0.10, rel=1e-5)
        assert metrics["1:b"]["relative_frobenius_error"] == pytest.approx(0.20, rel=1e-5)
        joint = first.clone()
    # The joint draw for one target matches that target's isolated sweep perturbation.
    with tool.perturb_weights({"0:a": first}, level=0.10, seed=4):
        assert torch.equal(first, joint)
    assert torch.all(first == 1) and torch.all(second == 3)
    with pytest.raises(ValueError, match="one target error per weight"):
        with tool.perturb_jointly(weights, {"0:a": 0.1}, seed=0):
            pass


def test_target_errors_refuse_extrapolated_projections() -> None:
    tool = _tool()
    assert tool.target_errors(_fit(), 0) == {"0:a": 0.1, "1:b": 0.2}
    with pytest.raises(ValueError, match="outside its measured range"):
        tool.target_errors(_fit("below_measured"), 0)


def test_measure_scores_all_projections_perturbed_together(monkeypatch) -> None:
    tool = _tool()
    weights = {"0:a": torch.ones((8, 8)), "1:b": torch.ones((8, 8))}
    calls = []

    def score(*_args, **_kwargs):
        changed = sum(not torch.all(weight == 1) for weight in weights.values())
        calls.append(changed)
        kl = 0.001 + 0.05 * changed
        return {"kl_nats_per_token": kl, "nll_nats_per_token": 2.0 + kl, "token_count": 7, "sequences": []}

    monkeypatch.setattr(tool, "score_model", score)
    result = tool.measure(None, None, weights, _fit(), seeds=[0, 1], device="cpu", token_chunk_size=2,
                          log=lambda _message: None)

    assert calls == [0, 2, 2]
    item, = result["results"]
    assert item["projection_count"] == 2 and item["isolated_sum"] == pytest.approx(0.02)
    assert item["mean_kl_above_control"] == pytest.approx(0.1)
    # Additive prediction is coefficient × error ** exponent summed over projections: 0.01 + 0.04.
    assert item["mean_additive_prediction"] == pytest.approx(0.05, rel=1e-4)
    assert item["mean_measured_over_additive"] == pytest.approx(2.0, rel=1e-4)
    assert all(torch.all(weight == 1) for weight in weights.values())

    report = {
        "protocol": {"teacher": {"source": "test/model", "model_dtype": "torch.float32"}},
        "measured_at": "now", "measurement_sweep": "sweep", "seeds": [0, 1], "json": "joint.json", **result,
    }
    markdown = tool.render_markdown(report)
    assert "| 0.01 | 2 | 15.00% (range 10.00%–20.00%) | 0.0500 | 0.1000 |" in markdown
    assert markdown.count("| 0.01 | 0 |") + markdown.count("| 0.01 | 1 |") == 2


def test_spec_lookup_matches_source_revision_and_precision() -> None:
    tool = _tool()
    spec = tool.GEMMA_270M
    protocol = {"teacher": {"source": spec.source, "revision": spec.revision, "model_dtype": "torch.float32"}}
    assert tool.spec_for_protocol(protocol) is spec
    protocol["teacher"]["model_dtype"] = "torch.bfloat16"
    with pytest.raises(ValueError, match="no pinned sensitivity model"):
        tool.spec_for_protocol(protocol)
