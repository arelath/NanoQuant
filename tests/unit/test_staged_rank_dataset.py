from types import SimpleNamespace

import pytest
import torch
from build_staged_rank_dataset import (
    allocator_curve,
    curve_error,
    error_metrics,
    fit_stage,
    measure_unit,
    monotone_curve,
    rank_grid,
    validate_model_hash,
)

from nanoquant.domain.factorization import AdmmParameters
from nanoquant.domain.models import BlockId, LayerId
from nanoquant.domain.planning import RankResponseSegment, integrated_rank_response


def test_grid_covers_stage_boundaries_for_each_physical_shape():
    for shape in ((6912, 1152), (1152, 6912), (1536, 1152), (1152, 1024)):
        ranks = rank_grid(shape)
        full = min(shape)
        assert len(ranks) == 17
        assert all(r % 32 == 0 for r in ranks)
        assert set((full, full * 2, full * 3)) <= set(ranks)
        assert ranks[-1] == full * 3


def test_curve_preserves_all_knots_and_allocator_matches_interpolation():
    ranks, errors = [32, 64, 96, 128, 160, 192], [.8, .5, .6, .2, .1, .05]
    curve = monotone_curve(ranks, errors)
    assert curve["measured_nmse"] == errors
    assert curve["fitted_nmse"][1] == pytest.approx(curve["fitted_nmse"][2])
    assert curve["fitted_nmse"][1] == pytest.approx((.5 * .6) ** .5)
    config = allocator_curve("fixture", 128, curve)
    segments = tuple(RankResponseSegment(**s) for s in config["segments"])
    assert len(segments) == 5
    for rank in (32, 48, 96, 112, 144, 192):
        integral = integrated_rank_response(128, rank, .25, segments)
        assert curve_error(curve, rank) == pytest.approx(.2 * torch.exp(torch.tensor(-2 * integral)).item())
    with pytest.raises(ValueError, match="outside"):
        curve_error(curve, 224)


def test_objective_and_column_noise_metrics_are_distinct_and_mean_normalized():
    target = torch.ones(2, 2)
    predicted = target.clone()
    predicted[0, 0] += 1
    result = error_metrics(predicted, target, torch.tensor([1., 3.]), torch.tensor([1., 2.]))
    assert result["raw_nmse"] == .25
    assert result["objective_nmse"] == pytest.approx(1 / 12)
    assert result["imatrix_noise_nmse"] == .125


def test_real_cpu_fit_returns_bf16_scale_reconstruction():
    torch.manual_seed(7)
    target = torch.randn(8, 8).bfloat16()
    fit, stored, details = fit_stage(target, torch.ones(8), torch.ones(8), 4, 0,
                                     AdmmParameters(outer_iterations=2, inner_iterations=1),
                                     {"enabled": True, "alternating_passes": 1})
    assert fit.dtype == stored.dtype == torch.float32
    assert torch.isfinite(stored).all()
    assert details["iterations_completed"] == 2


def test_interrupted_unit_reuses_only_completed_full_prefix(tmp_path, monkeypatch):
    import build_staged_rank_dataset as module

    for name in ("measurements", "work"):
        (tmp_path / name).mkdir()
    member = LayerId(BlockId(0), "mlp.gate_proj")
    case = SimpleNamespace(block=0, name=member.path, members=(member,), member_rows=(64,),
                           target=torch.ones(64, 64), input_importance=torch.ones(64),
                           output_importance=torch.ones(64), outlier_indices=None, patch=None)
    monkeypatch.setattr(module, "rank_grid", lambda shape: (32, 64, 96, 128, 160, 192))
    calls = []
    interrupted = True

    def fake_fit(target, columns, rows, rank, seed, parameters, scale_config):
        nonlocal interrupted
        calls.append(rank)
        if len(calls) == 3 and interrupted:
            interrupted = False
            raise InterruptedError("fixture")
        prediction = target.float() * .5
        return prediction, prediction, {"iterations_completed": 600}

    monkeypatch.setattr(module, "fit_stage", fake_fit)
    arguments = (case, tmp_path, None, {}, 0, "identity", ["weight-hash"], "cpu")
    with pytest.raises(InterruptedError):
        measure_unit(*arguments)
    result = measure_unit(*arguments)
    assert calls == [32, 64, 32, 32, 64, 32, 64]
    assert result["complete"] and len(result["points"]) == 6
    assert result["points"][-1]["stage_ranks"] == [64, 64, 64]
    assert result["points"][-1]["stored"]["raw_nmse"] == pytest.approx(1 / 64)
    assert not list((tmp_path / "work").glob("*.safetensors"))
    assert measure_unit(*arguments) == result
    assert len(calls) == 7
    import json

    path = next((tmp_path / "measurements").glob("*.json"))
    changed = json.loads(path.read_text())
    changed["points"][0]["stored"]["raw_nmse"] += .1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="checksum"):
        measure_unit(*arguments)


def test_wrong_source_model_is_rejected():
    validate_model_hash({"0:layer": ["sha256:matching#weight"]}, "matching")
    with pytest.raises(ValueError, match="source model"):
        validate_model_hash({"0:layer": ["sha256:matching#weight"]}, "different")


def test_published_dataset_is_self_contained_and_decodes_as_allocation_evidence(tmp_path):
    import copy

    from build_staged_rank_dataset import publish

    from nanoquant.config.codec import from_dict
    from nanoquant.infrastructure.allocation_profile_dataset import load_allocation_profile_dataset
    from nanoquant.resident_quantization import _RankProbeEvidence

    metrics = {"raw_squared_error": 2., "weight_norm_squared": 10., "raw_nmse": .2,
               "objective_nmse": .1, "imatrix_noise_nmse": .15, "sensitivity_energy": .4}
    points = [{"rank": rank, "stored": metrics, "members": [metrics], "stage_ranks": [32] * (rank // 32)}
              for rank in (32, 64, 96)]
    prototype = {"complete": True, "full_rank": 32, "member_rows": [32], "points": points,
                 "curves": {"objective_nmse": monotone_curve([32, 64, 96], [.3, .1, .04])}}
    records = []
    for i in range(3):
        record = copy.deepcopy(prototype)
        record["identity"] = {"unit_id": f"{i}:fixture", "source_weight_hashes": ["source"],
                              "logical_seed": i, "shape": [32, 32]}
        record["members"] = [{"block": {"index": i}, "path": "fixture"}]
        records.append(record)
    unit_ids = [r["identity"]["unit_id"] for r in records]
    identity = {"run_output": "retirable-source", "source_plan_id": "source-plan", "unit_ids": unit_ids}
    with pytest.raises(ValueError, match="incomplete"):
        publish(tmp_path, identity, records[:-1], {"model": {}}, {}, None)
    dataset = publish(tmp_path, identity,
                      records, {"model": {"source": "fixture", "revision": "fixed"}}, {}, None)
    loaded = load_allocation_profile_dataset(tmp_path, dataset.reference.artifact_id)
    assert len(loaded.payload["units"]) == 3
    assert loaded.kl_profile is None
    entry = loaded.payload["units"][0]
    assert entry["staged_measurements"]["points"][-1]["stage_ranks"] == [32, 32, 32]
    evidence = from_dict(_RankProbeEvidence, {**entry["evidence"], "probe_plan": {
        "artifact_type": "fixture", "artifact_id": "fixture", "schema_version": 1,
    }})
    assert len(evidence.response_curve.segments) == 2
    assert evidence.baseline_rank == 64
