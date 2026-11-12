import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_experiment070_campaign as campaign

import nanoquant.self_measured_d2_workflow as workflow
from nanoquant.application.planning import build_quantization_plan
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.domain.models import ArtifactRef
from nanoquant.domain.residual_factorization import residual_stage_ranks
from tests.support.experiments import config_diff_paths, load_experiment
from tests.unit.test_quantization_planning import _request
from tests.unit.test_shared_input_factorization import _planning_request


def test_071_preserves_070_except_requested_settings_and_identity():
    namespace = runpy.run_path(str(next(Path("experiments").glob("071-*.py"))))
    reference, candidate = namespace["REFERENCE"], namespace["CONFIG"]
    assert config_diff_paths(reference, candidate) == {
        "allocation.bounds.overcomplete_rank_ceiling_fraction",
        "allocation.maximum_rank_layer_patterns", "factorization.admm.outer_iterations",
        "factorization.admm.inner_iterations", "block_tuning.compile_forward",
    }
    before, after = load_experiment(70), load_experiment(71)
    assert config_diff_paths(before.workflow, after.workflow) == {
        "export.logical_output", "export.packed_output", "export.checkpoint_output", "export.gguf_output",
        "summary_output", "quality_output", "quality_markdown_output",
    }
    assert after.config.factorization.admm.outer_iterations == 600
    assert after.config.factorization.admm.inner_iterations == 2
    assert after.config.block_tuning.compile_forward
    assert after.config.distillation == before.config.distillation
    assert after.config.calibration == before.config.calibration
    assert after.config.factorization.shared_input == before.config.factorization.shared_input
    assert namespace["ALLOCATION_PROFILE_DATASET"].name == "gemma-3-1b-it-allocation-070"
    assert namespace["KL_TEACHER_DATASET"].name == "gemma-3-1b-it-kl-48x512"
    assert after.workflow.interrupt_after_block_commits == 1
    assert after.workflow.task_limit == 1000
    raise_for_issues(validate(after.config, ValidationPhase.RESOLVED))


@pytest.mark.parametrize("make_request", [_request, _planning_request])
@pytest.mark.parametrize("number", [71, 72])
def test_maximum_policy_gives_two_full_stages_for_ordinary_and_shared_owners(make_request, number):
    request = make_request()
    config = load_experiment(number).config
    allocation = replace(
        request.allocation, maximum_rank_layer_patterns=config.allocation.maximum_rank_layer_patterns,
        bounds=replace(request.allocation.bounds,
                       overcomplete_rank_ceiling_fraction=config.allocation.bounds.overcomplete_rank_ceiling_fraction),
    )
    plan = build_quantization_plan(
        replace(request, allocation=allocation, outliers=config.outliers, residual_stages=True)
    )
    for block in plan.blocks:
        for layer in block.layers:
            shape = layer.source_weight.spec.shape
            assert layer.rank == layer.retry.hard_rank_cap == 2 * min(shape)
            assert residual_stage_ranks(*shape, layer.rank) == (min(shape), min(shape))
        for group in block.shared_input_groups:
            dim = min(group.out_features, group.in_features)
            assert group.rank == group.retry.hard_rank_cap == 2 * dim
            assert residual_stage_ranks(group.out_features, group.in_features, group.rank) == (dim, dim)


@pytest.mark.parametrize("mismatch", [None, "tokens", "fingerprint", "baseline", "revision"])
def test_combined_datasets_are_cross_checked_without_running_in_dry_run(tmp_path, monkeypatch, mismatch):
    experiment = load_experiment(71)
    config = replace(experiment.config, allocation=replace(
        experiment.config.allocation, kl_profile_artifact=str(tmp_path / "profile"),
    ))
    protocol = SimpleNamespace(
        source=config.model.source, revision=str(config.model.revision), samples=48, sequence_length=512,
        batch_size=1, device=config.runtime.compute_device, token_hash="tokens", dataset_fingerprint="fingerprint",
    )
    if mismatch == "revision":
        protocol.revision = "wrong"
    dataset = SimpleNamespace(protocol=protocol, reference=ArtifactRef("kl-teacher-dataset", "fixture", 1),
                              cache=SimpleNamespace(baseline_negative_log_likelihood=2.0))
    profile = SimpleNamespace(
        provenance=SimpleNamespace(dataset_slice_hash="wrong" if mismatch == "tokens" else "tokens",
                                   dataset_fingerprint="wrong" if mismatch == "fingerprint" else "fingerprint"),
        baseline_negative_log_likelihood=3.0 if mismatch == "baseline" else 2.0,
    )
    seen = []
    monkeypatch.setattr(workflow, "load_kl_teacher_dataset", lambda path: seen.append(path) or dataset)
    monkeypatch.setattr(workflow, "config_with_allocation_dataset", lambda cfg, path: seen.append(path) or config)
    monkeypatch.setattr(workflow, "load_kl_budget_profile", lambda path: profile)

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not start control, compression, or evaluation")

    monkeypatch.setattr(workflow, "_prepare_automatic_kl_inputs", forbidden)
    monkeypatch.setattr(workflow, "run_compression_quality_experiment", forbidden)

    def invoke():
        return workflow.run_self_measured_d2_experiment(
            experiment, launcher_path=tmp_path / "experiments/071.py", arguments=["--dry-run"],
            profile_options=workflow.SelfMeasuredD2ProfileOptions(48, 512, True),
            allocation_profile_dataset=tmp_path / "allocation", kl_teacher_dataset=tmp_path / "teacher",
        )

    if mismatch:
        with pytest.raises(ValueError, match="dataset"):
            invoke()
    else:
        assert invoke() == 0
        assert seen == [tmp_path / "teacher", tmp_path / "allocation"]
    assert not (tmp_path / "evidence").exists()


def test_071_campaign_dry_run_only_forwards_to_071_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "__file__", str(tmp_path / "tools/run_experiment070_campaign.py"))
    seen = []
    monkeypatch.setattr(campaign.subprocess, "run", lambda command, **kwargs:
                        seen.append(command) or SimpleNamespace(returncode=0))
    namespace = runpy.run_path("tools/run_experiment071_campaign.py")
    assert namespace["main"](["--dry-run"]) == 0
    assert len(seen) == 1 and seen[0][-1] == "--dry-run"
    assert Path(seen[0][1]).name == "071-two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it.py"
    assert not (tmp_path / "evidence").exists()
