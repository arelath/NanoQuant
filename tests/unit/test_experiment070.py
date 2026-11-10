import runpy
from dataclasses import replace
from pathlib import Path

from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.self_measured_d2_workflow import _uniform_control_config
from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment070_changes_only_the_requested_representation_and_runtime_profile_binding(tmp_path: Path):
    namespace = runpy.run_path(str(next(Path("experiments").glob("070-*.py"))))
    reference, candidate = namespace["REFERENCE"], namespace["CONFIG"]
    assert config_diff_paths(reference, candidate) == {
        "allocation.bounds.overcomplete_rank_ceiling_fraction",
        "allocation.retry.outlier_count_increment_at_rank_cap",
        "allocation.kl_profile_artifact",
        "allocation.kl_profile_key",
        "factorization.residual_stages.enabled",
        "outliers.selector",
        "outliers.fraction",
    }
    experiment = load_experiment(70)
    assert experiment.workflow.task_limit == 1000
    assert experiment.workflow.interrupt_after_block_commits == 1
    assert experiment.workflow.export is not None
    assert experiment.config.distillation == reference.distillation
    control = _uniform_control_config(experiment.config, tmp_path, tuned_operating_point=True)
    raise_for_issues(validate(control, ValidationPhase.RESOLVED))
    assert control.factorization == candidate.factorization
    assert control.allocation.bounds.overcomplete_rank_ceiling_fraction is None
    invalid = replace(
        candidate,
        factorization=replace(
            candidate.factorization, residual_stages=replace(candidate.factorization.residual_stages, enabled=False)
        ),
    )
    assert any(issue.code == "CFG180" for issue in validate(invalid, ValidationPhase.RESOLVED))
