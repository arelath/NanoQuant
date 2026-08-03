import runpy
from pathlib import Path
from typing import Any

import pytest

from tests.support.experiments import config_diff_paths, load_experiment
from tools import run_experiment055_campaign as campaign


def test_experiment055_is_an_equal_budget_overcomplete_replay_of_054() -> None:
    baseline = load_experiment(54)
    candidate = load_experiment(55)

    assert candidate.identity.baseline is not None
    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert candidate.config.allocation.target_bpw == baseline.config.allocation.target_bpw
    assert candidate.config.allocation.bounds.ceiling_fraction_of_uniform == 1.5
    assert candidate.config.allocation.bounds.overcomplete_rank_ceiling_fraction == 1.5
    assert candidate.config.allocation.kl_profile_artifact == (
        "evidence/054/054-d2-uniform-control-kl-profile"
    )
    assert candidate.config.block_tuning.factorized.learning_rates.binary == 3e-5
    probe_admm = candidate.config.allocation.reconstruction.probe_admm
    assert probe_admm is not None
    assert probe_admm.outer_iterations == 100
    assert probe_admm.inner_iterations == 5
    assert probe_admm.regularization == 3e-2
    assert probe_admm.penalty_schedule == "cubic"
    assert probe_admm.transpose_wide is True
    search = candidate.config.factorization.binary_search
    assert search.enabled is True
    assert search.control_outer_passes == 8
    assert search.one_bit_passes == 16
    assert search.variable_depth_passes == 2
    assert search.variable_depth_length == 64
    assert search.tabu_outer_passes == 8
    assert search.tabu_passes == 2
    assert search.tabu_steps == 256
    assert search.tabu_tenure == 8
    assert search.tabu_tenure_jitter == 4
    assert candidate.workflow.task_limit == 1000
    assert candidate.workflow.interrupt_after_block_commits == 1
    assert config_diff_paths(baseline.config, candidate.config) == {
        "allocation.bounds.ceiling_fraction_of_uniform",
        "allocation.bounds.overcomplete_rank_ceiling_fraction",
        "allocation.kl_profile_artifact",
        "allocation.kl_profile_key",
        "allocation.reconstruction.probe_admm.outer_iterations",
        "factorization.binary_search.enabled",
        "intent.baseline_run",
        "intent.experiment_number",
        "intent.hypothesis",
        "intent.name",
        "intent.purpose",
        "intent.tags",
        "output.run_root",
    }


def test_experiment055_launcher_controls_slices_and_marks_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = next(Path("experiments").glob("055-*.py"))
    namespace = runpy.run_path(str(launcher))
    entry = namespace["_run_worker_or_campaign"]

    monkeypatch.delenv(campaign.WORKER_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(campaign, "main", lambda: 17)
    assert entry() == 17

    observed: dict[str, Any] = {}

    def run_worker(definition: object, *, launcher_path: str) -> int:
        observed.update(definition=definition, launcher_path=launcher_path)
        return 23

    monkeypatch.setenv(campaign.WORKER_ENVIRONMENT_VARIABLE, "1")
    monkeypatch.setitem(entry.__globals__, "run_experiment", run_worker)
    assert entry() == 23
    assert observed["definition"] is namespace["EXPERIMENT"]
    assert Path(observed["launcher_path"]).name == launcher.name
