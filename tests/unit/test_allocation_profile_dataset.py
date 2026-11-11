from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import nanoquant.self_measured_d2_workflow as workflow
from nanoquant.config.codec import config_hash
from nanoquant.config.schema import RankResponseSource
from nanoquant.domain.planning import RankResponseSegment, integrated_rank_response, rebase_rank_response
from tests.support.experiments import load_experiment


def test_rebased_curves_preserve_absolute_error_predictions() -> None:
    segments = (RankResponseSegment(1.0, 0.01), RankResponseSegment(1.5, 0.02))
    ratio, rebased = rebase_rank_response(100, 80, segments, floor_fraction=0.5, ceiling_fraction=2.0)
    assert ratio == pytest.approx(math.exp(0.4))
    for rank in (50, 80, 100, 120, 150):
        original = math.exp(-2 * integrated_rank_response(100, rank, 0.5, segments))
        prediction = ratio * math.exp(-2 * integrated_rank_response(80, rank, 0.5, rebased))
        assert prediction == pytest.approx(original)
    assert rebased[-1].maximum_rank_fraction == 2.0
    # Beyond the measured upper endpoint, continue the last observed slope.
    assert ratio * math.exp(-2 * integrated_rank_response(80, 160, 0.5, rebased)) == pytest.approx(math.exp(-2.4))


def test_fixed_rank_range_and_single_flat_point_are_supported() -> None:
    ratio, segments = rebase_rank_response(
        100, 200, (RankResponseSegment(1.0, 0.0),), floor_fraction=1.0, ceiling_fraction=1.0
    )
    assert ratio == 1.0
    assert integrated_rank_response(200, 200, 1.0, segments) == 0.0


def test_dataset_key_changes_semantic_config_identity() -> None:
    config = load_experiment(70).config
    first = replace(
        config,
        allocation=replace(
            config.allocation,
            reconstruction=replace(
                config.allocation.reconstruction,
                response_source=RankResponseSource.DATASET,
                allocation_dataset="Datasets/profile",
                allocation_dataset_key="first",
            ),
        ),
    )
    second = replace(
        first,
        allocation=replace(
            first.allocation,
            reconstruction=replace(
                first.allocation.reconstruction,
                allocation_dataset_key="second",
            ),
        ),
    )
    assert config_hash(first) != config_hash(second)


def test_dataset_launcher_skips_control_and_kl_profile_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = load_experiment(70)
    selected = replace(
        experiment.config, allocation=replace(experiment.config.allocation, kl_profile_key="dataset-key")
    )
    observed = []
    monkeypatch.setattr(workflow, "config_with_allocation_dataset", lambda config, root: selected)
    monkeypatch.setattr(
        workflow, "run_compression_quality_experiment", lambda config, *_args, **_kwargs: observed.append(config) or 0
    )

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the allocation dataset supplies control and KL planning evidence")

    monkeypatch.setattr(workflow, "_prepare_automatic_kl_inputs", unexpected)
    monkeypatch.setattr(workflow, "_validated_kl_profile", unexpected)
    assert (
        workflow.run_self_measured_d2_experiment(
            SimpleNamespace(config=experiment.config, workflow=experiment.workflow),
            launcher_path=tmp_path / "experiments/070.py",
            arguments=["--allocation-profile-dataset", "Datasets/profile"],
        )
        == 0
    )
    assert observed == [selected]
