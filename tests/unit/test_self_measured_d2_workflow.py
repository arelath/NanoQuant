from pathlib import Path
from types import SimpleNamespace

import pytest

import nanoquant.self_measured_d2_workflow as workflow
from nanoquant.compression_quality_workflow import CompressionQualityExperiment
from nanoquant.self_measured_d2_workflow import SelfMeasuredD2ProfileOptions
from tests.support.experiments import load_experiment


def test_uniform_control_propagates_block_bounded_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    definition = load_experiment(54)
    bounded = SimpleNamespace(
        config=definition.config,
        workflow=CompressionQualityExperiment(
            definition.workflow.export,
            definition.workflow.summary_output,
            definition.workflow.quality_output,
            definition.workflow.quality_markdown_output,
            interrupt_after_block_commits=1,
        ),
    )
    inputs = SimpleNamespace(output=tmp_path / "control", snapshot=tmp_path / "snapshot")
    observed = []

    monkeypatch.setattr(workflow, "_resolved_model_block_count", lambda _config: 26)
    monkeypatch.setattr(
        workflow,
        "resolve_resident_experiment_inputs",
        lambda *_args, **_kwargs: inputs,
    )

    class ExpectedInterruption(Exception):
        pass

    def execute(_config, _inputs, options):  # type: ignore[no-untyped-def]
        observed.append(options)
        raise ExpectedInterruption

    monkeypatch.setattr(workflow, "execute_resident_workflow", execute)

    with pytest.raises(ExpectedInterruption):
        workflow._prepare_automatic_kl_inputs(
            bounded,
            launcher_path=tmp_path / "experiments" / "054.py",
            campaign_root=tmp_path,
            control_config=definition.config,
            profile_options=SelfMeasuredD2ProfileOptions(),
        )

    assert observed[0].interrupt_after_block_commits == 1


def test_kl_profile_inherits_offline_dataset_setting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    definition = load_experiment(54)
    inputs = SimpleNamespace(output=tmp_path / "control", snapshot=tmp_path / "snapshot")
    observed = []

    monkeypatch.setattr(workflow, "_resolved_model_block_count", lambda _config: 26)
    monkeypatch.setattr(workflow, "_journal_identity", lambda *_args: object())
    monkeypatch.setattr(workflow, "_require_control_recipe", lambda *_args: None)
    monkeypatch.setattr(
        workflow,
        "resolve_resident_experiment_inputs",
        lambda *_args, **_kwargs: inputs,
    )
    monkeypatch.setattr(workflow, "execute_kl_budget", lambda args: observed.append(args) or 0)

    workflow._prepare_automatic_kl_inputs(
        definition,
        launcher_path=tmp_path / "experiments" / "054.py",
        campaign_root=tmp_path,
        control_config=definition.config,
        profile_options=SelfMeasuredD2ProfileOptions(),
    )

    assert observed[0].local_files_only is True
