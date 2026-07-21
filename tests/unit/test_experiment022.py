from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import nanoquant.self_measured_d2_workflow as d2_workflow
from nanoquant.config.schema import (
    AllocationStrategy,
    KlAllocationObjective,
    KlSensitivityGranularity,
    RankResponseSource,
)
from tests.support.experiments import load_experiment


def test_experiment022_is_self_measured_d2_matched_to_experiment017() -> None:
    experiment017 = load_experiment(17)
    experiment022 = load_experiment(22)
    config017 = experiment017.config
    config022 = experiment022.config

    assert config022.model.source == "google/gemma-3-1b-it"
    assert config022.model.revision == "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    assert config022.allocation.strategy is AllocationStrategy.KL_CALIBRATED
    assert config022.allocation.kl_profile_artifact == "runtime-kl-profile-required"
    assert config022.allocation.kl_profile_key == "runtime-kl-profile-key-required"
    assert config022.allocation.kl_sensitivity_granularity is KlSensitivityGranularity.EXACT
    reconstruction = config022.allocation.reconstruction
    assert reconstruction.response_source is RankResponseSource.MEASURED
    assert reconstruction.response_curves == ()
    assert reconstruction.objective_mode == "calibration_weighted"
    assert reconstruction.kl_objective is KlAllocationObjective.MEASURED_UNIT_KL
    assert reconstruction.sensitivity_strength == 1
    assert reconstruction.protect_sensitive_units is False
    assert reconstruction.importance.layer_multipliers == ()
    assert reconstruction.importance.protected_layer_patterns == ()
    assert reconstruction.rank_trust_reference_run is None
    assert reconstruction.rank_trust_fraction == 1

    adjusted022 = replace(
        config022,
        intent=config017.intent,
        output=config017.output,
        allocation=config017.allocation,
    )
    assert adjusted022 == config017
    assert config022.distillation.enabled is True
    assert config022.intent.baseline_run == "017-compress-and-benchmark-gemma-3-1b-it"
    assert experiment022.workflow.expected_blocks == 26
    assert experiment022.workflow.maximum_wddm_shared_gib == 0.75
    assert experiment022.workflow.export.huggingface is None
    assert experiment022.workflow.export.gguf_output == Path(
        "Results/022/gemma-3-1b-it-nanoquant.gguf"
    )

    workflow017 = experiment017.workflow
    workflow022 = experiment022.workflow
    assert (
        workflow022.wikitext_samples,
        workflow022.wikitext_sequence_length,
        workflow022.wikitext_batch_size,
        workflow022.task_names,
        workflow022.task_limit,
        workflow022.task_batch_size,
        workflow022.quality_backend,
    ) == (
        workflow017.wikitext_samples,
        workflow017.wikitext_sequence_length,
        workflow017.wikitext_batch_size,
        workflow017.task_names,
        workflow017.task_limit,
        workflow017.task_batch_size,
        workflow017.quality_backend,
    )


def test_experiment022_no_argument_run_owns_its_control_and_profile_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_experiment(22)
    launcher = tmp_path / "repo" / "experiments" / "022.py"
    campaign = (tmp_path / "repo" / "evidence" / "022").resolve()
    prepared: dict[str, object] = {}

    def prepare(
        definition: object,
        *,
        launcher_path: str | Path,
        campaign_root: Path,
        control_config: object,
    ) -> tuple[Path, Path]:
        prepared.update(
            definition=definition,
            launcher_path=launcher_path,
            campaign_root=campaign_root,
            control_config=control_config,
        )
        return campaign / "profile", campaign / "control"

    def consume(config: object, workflow: object, *, launcher_path: str | Path) -> int:
        prepared.update(config=config, workflow=workflow, final_launcher=launcher_path)
        return 0

    monkeypatch.setattr(d2_workflow, "_prepare_automatic_kl_inputs", prepare)
    monkeypatch.setattr(
        d2_workflow,
        "_validated_kl_profile",
        lambda *args, **kwargs: SimpleNamespace(complete=True, profile_key="sha256:fresh-1b"),
    )
    monkeypatch.setattr(d2_workflow, "run_compression_quality_experiment", consume)

    assert (
        d2_workflow.run_self_measured_d2_experiment(
            experiment,
            launcher_path=launcher,
            arguments=[],
        )
        == 0
    )

    assert prepared["campaign_root"] == campaign
    control_config = prepared["control_config"]
    assert control_config.intent.name == "022-d2-uniform-control-gemma-3-1b-it"
    assert control_config.intent.tags[1] == "experiment-022-preparation"
    assert control_config.model.source == "google/gemma-3-1b-it"
    assert control_config.allocation.strategy is AllocationStrategy.UNIFORM
    assert control_config.distillation.enabled is False
    runtime_config = prepared["config"]
    assert runtime_config.allocation.kl_profile_artifact == str(campaign / "profile")
    assert runtime_config.allocation.kl_profile_key == "sha256:fresh-1b"
