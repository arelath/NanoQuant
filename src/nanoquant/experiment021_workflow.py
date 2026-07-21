"""Fresh-input orchestration for the self-measured Experiment 021 D2 run."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import torch

from nanoquant.application.kl_budget import KlBudgetProfile, load_kl_budget_profile
from nanoquant.compression_quality_workflow import (
    CompressionQualityExperiment,
    run_compression_quality_experiment,
)
from nanoquant.config.codec import to_dict
from nanoquant.config.schema import (
    AllocationStrategy,
    KlSensitivityGranularity,
    ReconstructionRankPlanningConfig,
    RunConfig,
)
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.infrastructure.commits import CommitIdentity, latest_complete_identity
from nanoquant.kl_budget_workflow import execute_kl_budget
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    execute_resident_workflow,
    resolve_resident_experiment_inputs,
)

_CONTROL_NAME = "021-d2-uniform-control-gemma-3-270m-it"
_PROFILE_NAME = "021-d2-uniform-control-kl-profile"
_TEACHER_CACHE_NAME = "kl-teacher-cache"


class _Experiment021Definition(Protocol):
    config: RunConfig
    workflow: CompressionQualityExperiment


def _uniform_control_config(config: RunConfig, campaign_root: Path) -> RunConfig:
    """Derive a fresh uniform control without imported experimental values."""

    control = replace(
        config,
        intent=replace(
            config.intent,
            experiment_number=None,
            name=_CONTROL_NAME,
            purpose="Build the same-campaign uniform control for Experiment 021 KL measurement.",
            hypothesis="Direct exact-unit KL measurements provide the D2 allocation anchors.",
            baseline_run=None,
            tags=(
                "gemma-3-270m-it",
                "experiment-021-preparation",
                "uniform-control",
                "kl-profile-source",
            ),
        ),
        allocation=replace(
            config.allocation,
            strategy=AllocationStrategy.UNIFORM,
            kl_profile_artifact=None,
            kl_profile_key=None,
            kl_sensitivity_granularity=KlSensitivityGranularity.EXACT_OR_TYPE_BLOCK,
            reconstruction=ReconstructionRankPlanningConfig(),
        ),
        # The allocation profile measures the static uniform operating point.
        # Global tuning remains part of the final Experiment 021/016 comparison.
        distillation=replace(config.distillation, enabled=False),
        output=replace(config.output, run_root=str(campaign_root)),
    )
    raise_for_issues(validate(control, ValidationPhase.RESOLVED))
    return control


def _journal_identity(run_output: Path, expected_blocks: int) -> CommitIdentity:
    journal = run_output / "state" / "journal.jsonl"
    records = [
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identity, _blocks = latest_complete_identity(records, expected_blocks)
    return identity


def _source_identity(identity: CommitIdentity) -> str:
    return f"{identity.config_hash}|{identity.model_hash}|{identity.plan_hash}"


def _require_control_recipe(run_output: Path, expected: RunConfig) -> None:
    """Verify the canonical RunConfig embedded in the resident manifest."""

    payload = json.loads((run_output / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment 021 control manifest is invalid")
    resolved = payload.get("resolved_config")
    canonical = None if not isinstance(resolved, dict) else resolved.get("canonical_run_config")
    if canonical != to_dict(expected):
        raise ValueError(
            "Experiment 021 KL control does not match the current fresh uniform-control recipe"
        )


def _validated_kl_profile(
    profile_path: Path,
    control_run: Path,
    *,
    campaign_root: Path,
    expected_blocks: int,
    expected_control_config: RunConfig,
) -> KlBudgetProfile:
    for label, path in (("profile", profile_path), ("control run", control_run)):
        try:
            path.relative_to(campaign_root)
        except ValueError as exc:
            raise ValueError(
                f"Experiment 021 requires its {label} inside its own campaign root: {campaign_root}"
            ) from exc
    profile = load_kl_budget_profile(
        profile_path / "kl-budget-profile.json" if profile_path.is_dir() else profile_path
    )
    control_identity = _journal_identity(control_run, expected_blocks)
    _require_control_recipe(control_run, expected_control_config)
    expected_source_identity = _source_identity(control_identity)
    if not (
        profile.provenance.source_run_identity == expected_source_identity
        or profile.provenance.source_run_identity.startswith(expected_source_identity + "|")
    ):
        raise ValueError(
            "KL profile provenance does not belong to the supplied Experiment 021 control run"
        )
    return profile


def _prepare_automatic_kl_inputs(
    experiment: _Experiment021Definition,
    *,
    launcher_path: str | Path,
    campaign_root: Path,
    control_config: RunConfig,
) -> tuple[Path, Path]:
    """Create or resume the uniform control and its exact-unit KL profile."""

    profile_path = campaign_root / _PROFILE_NAME
    control_run = campaign_root / _CONTROL_NAME
    profile_file = profile_path / "kl-budget-profile.json"
    if profile_file.is_file():
        profile = _validated_kl_profile(
            profile_path,
            control_run,
            campaign_root=campaign_root,
            expected_blocks=experiment.workflow.expected_blocks,
            expected_control_config=control_config,
        )
        if profile.complete:
            print(f"Reusing completed Experiment 021 KL profile: {profile_file}", flush=True)
            return profile_path, control_run

    control_complete = False
    try:
        _journal_identity(control_run, experiment.workflow.expected_blocks)
    except (FileNotFoundError, ValueError):
        pass
    else:
        _require_control_recipe(control_run, control_config)
        control_complete = True

    if control_complete:
        print(f"Reusing completed Experiment 021 uniform control: {control_run}", flush=True)
        inputs = resolve_resident_experiment_inputs(control_config, launcher_path=launcher_path)
    else:
        print(f"Creating or resuming Experiment 021 uniform control: {control_run}", flush=True)
        inputs = resolve_resident_experiment_inputs(control_config, launcher_path=launcher_path)
        maximum_shared = experiment.workflow.maximum_wddm_shared_gib
        result = execute_resident_workflow(
            control_config,
            inputs,
            ResidentExecutionOptions(
                restore_completed_blocks=experiment.workflow.restore_completed_blocks,
                maximum_wddm_shared_bytes=(
                    None if maximum_shared is None else int(maximum_shared * 2**30)
                ),
            ),
        )
        if len(result.quantization.inventory.blocks) != experiment.workflow.expected_blocks:
            raise ValueError("Experiment 021 uniform control completed with the wrong block count")
        _require_control_recipe(control_run, control_config)
        del result
        gc.collect()
        if control_config.runtime.compute_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Creating or resuming Experiment 021 exact-unit KL profile: {profile_path}", flush=True)
    execute_kl_budget(
        argparse.Namespace(
            run_output=inputs.output,
            snapshot=inputs.snapshot,
            source=control_config.model.source,
            revision=str(control_config.model.revision),
            profile_output=profile_path,
            device=control_config.runtime.compute_device,
            wikitext_samples=12,
            sequence_length=512,
            batch_size=1,
            token_chunk_size=128,
            arm=[],
            teacher_cache_mode="cpu",
            teacher_cache_root=campaign_root / _TEACHER_CACHE_NAME,
            use_global_tuning=False,
            local_files_only=False,
        )
    )
    return profile_path, control_run


def run_experiment021(
    experiment: _Experiment021Definition,
    *,
    launcher_path: str | Path,
    arguments: list[str] | None = None,
) -> int:
    """Prepare, verify, and consume same-campaign KL inputs before compression."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kl-profile",
        type=Path,
        help="override the automatic fresh evaluator-v3 profile",
    )
    parser.add_argument(
        "--kl-control-run",
        type=Path,
        help="override the automatic matched uniform control run",
    )
    parsed = parser.parse_args(arguments)
    if (parsed.kl_profile is None) != (parsed.kl_control_run is None):
        parser.error("--kl-profile and --kl-control-run must be supplied together")

    repository = Path(launcher_path).resolve().parent.parent
    campaign_root = (repository / "evidence" / "021").resolve()
    control_config = _uniform_control_config(experiment.config, campaign_root)
    if parsed.kl_profile is None:
        profile_path, control_run = _prepare_automatic_kl_inputs(
            experiment,
            launcher_path=launcher_path,
            campaign_root=campaign_root,
            control_config=control_config,
        )
    else:
        profile_path = parsed.kl_profile.resolve()
        control_run = parsed.kl_control_run.resolve()

    profile = _validated_kl_profile(
        profile_path,
        control_run,
        campaign_root=campaign_root,
        expected_blocks=experiment.workflow.expected_blocks,
        expected_control_config=control_config,
    )
    if not profile.complete:
        raise ValueError("Experiment 021 requires a complete KL profile")
    runtime_config = replace(
        experiment.config,
        allocation=replace(
            experiment.config.allocation,
            kl_profile_artifact=str(profile_path),
            kl_profile_key=profile.profile_key,
        ),
    )
    return run_compression_quality_experiment(
        runtime_config,
        experiment.workflow,
        launcher_path=launcher_path,
    )


__all__ = ["run_experiment021"]
