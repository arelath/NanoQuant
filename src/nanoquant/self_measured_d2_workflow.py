"""Fresh-input orchestration for numbered self-measured D2 experiments."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass, replace
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

_TEACHER_CACHE_NAME = "kl-teacher-cache"


@dataclass(frozen=True, slots=True)
class SelfMeasuredD2ProfileOptions:
    """Opt-in controls for how the same-campaign KL profile is measured.

    The defaults reproduce the original Experiment 021/022 behavior exactly: a
    static (undistilled) uniform control profiled over 12x512 WikiText sequences.

    ``tuned_operating_point`` fixes the distillation-measurement mismatch. When it
    is ``True`` the uniform control keeps global distillation enabled and the KL
    profile is measured against the tuned reconstruction, so the per-unit
    sensitivities reflect the same globally distilled operating point as the final
    candidate rather than the static uniform point. When ``False`` (the default)
    the control is measured static, as Experiments 021 and 022 did.

    ``wikitext_samples`` / ``sequence_length`` size the profiling slice. The D2
    campaign review found the fixed 12x512 slice too small to resolve a 1%
    improvement decisively (a 48x512 slice did), so larger models should request
    at least 48 sequences.
    """

    wikitext_samples: int = 12
    sequence_length: int = 512
    tuned_operating_point: bool = False

    def __post_init__(self) -> None:
        if self.wikitext_samples <= 0 or self.sequence_length < 2:
            raise ValueError("KL profile dataset dimensions must be positive")


_DEFAULT_PROFILE_OPTIONS = SelfMeasuredD2ProfileOptions()


class _SelfMeasuredD2Definition(Protocol):
    @property
    def config(self) -> RunConfig: ...

    @property
    def workflow(self) -> CompressionQualityExperiment: ...


def _experiment_number(config: RunConfig) -> int:
    number = config.intent.experiment_number
    if number is None:
        raise ValueError("self-measured D2 workflow requires a numbered experiment")
    return number


def _model_slug(config: RunConfig) -> str:
    return config.model.source.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1].lower()


def _control_name(config: RunConfig) -> str:
    return f"{_experiment_number(config):03d}-d2-uniform-control-{_model_slug(config)}"


def _profile_name(config: RunConfig) -> str:
    return f"{_experiment_number(config):03d}-d2-uniform-control-kl-profile"


def _uniform_control_config(
    config: RunConfig,
    campaign_root: Path,
    *,
    tuned_operating_point: bool = False,
) -> RunConfig:
    """Derive a fresh uniform control without imported experimental values."""

    number = _experiment_number(config)
    # When ``tuned_operating_point`` is set the control keeps global distillation
    # enabled so the KL profile can be measured at the same tuned operating point
    # as the final candidate. Otherwise the control measures the static uniform
    # point, as Experiments 021 and 022 did; global tuning then remains part of
    # only the final candidate/baseline comparison.
    control_distillation = (
        config.distillation
        if tuned_operating_point
        else replace(config.distillation, enabled=False)
    )
    control = replace(
        config,
        intent=replace(
            config.intent,
            experiment_number=None,
            name=_control_name(config),
            purpose=f"Build the same-campaign uniform control for Experiment {number:03d} KL measurement.",
            hypothesis="Direct exact-unit KL measurements provide the D2 allocation anchors.",
            baseline_run=None,
            tags=(
                _model_slug(config),
                f"experiment-{number:03d}-preparation",
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
        distillation=control_distillation,
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
        raise ValueError("self-measured D2 control manifest is invalid")
    resolved = payload.get("resolved_config")
    canonical = None if not isinstance(resolved, dict) else resolved.get("canonical_run_config")
    if canonical != to_dict(expected):
        raise ValueError(
            "self-measured D2 KL control does not match the current fresh uniform-control recipe"
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
                f"self-measured D2 requires its {label} inside its own campaign root: {campaign_root}"
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
            "KL profile provenance does not belong to the supplied self-measured D2 control run"
        )
    return profile


def _prepare_automatic_kl_inputs(
    experiment: _SelfMeasuredD2Definition,
    *,
    launcher_path: str | Path,
    campaign_root: Path,
    control_config: RunConfig,
    profile_options: SelfMeasuredD2ProfileOptions,
) -> tuple[Path, Path]:
    """Create or resume the uniform control and its exact-unit KL profile."""

    number = _experiment_number(experiment.config)
    label = f"Experiment {number:03d}"
    profile_path = campaign_root / _profile_name(experiment.config)
    control_run = campaign_root / _control_name(experiment.config)
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
            print(f"Reusing completed {label} KL profile: {profile_file}", flush=True)
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
        print(f"Reusing completed {label} uniform control: {control_run}", flush=True)
        inputs = resolve_resident_experiment_inputs(control_config, launcher_path=launcher_path)
    else:
        print(f"Creating or resuming {label} uniform control: {control_run}", flush=True)
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
            raise ValueError(f"{label} uniform control completed with the wrong block count")
        _require_control_recipe(control_run, control_config)
        del result
        gc.collect()
        if control_config.runtime.compute_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Creating or resuming {label} exact-unit KL profile: {profile_path}", flush=True)
    execute_kl_budget(
        argparse.Namespace(
            run_output=inputs.output,
            snapshot=inputs.snapshot,
            source=control_config.model.source,
            revision=str(control_config.model.revision),
            profile_output=profile_path,
            device=control_config.runtime.compute_device,
            wikitext_samples=profile_options.wikitext_samples,
            sequence_length=profile_options.sequence_length,
            batch_size=1,
            token_chunk_size=128,
            arm=[],
            teacher_cache_mode="cpu",
            teacher_cache_root=campaign_root / _TEACHER_CACHE_NAME,
            use_global_tuning=profile_options.tuned_operating_point,
            local_files_only=False,
        )
    )
    return profile_path, control_run


def run_self_measured_d2_experiment(
    experiment: _SelfMeasuredD2Definition,
    *,
    launcher_path: str | Path,
    arguments: list[str] | None = None,
    profile_options: SelfMeasuredD2ProfileOptions = _DEFAULT_PROFILE_OPTIONS,
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
    number = _experiment_number(experiment.config)
    label = f"Experiment {number:03d}"
    campaign_root = (repository / "evidence" / f"{number:03d}").resolve()
    control_config = _uniform_control_config(
        experiment.config,
        campaign_root,
        tuned_operating_point=profile_options.tuned_operating_point,
    )
    if parsed.kl_profile is None:
        profile_path, control_run = _prepare_automatic_kl_inputs(
            experiment,
            launcher_path=launcher_path,
            campaign_root=campaign_root,
            control_config=control_config,
            profile_options=profile_options,
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
        raise ValueError(f"{label} requires a complete KL profile")
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


def run_experiment021(
    experiment: _SelfMeasuredD2Definition,
    *,
    launcher_path: str | Path,
    arguments: list[str] | None = None,
) -> int:
    """Backward-compatible adapter for the Experiment 021 launcher."""

    if _experiment_number(experiment.config) != 21:
        raise ValueError("run_experiment021 requires Experiment 021")
    return run_self_measured_d2_experiment(
        experiment,
        launcher_path=launcher_path,
        arguments=arguments,
    )


__all__ = [
    "SelfMeasuredD2ProfileOptions",
    "run_experiment021",
    "run_self_measured_d2_experiment",
]
