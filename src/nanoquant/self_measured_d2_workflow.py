"""Fresh-input orchestration for numbered self-measured D2 experiments."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from nanoquant.allocation_profile_dataset_workflow import config_with_allocation_dataset
from nanoquant.application.kl_budget import KlBudgetProfile, load_kl_budget_profile
from nanoquant.compression_quality_workflow import (
    CompressionQualityExperiment,
    run_compression_quality_experiment,
)
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.config.schema import (
    AllocationStrategy,
    KlSensitivityGranularity,
    ReconstructionRankPlanningConfig,
    RunConfig,
)
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.infrastructure.commits import CommitIdentity, latest_complete_identity
from nanoquant.infrastructure.device_lease import canonical_device_name
from nanoquant.infrastructure.global_tuning import active_global_tuning
from nanoquant.infrastructure.kl_teacher_dataset import load_kl_teacher_dataset
from nanoquant.infrastructure.memory_cleanup import release_memory
from nanoquant.infrastructure.model_adapters import decoder_block_count_from_config
from nanoquant.infrastructure.resolved_model_config import resolve_model_config
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
    control_distillation = replace(
        config.distillation,
        enabled=tuned_operating_point,
        foldable_mlp_multipliers=replace(
            config.distillation.foldable_mlp_multipliers,
            enabled=False,
        ),
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


def _journal_identity(run_output: Path, block_count: int) -> CommitIdentity:
    journal = run_output / "state" / "journal.jsonl"
    records = [
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identity, _blocks = latest_complete_identity(records, block_count)
    return identity


def _resolved_model_block_count(config: RunConfig) -> int:
    resolved = resolve_model_config(config.model.source, str(config.model.revision))
    return decoder_block_count_from_config(resolved.values)


def _source_identity(identity: CommitIdentity) -> str:
    return f"{identity.config_hash}|{identity.model_hash}|{identity.plan_hash}"


def _require_control_recipe(run_output: Path, expected: RunConfig) -> None:
    """Verify the canonical RunConfig embedded in the resident manifest."""

    payload = json.loads((run_output / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("self-measured D2 control manifest is invalid")
    resolved = payload.get("resolved_config")
    canonical = None if not isinstance(resolved, dict) else resolved.get("canonical_run_config")
    if canonical is None or to_dict(from_dict(RunConfig, canonical)) != to_dict(expected):
        raise ValueError(
            "self-measured D2 KL control does not match the current fresh uniform-control recipe"
        )


def _validated_kl_profile(
    profile_path: Path,
    control_run: Path,
    *,
    campaign_root: Path,
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
    control_identity = _journal_identity(
        control_run,
        _resolved_model_block_count(expected_control_config),
    )
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
    teacher_dataset: Path | None = None,
) -> tuple[Path, Path]:
    """Create or resume the uniform control and its exact-unit KL profile."""

    number = _experiment_number(experiment.config)
    label = f"Experiment {number:03d}"
    profile_path = campaign_root / _profile_name(experiment.config)
    control_run = campaign_root / _control_name(experiment.config)
    block_count = _resolved_model_block_count(control_config)
    profile_file = profile_path / "kl-budget-profile.json"
    if profile_file.is_file():
        profile = _validated_kl_profile(
            profile_path,
            control_run,
            campaign_root=campaign_root,
            expected_control_config=control_config,
        )
        if profile.complete:
            print(f"Reusing completed {label} KL profile: {profile_file}", flush=True)
            return profile_path, control_run

    control_blocks_complete = False
    try:
        _journal_identity(control_run, block_count)
    except (FileNotFoundError, ValueError):
        pass
    else:
        _require_control_recipe(control_run, control_config)
        control_blocks_complete = True

    control_complete = control_blocks_complete and (
        not profile_options.tuned_operating_point
        or active_global_tuning(control_run) is not None
    )

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
                interrupt_after_block_commits=experiment.workflow.interrupt_after_block_commits,
                maximum_wddm_shared_bytes=(
                    None if maximum_shared is None else int(maximum_shared * 2**30)
                ),
            ),
        )
        if len(result.quantization.inventory.blocks) != block_count:
            raise ValueError(f"{label} uniform control completed with the wrong block count")
        _require_control_recipe(control_run, control_config)
        del result
        release_memory(control_config.runtime.compute_device)

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
            teacher_cache_root=campaign_root / _TEACHER_CACHE_NAME if teacher_dataset is None else None,
            teacher_dataset=teacher_dataset,
            use_global_tuning=profile_options.tuned_operating_point,
            local_files_only=experiment.workflow.local_files_only,
        )
    )
    return profile_path, control_run


def run_self_measured_d2_experiment(
    experiment: _SelfMeasuredD2Definition,
    *,
    launcher_path: str | Path,
    arguments: list[str] | None = None,
    profile_options: SelfMeasuredD2ProfileOptions = _DEFAULT_PROFILE_OPTIONS,
    allocation_profile_dataset: Path | None = None,
    kl_teacher_dataset: Path | None = None,
) -> int:
    """Prepare, verify, and consume same-campaign KL inputs before compression."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocation-profile-dataset", type=Path, default=allocation_profile_dataset,
        help="reuse approximate rank responses and KL sensitivities from a dataset",
    )
    parser.add_argument(
        "--kl-teacher-dataset", type=Path, default=kl_teacher_dataset,
        help="reuse full-vocabulary teacher targets built by tools/build_kl_teacher_dataset.py",
    )
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
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate dataset-backed inputs and print the resolved recipe without compression or evaluation",
    )
    parsed = parser.parse_args(arguments)
    if parsed.dry_run and parsed.allocation_profile_dataset is None:
        parser.error("--dry-run requires --allocation-profile-dataset; automatic preparation will not be started")
    if parsed.allocation_profile_dataset is not None and (
        parsed.kl_profile is not None or parsed.kl_control_run is not None
    ):
        parser.error("--allocation-profile-dataset already supplies the profile; omit --kl-profile/--kl-control-run")
    if (parsed.kl_profile is None) != (parsed.kl_control_run is None):
        parser.error("--kl-profile and --kl-control-run must be supplied together")
    teacher_protocol = None
    teacher_reference = None
    teacher_baseline = None
    if parsed.kl_teacher_dataset is not None:
        dataset = load_kl_teacher_dataset(parsed.kl_teacher_dataset)
        teacher_protocol = dataset.protocol
        teacher_reference = dataset.reference
        teacher_baseline = dataset.cache.baseline_negative_log_likelihood
        del dataset
        if (
            teacher_protocol.source != experiment.config.model.source
            or teacher_protocol.revision != str(experiment.config.model.revision)
            or teacher_protocol.samples != profile_options.wikitext_samples
            or teacher_protocol.sequence_length != profile_options.sequence_length
            or teacher_protocol.batch_size != 1
            or canonical_device_name(teacher_protocol.device)
            != canonical_device_name(experiment.config.runtime.compute_device)
        ):
            raise ValueError("KL teacher dataset does not match the experiment model or profiling protocol")

    if parsed.allocation_profile_dataset is not None:
        runtime_config = config_with_allocation_dataset(experiment.config, parsed.allocation_profile_dataset)
        if teacher_protocol is not None:
            assert runtime_config.allocation.kl_profile_artifact is not None
            profile = load_kl_budget_profile(
                Path(runtime_config.allocation.kl_profile_artifact) / "kl-budget-profile.json"
            )
            if (
                profile.provenance.dataset_slice_hash != teacher_protocol.token_hash
                or profile.provenance.dataset_fingerprint != teacher_protocol.dataset_fingerprint
                or teacher_baseline is None
                or not math.isclose(profile.baseline_negative_log_likelihood, teacher_baseline, rel_tol=1e-12)
            ):
                raise ValueError("allocation profile and KL teacher datasets use different teacher inputs or baseline")
            print(
                f"Validated KL teacher dataset {teacher_reference}; the allocation dataset already contains "
                "its measured sensitivities, so teacher inference and KL profiling are skipped.", flush=True,
            )
        raise_for_issues(validate(runtime_config, ValidationPhase.RESOLVED))
        if parsed.dry_run:
            print(json.dumps({
                "dry_run": True, "config": to_dict(runtime_config), "workflow": to_dict(experiment.workflow),
                "kl_teacher_dataset": None if parsed.kl_teacher_dataset is None else str(parsed.kl_teacher_dataset),
                "kl_teacher_reference": None if teacher_reference is None else to_dict(teacher_reference),
                "allocation_profile_dataset": str(parsed.allocation_profile_dataset),
            }, indent=2))
            return 0
        print(f"Using approximate allocation dataset {parsed.allocation_profile_dataset}", flush=True)
        return run_compression_quality_experiment(runtime_config, experiment.workflow, launcher_path=launcher_path)

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
            teacher_dataset=parsed.kl_teacher_dataset,
        )
    else:
        profile_path = parsed.kl_profile.resolve()
        control_run = parsed.kl_control_run.resolve()

    profile = _validated_kl_profile(
        profile_path,
        control_run,
        campaign_root=campaign_root,
        expected_control_config=control_config,
    )
    if not profile.complete:
        raise ValueError(f"{label} requires a complete KL profile")
    if teacher_protocol is not None and (
        profile.provenance.dataset_slice_hash != teacher_protocol.token_hash
        or profile.provenance.dataset_fingerprint != teacher_protocol.dataset_fingerprint
    ):
        raise ValueError("completed KL profile uses different inputs from the requested teacher dataset")
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
