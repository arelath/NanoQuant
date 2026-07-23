"""Experiment 024: best-evidence composite compression and quality on Gemma 3 1B."""

from dataclasses import replace

from recipes import (
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
)

from nanoquant.config.schema import (
    AllocationStrategy,
    BiasCorrectionConfig,
    KlAllocationObjective,
    KlSensitivityGranularity,
    LowRankPatchConfig,
    RankResponseSource,
    ReconstructionImportanceConfig,
    SharedInputMemberMultiplierConfig,
)
from nanoquant.self_measured_d2_workflow import (
    SelfMeasuredD2ProfileOptions,
    run_self_measured_d2_experiment,
)

BASELINE = ExperimentRef(22, "d2-kl-compress-and-benchmark-gemma-3-1b-it")
_RUNTIME_KL_PROFILE = "runtime-kl-profile-required"
_RUNTIME_KL_PROFILE_KEY = "runtime-kl-profile-key-required"

BASE_CONFIG = ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE
SHARED_INPUT = BASE_CONFIG.factorization.shared_input

CONFIG = replace(
    BASE_CONFIG,
    allocation=replace(
        BASE_CONFIG.allocation,
        strategy=AllocationStrategy.KL_CALIBRATED,
        kl_profile_artifact=_RUNTIME_KL_PROFILE,
        kl_profile_key=_RUNTIME_KL_PROFILE_KEY,
        kl_sensitivity_granularity=KlSensitivityGranularity.EXACT,
        reconstruction=replace(
            BASE_CONFIG.allocation.reconstruction,
            objective_mode="calibration_weighted",
            response_source=RankResponseSource.MEASURED,
            response_curves=(),
            response_profile_provenance="",
            # Experiment 022's raw measured-unit objective retained better
            # WikiText quality than Experiment 023's interaction normalization.
            kl_objective=KlAllocationObjective.MEASURED_UNIT_KL,
            importance=ReconstructionImportanceConfig(),
            sensitivity_strength=1.0,
            protect_sensitive_units=False,
            target_protected_error_reduction_fraction=0.0,
            rank_trust_reference_run=None,
            rank_trust_fraction=1.0,
        ),
    ),
    factorization=replace(
        BASE_CONFIG.factorization,
        # D3 bias and D5 patches both failed their equal-budget 270M gates.
        bias_correction=BiasCorrectionConfig(enabled=False),
        low_rank_patch=LowRankPatchConfig(enabled=False),
        shared_input=replace(
            SHARED_INPUT,
            groups=tuple(
                replace(
                    group,
                    member_multipliers=(
                        SharedInputMemberMultiplierConfig("self_attn.v_proj", 2.0),
                    ),
                )
                for group in SHARED_INPUT.groups
            ),
        ),
    ),
    distillation=replace(BASE_CONFIG.distillation, enabled=True),
)

PROFILE_OPTIONS = SelfMeasuredD2ProfileOptions(
    wikitext_samples=48,
    sequence_length=512,
    tuned_operating_point=True,
)

_DEFINED_EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=24,
        name="best-methods-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Compress pinned Gemma 3 1B with the best retained quality methods: exact-unit "
            "same-run measured-KL allocation, calibration-weighted response probes, stacked QKV "
            "with a 2x value-member objective, and global scale distillation; then run the "
            "long quality benchmark."
        ),
        hypothesis=(
            "Raw exact-unit KL allocation, measured at the tuned operating point over 48x512 "
            "sequences, plus value-weighted stacked QKV improves WikiText and aggregate task "
            "quality over Experiment 022 without exceeding its effective BPW."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "best-methods",
            "d2",
            "kl-calibrated",
            "exact-unit-sensitivity",
            "same-run-rank-response",
            "measured-unit-kl",
            "tuned-operating-point",
            "wikitext-48x512",
            "stacked-qkv",
            "v-objective-2x",
            "no-bias",
            "no-low-rank-patch",
            "global-distillation",
            "task-limit-1000",
            "experiment-022-comparison",
        ),
    ),
    CONFIG,
    expected_blocks=26,
    maximum_wddm_shared_gib=0.75,
)

EXPERIMENT = replace(
    _DEFINED_EXPERIMENT,
    workflow=replace(
        _DEFINED_EXPERIMENT.workflow,
        task_limit=1000,
        local_files_only=True,
    ),
)


if __name__ == "__main__":
    raise SystemExit(
        run_self_measured_d2_experiment(
            EXPERIMENT,
            launcher_path=__file__,
            profile_options=PROFILE_OPTIONS,
        )
    )
