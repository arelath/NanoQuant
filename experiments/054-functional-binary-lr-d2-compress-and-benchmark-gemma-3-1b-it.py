"""Experiment 054: replay Experiment 024 with a conservative binary tuning rate."""

from dataclasses import replace

from recipes import (
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
    experiment_callable_main,
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

BASELINE = ExperimentRef(24, "best-methods-compress-and-benchmark-gemma-3-1b-it")
_RUNTIME_KL_PROFILE = "runtime-kl-profile-required"
_RUNTIME_KL_PROFILE_KEY = "runtime-kl-profile-key-required"

BASE_CONFIG = ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE
SHARED_INPUT = BASE_CONFIG.factorization.shared_input
FACTORIZED_TUNING = BASE_CONFIG.block_tuning.factorized

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
    block_tuning=replace(
        BASE_CONFIG.block_tuning,
        factorized=replace(
            FACTORIZED_TUNING,
            learning_rates=replace(
                FACTORIZED_TUNING.learning_rates,
                binary=3e-5,
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
        number=54,
        name="functional-binary-lr-d2-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Replay Experiment 024 at identical representation cost while allowing functional "
            "factorized tuning to move binary signs with a production-horizon 3e-5 rate, then "
            "complete export and the long quality benchmark."
        ),
        hypothesis=(
            "A conservative 3e-5 binary learning rate retains the block-0 gate improvement while "
            "remaining finite across the production 256-row optimizer horizon and improving the "
            "complete model without increasing effective BPW."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "experiment-024-replay",
            "functional-binary-tuning",
            "binary-lr-3e-5",
            "same-bpw",
            "held-out-kl-gated",
            "global-distillation",
            "task-limit-1000",
        ),
    ),
    CONFIG,
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


experiment_callable_main(
    __name__,
    lambda: run_self_measured_d2_experiment(
        EXPERIMENT,
        launcher_path=__file__,
        profile_options=PROFILE_OPTIONS,
    ),
)
