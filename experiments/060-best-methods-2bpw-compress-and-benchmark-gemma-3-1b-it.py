"""Experiment 060: best-evidence 2-BPW compression and quality on Gemma 3 1B."""

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
    DType,
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

BASELINE = ExperimentRef(
    56,
    "physical-cap-rank-redistribution-d2-compress-and-benchmark-gemma-3-1b-it",
)
_RUNTIME_KL_PROFILE = "runtime-kl-profile-required"
_RUNTIME_KL_PROFILE_KEY = "runtime-kl-profile-key-required"

# Full aligned physical rank costs about 1.258 BPW on Gemma 3 1B. An 8.8%
# charged INT8 residual-column policy brings the conservative full-rank bound
# to 1.966 BPW and leaves the same 0.02-BPW retry reserve used by Experiment
# 059. This shape-matched fraction avoids stranding the 2-BPW budget.
RESIDUAL_OUTLIER_FRACTION = 0.088
TARGET_BPW = 2.0
RETRY_EXTRA_BUDGET_FRACTION = 0.01

BASE_CONFIG = ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE
SHARED_INPUT = BASE_CONFIG.factorization.shared_input
FACTORIZED_TUNING = BASE_CONFIG.block_tuning.factorized

CONFIG = replace(
    BASE_CONFIG,
    allocation=replace(
        BASE_CONFIG.allocation,
        target_bpw=TARGET_BPW,
        strategy=AllocationStrategy.KL_CALIBRATED,
        kl_profile_artifact=_RUNTIME_KL_PROFILE,
        kl_profile_key=_RUNTIME_KL_PROFILE_KEY,
        kl_sensitivity_granularity=KlSensitivityGranularity.EXACT,
        bounds=replace(
            BASE_CONFIG.allocation.bounds,
            ceiling_fraction_of_uniform=1.5,
            overcomplete_rank_ceiling_fraction=1.0,
        ),
        retry=replace(
            BASE_CONFIG.allocation.retry,
            enabled=True,
            extra_bit_budget_fraction=RETRY_EXTRA_BUDGET_FRACTION,
            outlier_count_increment_at_rank_cap=2,
        ),
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
    outliers=replace(
        BASE_CONFIG.outliers,
        fraction=RESIDUAL_OUTLIER_FRACTION,
        storage_dtype=DType.INT8,
        charge_to_bit_budget=True,
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
        number=60,
        name="best-methods-2bpw-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Replay Experiment 059's charged 2-BPW best-methods campaign on pinned "
            "Gemma 3 1B: same-run exact-unit D2 allocation, measured calibration-weighted "
            "responses, physical-rank redistribution, stacked QKV, functional binary "
            "tuning, shape-matched INT8 residual columns, rank-cap outlier retries, global "
            "distillation, complete export, and the long quality benchmark."
        ),
        hypothesis=(
            "Reinvesting Gemma 3 1B's physically unreachable binary-rank budget in charged "
            "residual INT8 columns at a shape-matched 8.8% rate materially improves quality "
            "over Experiment 056 without exceeding two represented bits per quantized weight."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "best-methods",
            "2-bpw",
            "charged-budget",
            "d2",
            "kl-calibrated",
            "exact-unit-sensitivity",
            "same-run-rank-response",
            "measured-unit-kl",
            "tuned-operating-point",
            "wikitext-48x512",
            "physical-rank-cap",
            "stacked-qkv",
            "functional-binary-tuning",
            "int8-outliers",
            "rank-cap-outlier-retry",
            "global-distillation",
            "task-limit-1000",
            "experiment-056-comparison",
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
        interrupt_after_block_commits=1,
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
