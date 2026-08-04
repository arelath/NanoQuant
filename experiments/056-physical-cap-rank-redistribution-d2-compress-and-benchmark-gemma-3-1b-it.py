"""Experiment 056: redistribute saturated D2 rank under the physical cap."""

from dataclasses import replace

from recipes import (
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
    experiment_callable_main,
    run_experiment,
)

from nanoquant.config.schema import (
    ADMMConfig,
    AllocationStrategy,
    BiasCorrectionConfig,
    BinaryFactorSearchConfig,
    KlAllocationObjective,
    KlSensitivityGranularity,
    LowRankPatchConfig,
    RankResponseSource,
    ReconstructionImportanceConfig,
    SharedInputMemberMultiplierConfig,
)

BASELINE = ExperimentRef(55, "overcomplete-rank-d2-compress-and-benchmark-gemma-3-1b-it")
PROFILE = "evidence/054/054-d2-uniform-control-kl-profile"
PROFILE_KEY = "sha256:4a67e45d5266763b09e3b487a3820f4ad8520201b144807241e2744d9c271bf9"

BASE_CONFIG = ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE
SHARED_INPUT = BASE_CONFIG.factorization.shared_input
FACTORIZED_TUNING = BASE_CONFIG.block_tuning.factorized

CONFIG = replace(
    BASE_CONFIG,
    allocation=replace(
        BASE_CONFIG.allocation,
        strategy=AllocationStrategy.KL_CALIBRATED,
        kl_profile_artifact=PROFILE,
        kl_profile_key=PROFILE_KEY,
        kl_sensitivity_granularity=KlSensitivityGranularity.EXACT,
        bounds=replace(
            BASE_CONFIG.allocation.bounds,
            # Keep 055's broad response/allocation range, but cap every owner
            # at its aligned physical dimension. The global allocator then
            # continues buying rank in other unsaturated owners.
            ceiling_fraction_of_uniform=1.5,
            overcomplete_rank_ceiling_fraction=1.0,
        ),
        reconstruction=replace(
            BASE_CONFIG.allocation.reconstruction,
            objective_mode="calibration_weighted",
            probe_admm=ADMMConfig(
                outer_iterations=100,
                inner_iterations=5,
                regularization=3e-2,
                penalty_schedule="cubic",
                convergence_check_interval=100,
                transpose_wide=True,
            ),
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
        binary_search=BinaryFactorSearchConfig(enabled=True),
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

_DEFINED_EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=56,
        name="physical-cap-rank-redistribution-d2-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Replay Experiment 055 with the physical binary-rank ceiling restored, keep its "
            "wider 1.5x measured allocation range, and redirect capacity that saturated owners "
            "cannot consume into the best remaining ranks in other blocks without changing the "
            "fixed outlier policy."
        ),
        hypothesis=(
            "The quality regression in Experiment 055 came from assigning over-complete binary "
            "rank to saturated owners; globally reallocating those bits to unsaturated owners "
            "improves retained WikiText perplexity at the same effective BPW and outlier count."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "d2",
            "physical-rank-cap",
            "global-rank-redistribution",
            "fixed-outlier-policy",
            "rank-ceiling-1p5x",
            "same-bpw",
            "reactive-tabu",
            "functional-binary-tuning",
            "experiment-055-ablation",
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
    lambda: run_experiment(EXPERIMENT, launcher_path=__file__),
)
