"""Experiment 055: equal-budget D2 allocation with bounded over-complete binary rank."""

from dataclasses import replace

from recipes import (
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
    experiment_main,
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

BASELINE = ExperimentRef(54, "functional-binary-lr-d2-compress-and-benchmark-gemma-3-1b-it")
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
            # Measure the response through 1.5x uniform rank and permit binary
            # factor owners to exceed min(in_features, out_features) by 50%.
            # Target BPW remains 1.0, so any over-complete rank is funded by
            # measured equal-budget redistribution rather than additive bits.
            ceiling_fraction_of_uniform=1.5,
            overcomplete_rank_ceiling_fraction=1.5,
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

_DEFINED_EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=55,
        name="overcomplete-rank-d2-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Remove the algebraic-dimension rank ceiling from the measured D2 allocation, "
            "permit bounded over-complete binary factors at unchanged target BPW, and determine "
            "whether saturated projections—especially down_proj—earn and use the extra capacity."
        ),
        hypothesis=(
            "Because rank=min(m,n) does not span arbitrary matrices under binary factors and "
            "shared diagonal scales, allowing ranks through 1.5x the physical dimension lowers "
            "held-out functional error and final WikiText perplexity at no greater effective BPW "
            "than Experiment 054."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "d2",
            "overcomplete-rank",
            "rank-ceiling-1p5x",
            "same-bpw",
            "functional-binary-tuning",
            "experiment-054-comparison",
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


experiment_main(__name__, __file__, EXPERIMENT)
