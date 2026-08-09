"""Experiment 057: replay 056 with a measured mixed k16 right-factor encoding."""

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
    ProductCodebookConfig,
    RankResponseSource,
    ReconstructionImportanceConfig,
    SharedInputMemberMultiplierConfig,
)

BASELINE = ExperimentRef(56, "physical-cap-rank-redistribution-d2-compress-and-benchmark-gemma-3-1b-it")
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
        implementation="nanoquant_admm_product_k16",
        binary_search=BinaryFactorSearchConfig(enabled=True),
        product_codebook=ProductCodebookConfig(enabled=True),
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
        number=57,
        name="matched-056-product-k16-codebook-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Replay Experiment 056's complete resident compression and retained quality protocol, "
            "changing only eligible right-factor storage from free 32-bit sign words to a measured "
            "mixture of free rows and two learned 8-bit half-codebooks without bit flips."
        ),
        hypothesis=(
            "A globally allocated mixed k16 product encoding can spend the same 1.0-BPW budget more "
            "effectively than free sign words while preserving Experiment 056's ranks, outliers, "
            "binary search, block tuning, post-block refit, and top-k KL distillation protocol."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "experiment-056-matched-control",
            "product-codebook",
            "k16",
            "two-8-bit-codebooks",
            "no-bit-flips",
            "mixed-encoding",
            "measured-global-allocation",
            "same-bpw",
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
