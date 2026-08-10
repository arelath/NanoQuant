"""Experiment 058: replay 057 with rate-matched expanded INT8 outliers."""

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
    DType,
    KlAllocationObjective,
    KlSensitivityGranularity,
    LowRankPatchConfig,
    ProductCodebookConfig,
    RankResponseSource,
    ReconstructionImportanceConfig,
    SharedInputMemberMultiplierConfig,
)

BASELINE = ExperimentRef(57, "matched-056-product-k16-codebook-compress-and-benchmark-gemma-3-1b-it")
PROFILE = "evidence/054/054-d2-uniform-control-kl-profile"
PROFILE_KEY = "sha256:4a67e45d5266763b09e3b487a3820f4ad8520201b144807241e2744d9c271bf9"
DOWN_PROJECTION_INPUTS = 6912
RATE_MATCHED_DOWN_OUTLIERS = 13
RATE_MATCHED_OUTLIER_FRACTION = RATE_MATCHED_DOWN_OUTLIERS / DOWN_PROJECTION_INPUTS

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
        retry=replace(
            BASE_CONFIG.allocation.retry,
            outlier_count_increment_at_rank_cap=1,
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
    outliers=replace(
        BASE_CONFIG.outliers,
        fraction=RATE_MATCHED_OUTLIER_FRACTION,
        storage_dtype=DType.INT8,
    ),
    factorization=replace(
        BASE_CONFIG.factorization,
        implementation="nanoquant_admm_product_k16",
        binary_search=BinaryFactorSearchConfig(enabled=True),
        product_codebook=ProductCodebookConfig(
            enabled=True,
            probe_outer_iterations=100,
            probe_frontier_outer_iterations=400,
            probe_final_outer_iterations=1_200,
            probe_final_options_per_layer=2,
        ),
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
        number=58,
        name="matched-057-rate-matched-int8-outliers-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Replay Experiment 057's complete product-k16 compression and quality protocol, "
            "changing the global outlier fraction and storage so each down projection uses "
            "thirteen calibration-weighted INT8 columns at no greater sidecar rate than 057's "
            "seven BF16 columns, and selecting codebook options with a 100/400/1200-step "
            "multi-fidelity screen; a failed full-rank retry adds one outlier column."
        ),
        hypothesis=(
            "Reinvesting BF16 outlier value bits into nearly twice as many residual-selected "
            "INT8 columns will preserve the 057 representation and improve full-model quality, "
            "matching the positive held-out down-projection screen, while final codebook "
            "allocation uses converged 1200-step option receipts instead of coarse receipts."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "experiment-057-matched-control",
            "product-codebook",
            "k16",
            "int8-outliers",
            "rate-matched-outliers",
            "calibration-weighted",
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
