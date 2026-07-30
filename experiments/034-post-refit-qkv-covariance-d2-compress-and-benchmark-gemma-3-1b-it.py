"""Experiment 034: selected post-refit QKV covariance refinement on Gemma 3 1B."""

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
    KlAllocationObjective,
    KlSensitivityGranularity,
    PostRefitCovarianceRefinementConfig,
    RankResponseSource,
    ReconstructionImportanceConfig,
)
from nanoquant.self_measured_d2_workflow import run_self_measured_d2_experiment

BASELINE = ExperimentRef(22, "d2-kl-compress-and-benchmark-gemma-3-1b-it")
_RUNTIME_KL_PROFILE = "runtime-kl-profile-required"
_RUNTIME_KL_PROFILE_KEY = "runtime-kl-profile-key-required"

BASE_CONFIG = ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE

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
    block_tuning=replace(
        BASE_CONFIG.block_tuning,
        post_refit_covariance_refinement=PostRefitCovarianceRefinementConfig(
            enabled=True,
            block_indices=(5, 11, 24, 25),
            shared_input_groups=("self_attn.attn_qkv",),
        ),
    ),
    distillation=replace(BASE_CONFIG.distillation, enabled=True),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=34,
        name="post-refit-qkv-covariance-d2-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Run the complete Experiment 022 D2 workflow while applying same-rank covariance "
            "refinement only to fused QKV in blocks 5, 11, 24, and 25 after block-local refit."
        ),
        hypothesis=(
            "The four-block placement selected by three disjoint functional slices and exact "
            "pre-KD WikiText improves final retained quality without changing effective BPW."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "post-refit-covariance",
            "selected-qkv",
            "same-rank",
            "d2",
            "kl-calibrated",
            "exact-unit-sensitivity",
            "same-run-rank-response",
            "measured-unit-kl",
            "global-distillation",
            "experiment-022-comparison",
            "wikitext2",
        ),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
)


experiment_callable_main(
    __name__,
    lambda: run_self_measured_d2_experiment(EXPERIMENT, launcher_path=__file__),
)
