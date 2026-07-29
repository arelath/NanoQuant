"""Experiment 033: covariance-refined D2 compression on pinned Gemma 3 1B."""

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
    ObjectiveKind,
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
    calibration=replace(
        BASE_CONFIG.calibration,
        objective=replace(
            BASE_CONFIG.calibration.objective,
            kind=ObjectiveKind.DENSE_HESSIAN,
            sampling=replace(
                BASE_CONFIG.calibration.objective.sampling,
                max_tokens_per_layer=8192,
            ),
        ),
    ),
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
    distillation=replace(BASE_CONFIG.distillation, enabled=True),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=33,
        name="covariance-refined-d2-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Run the complete Experiment 022 exact-unit D2 workflow while refining accepted "
            "binary factors under an 8,192-row input covariance, then validate, export, and "
            "compare retained quality at the same factor-format budget."
        ),
        hypothesis=(
            "The independently held-out covariance and KL gains from the 26-block static screen "
            "survive resident tuning and global distillation, improving WikiText quality without "
            "increasing effective BPW relative to Experiment 022."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "covariance-refinement",
            "dense-hessian",
            "binary-coordinate-descent",
            "d2",
            "kl-calibrated",
            "exact-unit-sensitivity",
            "same-run-rank-response",
            "measured-unit-kl",
            "stacked-qkv",
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
