"""Experiment 037: fresh common D2 state for matched conditional/tail-aware KD."""

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
    KlAllocationObjective,
    KlSensitivityGranularity,
    RankResponseSource,
    ReconstructionImportanceConfig,
)

BASELINE = ExperimentRef(35, "foldable-mlp-d2-compress-and-benchmark-gemma-3-1b-it")
PROFILE = "evidence/035/035-d2-uniform-control-kl-profile"
PROFILE_KEY = "sha256:8878a1bcc5cf2301a0e5c1cc21b2691950017e4c0fab4d9ea3f1eddb2b6e5f21"

BASE_CONFIG = ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE

CONFIG = replace(
    BASE_CONFIG,
    allocation=replace(
        BASE_CONFIG.allocation,
        strategy=AllocationStrategy.KL_CALIBRATED,
        kl_profile_artifact=PROFILE,
        kl_profile_key=PROFILE_KEY,
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
    distillation=replace(
        BASE_CONFIG.distillation,
        enabled=False,
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=37,
        name="matched-tail-aware-d2-base-gemma-3-1b-it",
        purpose=(
            "Create one fresh pinned-Gemma D2 frozen state, without global KD, for exact "
            "hard-link branches into matched conditional and tail-aware distillation arms."
        ),
        hypothesis=(
            "On one byte-identical fresh factorization, tail-aware 0.5 KD plus identity-bound "
            "final-norm calibration improves broad NLL/KL and retained quality over conditional KD."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "d2",
            "kl-calibrated",
            "matched-kd-control",
            "tail-aware-distillation",
            "pre-kd-common-state",
            "wikitext2",
        ),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
)


experiment_main(__name__, __file__, EXPERIMENT)
