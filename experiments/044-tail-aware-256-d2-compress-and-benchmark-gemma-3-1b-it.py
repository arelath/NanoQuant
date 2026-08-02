"""Experiment 044: fresh D2 compression with matched 256-step tail-aware KD."""

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
    DistillationLoss,
    KlAllocationObjective,
    KlSensitivityGranularity,
    RankResponseSource,
    ReconstructionImportanceConfig,
)

BASELINE = ExperimentRef(42, "low-pressure-correction-d2-compress-and-benchmark-gemma-3-1b-it")
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
        enabled=True,
        loss=DistillationLoss.TOP_K_TAIL,
        epochs=8,
        maximum_batches_per_epoch=32,
        tail_mass_weight=0.5,
        mass_floor_correction=replace(
            BASE_CONFIG.distillation.mass_floor_correction,
            enabled=False,
        ),
        final_norm_calibration=replace(
            BASE_CONFIG.distillation.final_norm_calibration,
            enabled=False,
        ),
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=44,
        name="tail-aware-256-d2-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Run a fresh pinned-Gemma D2 compression with the Experiment 043 matched "
            "256-step top-64-plus-tail primary objective, then complete strict resident "
            "validation, packed quality, llama.cpp quality, and GGUF export."
        ),
        hypothesis=(
            "An explicit 256-step horizon and always-on aggregated-tail objective transfer "
            "the Experiment 043 NLL/KL gains to a fresh factorization without the rejected "
            "mass-floor correction, fixed final-norm fold, or an established task regression."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "d2",
            "kl-calibrated",
            "global-distillation",
            "top-k-tail",
            "matched-256-step-horizon",
            "packed-quality",
            "llama.cpp-quality",
            "gguf",
            "wikitext2",
        ),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    llamacpp_quality=True,
)


experiment_main(__name__, __file__, EXPERIMENT)
