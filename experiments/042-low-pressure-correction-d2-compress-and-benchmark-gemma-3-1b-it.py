"""Experiment 042: fresh full D2 campaign with the accepted low-pressure correction."""

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

BASELINE = ExperimentRef(37, "matched-tail-aware-d2-base-gemma-3-1b-it")
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
        mass_floor_correction=replace(
            BASE_CONFIG.distillation.mass_floor_correction,
            enabled=True,
            epochs=1,
            learning_rate=1e-5,
            maximum_batches_per_epoch=32,
            scheduler_total_steps=128,
            minimum_teacher_mass_ratio=0.8,
            mass_loss_weight=2.0,
        ),
        final_norm_calibration=replace(
            BASE_CONFIG.distillation.final_norm_calibration,
            enabled=True,
            scale=1.015,
        ),
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=42,
        name="low-pressure-correction-d2-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Run one fresh pinned-Gemma D2 compression through conditional KD, the accepted "
            "32-step one-sided mass correction, immutable 1.015 final-norm fold, packed quality, "
            "llama.cpp quality, and the complete validated GGUF export lifecycle."
        ),
        hypothesis=(
            "On a fresh D2 factorization, the Experiment 040 low-pressure correction and minimal "
            "confidence fold preserve the selected-mass floor and improve broad NLL/KL without "
            "an established task regression at unchanged effective BPW."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "d2",
            "kl-calibrated",
            "global-distillation",
            "mass-floor-correction",
            "final-norm-calibration",
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
