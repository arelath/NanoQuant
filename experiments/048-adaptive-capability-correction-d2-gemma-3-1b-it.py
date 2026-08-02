"""Experiment 048: fresh D2 run with adaptive correction-checkpoint selection."""

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
    DistillationLoss,
    KlAllocationObjective,
    KlSensitivityGranularity,
    RankResponseSource,
    ReconstructionImportanceConfig,
)

BASELINE = ExperimentRef(44, "tail-aware-256-d2-compress-and-benchmark-gemma-3-1b-it")
PROFILE = "evidence/035/035-d2-uniform-control-kl-profile"
PROFILE_KEY = "sha256:8878a1bcc5cf2301a0e5c1cc21b2691950017e4c0fab4d9ea3f1eddb2b6e5f21"
PRIMARY_PROTOCOL_HASH = (
    "sha256:0ed7993a02eb980403ebeb97ff2d2cbf738242e64e6a7d07ad9f2900ef611936"
)

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
            enabled=True,
            expected_initializer_protocol_hash=PRIMARY_PROTOCOL_HASH,
            expected_initializer_steps=256,
            epochs=4,
            learning_rate=1e-5,
            maximum_batches_per_epoch=32,
            scheduler_total_steps=128,
            minimum_teacher_mass_ratio=0.8,
            mass_loss_weight=2.0,
        ),
        final_norm_calibration=replace(
            BASE_CONFIG.distillation.final_norm_calibration,
            enabled=False,
        ),
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=48,
        name="adaptive-capability-correction-d2-gemma-3-1b-it",
        purpose=(
            "Run one fresh pinned-Gemma D2 factorization through the exact 256-step "
            "tail-aware primary regime and a complete four-checkpoint correction curve, "
            "then select, materialize, validate, export, and benchmark at most one "
            "checkpoint under the frozen C4 policy."
        ),
        hypothesis=(
            "Selecting the earliest jointly near-optimal correction checkpoint on an "
            "independent C4 calibration slice improves same-run raw NLL and full KL over "
            "the uncorrected 256-step endpoint without transferring a fixed checkpoint or "
            "calibration scalar across factorization trajectories."
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
            "adaptive-mass-floor-correction",
            "c4-checkpoint-selection",
            "temperature-separated-reporting",
            "packed-quality",
            "llama.cpp-quality",
            "gguf",
        ),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    llamacpp_quality=True,
)


def _paused_main() -> int:
    raise RuntimeError(
        "Experiment 048 remains paused until its adaptive campaign orchestrator, "
        "campaign receipt, and immutable C4 slice reservations are complete"
    )


experiment_callable_main(__name__, _paused_main)
