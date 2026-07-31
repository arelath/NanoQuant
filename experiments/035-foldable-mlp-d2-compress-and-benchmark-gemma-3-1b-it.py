"""Experiment 035: post-KD foldable MLP continuation on pinned Gemma 3 1B."""

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
    FoldableMlpMultiplierTuningConfig,
    KlAllocationObjective,
    KlSensitivityGranularity,
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
    distillation=replace(
        BASE_CONFIG.distillation,
        enabled=True,
        foldable_mlp_multipliers=FoldableMlpMultiplierTuningConfig(
            enabled=True,
            steps=64,
            learning_rate=1e-4,
            identity_penalty=100.0,
            gradient_clip=1.0,
            multiplier_limit=4.0,
            checkpoint_interval_steps=16,
            gradient_checkpointing=False,
        ),
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=35,
        name="foldable-mlp-d2-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Run the complete Experiment 022 D2 compression, global distillation, zero-byte "
            "foldable MLP continuation, validated export, and retained quality workflow."
        ),
        hypothesis=(
            "A conservative identity-initialized 64-step composed top-k continuation over only "
            "covariantly foldable MLP multipliers improves retained quality without changing "
            "packed bytes or effective BPW."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "d2",
            "kl-calibrated",
            "global-distillation",
            "foldable-mlp-multipliers",
            "zero-byte-refit",
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
