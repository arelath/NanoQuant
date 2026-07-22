"""Experiment 021: self-measured D2 allocation with an Experiment 016-matched quality run."""

from dataclasses import replace

from recipes import (
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
)

from nanoquant.config.schema import (
    AllocationStrategy,
    KlAllocationObjective,
    KlSensitivityGranularity,
    RankResponseSource,
    ReconstructionImportanceConfig,
)
from nanoquant.self_measured_d2_workflow import run_experiment021

BASELINE = ExperimentRef(16, "compress-and-benchmark-gemma-3-270m-it")
_RUNTIME_KL_PROFILE = "runtime-kl-profile-required"
_RUNTIME_KL_PROFILE_KEY = "runtime-kl-profile-key-required"

BASE_CONFIG = replace(
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    model=GEMMA_3_270M_COMPRESSION_TEMPLATE.model,
)

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
    # Experiment 016 evaluated its globally distilled state. Keep distillation
    # explicit here so the final packed comparison cannot become static-vs-tuned.
    distillation=replace(BASE_CONFIG.distillation, enabled=True),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=21,
        name="d2-kl-compress-and-benchmark-gemma-3-270m-it",
        purpose=(
            "Measure self-contained exact-unit D2 KL allocation with per-unit rank responses "
            "measured on this model, and compare the globally "
            "distilled packed result with Experiment 016 under the same quality protocol."
        ),
        hypothesis=(
            "Exact physical-unit KL anchors plus same-run calibration-weighted rank-response probes "
            "improves globally tuned WikiText quality at no greater effective BPW than Experiment 016."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "d2",
            "kl-calibrated",
            "exact-unit-sensitivity",
            "same-run-rank-response",
            "measured-unit-kl",
            "global-distillation",
            "experiment-016-comparison",
            "wikitext2",
            "ultrachat",
        ),
    ),
    CONFIG,
    expected_blocks=18,
    maximum_wddm_shared_gib=0.75,
)


if __name__ == "__main__":
    raise SystemExit(run_experiment021(EXPERIMENT, launcher_path=__file__))
