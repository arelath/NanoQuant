"""Experiment 022: self-measured D2 allocation on pinned Gemma 3 1B."""

from dataclasses import replace

from recipes import (
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
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
from nanoquant.self_measured_d2_workflow import run_self_measured_d2_experiment

BASELINE = ExperimentRef(17, "compress-and-benchmark-gemma-3-1b-it")
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
    # Experiment 017 evaluated its globally distilled state. Keep the same
    # final operating mode for the matched quality comparison.
    distillation=replace(BASE_CONFIG.distillation, enabled=True),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=22,
        name="d2-kl-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Measure self-contained exact-unit D2 KL allocation with per-unit rank responses "
            "measured on pinned Gemma 3 1B, and compare the globally distilled packed result "
            "with Experiment 017 under the same quality protocol."
        ),
        hypothesis=(
            "Exact physical-unit KL anchors plus same-run calibration-weighted rank-response probes "
            "improve globally tuned WikiText quality at no greater effective BPW than Experiment 017."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "d2",
            "kl-calibrated",
            "exact-unit-sensitivity",
            "same-run-rank-response",
            "measured-unit-kl",
            "global-distillation",
            "experiment-017-comparison",
            "wikitext2",
            "ultrachat",
        ),
    ),
    CONFIG,
    expected_blocks=26,
    maximum_wddm_shared_gib=0.75,
)


if __name__ == "__main__":
    raise SystemExit(
        run_self_measured_d2_experiment(
            EXPERIMENT,
            launcher_path=__file__,
        )
    )
