"""Experiment 023: interaction-corrected, tuned-operating-point D2 on Gemma 3 1B."""

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
from nanoquant.self_measured_d2_workflow import (
    SelfMeasuredD2ProfileOptions,
    run_self_measured_d2_experiment,
)

BASELINE = ExperimentRef(22, "d2-kl-compress-and-benchmark-gemma-3-1b-it")
_RUNTIME_KL_PROFILE = "runtime-kl-profile-required"
_RUNTIME_KL_PROFILE_KEY = "runtime-kl-profile-key-required"

BASE_CONFIG = ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE

# Experiment 022 measured raw single-unit KL anchors on a static (undistilled)
# uniform control over a 12x512 slice. This experiment applies the three D2
# follow-up corrections while keeping every other setting identical to 022:
#   1. Interaction-corrected anchors (INTERACTION_NORMALIZED_UNIT_KL): each type's
#      per-unit anchors are rescaled to sum to its measured joint type-arm KL,
#      replacing the additive-across-units assumption that over-counted joint
#      damage by ~3x and misweighted super-additive types (down_proj, o_proj).
#   2. Tuned operating point: the uniform control keeps global distillation
#      enabled and the KL profile is measured against the tuned reconstruction, so
#      the anchors reflect the same globally distilled point as the final packed
#      candidate rather than the static uniform point.
#   3. A larger 48x512 profiling slice, per the D2 campaign review's finding that
#      12x512 could not resolve a 1% improvement decisively while 48x512 could.
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
            kl_objective=KlAllocationObjective.INTERACTION_NORMALIZED_UNIT_KL,
            importance=ReconstructionImportanceConfig(),
            sensitivity_strength=1.0,
            protect_sensitive_units=False,
            target_protected_error_reduction_fraction=0.0,
            rank_trust_reference_run=None,
            rank_trust_fraction=1.0,
        ),
    ),
    # Global distillation is the final operating mode for the matched quality
    # comparison, and (via the tuned-operating-point option below) the point at
    # which the KL anchors are measured.
    distillation=replace(BASE_CONFIG.distillation, enabled=True),
)

# Measure the KL profile at the same globally distilled operating point as the
# final candidate, over the larger 48x512 slice the D2 review recommended.
PROFILE_OPTIONS = SelfMeasuredD2ProfileOptions(
    wikitext_samples=48,
    sequence_length=512,
    tuned_operating_point=True,
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=23,
        name="d2-kl-interaction-tuned-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Measure interaction-corrected exact-unit D2 KL allocation on pinned Gemma 3 1B, "
            "with anchors measured at the globally distilled operating point over a 48x512 "
            "slice, and compare the globally distilled packed result with Experiment 022 "
            "under the same quality protocol."
        ),
        hypothesis=(
            "Rescaling per-unit KL anchors to their measured joint type effect, measuring them "
            "at the tuned operating point, and profiling over 48x512 sequences improve globally "
            "tuned WikiText quality over Experiment 022 at no greater effective BPW."
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
            "interaction-normalized-unit-kl",
            "tuned-operating-point",
            "wikitext-48x512",
            "global-distillation",
            "experiment-022-comparison",
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
            profile_options=PROFILE_OPTIONS,
        )
    )
