"""Experiment 021: corrected D2 allocation with an Experiment 016-matched quality run."""

from dataclasses import replace

from recipes import (
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment
from nanoquant.config.schema import AllocationStrategy, KlSensitivityGranularity

BASELINE = ExperimentRef(16, "compress-and-benchmark-gemma-3-270m-it")
KL_PROFILE = "evidence/020/016-kl-budget-profile-v3"
KL_PROFILE_KEY = "sha256:e62295bb78c07fa7435560f7ba2463e2bbb7164e36963ab61dac9972b2f5a324"

BASE_CONFIG = replace(
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    model=GEMMA_3_270M_COMPRESSION_TEMPLATE.model,
)

CONFIG = replace(
    BASE_CONFIG,
    allocation=replace(
        BASE_CONFIG.allocation,
        strategy=AllocationStrategy.KL_CALIBRATED,
        kl_profile_artifact=KL_PROFILE,
        kl_profile_key=KL_PROFILE_KEY,
        kl_sensitivity_granularity=KlSensitivityGranularity.EXACT_OR_TYPE_BLOCK,
        reconstruction=replace(
            BASE_CONFIG.allocation.reconstruction,
            rank_trust_reference_run=BASELINE.run_output.as_posix(),
            rank_trust_fraction=0.25,
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
            "Measure corrected exact-unit D2 KL-calibrated rank allocation and compare the globally "
            "distilled packed result with Experiment 016 under the same quality protocol."
        ),
        hypothesis=(
            "A 0.25 trust-region step using evaluator-v3 normalized weighted-error sensitivities "
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
            "normalized-weighted-error",
            "rank-trust-region",
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
    raise SystemExit(
        run_compression_quality_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
