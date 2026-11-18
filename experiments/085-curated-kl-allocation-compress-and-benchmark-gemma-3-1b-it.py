"""Experiment 085: allocate Gemma 3 1B IT ranks with curated sensitivity."""

from dataclasses import replace
from pathlib import Path

from recipes import (
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
    experiment_callable_main,
)
from recipes._delta import config_delta, retained_config
from recipes._experiment import run_kl_rank_option_experiment

from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    AllocationStrategy,
    KlSensitivityGranularity,
    RankResponseSource,
    ReconstructionRankPlanningConfig,
    TuningBatchScheduleConfig,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = retained_config(
    ROOT / "experiments/recipes/references/experiment073.json",
    "sha256:c289aeaf4072b773ffa7f814a97a6958070f6407d80c02667c51701f6f2bc468",
)
DATASET = Path("Datasets/gemma-3-1b-it-curated-kl-rank-options-085")
DECODER_FACTOR_BITS = 1_754_947_584
DECODER_WEIGHTS = 697_761_792

CONFIG = config_delta(
    REFERENCE,
    factorization=config_delta(
        REFERENCE.factorization,
        admm=config_delta(REFERENCE.factorization.admm, outer_iterations=800, penalty_schedule="linear"),
    ),
    allocation=config_delta(
        REFERENCE.allocation,
        target_bpw=DECODER_FACTOR_BITS / DECODER_WEIGHTS,
        strategy=AllocationStrategy.RECONSTRUCTION_AWARE,
        kl_profile_artifact=None,
        kl_profile_key=None,
        kl_sensitivity_granularity=KlSensitivityGranularity.EXACT_OR_TYPE_BLOCK,
        maximum_rank_layer_patterns=(),
        bounds=config_delta(
            REFERENCE.allocation.bounds,
            floor_fraction_of_uniform=0.25,
            ceiling_fraction_of_uniform=3.0,
            overcomplete_rank_ceiling_fraction=3.0,
            edge_block_boost=0.0,
        ),
        retry=config_delta(REFERENCE.allocation.retry, enabled=False),
        reconstruction=ReconstructionRankPlanningConfig(
            enabled=True,
            objective_mode="calibration_weighted",
            response_source=RankResponseSource.KL_DATASET,
            sensitivity_strength=0.0,
            protect_sensitive_units=False,
            target_protected_error_reduction_fraction=0.0,
        ),
    ),
    distillation=config_delta(REFERENCE.distillation, enabled=False),
    block_tuning=config_delta(
        REFERENCE.block_tuning,
        factorized=config_delta(
            REFERENCE.block_tuning.factorized,
            optimizer=config_delta(REFERENCE.block_tuning.factorized.optimizer, start_block=0),
        ),
        batch_schedule=TuningBatchScheduleConfig(enabled=True, start_block=0, maximum_batch_size=8),
    ),
    runtime=config_delta(
        REFERENCE.runtime,
        block_forward_batch_size=8,
        activations=config_delta(REFERENCE.runtime.activations, gpu_cache=ActivationGpuCacheMode.AUTO),
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=85,
        name="curated-kl-allocation-compress-and-benchmark-gemma-3-1b-it",
        purpose="Repeat the curated KL allocation with an 800-iteration linear ADMM penalty ramp.",
        hypothesis="Linear 800 improves joint decoder KL and benchmark quality at the same ranks as cubic 600.",
        baseline=BaselineRef.experiment(ExperimentRef(
            83, "in-range-kl-allocation-compress-and-benchmark-gemma-3-1b-it",
        )),
        tags=("gemma-3-1b-it", "imatrix", "in-range-kl-allocation", "curated-sensitivity", "no-global-kd", "mmlu"),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend="dense",
    task_limit=1000,
    include_mmlu=True,
)
EXPERIMENT = replace(EXPERIMENT, workflow=replace(EXPERIMENT.workflow, local_files_only=True))

experiment_callable_main(
    __name__,
    lambda: run_kl_rank_option_experiment(EXPERIMENT, launcher_path=__file__, dataset_path=DATASET),
)
