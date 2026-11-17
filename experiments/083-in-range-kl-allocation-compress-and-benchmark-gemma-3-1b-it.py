"""Experiment 083: use the in-range isolated KL rank allocation."""

from dataclasses import replace
from pathlib import Path

from recipes import (
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
    experiment_main,
)
from recipes._delta import config_delta, retained_config

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
DATASET = "Datasets/gemma-3-1b-it-kl-rank-plan-083"
DATASET_KEY = "sha256-39a90bec78d7569db32182aeeac874b70cb2efb48e8181c27b69868fc5d7cf15"
DECODER_FACTOR_BITS = 1_754_947_584
DECODER_WEIGHTS = 697_761_792

CONFIG = config_delta(
    REFERENCE,
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
            response_source=RankResponseSource.PINNED_DATASET,
            allocation_dataset=DATASET,
            allocation_dataset_key=DATASET_KEY,
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
        number=83,
        name="in-range-kl-allocation-compress-and-benchmark-gemma-3-1b-it",
        purpose="Test the in-range isolated KL allocation at the 082 factor budget.",
        hypothesis="The KL-priced rank plan lowers measured joint KL and improves benchmark quality.",
        baseline=BaselineRef.experiment(ExperimentRef(
            82, "staged-rank-allocation-compress-and-benchmark-gemma-3-1b-it",
        )),
        tags=("gemma-3-1b-it", "imatrix", "in-range-kl-allocation", "no-global-kd", "mmlu"),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend="dense",
    task_limit=1000,
    include_mmlu=True,
)
EXPERIMENT = replace(EXPERIMENT, workflow=replace(EXPERIMENT.workflow, local_files_only=True))

experiment_main(__name__, __file__, EXPERIMENT)
