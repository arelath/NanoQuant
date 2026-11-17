"""Experiment 084: allocate Gemma 3 270M IT ranks with curated sensitivity."""

from dataclasses import replace
from pathlib import Path

from recipes import (
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
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
    ImatrixConfig,
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
DATASET = Path("Datasets/gemma-3-270m-it-curated-kl-rank-options-084")
DECODER_FACTOR_BITS = 282_968_064
DECODER_WEIGHTS = 100_270_080

CONFIG = config_delta(
    REFERENCE,
    model=GEMMA_3_270M_COMPRESSION_TEMPLATE.model,
    calibration=config_delta(
        REFERENCE.calibration,
        imatrix=ImatrixConfig(
            path=str(ROOT / "Datasets/gemma-3-270m-it-imatrix-unsloth/imatrix_unsloth.gguf_file"),
            sha256="2f88200eae0588cf7b3030f34e5469a529758a1c69f7a13453b7d3a91de017f3",
        ),
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
        number=84,
        name="curated-kl-allocation-compress-and-benchmark-gemma-3-270m-it",
        purpose="Test curated in-range KL allocation at the two-stage 074 factor budget.",
        hypothesis="The KL-priced rank plan lowers measured joint KL and improves benchmark quality.",
        baseline=BaselineRef.experiment(ExperimentRef(
            74, "fast-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
        )),
        tags=("gemma-3-270m-it", "imatrix", "curated-kl-allocation", "no-global-kd", "mmlu"),
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
