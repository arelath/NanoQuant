"""Experiment 087: adaptive block ADMM with three independent full-rank stages on Gemma 3 270M IT."""

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

from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    BlockADMMRetryConfig,
    ImatrixConfig,
    TuningBatchScheduleConfig,
)
from nanoquant.self_measured_d2_workflow import SelfMeasuredD2ProfileOptions, run_self_measured_d2_experiment

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = retained_config(
    ROOT / "experiments/recipes/references/experiment073.json",
    "sha256:c289aeaf4072b773ffa7f814a97a6958070f6407d80c02667c51701f6f2bc468",
)
IMATRIX = ROOT / "Datasets/gemma-3-270m-it-imatrix-unsloth/imatrix_unsloth.gguf_file"
CONFIG = config_delta(
    REFERENCE,
    model=GEMMA_3_270M_COMPRESSION_TEMPLATE.model,
    calibration=config_delta(
        REFERENCE.calibration,
        imatrix=ImatrixConfig(
            path=str(IMATRIX), sha256="2f88200eae0588cf7b3030f34e5469a529758a1c69f7a13453b7d3a91de017f3",
        ),
    ),
    factorization=config_delta(
        REFERENCE.factorization,
        admm=config_delta(
            REFERENCE.factorization.admm, outer_iterations=3200,
            block_retry=BlockADMMRetryConfig(target_error=0.01, stop_iterations=50_000),
        ),
    ),
    allocation=config_delta(
        REFERENCE.allocation,
        bounds=config_delta(REFERENCE.allocation.bounds, overcomplete_rank_ceiling_fraction=3.0),
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
PROFILE_OPTIONS = SelfMeasuredD2ProfileOptions(wikitext_samples=48, sequence_length=512, tuned_operating_point=False)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=87,
        name="adaptive-admm-three-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
        purpose="Measure adaptive ADMM budgets at the fixed ranks and quality protocol of 081.",
        hypothesis="Longer solves on high-error blocks improve quality at the same factor storage cost.",
        baseline=BaselineRef.experiment(ExperimentRef(
            81, "three-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
        )),
        tags=("gemma-3-270m-it", "imatrix", "three-full-rank-stages", "no-global-kd", "mmlu"),
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
    lambda: run_self_measured_d2_experiment(
        EXPERIMENT, launcher_path=__file__, profile_options=PROFILE_OPTIONS, reuse_maximum_rank_control=True,
    ),
)
