"""Experiment 081: three independent full-rank stages on Gemma 3 270M IT."""

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

from nanoquant.config.schema import ActivationGpuCacheMode, ImatrixConfig, TuningBatchScheduleConfig
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
        number=81,
        name="three-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
        purpose="Compare three independently scaled full-rank stages against 074's two-stage pre-KD model.",
        hypothesis="A third full-rank residual stage improves Gemma 3 270M quality at higher storage cost.",
        baseline=BaselineRef.experiment(ExperimentRef(
            74, "fast-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
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
