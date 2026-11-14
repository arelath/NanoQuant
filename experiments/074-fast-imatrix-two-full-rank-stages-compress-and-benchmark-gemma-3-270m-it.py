"""Experiment 074: Experiment 073 on Gemma 3 270M IT with every available speed setting enabled."""

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

REFERENCE = retained_config(
    Path(__file__).parent / "recipes" / "references" / "experiment073.json",
    "sha256:c289aeaf4072b773ffa7f814a97a6958070f6407d80c02667c51701f6f2bc468",
)
DATASETS = Path(__file__).resolve().parents[1] / "Datasets"
# unsloth/gemma-3-270m-it-GGUF@c90975dbd40c0c7b275fefaae758c3415c906238; see source.json beside it.
IMATRIX_PATH = DATASETS / "gemma-3-270m-it-imatrix-unsloth" / "imatrix_unsloth.gguf_file"
IMATRIX_SHA256 = "2f88200eae0588cf7b3030f34e5469a529758a1c69f7a13453b7d3a91de017f3"
CONFIG = config_delta(
    REFERENCE,
    # The GGUF repository holds the imatrix; compression reads the matching safetensors snapshot.
    model=GEMMA_3_270M_COMPRESSION_TEMPLATE.model,
    calibration=config_delta(
        REFERENCE.calibration,
        imatrix=ImatrixConfig(path=str(IMATRIX_PATH), sha256=IMATRIX_SHA256),
    ),
    block_tuning=config_delta(
        REFERENCE.block_tuning,
        factorized=config_delta(
            REFERENCE.block_tuning.factorized,
            # Use 072's quality-gated fused AdamW projections from the first block
            # instead of from block 19. Up and shared QKV stay on parity, as in 072.
            optimizer=config_delta(
                REFERENCE.block_tuning.factorized.optimizer,
                start_block=0,
            ),
        ),
        # Every block tunes with a measured logical batch of up to 8 (071's 1B batch),
        # executed without gradient accumulation. No committed prefix to continue.
        batch_schedule=TuningBatchScheduleConfig(enabled=True, start_block=0, maximum_batch_size=8),
    ),
    runtime=config_delta(
        REFERENCE.runtime,
        block_forward_batch_size=8,
        activations=config_delta(REFERENCE.runtime.activations, gpu_cache=ActivationGpuCacheMode.AUTO),
    ),
)
PROFILE_OPTIONS = SelfMeasuredD2ProfileOptions(wikitext_samples=48, sequence_length=512, tuned_operating_point=True)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=74,
        name="fast-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
        purpose=("Compress Gemma 3 270M IT with Experiment 073's recipe and Unsloth's pinned 270M imatrix, "
                 "enabling every available speed setting: measured tuning batch, fused AdamW from block 0, "
                 "larger block forward batch, GPU activation cache and an uninterrupted single process."),
        hypothesis=("Experiment 073's two full-rank imatrix recipe transfers to Gemma 3 270M IT and the speed "
                    "settings shorten compression without breaking the run."),
        baseline=BaselineRef.experiment(ExperimentRef(
            73, "imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it",
        )),
        tags=("gemma-3-270m-it", "compression", "quality", "imatrix", "two-full-rank-stages", "admm-600x2",
              "compiled-block-tuning", "stacked-qkv", "global-distillation", "task-limit-1000",
              "measured-tuning-batch", "fused-adamw", "speed"),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    # 270M fits comfortably, so run every block in one process instead of one-block slices.
    interrupt_after_block_commits=None,
    quality_backend="dense",
    task_limit=1000,
)
EXPERIMENT = replace(EXPERIMENT, workflow=replace(EXPERIMENT.workflow, local_files_only=True))

experiment_callable_main(
    __name__,
    lambda: run_self_measured_d2_experiment(
        EXPERIMENT, launcher_path=__file__, profile_options=PROFILE_OPTIONS, reuse_maximum_rank_control=True,
    ),
)
