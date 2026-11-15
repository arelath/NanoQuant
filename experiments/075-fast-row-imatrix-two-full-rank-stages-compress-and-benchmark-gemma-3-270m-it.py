"""Experiment 075: Experiment 074 with a row and column imatrix."""

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
# See source.json and validation.json beside the locally generated matrix.
IMATRIX_PATH = DATASETS / "gemma-3-270m-it-imatrix-eaddario-huge-rows" / "imatrix.gguf"
IMATRIX_SHA256 = "3deceafea3aadd5fb6b75e22dfa0123bb99f22e0be95816354b374e80ebcc0c9"
CONFIG = config_delta(
    REFERENCE,
    model=GEMMA_3_270M_COMPRESSION_TEMPLATE.model,
    calibration=config_delta(
        REFERENCE.calibration,
        imatrix=ImatrixConfig(path=str(IMATRIX_PATH), sha256=IMATRIX_SHA256, use_rows=True),
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
        number=75,
        name="fast-row-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
        purpose="Repeat Experiment 074 with a locally generated imatrix that weights both rows and columns.",
        hypothesis="The locally measured row and column weights improve quality under the same compression settings.",
        baseline=BaselineRef.experiment(ExperimentRef(
            74, "fast-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
        )),
        tags=("gemma-3-270m-it", "compression", "quality", "imatrix", "two-full-rank-stages", "admm-600x2",
              "compiled-block-tuning", "stacked-qkv", "global-distillation", "task-limit-1000",
              "measured-tuning-batch", "fused-adamw", "speed", "imatrix-rows"),
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
