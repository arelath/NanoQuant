"""Experiment 073: Experiment 072 settings on Gemma 3 1B IT with pinned imatrix weights."""

from dataclasses import replace
from pathlib import Path

from recipes import (
    BASE_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
    experiment_callable_main,
)
from recipes._delta import config_delta, retained_config

from nanoquant.config.schema import ImatrixConfig
from nanoquant.self_measured_d2_workflow import SelfMeasuredD2ProfileOptions, run_self_measured_d2_experiment

REFERENCE = retained_config(
    Path(__file__).parent / "recipes" / "references" / "experiment072.json",
    "sha256:e9a730f78beaeb8f0b1b2813b331ad36ebb0bf6a67a274acca0fa32fcc629dc5",
)
DATASETS = Path(__file__).resolve().parents[1] / "Datasets"
IMATRIX_PATH = DATASETS / "gemma-3-1b-it-imatrix-unsloth" / "imatrix_unsloth.dat"
IMATRIX_SHA256 = "86a81192de49a6480f8dc19c5a7fd3f4377af59172fd4c7b8d93d296a0a18515"
CONFIG = config_delta(
    REFERENCE,
    model=BASE_COMPRESSION_TEMPLATE.model,
    calibration=config_delta(
        REFERENCE.calibration,
        imatrix=ImatrixConfig(path=str(IMATRIX_PATH), sha256=IMATRIX_SHA256),
    ),
    block_tuning=config_delta(
        REFERENCE.block_tuning,
        factorized=config_delta(
            REFERENCE.block_tuning.factorized,
            # Retain 072's numerical schedule, including the block-19 switch.
            # A new campaign must not bind to 072's previously committed journal.
            optimizer=config_delta(
                REFERENCE.block_tuning.factorized.optimizer,
                continuation_journal_sha256=None,
            ),
        ),
    ),
)
KL_TEACHER_DATASET = DATASETS / "gemma-3-1b-it-kl-48x512"
PROFILE_OPTIONS = SelfMeasuredD2ProfileOptions(wikitext_samples=48, sequence_length=512, tuned_operating_point=True)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=73,
        name="imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it",
        purpose=("Compress Gemma 3 1B IT using Experiment 072's fitting, tuning and runtime settings, "
                 "with Unsloth's pinned llama.cpp imatrix as input-channel importance."),
        hypothesis="Activation importance from the external imatrix improves quality at two full-rank stages.",
        baseline=BaselineRef.experiment(ExperimentRef(
            71, "two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it",
        )),
        tags=("gemma-3-1b-it", "compression", "quality", "imatrix", "two-full-rank-stages", "admm-600x2",
              "compiled-block-tuning", "stacked-qkv", "global-distillation", "task-limit-1000"),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    interrupt_after_block_commits=1,
    quality_backend="dense",
    task_limit=1000,
)
EXPERIMENT = replace(EXPERIMENT, workflow=replace(EXPERIMENT.workflow, local_files_only=True))

experiment_callable_main(
    __name__,
    lambda: run_self_measured_d2_experiment(
        EXPERIMENT, launcher_path=__file__, profile_options=PROFILE_OPTIONS, kl_teacher_dataset=KL_TEACHER_DATASET,
        reuse_maximum_rank_control=True,
    ),
)
