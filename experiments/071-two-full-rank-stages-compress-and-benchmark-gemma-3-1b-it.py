"""Experiment 071: two full-rank stages, 600x2 ADMM, compiled tuning and reusable datasets."""

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

from nanoquant.self_measured_d2_workflow import SelfMeasuredD2ProfileOptions, run_self_measured_d2_experiment

REFERENCE = retained_config(
    Path(__file__).parent / "recipes" / "references" / "experiment070.json",
    "sha256:97f03cb6e399438814dee7bdfeaec3108ec6bce519da842ada9099ca9ad4c509",
)
CONFIG = config_delta(
    REFERENCE,
    allocation=config_delta(
        REFERENCE.allocation,
        # Promotion follows allocation and explicitly permits exceeding the nominal 2 BPW target.
        maximum_rank_layer_patterns=("*",),
        bounds=config_delta(REFERENCE.allocation.bounds, overcomplete_rank_ceiling_fraction=2.0),
    ),
    factorization=config_delta(
        REFERENCE.factorization,
        admm=config_delta(REFERENCE.factorization.admm, outer_iterations=600, inner_iterations=2),
    ),
    block_tuning=config_delta(REFERENCE.block_tuning, compile_forward=True),
)
DATASETS = Path(__file__).resolve().parents[1] / "Datasets"
ALLOCATION_PROFILE_DATASET = DATASETS / "gemma-3-1b-it-allocation-070"
KL_TEACHER_DATASET = DATASETS / "gemma-3-1b-it-kl-48x512"
PROFILE_OPTIONS = SelfMeasuredD2ProfileOptions(wikitext_samples=48, sequence_length=512, tuned_operating_point=True)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=71,
        name="two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Repeat Experiment 070 with two full-rank independent stages per owner, 600x2 fitting, "
            "compiled block tuning and reusable planning datasets."
        ),
        hypothesis=(
            "Two full-rank stages improve quality with more represented bits while shorter ADMM, "
            "compiled tuning and cached planning inputs reduce preparation and compute time."
        ),
        baseline=BaselineRef.experiment(ExperimentRef(
            70, "independent-residual-rank-2bpw-compress-and-benchmark-gemma-3-1b-it",
        )),
        tags=("gemma-3-1b-it", "compression", "quality", "two-full-rank-stages", "admm-600x2",
              "compiled-block-tuning", "reusable-datasets", "stacked-qkv", "global-distillation", "task-limit-1000"),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    interrupt_after_block_commits=1,
    task_limit=1000,
)
EXPERIMENT = replace(EXPERIMENT, workflow=replace(EXPERIMENT.workflow, local_files_only=True))

experiment_callable_main(
    __name__,
    lambda: run_self_measured_d2_experiment(
        EXPERIMENT, launcher_path=__file__, profile_options=PROFILE_OPTIONS,
        allocation_profile_dataset=ALLOCATION_PROFILE_DATASET, kl_teacher_dataset=KL_TEACHER_DATASET,
    ),
)
