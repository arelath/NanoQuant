"""Experiment 077: Experiment 075 with the curated Gemma calibration corpus."""

from dataclasses import replace
from pathlib import Path

from recipes import (
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
    experiment_pre_post_callable_main,
)
from recipes._delta import config_delta, retained_config

from nanoquant.config.schema import ImatrixConfig
from nanoquant.self_measured_d2_workflow import SelfMeasuredD2ProfileOptions, run_self_measured_d2_experiment

REFERENCE = retained_config(
    Path(__file__).parent / "recipes" / "references" / "experiment075.json",
    "sha256:06b8f9c7f5796e80c95cca6969ea3f1e1514cce6691704536b6e147d3df92b40",
)
DATASETS = Path(__file__).resolve().parents[1] / "Datasets"
IMATRIX_PATH = DATASETS / "gemma-3-270m-it-imatrix-calibration-v1-rows" / "imatrix.gguf"
IMATRIX_SHA256 = "e1e271f6a1b35dea9a372713e7ab403094ee848b174648b5dd679a70a44b7283"
CONFIG = config_delta(
    REFERENCE,
    calibration=config_delta(
        REFERENCE.calibration,
        imatrix=ImatrixConfig(path=str(IMATRIX_PATH), sha256=IMATRIX_SHA256, use_rows=True),
    ),
)
PROFILE_OPTIONS = SelfMeasuredD2ProfileOptions(wikitext_samples=48, sequence_length=512, tuned_operating_point=True)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=77,
        name="curated-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
        purpose="Measure the curated calibration corpus with Experiment 075's unchanged compression settings.",
        hypothesis="Curated prose and native Gemma prompts improve importance estimates and quantized quality.",
        baseline=BaselineRef.experiment(ExperimentRef(
            75, "fast-row-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
        )),
        tags=("gemma-3-270m-it", "compression", "quality", "imatrix", "two-full-rank-stages", "admm-600x2",
              "compiled-block-tuning", "stacked-qkv", "global-distillation", "task-limit-1000",
              "measured-tuning-batch", "fused-adamw", "imatrix-rows", "curated-calibration", "matched-1953-contexts"),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    interrupt_after_block_commits=None,
    quality_backend="dense",
    task_limit=1000,
)
EXPERIMENT = replace(EXPERIMENT, workflow=replace(EXPERIMENT.workflow, local_files_only=True))

experiment_pre_post_callable_main(
    __name__,
    __file__,
    lambda: run_self_measured_d2_experiment(
        EXPERIMENT, launcher_path=__file__, profile_options=PROFILE_OPTIONS, reuse_maximum_rank_control=True,
        audit_before_global_tuning=True,
    ),
    uniform_control=True,
)
