"""Experiment 076: Experiment 075 with Unsloth columns and local Eaddario rows."""

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

from nanoquant.config.schema import ImatrixConfig
from nanoquant.self_measured_d2_workflow import SelfMeasuredD2ProfileOptions, run_self_measured_d2_experiment

# Freeze 075's resolved input recipe; the only numerical setting changed is the imatrix.
REFERENCE = retained_config(
    Path(__file__).parent / "recipes" / "references" / "experiment075.json",
    "sha256:06b8f9c7f5796e80c95cca6969ea3f1e1514cce6691704536b6e147d3df92b40",
)
DATASETS = Path(__file__).resolve().parents[1] / "Datasets"
IMATRIX_PATH = DATASETS / "gemma-3-270m-it-imatrix-unsloth-columns-eaddario-rows" / "imatrix.gguf"
IMATRIX_SHA256 = "f4b1e6e371afbc34d3296e16559d4ad707be41075f88f4a5049fcfaaf8ef678d"
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
        number=76,
        name="fast-hybrid-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
        purpose="Repeat Experiment 075 with Unsloth activation columns and the same local Eaddario activation rows.",
        hypothesis=("Unsloth column weights improve quality while holding Experiment 075's "
                    "row weights and recipe fixed."),
        baseline=BaselineRef.experiment(ExperimentRef(
            75, "fast-row-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
        )),
        tags=("gemma-3-270m-it", "compression", "quality", "imatrix", "two-full-rank-stages", "admm-600x2",
              "compiled-block-tuning", "stacked-qkv", "global-distillation", "task-limit-1000",
              "measured-tuning-batch", "fused-adamw", "speed", "imatrix-rows", "hybrid-imatrix",
              "unsloth-columns", "eaddario-rows"),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
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
