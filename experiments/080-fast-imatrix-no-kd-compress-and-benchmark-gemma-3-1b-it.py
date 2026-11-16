"""Experiment 080: Experiment 074's speed settings on Gemma 3 1B IT without global KD."""

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
from recipes._experiment import run_imatrix_attribution_experiment

from nanoquant.config.schema import ActivationGpuCacheMode, TuningBatchScheduleConfig
from nanoquant.self_measured_d2_workflow import SelfMeasuredD2ProfileOptions

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = retained_config(
    ROOT / "experiments/recipes/references/experiment073.json",
    "sha256:c289aeaf4072b773ffa7f814a97a6958070f6407d80c02667c51701f6f2bc468",
)
CONFIG = config_delta(
    REFERENCE,
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
SWEEP = (
    ROOT / "Datasets/gemma-3-1b-it-fp16-sensitivity-16x256/measurements"
    / "022fc4f64542b546eebd3bbbe21ee50cf36d36946dcd0a871d20ae5e24abfc6f/results.json"
)
FITS_SHA256 = "a2dced15915832d2d2fc7f4064cb80fa685ae4cd3ea78bdaba0203ffe466ce57"
ATTRIBUTION = ROOT / "Results/080/080-imatrix-error-attribution.md"

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=80,
        name="fast-imatrix-no-kd-compress-and-benchmark-gemma-3-1b-it",
        purpose="Apply 074's two full-rank stages and speed settings to Gemma 3 1B IT without global KD.",
        hypothesis="Measure quality and estimate which decoder projections dominate error before global KD.",
        baseline=BaselineRef.experiment(ExperimentRef(
            73, "imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it",
        )),
        tags=("gemma-3-1b-it", "imatrix", "two-full-rank-stages", "no-global-kd", "mmlu", "error-attribution"),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    interrupt_after_block_commits=None,
    quality_backend="dense",
    task_limit=1000,
    include_mmlu=True,
)
EXPERIMENT = replace(EXPERIMENT, workflow=replace(EXPERIMENT.workflow, local_files_only=True))


experiment_callable_main(
    __name__,
    lambda: run_imatrix_attribution_experiment(
        EXPERIMENT,
        launcher_path=__file__,
        profile_options=PROFILE_OPTIONS,
        sweep=SWEEP,
        fits_sha256=FITS_SHA256,
        attribution=ATTRIBUTION,
    ),
)
