"""Experiment 086 repeats 080's uniform-rank compression with a linear ADMM penalty ramp."""

from dataclasses import replace
from pathlib import Path

from recipes import (
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
    experiment_main,
)
from recipes._delta import config_delta, retained_config

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = retained_config(
    ROOT / "experiments/recipes/references/experiment080.json",
    "sha256:e0ff7e41d3f6a03282add89ad7bf5a84037fe9380c3112b2ab9bc5061181e0c6",
)
CONFIG = config_delta(
    REFERENCE,
    factorization=config_delta(
        REFERENCE.factorization,
        admm=config_delta(REFERENCE.factorization.admm, penalty_schedule="linear"),
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=86,
        name="linear-600-uniform-compress-and-benchmark-gemma-3-1b-it",
        purpose="Measure Linear 600 on the Experiment 080 uniform allocation.",
        hypothesis="Linear 600 lowers joint KL at the same ranks and compression budget as Cubic 600.",
        baseline=BaselineRef.experiment(ExperimentRef(
            80, "fast-imatrix-no-kd-compress-and-benchmark-gemma-3-1b-it",
        )),
        tags=("gemma-3-1b-it", "uniform-rank", "admm-linear-600", "no-global-kd", "mmlu"),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    quality_backend="dense",
    task_limit=1000,
    include_mmlu=True,
)
EXPERIMENT = replace(EXPERIMENT, workflow=replace(EXPERIMENT.workflow, local_files_only=True))

experiment_main(__name__, __file__, EXPERIMENT)
