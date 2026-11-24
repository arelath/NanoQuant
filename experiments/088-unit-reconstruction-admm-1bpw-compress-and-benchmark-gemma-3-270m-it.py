"""Experiment 088 uses per-unit weighted reconstruction error with a 1 BPW decoder projection budget."""

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
    ROOT / "experiments/recipes/references/experiment087.json",
    "sha256:d8bc1c87b2d4c405fffe2791c6705393b23f16b1366b06338cd90b7431767fbb",
)
CONFIG = config_delta(
    REFERENCE,
    factorization=config_delta(
        REFERENCE.factorization,
        admm=config_delta(
            REFERENCE.factorization.admm,
            unit_reconstruction_retry=REFERENCE.factorization.admm.block_retry,
            block_retry=None,
        ),
    ),
    allocation=config_delta(REFERENCE.allocation, target_bpw=1.0, maximum_rank_layer_patterns=()),
)
# The retained receipt records the absolute checkout path on the experiment machine.
imatrix = CONFIG.calibration.imatrix
assert imatrix is not None
local_imatrix = ROOT / "Datasets/gemma-3-270m-it-imatrix-unsloth/imatrix_unsloth.gguf_file"
if Path(imatrix.path) != local_imatrix:
    CONFIG = config_delta(CONFIG, calibration=config_delta(
        CONFIG.calibration, imatrix=config_delta(imatrix, path=str(local_imatrix)),
    ))

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=88,
        name="unit-reconstruction-admm-1bpw-compress-and-benchmark-gemma-3-270m-it",
        purpose=("Measure Gemma 3 270M quality at a 1 BPW projection budget "
                 "with per-unit weighted reconstruction retries."),
        hypothesis="Longer ADMM fits reduce weighted reconstruction error at a fixed projection budget.",
        baseline=BaselineRef.experiment(ExperimentRef(
            87, "adaptive-admm-three-full-rank-stages-compress-and-benchmark-gemma-3-270m-it",
        )),
        tags=("gemma-3-270m-it", "imatrix", "uniform-rank", "1bpw", "adaptive-admm", "no-global-kd", "mmlu"),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend="dense",
    task_limit=1000,
    include_mmlu=True,
)
EXPERIMENT = replace(EXPERIMENT, workflow=replace(EXPERIMENT.workflow, local_files_only=True))

experiment_main(__name__, __file__, EXPERIMENT)
