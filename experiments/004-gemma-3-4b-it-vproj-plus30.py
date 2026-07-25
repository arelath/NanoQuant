"""Experiment 004: selectively expand Experiment 003 v_proj rank and benchmark quality."""

from recipes import (
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_rank_expansion_experiment,
    experiment_main,
)

PARENT = ExperimentRef(3, "compress-and-benchmark-gemma-3-4b-it")

EXPERIMENT = define_rank_expansion_experiment(
    ExperimentIdentity(
        number=4,
        name="gemma-3-4b-it-vproj-plus30",
        purpose="Measure whether additive v_proj rank improves Experiment 003 reconstruction and quality.",
        hypothesis=(
            "Adding 30% packed bits only to final Experiment 003 v_proj states lowers weighted reconstruction "
            "error and improves matched WikiText/task quality while every non-v_proj tensor remains exact."
        ),
        baseline=BaselineRef.experiment(PARENT),
        tags=("rank-allocation", "v-proj", "selective-replay", "gemma-3-4b-it"),
    ),
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    parent=PARENT,
    release_name="gemma-3-4b-it-vproj-plus30",
)


experiment_main(__name__, __file__, EXPERIMENT)
