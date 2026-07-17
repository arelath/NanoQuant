"""Experiment 005: request twice the Experiment 003 v_proj bits and benchmark quality."""

from recipes import (
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_rank_expansion_experiment,
)

from nanoquant.rank_expansion_experiment import run_rank_expansion_experiment

PARENT = ExperimentRef(3, "compress-and-benchmark-gemma-3-4b-it")

EXPERIMENT = define_rank_expansion_experiment(
    ExperimentIdentity(
        number=5,
        name="gemma-3-4b-it-vproj-double-request",
        purpose="Upper-bound the Experiment 003 v_proj allocation hypothesis at maximum physical rank.",
        hypothesis=(
            "Requesting twice the packed v_proj bits, capped at rank 1024, may produce enough downstream quality "
            "gain to overturn the negative Experiment 004 result."
        ),
        baseline=BaselineRef.experiment(PARENT),
        tags=("rank-allocation", "v-proj", "maximum-rank", "gemma-3-4b-it"),
    ),
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    parent=PARENT,
    release_name="gemma-3-4b-it-vproj-maxrank",
    bit_multiplier=2.0,
)


if __name__ == "__main__":
    raise SystemExit(
        run_rank_expansion_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
