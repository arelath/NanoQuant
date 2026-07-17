"""Experiment 004: selectively add 30% packed bits to Experiment 003 v_proj layers."""

from ._experiment import (
    BaselineRef,
    ExperimentIdentity,
    define_rank_expansion_experiment,
)
from .experiment003 import EXPERIMENT_003

_IDENTITY = ExperimentIdentity(
        number=4,
        name="gemma-3-4b-it-vproj-plus30",
        purpose="Measure whether additive v_proj rank improves Experiment 003 reconstruction and quality.",
        hypothesis=(
            "Adding 30% packed bits only to final Experiment 003 v_proj states lowers weighted reconstruction "
            "error and improves matched WikiText/task quality while every non-v_proj tensor remains exact."
        ),
        baseline=BaselineRef.experiment(EXPERIMENT_003.identity),
        tags=("rank-allocation", "v-proj", "selective-replay", "gemma-3-4b-it"),
)

EXPERIMENT_004 = define_rank_expansion_experiment(
    _IDENTITY,
    EXPERIMENT_003.config,
    parent=EXPERIMENT_003,
    release_name="gemma-3-4b-it-vproj-plus30",
)

__all__ = ["EXPERIMENT_004"]
