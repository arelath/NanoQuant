"""Experiment 005: request 2x v_proj bits, saturating all layers at maximum rank."""

from ._experiment import (
    BaselineRef,
    ExperimentIdentity,
    define_rank_expansion_experiment,
)
from .experiment003 import EXPERIMENT_003

_IDENTITY = ExperimentIdentity(
        number=5,
        name="gemma-3-4b-it-vproj-double-request",
        purpose="Upper-bound the Experiment 003 v_proj allocation hypothesis at maximum physical rank.",
        hypothesis=(
            "Requesting twice the packed v_proj bits, capped at rank 1024, may produce enough downstream quality "
            "gain to overturn the negative Experiment 004 result."
        ),
        baseline=BaselineRef.experiment(EXPERIMENT_003.identity),
        tags=("rank-allocation", "v-proj", "maximum-rank", "gemma-3-4b-it"),
)

EXPERIMENT_005 = define_rank_expansion_experiment(
    _IDENTITY,
    EXPERIMENT_003.config,
    parent=EXPERIMENT_003,
    release_name="gemma-3-4b-it-vproj-maxrank",
    bit_multiplier=2.0,
)

__all__ = ["EXPERIMENT_005"]
