"""Experiment 006: Gemma 3 1B attention-rank compression quality baseline."""

from ._experiment import (
    BaselineRef,
    ExperimentIdentity,
    define_compression_quality_experiment,
)
from .base_compression import BASE_COMPRESSION_TEMPLATE

EXPERIMENT_006 = define_compression_quality_experiment(
    ExperimentIdentity(
        number=6,
        name="compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Establish a complete Gemma 3 1B compression and quality baseline for the full-rank "
            "v_proj/k_proj and enlarged q_proj allocation policy."
        ),
        hypothesis=(
            "Additional attention-projection capacity lowers reconstruction error and improves matched "
            "WikiText-2 and task quality after the complete tuning and distillation pipeline."
        ),
        baseline=BaselineRef.external("bf16-google-gemma-3-1b-it"),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "attention-rank",
            "wikitext2",
            "ultrachat",
        ),
    ),
    BASE_COMPRESSION_TEMPLATE,
    expected_blocks=26,
    maximum_wddm_shared_gib=0.75,
)

__all__ = ["EXPERIMENT_006"]
