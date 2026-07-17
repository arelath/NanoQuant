"""Experiment 007: compress and quality-benchmark pinned Gemma 3 270M."""

from recipes import (
    BASE_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    define_compression_quality_experiment,
)
from recipes._delta import config_delta

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

MODEL_REVISION = "23cf460f6bb16954176b3ddcc8d4f250501458a9"

_TEMPLATE = config_delta(
    BASE_COMPRESSION_TEMPLATE,
    model=config_delta(
        BASE_COMPRESSION_TEMPLATE.model,
        source="unsloth/gemma-3-270m-it",
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=7,
        name="compress-and-benchmark-gemma-3-270m-it",
        purpose=(
            "Establish a complete Gemma 3 270M compression and quality benchmark using the promoted "
            "attention-projection allocation policy."
        ),
        hypothesis=(
            "The full-rank v_proj/k_proj and enlarged q_proj recipe remains effective at 270M scale "
            "after complete tuning and distillation."
        ),
        baseline=BaselineRef.external("bf16-unsloth-gemma-3-270m-it"),
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "attention-rank",
            "wikitext2",
            "ultrachat",
        ),
    ),
    _TEMPLATE,
    expected_blocks=18,
    maximum_wddm_shared_gib=0.75,
)


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
