"""Experiment 008: compress and quality-benchmark pinned Unsloth Gemma 3 12B."""

from recipes import (
    LARGE_MODEL_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    define_compression_quality_experiment,
)
from recipes._delta import config_delta

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

REQUESTED_GGUF_REPOSITORY = "unsloth/gemma-3-12b-it-GGUF"
REQUESTED_GGUF_REVISION = "d15e4c7dc21dc55d56bf8549db57a71ad8a2a35d"
REQUESTED_GGUF_FILENAME = "gemma-3-12b-it-BF16.gguf"
MODEL_SOURCE = "unsloth/gemma-3-12b-it"
MODEL_REVISION = "9478e665381f42974aa06177b019352fb6291876"

_TEMPLATE = config_delta(
    LARGE_MODEL_COMPRESSION_TEMPLATE,
    model=config_delta(
        LARGE_MODEL_COMPRESSION_TEMPLATE.model,
        source=MODEL_SOURCE,
        revision=MODEL_REVISION,
        tokenizer_source=MODEL_SOURCE,
        tokenizer_revision=MODEL_REVISION,
    ),
    block_tuning=config_delta(
        LARGE_MODEL_COMPRESSION_TEMPLATE.block_tuning,
        microbatch_size=1,
    ),
    runtime=config_delta(
        LARGE_MODEL_COMPRESSION_TEMPLATE.runtime,
        block_forward_batch_size=1,
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=8,
        name="compress-and-benchmark-gemma-3-12b-it",
        purpose=(
            "Compress and quality-benchmark the BF16 weights corresponding to "
            "unsloth/gemma-3-12b-it-GGUF without entering WDDM shared memory."
        ),
        hypothesis=(
            "CPU-offloaded block compression and packed evaluation can process the 48-block 12B model "
            "within 12 GiB of dedicated VRAM."
        ),
        baseline=BaselineRef.external(
            f"{REQUESTED_GGUF_REPOSITORY}@{REQUESTED_GGUF_REVISION}:{REQUESTED_GGUF_FILENAME}"
        ),
        tags=(
            "gemma-3-12b-it",
            "unsloth",
            "compression",
            "quality",
            "large-model",
            "cpu-offload",
            "packed-quality",
            "wikitext2",
            "ultrachat",
        ),
    ),
    _TEMPLATE,
    expected_blocks=48,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    large_model_guards=True,
)


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
