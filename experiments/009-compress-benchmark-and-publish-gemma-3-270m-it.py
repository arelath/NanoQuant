"""Experiment 009: compress, quality-benchmark, and publish pinned Gemma 3 270M."""

from recipes import (
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentIdentity,
    HuggingFaceUploadConfig,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=9,
        name="compress-benchmark-and-publish-gemma-3-270m-it",
        purpose=(
            "Compress pinned Gemma 3 270M, measure the complete BF16-versus-NanoQuant quality "
            "benchmark, and publish the validated deployment artifacts to Hugging Face."
        ),
        hypothesis=(
            "The promoted attention-rank recipe produces a publishable 270M NanoQuant GGUF while "
            "retaining measured quality across WikiText-2 and the common task suite."
        ),
        baseline=BaselineRef.external("bf16-unsloth-gemma-3-270m-it"),
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "huggingface",
            "gguf",
            "wikitext2",
            "ultrachat",
        ),
    ),
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    expected_blocks=18,
    maximum_wddm_shared_gib=0.75,
    export=CompressionExportPolicy(
        huggingface=HuggingFaceUploadConfig(
            "gemma-3-270m-it-nanoquant-GGUF",
            private=False,
            commit_message="Publish NanoQuant Experiment 009",
        ),
    ),
)


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
