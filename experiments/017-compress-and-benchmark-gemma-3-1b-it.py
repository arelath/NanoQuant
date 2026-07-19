"""Experiment 017: architecture-protected reconstruction ranks for Gemma 3 1B."""

from dataclasses import replace

from recipes import (
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentIdentity,
    ExperimentRef,
    HuggingFaceUploadConfig,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

PARENT = ExperimentRef(12, "compress-and-benchmark-gemma-3-1b-it")

CONFIG = replace(
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    allocation=replace(
        ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE.allocation,
        reconstruction=replace(
            ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE.allocation.reconstruction,
            sensitivity_strength=0.5,
        ),
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=17,
        name="compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Apply the Experiment 016 architecture-protected stacked-QKV reconstruction policy "
            "to Gemma 3 1B with moderately tempered sensitivity."
        ),
        hypothesis=(
            "Sensitivity strength 0.5 preserves the Q/K/V/O/down and edge-block reconstruction "
            "advantages while reducing the gate/up quality tradeoff observed at strength 0.75."
        ),
        baseline=BaselineRef.experiment(PARENT),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "shared-input-qkv",
            "reconstruction-aware-ranks",
            "architecture-protected-ranks",
            "sensitivity-0.5",
            "down-projection-priority",
            "edge-block-protection",
            "huggingface",
            "gguf",
            "wikitext2",
            "ultrachat",
        ),
    ),
    CONFIG,
    expected_blocks=26,
    maximum_wddm_shared_gib=0.75,
    export=CompressionExportPolicy(
        huggingface=HuggingFaceUploadConfig(
            "gemma-3-1b-it-nanoquant-GGUF",
            private=False,
            commit_message="Publish NanoQuant Experiment 017",
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
