"""Experiment 018: architecture-protected reconstruction ranks for Gemma 3 4B."""

from dataclasses import replace

from recipes import (
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentIdentity,
    ExperimentRef,
    HuggingFaceUploadConfig,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

PARENT = ExperimentRef(3, "compress-and-benchmark-gemma-3-4b-it")

POLICY_CONFIG = replace(
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    allocation=replace(
        ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE.allocation,
        reconstruction=replace(
            ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE.allocation.reconstruction,
            sensitivity_strength=0.5,
        ),
    ),
)

CONFIG = replace(
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    allocation=POLICY_CONFIG.allocation,
    factorization=POLICY_CONFIG.factorization,
    block_tuning=replace(
        GEMMA_3_4B_COMPRESSION_TEMPLATE.block_tuning,
        non_factorized=replace(
            GEMMA_3_4B_COMPRESSION_TEMPLATE.block_tuning.non_factorized,
            epochs_by_layer_position=(
                POLICY_CONFIG.block_tuning.non_factorized.epochs_by_layer_position
            ),
        ),
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=18,
        name="compress-and-benchmark-gemma-3-4b-it",
        purpose=(
            "Apply the architecture-protected stacked-QKV reconstruction policy at sensitivity "
            "0.5 to Gemma 3 4B while retaining its bounded-memory execution recipe."
        ),
        hypothesis=(
            "The moderate sensitivity policy transfers to Gemma 3 4B, reduces Q/K/V/O/down and "
            "edge-block reconstruction error, and remains within the fixed bit and WDDM budgets."
        ),
        baseline=BaselineRef.experiment(PARENT),
        tags=(
            "gemma-3-4b-it",
            "compression",
            "quality",
            "shared-input-qkv",
            "reconstruction-aware-ranks",
            "architecture-protected-ranks",
            "sensitivity-0.5",
            "down-projection-priority",
            "edge-block-protection",
            "shared-vram-guard",
            "huggingface",
            "gguf",
            "wikitext2",
            "ultrachat",
        ),
    ),
    CONFIG,
    expected_blocks=34,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend="dense",
    export=CompressionExportPolicy(
        huggingface=HuggingFaceUploadConfig(
            "gemma-3-4b-it-nanoquant-GGUF",
            private=False,
            commit_message="Publish NanoQuant Experiment 018",
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
