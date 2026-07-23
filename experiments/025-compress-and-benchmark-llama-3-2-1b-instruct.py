"""Experiment 025: rerun Llama 3.2 1B Instruct on the current NanoQuant pipeline."""

from dataclasses import replace

from recipes import (
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentIdentity,
    ExperimentRef,
    HuggingFaceUploadConfig,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

BASELINE = ExperimentRef(19, "compress-and-benchmark-llama-3-2-1b-instruct")

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
    LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE,
    allocation=POLICY_CONFIG.allocation,
    factorization=POLICY_CONFIG.factorization,
    block_tuning=replace(
        LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE.block_tuning,
        non_factorized=replace(
            LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE.block_tuning.non_factorized,
            epochs_by_layer_position=(
                POLICY_CONFIG.block_tuning.non_factorized.epochs_by_layer_position
            ),
        ),
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=25,
        name="compress-and-benchmark-llama-3-2-1b-instruct",
        purpose=(
            "Re-run Experiment 019's pinned Llama 3.2 1B Instruct compression and benchmark "
            "recipe on the current NanoQuant implementation and publish its validated GGUF."
        ),
        hypothesis=(
            "The current pipeline reproduces or improves Experiment 019 quality at the same "
            "effective bit budget while preserving bounded memory, resume, packed-runtime, "
            "benchmark, and export contracts."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "llama-3-2-1b-instruct",
            "compression",
            "quality",
            "experiment-019-replication",
            "cross-architecture",
            "shared-input-qkv",
            "reconstruction-aware-ranks",
            "architecture-protected-ranks",
            "sensitivity-0.5",
            "down-projection-priority",
            "edge-block-protection",
            "shared-vram-guard",
            "runpod-default",
            "huggingface",
            "gguf",
            "wikitext2",
            "ultrachat",
        ),
    ),
    CONFIG,
    expected_blocks=16,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend="dense",
    export=CompressionExportPolicy(
        release_name="llama-3-2-1b-instruct",
        runtime_family="llama",
        huggingface=HuggingFaceUploadConfig(
            "Llama-3.2-1B-Instruct-nanoquant-GGUF",
            private=False,
            commit_message="Publish NanoQuant Experiment 025",
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
