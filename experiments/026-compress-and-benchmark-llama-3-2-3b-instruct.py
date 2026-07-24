"""Experiment 026: apply Experiment 025 unchanged to Llama 3.2 3B Instruct."""

from recipes import (
    LLAMA_3_2_3B_INSTRUCT_COMPRESSION_TEMPLATE,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentIdentity,
    ExperimentRef,
    HuggingFaceUploadConfig,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

BASELINE = ExperimentRef(25, "compress-and-benchmark-llama-3-2-1b-instruct")

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=26,
        name="compress-and-benchmark-llama-3-2-3b-instruct",
        purpose=(
            "Apply Experiment 025's complete compression, quality, and publication settings "
            "unchanged to the pinned Llama 3.2 3B Instruct model."
        ),
        hypothesis=(
            "Experiment 025's architecture-protected shared-QKV policy transfers from Llama "
            "3.2 1B to 3B while preserving its bit budget, quality protocol, bounded execution, "
            "resume behavior, and export contracts."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "llama-3-2-3b-instruct",
            "compression",
            "quality",
            "experiment-025-settings",
            "cross-scale",
            "shared-input-qkv",
            "reconstruction-aware-ranks",
            "architecture-protected-ranks",
            "sensitivity-0.5",
            "down-projection-priority",
            "edge-block-protection",
            "shared-vram-guard",
            "runpod",
            "huggingface",
            "gguf",
            "wikitext2",
            "ultrachat",
        ),
    ),
    LLAMA_3_2_3B_INSTRUCT_COMPRESSION_TEMPLATE,
    expected_blocks=28,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend="dense",
    export=CompressionExportPolicy(
        release_name="llama-3-2-3b-instruct",
        runtime_family="llama",
        huggingface=HuggingFaceUploadConfig(
            "Llama-3.2-3B-Instruct-nanoquant-GGUF",
            private=False,
            commit_message="Publish NanoQuant Experiment 026",
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
