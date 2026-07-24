"""Experiment 019: transfer Experiment 018 compression to Llama 3.2 1B Instruct."""

from recipes import (
    LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentIdentity,
    HuggingFaceUploadConfig,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=19,
        name="compress-and-benchmark-llama-3-2-1b-instruct",
        purpose=(
            "Transfer Experiment 018's architecture-protected stacked-QKV reconstruction "
            "recipe unchanged to the Llama 3.2 1B Instruct architecture."
        ),
        hypothesis=(
            "The Experiment 018 policy transfers across the Llama decoder architecture while "
            "preserving its fixed bit budget, WDDM guard, tuning, quality, and export contracts."
        ),
        baseline=BaselineRef.none("first NanoQuant experiment for the Llama architecture"),
        tags=(
            "llama-3-2-1b-instruct",
            "compression",
            "quality",
            "cross-architecture",
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
    LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE,
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
            commit_message="Publish NanoQuant Experiment 019",
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
