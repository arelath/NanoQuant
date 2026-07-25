"""Experiment 028: apply Experiment 027's settings to Qwen3 0.6B."""

from recipes import (
    QWEN_3_0_6B_COMPRESSION_TEMPLATE,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentIdentity,
    ExperimentRef,
    HuggingFaceUploadConfig,
    define_compression_quality_experiment,
    experiment_main,
)

BASELINE = ExperimentRef(27, "compress-and-benchmark-meta-llama-3-8b-instruct")

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=28,
        name="compress-and-benchmark-qwen3-0-6b",
        purpose=(
            "Apply Experiment 027's compression, adaptive execution, quality, and publication "
            "settings unchanged to the pinned Qwen3 0.6B model."
        ),
        hypothesis=(
            "Experiment 027's architecture-protected rank policy, 32-sample logical tuning "
            "batches, adaptive physical microbatches, and GPU activation caching transfer to "
            "Qwen3 0.6B while preserving the bit budget, quality protocol, resume behavior, "
            "and export contracts."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "qwen3-0-6b",
            "compression",
            "quality",
            "experiment-027-settings",
            "cross-architecture",
            "adaptive-memory",
            "logical-batch-32",
            "throughput-memory-profile",
            "activation-gpu-cache-auto",
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
    QWEN_3_0_6B_COMPRESSION_TEMPLATE,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend=None,
    llamacpp_quality=True,
    export=CompressionExportPolicy(
        release_name="qwen3-0-6b",
        runtime_family="qwen",
        huggingface=HuggingFaceUploadConfig(
            "Qwen3-0.6B-nanoquant-GGUF",
            private=False,
            commit_message="Publish NanoQuant Experiment 028",
        ),
    ),
)


experiment_main(__name__, __file__, EXPERIMENT)
