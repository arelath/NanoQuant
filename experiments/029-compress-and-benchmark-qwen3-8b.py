"""Experiment 029: apply Experiment 028's settings to Qwen3 8B."""

from recipes import (
    QWEN_3_8B_COMPRESSION_TEMPLATE,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentIdentity,
    ExperimentRef,
    HuggingFaceUploadConfig,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

BASELINE = ExperimentRef(28, "compress-and-benchmark-qwen3-0-6b")

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=29,
        name="compress-and-benchmark-qwen3-8b",
        purpose=(
            "Apply Experiment 028's compression, adaptive execution, quality, and publication "
            "settings unchanged to the pinned Qwen3 8B model."
        ),
        hypothesis=(
            "Experiment 028's architecture-protected rank policy, 32-sample logical tuning "
            "batches, adaptive physical microbatches, and GPU activation caching transfer from "
            "Qwen3 0.6B to Qwen3 8B while preserving the bit budget, quality protocol, resume "
            "behavior, and export contracts."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "qwen3-8b",
            "compression",
            "quality",
            "experiment-028-settings",
            "cross-scale",
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
    QWEN_3_8B_COMPRESSION_TEMPLATE,
    expected_blocks=36,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend="dense",
    export=CompressionExportPolicy(
        release_name="qwen3-8b",
        runtime_family="qwen",
        huggingface=HuggingFaceUploadConfig(
            "Qwen3-8B-nanoquant-GGUF",
            private=False,
            commit_message="Publish NanoQuant Experiment 029",
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
