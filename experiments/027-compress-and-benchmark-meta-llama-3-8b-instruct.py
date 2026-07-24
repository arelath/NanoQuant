"""Experiment 027: apply Experiment 025 numerics with adaptive 8B execution."""

from recipes import (
    META_LLAMA_3_8B_INSTRUCT_COMPRESSION_TEMPLATE,
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
        number=27,
        name="compress-and-benchmark-meta-llama-3-8b-instruct",
        purpose=(
            "Apply Experiment 025's numerical compression, quality, and publication settings "
            "to the pinned Meta Llama 3 8B Instruct model while adapting physical batches and "
            "activation placement to the available CUDA memory."
        ),
        hypothesis=(
            "Experiment 025's architecture-protected shared-QKV policy transfers from Llama "
            "3.2 1B to Meta Llama 3 8B while adaptive throughput sizing and GPU activation "
            "caching improve device utilization without changing logical optimizer batches, "
            "the bit budget, quality protocol, resume behavior, or export contracts."
        ),
        baseline=BaselineRef.experiment(BASELINE),
        tags=(
            "meta-llama-3-8b-instruct",
            "compression",
            "quality",
            "experiment-025-numerics",
            "cross-scale",
            "cross-generation",
            "adaptive-memory",
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
    META_LLAMA_3_8B_INSTRUCT_COMPRESSION_TEMPLATE,
    expected_blocks=32,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend="dense",
    export=CompressionExportPolicy(
        release_name="meta-llama-3-8b-instruct",
        runtime_family="llama",
        huggingface=HuggingFaceUploadConfig(
            "Meta-Llama-3-8B-Instruct-nanoquant-GGUF",
            private=False,
            commit_message="Publish NanoQuant Experiment 027",
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
