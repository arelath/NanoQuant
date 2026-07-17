"""Experiment 003: compress and quality-benchmark pinned Gemma 3 4B."""

from recipes import (
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=3,
        name="compress-and-benchmark-gemma-3-4b-it",
        purpose=(
            "Prove that the multimodal Gemma 3 4B checkpoint's text model still compresses "
            "within dedicated VRAM and measure its BF16-versus-NanoQuant quality."
        ),
        hypothesis=(
            "CPU extraction of the language model plus pageable activation streaming keeps WDDM "
            "shared memory below the hard limit while producing a complete 34-block candidate."
        ),
        baseline=BaselineRef.external("bf16-google-gemma-3-4b-it"),
        tags=("gemma-3-4b-it", "compression", "quality", "shared-vram-guard", "profiling"),
    ),
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    expected_blocks=34,
    maximum_wddm_shared_gib=0.75,
    restore_completed_blocks=False,
    quality_backend="dense",
)


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
