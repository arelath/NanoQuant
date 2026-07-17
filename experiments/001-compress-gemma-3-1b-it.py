"""Experiment 001: compress, export, and benchmark pinned Gemma 3 1B."""

from recipes import (
    GEMMA_3_1B_PARITY_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    define_compression_benchmark_experiment,
)

from nanoquant.compression_benchmark_workflow import run_compression_benchmark_experiment

EXPERIMENT = define_compression_benchmark_experiment(
    ExperimentIdentity(
        number=1,
        name="compress-gemma-3-1b-it",
        purpose=(
            "Compress pinned Gemma 3 1B with the current parity recipe, export a deployable "
            "NanoQuant GGUF, and compare it with the BF16 source model."
        ),
        hypothesis=(
            "The current resident recipe produces a validated GGUF while retaining measured "
            "quality parity on WikiText-2 and the six-task evaluation suite."
        ),
        baseline=BaselineRef.external("bf16-google-gemma-3-1b-it"),
        tags=("gemma-3-1b-it", "compression", "gguf", "bf16-comparison", "quality"),
    ),
    GEMMA_3_1B_PARITY_TEMPLATE,
)


if __name__ == "__main__":
    raise SystemExit(
        run_compression_benchmark_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
