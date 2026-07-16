"""Canonical recipe for legacy Experiment 011's generation-throughput workload."""

from pathlib import Path

from nanoquant.benchmark_workflow import RuntimeBenchmarkExperiment
from nanoquant.config.schema import (
    EvaluationConfig,
    IntentConfig,
    ModelConfig,
    RunConfig,
    RuntimeConfig,
)
from nanoquant.runtime_benchmark import RuntimeBenchmarkRequest

MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"

EXPERIMENT_011_CONFIG = RunConfig(
    model=ModelConfig(
        source="google/gemma-3-1b-it",
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        sequence_length=512,
    ),
    intent=IntentConfig(
        experiment_number=11,
        name="011-benchmark-generation-tps",
        purpose="Measure generation-only throughput after load, preparation, tokenization, and warmup.",
        hypothesis="The production packed runtime replaces the legacy mutable GEMV path at the same workload.",
        baseline_run="legacy-experiment-011",
        tags=("gemma-3-1b-it", "runtime", "generation", "throughput"),
    ),
    runtime=RuntimeConfig(compute_device="cuda:0"),
    evaluation=EvaluationConfig(suites=("runtime-generation-v1",)),
)

EXPERIMENT_011_BENCHMARK = RuntimeBenchmarkExperiment(
    RuntimeBenchmarkRequest(
        packed_artifact=Path("evidence/m6/gemma-pageable-v28-runtime-bundle/packed"),
        # The material snapshot is resolved from the pinned model/revision above.
        model=Path("google/gemma-3-1b-it"),
        run_output=Path("evidence/m4/gemma-pageable-v28-four-block-canary"),
        expected_blocks=26,
        device="cuda:0",
        input_dtype="bfloat16",
        cache_dtype="bfloat16",
        suite=("end-to-end",),
        warmups=1,
        repetitions=3,
        max_new_tokens=128,
        stopping_check_interval=8,
        chat_template=False,
        ignore_eos=True,
        prompt=("Explain why compact language models are useful for local inference.",),
    ),
    Path("evidence/m9/011-generation-tps.json"),
    resolve_model_from_config=True,
)

__all__ = ["EXPERIMENT_011_BENCHMARK", "EXPERIMENT_011_CONFIG", "MODEL_REVISION"]
