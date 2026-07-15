"""Experiment 011: packed-runtime generation throughput benchmark."""

from nanoquant.benchmark_workflow import run_runtime_benchmark_experiment
from nanoquant.recipes import EXPERIMENT_011_BENCHMARK, EXPERIMENT_011_CONFIG

CONFIG = EXPERIMENT_011_CONFIG
BENCHMARK = EXPERIMENT_011_BENCHMARK


if __name__ == "__main__":
    raise SystemExit(
        run_runtime_benchmark_experiment(CONFIG, BENCHMARK, launcher_path=__file__)
    )
