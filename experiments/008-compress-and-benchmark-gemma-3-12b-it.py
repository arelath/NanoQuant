"""Experiment 008: compress and quality-benchmark pinned Unsloth Gemma 3 12B."""

from recipes import EXPERIMENT_008, EXPERIMENT_008_CONFIG

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

CONFIG = EXPERIMENT_008_CONFIG
EXPERIMENT = EXPERIMENT_008


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(CONFIG, EXPERIMENT, launcher_path=__file__)
    )
