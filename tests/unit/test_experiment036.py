from pathlib import Path

from nanoquant.infrastructure.io_utils import hash_file
from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment036_adds_only_the_pinned_initializer_to_035() -> None:
    baseline = load_experiment(35)
    candidate = load_experiment(36)
    tuning = candidate.config.distillation.foldable_mlp_multipliers
    initializer = Path(str(tuning.initializer_artifact))

    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert tuning.initializer_sha256 == hash_file(initializer / "multipliers.safetensors")
    assert tuning.initializer_multiplier_limit == 128.0
    assert config_diff_paths(baseline.config, candidate.config) == {
        "distillation.foldable_mlp_multipliers.initializer_artifact",
        "distillation.foldable_mlp_multipliers.initializer_sha256",
        "intent.baseline_run",
        "intent.experiment_number",
        "intent.hypothesis",
        "intent.name",
        "intent.purpose",
        "intent.tags",
        "output.run_root",
    }
    assert candidate.workflow.maximum_wddm_shared_gib == 0.75
    assert candidate.workflow.export.gguf_output == Path(
        "Results/036/gemma-3-1b-it-nanoquant.gguf"
    )
    assert candidate.workflow.wikitext_samples == 64
    assert candidate.workflow.task_limit == 200
