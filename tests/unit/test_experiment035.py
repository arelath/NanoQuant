from pathlib import Path

from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment035_adds_only_identity_and_foldable_mlp_stage_to_022() -> None:
    baseline = load_experiment(22)
    candidate = load_experiment(35)
    tuning = candidate.config.distillation.foldable_mlp_multipliers

    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert tuning.enabled
    assert tuning.steps == 64
    assert tuning.learning_rate == 1e-4
    assert tuning.identity_penalty == 100.0
    assert tuning.gradient_clip == 1.0
    assert tuning.multiplier_limit == 4.0
    assert tuning.checkpoint_interval_steps == 16
    assert not tuning.gradient_checkpointing
    assert config_diff_paths(baseline.config, candidate.config) == {
        "distillation.foldable_mlp_multipliers.enabled",
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
        "Results/035/gemma-3-1b-it-nanoquant.gguf"
    )
    assert candidate.workflow.wikitext_samples == 64
    assert candidate.workflow.task_limit == 200
