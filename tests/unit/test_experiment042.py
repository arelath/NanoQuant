from pathlib import Path

from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment042_is_the_fresh_canonical_experiment040_policy() -> None:
    baseline = load_experiment(37)
    candidate = load_experiment(42)

    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert candidate.config.distillation.enabled
    assert candidate.config.distillation.loss.value == "top_k"
    assert not candidate.config.distillation.foldable_mlp_multipliers.enabled
    correction = candidate.config.distillation.mass_floor_correction
    assert correction.enabled
    assert correction.epochs == 1
    assert correction.learning_rate == 1e-5
    assert correction.maximum_batches_per_epoch == 32
    assert correction.scheduler_total_steps == 128
    assert correction.minimum_teacher_mass_ratio == 0.8
    assert correction.mass_loss_weight == 2.0
    calibration = candidate.config.distillation.final_norm_calibration
    assert calibration.enabled
    assert calibration.scale == 1.015
    assert config_diff_paths(baseline.config, candidate.config) == {
        "distillation.enabled",
        "distillation.final_norm_calibration.enabled",
        "distillation.mass_floor_correction.enabled",
        "intent.baseline_run",
        "intent.experiment_number",
        "intent.hypothesis",
        "intent.name",
        "intent.purpose",
        "intent.tags",
        "output.run_root",
    }
    assert candidate.workflow.maximum_wddm_shared_gib == 0.75
    assert candidate.workflow.llamacpp_quality
    assert candidate.workflow.interrupt_after_block_commits is None
    assert candidate.workflow.export.gguf_output == Path(
        "Results/042/gemma-3-1b-it-nanoquant.gguf"
    )
