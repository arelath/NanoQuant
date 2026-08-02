from pathlib import Path

from nanoquant.global_distillation import distillation_protocol_hash
from nanoquant.resident_workflow import primary_distillation_config_from_run_config
from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment048_freezes_one_exact_deployment_regime() -> None:
    baseline = load_experiment(44)
    candidate = load_experiment(48)

    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    distillation = candidate.config.distillation
    assert distillation.enabled
    assert distillation.loss.value == "top_k_tail"
    assert distillation.epochs == 8
    assert distillation.maximum_batches_per_epoch == 32
    assert distillation.tail_mass_weight == 0.5
    assert distillation_protocol_hash(
        primary_distillation_config_from_run_config(candidate.config)
    ) == "sha256:0ed7993a02eb980403ebeb97ff2d2cbf738242e64e6a7d07ad9f2900ef611936"

    correction = distillation.mass_floor_correction
    assert correction.enabled
    assert correction.expected_initializer_protocol_hash == (
        "sha256:0ed7993a02eb980403ebeb97ff2d2cbf738242e64e6a7d07ad9f2900ef611936"
    )
    assert correction.expected_initializer_steps == 256
    assert correction.epochs == 4
    assert correction.maximum_batches_per_epoch == 32
    assert correction.scheduler_total_steps == 128
    assert correction.learning_rate == 1e-5
    assert correction.minimum_teacher_mass_ratio == 0.8
    assert correction.mass_loss_weight == 2.0
    assert not distillation.final_norm_calibration.enabled
    assert not distillation.foldable_mlp_multipliers.enabled

    assert config_diff_paths(baseline.config, candidate.config) == {
        "distillation.mass_floor_correction.enabled",
        "distillation.mass_floor_correction.epochs",
        "distillation.mass_floor_correction.expected_initializer_protocol_hash",
        "distillation.mass_floor_correction.expected_initializer_steps",
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
    assert candidate.workflow.export.gguf_output == Path(
        "Results/048/gemma-3-1b-it-nanoquant.gguf"
    )
