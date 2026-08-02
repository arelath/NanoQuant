from pathlib import Path

from tests.support.experiments import load_experiment


def test_experiment044_freezes_the_matched_tail_aware_deployment_regime() -> None:
    baseline = load_experiment(42)
    candidate = load_experiment(44)

    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    distillation = candidate.config.distillation
    assert distillation.enabled
    assert distillation.loss.value == "top_k_tail"
    assert distillation.epochs == 8
    assert distillation.maximum_batches_per_epoch == 32
    assert distillation.epochs * distillation.maximum_batches_per_epoch == 256
    assert distillation.tail_mass_weight == 0.5
    assert not distillation.mass_floor_correction.enabled
    assert not distillation.final_norm_calibration.enabled
    assert not distillation.foldable_mlp_multipliers.enabled
    assert candidate.workflow.maximum_wddm_shared_gib == 0.75
    assert candidate.workflow.llamacpp_quality
    assert candidate.workflow.export.gguf_output == Path(
        "Results/044/gemma-3-1b-it-nanoquant.gguf"
    )
