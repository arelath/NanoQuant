from pathlib import Path

from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment037_is_a_pre_kd_common_state_using_the_035_profile() -> None:
    baseline = load_experiment(35)
    candidate = load_experiment(37)

    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert not candidate.config.distillation.enabled
    assert not candidate.config.distillation.foldable_mlp_multipliers.enabled
    assert candidate.config.allocation.kl_profile_artifact == (
        "evidence/035/035-d2-uniform-control-kl-profile"
    )
    assert candidate.config.allocation.kl_profile_key == (
        "sha256:8878a1bcc5cf2301a0e5c1cc21b2691950017e4c0fab4d9ea3f1eddb2b6e5f21"
    )
    assert config_diff_paths(baseline.config, candidate.config) == {
        "allocation.kl_profile_artifact",
        "allocation.kl_profile_key",
        "distillation.enabled",
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
        "Results/037/gemma-3-1b-it-nanoquant.gguf"
    )
