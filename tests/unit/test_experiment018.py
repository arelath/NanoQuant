from pathlib import Path

from nanoquant.config.schema import ProfilingLevel
from tests.support.experiments import load_experiment


def test_experiment018_applies_experiment017_policy_with_gemma4b_guards_and_upload() -> None:
    experiment017 = load_experiment(17)
    experiment018 = load_experiment(18)
    config017 = experiment017.config
    config018 = experiment018.config
    workflow = experiment018.workflow

    assert config018.model.source == "google/gemma-3-4b-it"
    assert config018.model.revision == "093f9f388b31de276ce2de164bdc2081324b9767"
    assert config018.allocation == config017.allocation
    assert config018.factorization == config017.factorization
    assert config018.allocation.reconstruction.sensitivity_strength == 0.5
    assert config018.block_tuning.non_factorized.epochs_by_layer_position == (8, 4, 3, 6, 2)
    assert config018.runtime.block_forward_batch_size == 4
    assert config018.block_tuning.non_factorized.loop.batch_size == 4
    assert config018.block_tuning.factorized.loop.batch_size == 1
    assert config018.block_tuning.microbatch_size == 1
    assert config018.evaluation.inline_quality is False
    assert config018.profiling.level is ProfilingLevel.MACRO
    assert workflow.expected_blocks == 34
    assert workflow.maximum_wddm_shared_gib == 0.75
    assert workflow.restore_completed_blocks is False
    assert workflow.quality_backend == "dense"
    assert workflow.export.gguf_output == Path("Results/018/gemma-3-4b-it-nanoquant.gguf")
    assert config018.intent.baseline_run == "003-compress-and-benchmark-gemma-3-4b-it"
    upload = workflow.export.huggingface
    assert upload is not None
    assert upload.repo_id == "gemma-3-4b-it-nanoquant-GGUF"
    assert upload.private is False
    assert upload.commit_message == "Publish NanoQuant Experiment 018"
