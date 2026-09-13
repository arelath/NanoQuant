"""Check paired task alignment and accuracy-difference reporting."""

from copy import deepcopy

import pytest
from render_downloaded_gguf_profile import paired_task_comparison, verify_deployment_protocol


def task(correct, metric="acc_norm"):
    return {"primary_metric": metric, "examples": [
        {"sample_id": f"sample:{index}", "correct_choice": index % 2,
         "normalized_correct": value, "raw_correct": not value}
        for index, value in enumerate(correct)
    ]}


def test_paired_comparison_uses_primary_metric_and_discordant_pairs():
    left, right = task([True, True, False, True]), task([False, True, True, False])
    result = paired_task_comparison(left, right)
    assert result["external_only_correct"] == 2
    assert result["nanoquant_only_correct"] == 1
    assert result["accuracy_delta"] == 0.25
    left["primary_metric"] = right["primary_metric"] = "acc"
    assert paired_task_comparison(left, right)["accuracy_delta"] == -0.25


def test_identical_paired_results_have_zero_difference_and_interval():
    result = paired_task_comparison(task([True, False, True]), task([True, False, True]))
    assert result["accuracy_delta"] == 0
    assert result["paired_bootstrap_95pct_interval"] == [0, 0]


def test_misaligned_examples_and_metrics_fail_closed():
    left, right = task([True, True]), task([True, False])
    right["examples"][0]["sample_id"] = "different"
    with pytest.raises(ValueError, match="aligned"):
        paired_task_comparison(left, right)
    with pytest.raises(ValueError, match="metrics"):
        paired_task_comparison(task([True]), task([True], metric="acc"))


def deployment_receipt():
    return {"identity": dict.fromkeys(("input_sha256", "runner_sha256", "runtime_sha256", "scoring",
                                      "llama_cpp_commit", "gpu_layers", "parallel", "threads", "batch_threads"),
                                     "same"),
            "protocol": {"wikitext_fingerprint": "old", "task_limit": 1000}}


def test_cache_fingerprint_change_is_allowed_with_identical_scorer_inputs():
    left = deployment_receipt()
    right = deepcopy(left)
    right["protocol"]["wikitext_fingerprint"] = "new"
    verify_deployment_protocol(left, right)


@pytest.mark.parametrize("field", ["input_sha256", "runtime_sha256", "parallel"])
def test_scorer_identity_changes_cannot_hide_behind_cache_fingerprint_exception(field):
    left = deployment_receipt()
    right = deepcopy(left)
    right["identity"][field] = "different"
    with pytest.raises(ValueError, match="identity"):
        verify_deployment_protocol(left, right)


def test_semantic_protocol_changes_fail_even_when_serialized_input_hash_matches():
    left = deployment_receipt()
    right = deepcopy(left)
    right["protocol"]["task_limit"] = 500
    with pytest.raises(ValueError, match="semantic"):
        verify_deployment_protocol(left, right)
