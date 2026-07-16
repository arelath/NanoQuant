import runpy
from pathlib import Path

from recipes import BASE_COMPRESSION_CONFIG, EXPERIMENT_006, EXPERIMENT_006_CONFIG

from nanoquant.config.codec import to_dict


def _diff(left: object, right: object, prefix: str = "") -> set[str]:
    left = to_dict(left)
    right = to_dict(right)
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else key
            paths.update(_diff(left.get(key), right.get(key), path))
        return paths
    return set() if left == right else {prefix}


def test_experiment006_is_the_new_gemma1b_attention_rank_quality_baseline() -> None:
    config = EXPERIMENT_006_CONFIG

    assert _diff(BASE_COMPRESSION_CONFIG, config) == {
        "intent.experiment_number",
        "intent.name",
        "intent.purpose",
        "intent.hypothesis",
        "intent.baseline_run",
        "intent.tags",
        "output.run_root",
    }
    assert config.model.source == "google/gemma-3-1b-it"
    assert config.allocation.maximum_rank_layer_patterns == (
        "self_attn.v_proj",
        "self_attn.k_proj",
    )
    assert tuple(
        (item.pattern, item.multiplier) for item in config.allocation.layer_budget_multipliers
    ) == (("self_attn.q_proj", 1.25),)
    assert tuple(source.name for source in config.dataset.sources) == (
        "HuggingFaceH4/ultrachat_200k",
        "Salesforce/wikitext",
    )
    assert EXPERIMENT_006.expected_blocks == 26
    assert EXPERIMENT_006.maximum_wddm_shared_gib == 0.75
    assert EXPERIMENT_006.export.gguf_output == Path(
        "outputs/006-gemma-3-1b-it/gemma-3-1b-it-nanoquant.gguf"
    )
    assert EXPERIMENT_006.wikitext_samples == 64
    assert len(EXPERIMENT_006.task_names) == 6
    assert EXPERIMENT_006.task_limit == 200


def test_experiment006_runfile_imports_canonical_recipe() -> None:
    namespace = runpy.run_path("experiments/006-compress-and-benchmark-gemma-3-1b-it.py")

    assert namespace["CONFIG"] is EXPERIMENT_006_CONFIG
    assert namespace["EXPERIMENT"] is EXPERIMENT_006
