from pathlib import Path

from recipes import BASE_COMPRESSION_TEMPLATE

from nanoquant.config.codec import to_dict
from tests.support.experiments import load_experiment

_DEFINITION = load_experiment(6)


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
    config = _DEFINITION.config
    experiment = _DEFINITION.workflow

    assert _diff(BASE_COMPRESSION_TEMPLATE, config) == {
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
    assert experiment.expected_blocks == 26
    assert experiment.maximum_wddm_shared_gib == 0.75
    assert experiment.export.gguf_output == Path("outputs/006/gemma-3-1b-it-nanoquant.gguf")
    assert experiment.wikitext_samples == 64
    assert len(experiment.task_names) == 6
    assert experiment.task_limit == 200
