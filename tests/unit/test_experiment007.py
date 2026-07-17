from pathlib import Path

from nanoquant.config.codec import to_dict
from tests.support.experiments import load_experiment

_DEFINITION_006 = load_experiment(6)
_DEFINITION_007 = load_experiment(7)


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


def test_experiment007_is_the_270m_counterpart_to_experiment006() -> None:
    config = _DEFINITION_007.config
    previous = _DEFINITION_006.config
    experiment = _DEFINITION_007.workflow

    assert _diff(previous, config) == {
        "model.source",
        "model.revision",
        "model.tokenizer_revision",
        "intent.experiment_number",
        "intent.name",
        "intent.purpose",
        "intent.hypothesis",
        "intent.baseline_run",
        "intent.tags",
        "output.run_root",
    }
    assert config.model.source == "unsloth/gemma-3-270m-it"
    assert config.model.revision == "23cf460f6bb16954176b3ddcc8d4f250501458a9"
    assert config.allocation == previous.allocation
    assert config.dataset == previous.dataset
    assert experiment.expected_blocks == 18
    assert experiment.maximum_wddm_shared_gib == 0.75
    assert experiment.export.gguf_output == Path("Results/007/gemma-3-270m-it-nanoquant.gguf")
    assert experiment.wikitext_samples == 64
    assert len(experiment.task_names) == 6
    assert experiment.task_limit == 200
