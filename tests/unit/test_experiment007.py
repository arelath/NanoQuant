import runpy
from pathlib import Path

from recipes import EXPERIMENT_006_CONFIG, EXPERIMENT_007, EXPERIMENT_007_CONFIG

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


def test_experiment007_is_the_270m_counterpart_to_experiment006() -> None:
    config = EXPERIMENT_007_CONFIG

    assert _diff(EXPERIMENT_006_CONFIG, config) == {
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
    assert config.allocation == EXPERIMENT_006_CONFIG.allocation
    assert config.dataset == EXPERIMENT_006_CONFIG.dataset
    assert EXPERIMENT_007.expected_blocks == 18
    assert EXPERIMENT_007.maximum_wddm_shared_gib == 0.75
    assert EXPERIMENT_007.export.gguf_output == Path(
        "outputs/007-gemma-3-270m-it/gemma-3-270m-it-nanoquant.gguf"
    )
    assert EXPERIMENT_007.wikitext_samples == 64
    assert len(EXPERIMENT_007.task_names) == 6
    assert EXPERIMENT_007.task_limit == 200


def test_experiment007_runfile_imports_canonical_recipe() -> None:
    namespace = runpy.run_path("experiments/007-compress-and-benchmark-gemma-3-270m-it.py")

    assert namespace["CONFIG"] is EXPERIMENT_007_CONFIG
    assert namespace["EXPERIMENT"] is EXPERIMENT_007
